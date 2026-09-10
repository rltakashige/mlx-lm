# Copyright © 2026 Apple Inc.

"""Expert-grouped small-M gather matvec for MoE decode: the gather variant of ``qmv_small``.

The (token, slot) pairs of a step are token-major, ``P = M * (top_k + 1)`` pairs, and
slot ``top_k`` is the shared expert stored as the last expert of the weights. The prep
kernel of x also builds a plan: for each distinct expert its id, its row count and up to
M pair indices. The gather kernel runs one threadgroup per (plan slot, N tile), so the
rows that share an expert read and dequantize its weights once. The down projection
scales each row by its routing score in the epilogue (softmax over the token's top-k
logits, sigmoid of the shared gate logit) and the block output is the sum over the slots.
"""

import math

import mlx.core as mx
import mlx.nn as nn

from .fused_ops import enabled
from .qmv_small import (
    _HEADER,
    _UNROLL,
    _VPL,
    _kernel,
    _m5,
    _mp,
    _prep_segments,
    _prep_source,
    _scan_threads,
    _tag,
)
from .switch_layers import QuantizedSwitchLinear

_MAX_M = 8
_KSTEP = 32 * _VPL
_routes = {}
_calls = {}


def _config(K):
    """(rows per simdgroup, simdgroups per threadgroup), measured on M5 for K = 2048 and 512."""
    return (4, 2) if K >= 2048 else (4, 4)


def routes(block, x):
    """True when ``experts`` runs this step: M5, 4-bit g64 affine experts, 1 <= M <= 8."""
    projs = (block.switch_mlp.gate_up_proj, block.switch_mlp.down_proj)
    key = (id(block), type(projs[0]), x.shape, x.dtype)
    use = _routes.get(key)
    if use is None:
        use = _routes[key] = _supported(block, projs, x)
    return use


def _supported(block, projs, x):
    *batch, k = x.shape
    m = math.prod(batch)
    if not (
        _m5()
        and block.norm_topk_prob
        and 1 <= m <= _MAX_M
        and x.dtype in (mx.bfloat16, mx.float16)
    ):
        return False
    for p in projs:
        if (
            not isinstance(p, QuantizedSwitchLinear)
            or p.bits != 4
            or p.group_size != 64
            or p.mode != "affine"
            or "bias" in p
            or p.scales.dtype != x.dtype
            or p.input_dims % _KSTEP
            or p.output_dims % (math.prod(_config(p.input_dims)))
        ):
            return False
    return projs[0].input_dims == k


def _plan_source(M, S, TOPK, ESHARED):
    """Appended to the prep kernel: threadgroup 0 builds the plan.

    Pair t is a "first" when no earlier pair has its expert. Slot d of the plan holds
    [expert, count, pair_0 .. pair_{M-1}] for the d-th first pair; later slots count 0.
    """
    return f"""
    {{
      constexpr int PT = {M} * {S}, PW = {M} + 2;
      threadgroup uint ex[PT];
      threadgroup int first[PT];
      if (threadgroup_position_in_grid.x == 0) {{
        if (t < PT) ex[t] = (t % {S} < {TOPK}) ? uint(inds[(t / {S}) * {TOPK} + t % {S}]) : uint({ESHARED});
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (t < PT) {{
          int f = 1;
          for (int q = 0; q < t; q++) f &= (ex[q] != ex[t]);
          first[t] = f;
        }}
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (t < PT) {{
          int d = 0, D = 0;
          for (int q = 0; q < PT; q++) {{
            D += first[q];
            d += (q < t) ? first[q] : 0;
          }}
          if (first[t]) {{
            const uint e = ex[t];
            int c = 0;
            for (int q = t; q < PT; q++) if (ex[q] == e) {{ if (c < {M}) plan[d * PW + 2 + c] = q; c++; }}
            plan[d * PW] = int(e);
            plan[d * PW + 1] = c;
          }}
          if (t >= D) plan[t * PW + 1] = 0;
        }}
      }}
    }}
"""


def _prep(x, kind, inds=None, m=0, top_k=0, eshared=0):
    """``qmv_small``'s prep of (rows, K) x; with ``inds`` also the plan of the m tokens' pairs."""
    M, K = x.shape
    if kind == "swiglu":
        K //= 2
    Mp = _mp(M)
    S = top_k + 1
    plan = inds is not None

    def source():
        src = _prep_source(K, M, kind)
        return src + _plan_source(m, S, top_k, eshared) if plan else src

    key = ("prep", kind, plan, K, M, _tag(x.dtype), m, top_k, eshared)
    call = _calls.get(key)
    if call is None:
        kern = _kernel(
            "moe_small_prep_" + kind + ("_plan" if plan else ""),
            key[3:],
            source,
            ["x"] + (["inds"] if plan else []),
            ["x16", "xsum", "rscale"] + (["plan"] if plan else []),
            _HEADER,
        )
        kwargs = dict(
            template=[("T", x.dtype)],
            grid=(_scan_threads(K) * _prep_segments(K) * M, 1, 1),
            threadgroup=(_scan_threads(K), 1, 1),
            output_shapes=[(M, K), (K // 16, Mp), (Mp,)]
            + ([(m * S, m + 2)] if plan else []),
            output_dtypes=[mx.float16, mx.float32, mx.float32]
            + ([mx.int32] if plan else []),
        )
        call = _calls[key] = (kern, kwargs)
    kern, kwargs = call
    return kern(inputs=[x] + ([inds] if plan else []), **kwargs)


def _body(c, R, MC):
    """The main loop over K for ``c`` gathered rows (the ``qmv_small`` loop with per-row x)."""
    wload = "\n".join(
        f"        {{dst}}[{r}] = *(const device uint2*)(wp + {r} * KW);"
        for r in range(R)
    )
    deq = []
    for r in range(R):
        for wi in range(2):
            wd = f"wv[{r}][{wi}]"
            deq.append(f"        {{ const uint lo = {wd}, hi = {wd} >> 8;")
            for j, (src, mask) in enumerate(
                (
                    ("lo", "0x000F000Fu"),
                    ("lo", "0x00F000F0u"),
                    ("hi", "0x000F000Fu"),
                    ("hi", "0x00F000F0u"),
                )
            ):
                deq.append(
                    f"          q2[{r}][{wi * 4 + j}] = as_type<half2>(({src} & {mask}) | 0x64006400u) - half2(1024.0h);"
                )
            deq[-1] += " }"
    chunks = []
    for m0 in range(0, c, MC):
        rows = range(m0, min(m0 + MC, c))
        xl = "\n".join(
            f"        xv[{m - m0}][{cc}] = *(const device uint4*)(xp + xo[{m}] + {cc} * 8);"
            for m in rows
            for cc in range(2)
        )
        xs = "\n".join(f"        xs[{m - m0}] = xsp[xrow[{m}]];" for m in rows)
        fm = []
        for r in range(R):
            for m in rows:
                mm = m - m0
                fm.append(
                    f"        {{ half2 p = q2[{r}][0] * x2({mm}, 0);\n"
                    + "\n".join(
                        f"          p = fma(q2[{r}][{j}], x2({mm}, {j}), p);"
                        for j in range(1, 8)
                    )
                    + f"\n          acc[{r}][{m}] = fma(s[{r}], float(p.x + p.y), fma(bb[{r}], xs[{mm}], acc[{r}][{m}])); }}"
                )
        chunks.append(xl + "\n" + xs + "\n" + "\n".join(fm))
    store = "\n".join(
        f"        {{ const float sc = rscale[xrow[{m}]] * score(prow[{m}]);\n"
        + "\n".join(
            f"          y[(size_t)prow[{m}] * N + row0 + {r}] = T(acc[{r}][{m}] * sc);"
            for r in range(R)
        )
        + " }"
        for m in range(c)
    )
    nl = "\n"
    return f"""
{wload.format(dst="wv")}
      for (int b = 0; b < NB; b++) {{
        // Request the next step's weights before this step's math.
        wp += 32 * VPL / 8;
        if (b + 1 < NB) {{
{wload.format(dst="wn")}
        }}
{nl.join(f"        s[{r}] = float(sp[{r} * KG]); bb[{r}] = float(bp[{r} * KG]);" for r in range(R))}
{nl.join(deq)}
{nl.join(chunks)}
        sp += 32 * VPL / 64;
        bp += 32 * VPL / 64;
        xp += 32 * VPL;
        xsp += 32 * Mp;
{nl.join(f"        wv[{r}] = wn[{r}];" for r in range(R))}
      }}
{nl.join(f"      acc[{r}][{m}] = simd_sum(acc[{r}][{m}]);" for r in range(R) for m in range(c))}
      if (lane == 0) {{
{store}
      }}
"""


def _score_source(LW):
    """Routing score of pair q from the logits (width LW, the shared gate last), or 1."""
    if not LW:
        return "    #define score(q) 1.0f\n"
    return f"""
    auto score = [&](int q) -> float {{
      const int t = q / S, sl = q % S;
      const device T* lg = logits + t * {LW};
      if (sl >= TOPK) return 1.0f / (1.0f + metal::exp(-float(lg[{LW} - 1])));
      float mx_ = -1e30f;
      for (int j = 0; j < TOPK; j++) mx_ = max(mx_, float(lg[inds[t * TOPK + j]]));
      float den = 0.0f;
      for (int j = 0; j < TOPK; j++) den += metal::exp(float(lg[inds[t * TOPK + j]]) - mx_);
      return metal::exp(float(lg[inds[t * TOPK + sl]]) - mx_) / den;
    }};
"""


def _gather_source(M, N, K, R, NSG, MC, Mp, S, TOPK, RDIV, LW):
    NB = K // _KSTEP
    cases = "\n".join(
        f"      case {c}: {{{_body(c, R, MC)}      break; }}" for c in range(1, M + 1)
    )
    return f"""
    constexpr int M = {M}, N = {N}, K = {K}, R = {R}, NSG = {NSG}, VPL = {_VPL}, Mp = {Mp}, MC = {MC};
    constexpr int TOPK = {TOPK}, S = {S}, RDIV = {RDIV}, PW = M + 2;
    constexpr int KW = K / 8, KG = K / 64, NB = {NB}, NTILE = N / (R * NSG);
    const int tg = threadgroup_position_in_grid.x;
    const device int* pl = plan + (tg / NTILE) * PW;
    const int c = pl[1];
    if (c == 0) return;
    const uint e = uint(pl[0]);
    int prow[M], xrow[M];
    {_UNROLL}
    for (int m = 0; m < M; m++) {{ prow[m] = pl[2 + min(m, c - 1)]; xrow[m] = prow[m] / RDIV; }}
    const int lane = thread_index_in_simdgroup;
    const int sg = simdgroup_index_in_threadgroup;
    const int row0 = ((tg % NTILE) * NSG + sg) * R;
    const device uint32_t* wp = w + ((size_t)e * N + row0) * KW + lane * 2;
    const device T* sp = scales + ((size_t)e * N + row0) * KG + lane / 4;
    const device T* bp = biases + ((size_t)e * N + row0) * KG + lane / 4;
    const device half* xp = x16 + lane * VPL;
    const device float* xsp = xsum + (size_t)lane * Mp;
    size_t xo[M];
    {_UNROLL}
    for (int m = 0; m < M; m++) xo[m] = (size_t)xrow[m] * K;
    #define x2(m, j) as_type<half2>(xv[m][(j) / 4][(j) % 4])
    float acc[R][M];
    {_UNROLL}
    for (int r = 0; r < R; r++)
      {_UNROLL}
      for (int m = 0; m < M; m++) acc[r][m] = 0.0f;
    uint2 wv[R], wn[R];
    uint4 xv[MC][2];
    half2 q2[R][8];
    float s[R], bb[R], xs[MC];
{_score_source(LW)}
    switch (c) {{
{cases}
    }}
    #undef x2
"""


def _gather(prepped, plan, proj, inds, m, top_k, tokens, logits=None):
    """y (P, N): every pair's expert matvec; the x rows are tokens (``tokens``) or pairs."""
    x16, xsum, rscale = prepped
    w, scales, biases = proj.weight, proj.scales, proj.biases
    N, K = w.shape[1], w.shape[2] * 8
    S = top_k + 1
    R, NSG = _config(K)
    MC = min(4, m)
    Mp = xsum.shape[1]
    RDIV = S if tokens else 1
    LW = logits.shape[-1] if logits is not None else 0
    key = ("gather", m, N, K, R, NSG, MC, Mp, S, top_k, RDIV, LW, _tag(scales.dtype))
    call = _calls.get(key)
    if call is None:
        kern = _kernel(
            "moe_small_gather",
            key[1:],
            lambda: _gather_source(m, N, K, R, NSG, MC, Mp, S, top_k, RDIV, LW),
            [
                "x16",
                "xsum",
                "rscale",
                "plan",
                "w",
                "scales",
                "biases",
                "inds",
                "logits",
            ],
            ["y"],
        )
        kwargs = dict(
            template=[("T", scales.dtype)],
            grid=(32 * NSG * m * S * (N // (R * NSG)), 1, 1),
            threadgroup=(32 * NSG, 1, 1),
            output_shapes=[(m * S, N)],
            output_dtypes=[scales.dtype],
        )
        call = _calls[key] = (kern, kwargs)
    kern, kwargs = call
    # An unused input (scores of 1) reuses an existing array so no op is added
    logits = scales if logits is None else logits
    (y,) = kern(
        inputs=[x16, xsum, rscale, plan, w, scales, biases, inds, logits], **kwargs
    )
    return y


def experts(block, x, logits, inds, slots=False):
    """The routed and the shared expert of x (.., K) -> (.., K); ``logits`` (.., E + 1).

    With ``slots`` the top_k + 1 expert outputs of a token are returned unsummed (.., S, K).
    """
    *batch, K = x.shape
    m = math.prod(batch)
    E, top_k = block.num_experts, block.top_k
    inds = inds.reshape(-1)
    x16, xsum, rscale, plan = _prep(x.reshape(m, K), "copy", inds, m, top_k, E)
    gu = _gather(
        (x16, xsum, rscale), plan, block.switch_mlp.gate_up_proj, inds, m, top_k, True
    )
    h = _prep(gu, "swiglu")
    y = _gather(
        h,
        plan,
        block.switch_mlp.down_proj,
        inds,
        m,
        top_k,
        False,
        logits.reshape(m, -1),
    )
    y = y.reshape(*batch, top_k + 1, K)
    return y if slots else y.sum(axis=-2)


_ROUTER_HEADER = """
// qdot of MLX's qmv kernel for 8-bit weights (quantized.h), 4 values per lane per step.
inline float qdot8(uint w, const thread float* x_thread, float scale, float bias, float sum) {
  float accum = 0;
  #pragma clang loop unroll(full)
  for (int i = 0; i < 4; i++) {
    accum += x_thread[i] * ((w >> (8 * i)) & 0xffu);
  }
  return scale * accum + sum * bias;
}
"""


def _router_source(K, N, NSG):
    """One simdgroup per row; the row's weights, the scales/biases and x are staged in threadgroup
    memory with 16-byte loads (7 device loads per lane), then the math is MLX's ``qmv`` kernel for
    8-bit weights step by step (lane l takes values 128 t + 4 l .. + 3 of step t, sums in that
    order, ``simd_sum`` at the end), so the logits are bitwise ``mx.quantized_matmul``'s. MLX's
    kernel walks the row in 16 dependent steps of scalar loads; this one issues a few wide loads.
    """
    NB = K // 128
    KW4 = K // 16  # uint4 words per weight row
    XW4 = K // 8  # uint4 words of x (T = 2 bytes)
    return f"""
    constexpr int K = {K}, N = {N}, NSG = {NSG}, NB = {NB}, KW = K / 4, KG = K / 64, KW4 = {KW4}, XW4 = {XW4};
    const int lane = thread_index_in_simdgroup;
    const int sg = simdgroup_index_in_threadgroup;
    const int tid = thread_position_in_threadgroup.x;
    const int row = threadgroup_position_in_grid.x * NSG + sg;
    threadgroup uint4 wbuf[NSG][KW4];
    threadgroup uint4 xbuf[XW4];
    threadgroup uint4 sbuf[NSG][KG / 8];
    threadgroup uint4 bbuf[NSG][KG / 8];
    if (row < N) {{
      const device uint4* wr = (const device uint4*)(w + (size_t)row * KW);
      {_UNROLL}
      for (int j = 0; j < KW4 / 32; j++) wbuf[sg][lane + 32 * j] = wr[lane + 32 * j];
      if (lane < KG / 8) {{
        sbuf[sg][lane] = ((const device uint4*)(scales + (size_t)row * KG))[lane];
        bbuf[sg][lane] = ((const device uint4*)(biases + (size_t)row * KG))[lane];
      }}
    }}
    for (int j = tid; j < XW4; j += 32 * NSG) xbuf[j] = ((const device uint4*)x)[j];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (row >= N) return;
    const int hi = lane / 16;
    const threadgroup uint* wt = (const threadgroup uint*)wbuf[sg];
    const threadgroup vec<T, 4>* xt = (const threadgroup vec<T, 4>*)xbuf;
    const threadgroup T* st = (const threadgroup T*)sbuf[sg];
    const threadgroup T* bt = (const threadgroup T*)bbuf[sg];
    float result = 0.0f;
    {_UNROLL}
    for (int t = 0; t < NB; t++) {{
      const vec<T, 4> xv = xt[32 * t + lane];
      float sum = 0.0f;
      float xt4[4];
      {_UNROLL}
      for (int i = 0; i < 4; i++) {{ sum += xv[i]; xt4[i] = xv[i]; }}
      const float s = st[2 * t + hi];
      const float b = bt[2 * t + hi];
      result += qdot8(wt[32 * t + lane], xt4, s, b, sum);
    }}
    result = simd_sum(result);
    if (lane == 0) y[row] = static_cast<T>(result);
"""


def router_ok(gate, x):
    """True when ``router`` runs: one row, an 8-bit g64 affine ``QuantizedLinear`` without bias."""
    *batch, k = x.shape
    return (
        enabled()
        and math.prod(batch) == 1
        and isinstance(gate, nn.QuantizedLinear)
        and gate.bits == 8
        and gate.group_size == 64
        and getattr(gate, "mode", "affine") == "affine"
        and "bias" not in gate
        and x.dtype in (mx.bfloat16, mx.float16)
        and gate.scales.dtype == x.dtype
        and k % 512 == 0
        and gate.weight.shape[1] * 4 == k
    )


def router(gate, x, nsg=4):
    """``gate(x)``: the router logits of one token through the low-latency 8-bit matvec."""
    if not router_ok(gate, x):
        return gate(x)
    *batch, K = x.shape
    N = gate.weight.shape[0]
    key = ("router", N, K, nsg, _tag(x.dtype))
    call = _calls.get(key)
    if call is None:
        kern = _kernel(
            "moe_small_router",
            key[1:],
            lambda: _router_source(K, N, nsg),
            ["x", "w", "scales", "biases"],
            ["y"],
            _ROUTER_HEADER,
        )
        kwargs = dict(
            template=[("T", x.dtype)],
            grid=(32 * nsg * (-(-N // nsg)), 1, 1),
            threadgroup=(32 * nsg, 1, 1),
            output_shapes=[(1, N)],
            output_dtypes=[x.dtype],
        )
        call = _calls[key] = (kern, kwargs)
    kern, kwargs = call
    (y,) = kern(inputs=[x.reshape(1, K), gate.weight, gate.scales, gate.biases], **kwargs)
    return y.reshape(*batch, N)


_SORT_HEADER = """
// MLX's LessThan of sort.h (NaN sorts last).
template <typename T> struct SortLess {
  bool operator()(T a, T b) const {
    bool an = metal::isnan(a), bn = metal::isnan(b);
    if (an | bn) return (!an) & bn;
    return a < b;
  }
};
"""


def _topk_plan_source(E, TOPK, S, NT):
    """Appended to the copy prep of one token: threads 0..63 argsort the E logits as MLX's
    single-block sort does (carg_block_sort, 64 threads x 4 values, the same merges and tie
    order), the last TOPK indices are the experts of mx.argpartition, and the plan is one
    pair per slot with the shared expert last."""
    return f"""
    {{
      constexpr int E = {E}, TOPK = {TOPK}, S = {S}, PW = 3, BN = 64, TN = 4, NPB = BN * TN;
      threadgroup T tv[NPB];
      threadgroup uint ti[NPB];
      SortLess<T> op;
      const T init = metal::numeric_limits<T>::quiet_NaN();
      for (int i = t; i < NPB; i += {NT}) {{ tv[i] = i < E ? logits[i] : init; ti[i] = uint(i); }}
      threadgroup_barrier(mem_flags::mem_threadgroup);
      const bool sorter = t < BN;
      const int idx = t * TN;
      T vals[TN];
      uint idxs[TN];
      if (sorter) {{
        for (int i = 0; i < TN; ++i) {{ vals[i] = tv[idx + i]; idxs[i] = ti[idx + i]; }}
        if (idx < E) {{
          for (short i = 0; i < TN; ++i) {{
            for (short j = i & 1; j < TN - 1; j += 2) {{
              if (op(vals[j + 1], vals[j])) {{
                T w = vals[j + 1]; vals[j + 1] = vals[j]; vals[j] = w;
                uint wi = idxs[j + 1]; idxs[j + 1] = idxs[j]; idxs[j] = wi;
              }}
            }}
          }}
        }}
      }}
      for (int merge_threads = 2; merge_threads <= BN; merge_threads *= 2) {{
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (sorter) for (int i = 0; i < TN; ++i) {{ tv[idx + i] = vals[i]; ti[idx + i] = idxs[i]; }}
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (sorter) {{
          const int merge_group = t / merge_threads, merge_lane = t % merge_threads;
          const int sort_sz = TN * merge_threads, sort_st = sort_sz * merge_group;
          const int A_st = sort_st, A_ed = sort_st + sort_sz / 2, B_st = A_ed, B_ed = sort_st + sort_sz;
          const threadgroup T* As = tv + A_st;
          const threadgroup T* Bs = tv + B_st;
          short A_sz = A_ed - A_st, B_sz = B_ed - B_st;
          const short sort_md = TN * merge_lane;
          short pa = max(0, sort_md - B_sz), pe = min(sort_md, A_sz);
          while (pa < pe) {{
            const short md = pa + (pe - pa) / 2;
            const T a = As[md];
            const T b = Bs[sort_md - 1 - md];
            if (op(b, a)) pe = md; else pa = md + 1;
          }}
          const short partition = pe;
          As += partition;
          Bs += sort_md - partition;
          A_sz -= partition;
          B_sz -= sort_md - partition;
          const threadgroup uint* As_idx = ti + A_st + partition;
          const threadgroup uint* Bs_idx = ti + B_st + sort_md - partition;
          short a_idx = 0, b_idx = 0;
          for (int i = 0; i < TN; ++i) {{
            const T a = (a_idx < A_sz) ? As[a_idx] : init;
            const T b = (b_idx < B_sz) ? Bs[b_idx] : init;
            const bool pred = (b_idx < B_sz) && (a_idx >= A_sz || op(b, a));
            vals[i] = pred ? b : a;
            idxs[i] = pred ? Bs_idx[b_idx] : ((a_idx < A_sz) ? As_idx[a_idx] : 0u);
            b_idx += short(pred);
            a_idx += short(!pred);
          }}
        }}
      }}
      threadgroup_barrier(mem_flags::mem_threadgroup);
      if (sorter) for (int i = 0; i < TN; ++i) {{ tv[idx + i] = vals[i]; ti[idx + i] = idxs[i]; }}
      threadgroup_barrier(mem_flags::mem_threadgroup);
      if (t < TOPK) inds[t] = ti[E - TOPK + t];
      if (t < S) {{
        plan[t * PW] = t < TOPK ? int(ti[E - TOPK + t]) : E;
        plan[t * PW + 1] = 1;
        plan[t * PW + 2] = t;
      }}
    }}
"""


def topk_ok(block, x):
    """True for one token, 128 < experts <= 256 (MLX's 64 x 4 single-block sort) and K <= 8192."""
    *batch, k = x.shape
    return math.prod(batch) == 1 and 128 < block.num_experts <= 256 and k <= 8192


def topk_prep(x, logits, E, top_k):
    """The copy prep of one token x (1, K) plus the top-k experts of its logits and the plan.

    Returns (x16, xsum, rscale, plan, inds); ``inds`` (top_k,) equals
    ``mx.argpartition(logits[..., :E], kth=-top_k)[..., -top_k:]`` bit for bit.
    """
    M, K = x.shape
    S = top_k + 1
    NT = _scan_threads(K)
    key = ("topk_prep", K, E, top_k, _tag(x.dtype))
    call = _calls.get(key)
    if call is None:
        kern = _kernel(
            "moe_small_topk_prep",
            key[1:],
            lambda: _prep_source(K, 1, "copy") + _topk_plan_source(E, top_k, S, NT),
            ["x", "logits"],
            ["x16", "xsum", "rscale", "plan", "inds"],
            _HEADER + _SORT_HEADER,
        )
        kwargs = dict(
            template=[("T", x.dtype)],
            grid=(NT * _prep_segments(K), 1, 1),
            threadgroup=(NT, 1, 1),
            output_shapes=[(1, K), (K // 16, _mp(1)), (_mp(1),), (S, 3), (top_k,)],
            output_dtypes=[mx.float16, mx.float32, mx.float32, mx.int32, mx.uint32],
        )
        call = _calls[key] = (kern, kwargs)
    kern, kwargs = call
    return kern(inputs=[x, logits.reshape(1, -1)], **kwargs)


def experts_topk(block, x, logits, slots=False):
    """``experts`` for one token with the top-k, the plan and the prep in one kernel."""
    *batch, K = x.shape
    E, top_k = block.num_experts, block.top_k
    x16, xsum, rscale, plan, inds = topk_prep(x.reshape(1, K), logits, E, top_k)
    gu = _gather((x16, xsum, rscale), plan, block.switch_mlp.gate_up_proj, inds, 1, top_k, True)
    h = _prep(gu, "swiglu")
    y = _gather(h, plan, block.switch_mlp.down_proj, inds, 1, top_k, False, logits.reshape(1, -1))
    y = y.reshape(*batch, top_k + 1, K)
    return y if slots else y.sum(axis=-2)
