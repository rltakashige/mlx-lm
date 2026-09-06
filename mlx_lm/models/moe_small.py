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


def _config(K):
    """(rows per simdgroup, simdgroups per threadgroup), measured on M5 for K = 2048 and 512."""
    return (4, 2) if K >= 2048 else (4, 4)


def routes(block, x):
    """True when ``experts`` runs this step: M5, 4-bit g64 affine experts, 1 <= M <= 8."""
    projs = (block.switch_mlp.gate_up_proj, block.switch_mlp.down_proj)
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

    kern = _kernel(
        "moe_small_prep_" + kind + ("_plan" if plan else ""),
        (K, M, _tag(x.dtype), m, top_k, eshared),
        source,
        ["x"] + (["inds"] if plan else []),
        ["x16", "xsum", "rscale"] + (["plan"] if plan else []),
        _HEADER,
    )
    return kern(
        inputs=[x] + ([inds] if plan else []),
        template=[("T", x.dtype)],
        grid=(_scan_threads(K) * _prep_segments(K) * M, 1, 1),
        threadgroup=(_scan_threads(K), 1, 1),
        output_shapes=[(M, K), (K // 16, Mp), (Mp,)]
        + ([(m * S, m + 2)] if plan else []),
        output_dtypes=[mx.float16, mx.float32, mx.float32]
        + ([mx.int32] if plan else []),
    )


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
    kern = _kernel(
        "moe_small_gather",
        (m, N, K, R, NSG, MC, Mp, S, top_k, RDIV, LW, _tag(scales.dtype)),
        lambda: _gather_source(m, N, K, R, NSG, MC, Mp, S, top_k, RDIV, LW),
        ["x16", "xsum", "rscale", "plan", "w", "scales", "biases", "inds", "logits"],
        ["y"],
    )
    # An unused input (scores of 1) reuses an existing array so no op is added
    logits = scales if logits is None else logits
    (y,) = kern(
        inputs=[x16, xsum, rscale, plan, w, scales, biases, inds, logits],
        template=[("T", scales.dtype)],
        grid=(32 * NSG * m * S * (N // (R * NSG)), 1, 1),
        threadgroup=(32 * NSG, 1, 1),
        output_shapes=[(m * S, N)],
        output_dtypes=[scales.dtype],
    )
    return y


def experts(block, x, logits, inds):
    """The routed and the shared expert of x (.., K) -> (.., K); ``logits`` (.., E + 1)."""
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
    return y.reshape(*batch, top_k + 1, K).sum(axis=-2)
