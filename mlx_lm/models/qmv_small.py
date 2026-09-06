# Copyright © 2026 Apple Inc.

"""Small-M (3..8 rows) 4-bit affine g64 quantized matvec for the MTP verify pass.

``mx.quantized_matmul`` routes 2 <= M < 13 to ``qmv_wide``, which dequantizes
each weight in fp32 per input row and is ALU bound on M5. This kernel keeps the
``qmv_fast`` geometry (R rows per simdgroup, 16 K values per lane per step),
dequantizes each nibble once into fp16 and applies it to all M rows with half2
FMAs. Half partials are flushed into fp32 accumulators every step, so one group
of 64 per lane per step.

``x`` is first converted by a prep kernel into fp16 scaled per row by a power
of two (so the fp16 partials stay in range), plus per-16-chunk sums for the
bias term. The kernel reads the nibble pairs (k, k+4) of each 8-value word
through the fp16 "magic number" trick, so the prep permutes x the same way and
pre-scales the odd pairs by 1/16. The prep can fuse the producer of x (RMSNorm,
swiglu, the attention output gate, the GDN gated norm) so no extra kernel is
launched.
"""

import mlx.core as mx
import mlx.nn as nn

_BITS = 4
_GROUP = 64
_VPL = 16  # K values per lane per step
# mx.quantized_matmul is already weight-bandwidth bound at M = 2.
_MIN_M, _MAX_M = 3, 8
_SEG = 1024  # x values per prep threadgroup (64 threads x 16)
_UNROLL = "#pragma clang loop unroll(full)"
_kernels = {}


def _tag(dtype):
    return {mx.bfloat16: "bf16", mx.float16: "f16", mx.float32: "f32"}[dtype]


def _mp(m):
    return 2 if m <= 2 else 4 if m <= 4 else 8


# x order inside each 8-value word, and the pre-scale of each stored position.
_ORDER = (0, 4, 1, 5, 2, 6, 3, 7)
_SCALE = (1.0, 1.0, 1 / 16, 1 / 16, 1.0, 1.0, 1 / 16, 1 / 16)


_HEADER = """
template <typename T> struct Unpack;
template <> struct Unpack<bfloat16_t> {
  static inline float2 f(uint u) { return float2(as_type<float>(u << 16), as_type<float>(u & 0xffff0000u)); }
};
template <> struct Unpack<half> {
  static inline float2 f(uint u) { return float2(as_type<half2>(u)); }
};

// 16 consecutive T values as floats through two 16-byte loads.
template <typename T>
inline void load16(const device T* p, thread float* v) {
  const device uint4* q = (const device uint4*)p;
  const uint4 a = q[0], b = q[1];
  const uint u[8] = {a.x, a.y, a.z, a.w, b.x, b.y, b.z, b.w};
  #pragma clang loop unroll(full)
  for (int i = 0; i < 8; i++) {
    const float2 f = Unpack<T>::f(u[i]);
    v[2 * i] = f.x;
    v[2 * i + 1] = f.y;
  }
}

inline float sigmoid_f(float g) { return 1.0f / (1.0f + metal::exp(-g)); }
"""

_PREP_INPUTS = {
    "copy": ["x"],
    "rms_norm": ["x", "weight"],
    "swiglu": ["x"],
    "gate": ["x", "gate"],
    "gated_norm": ["x", "gate", "weight"],
}


def _values(kind, dst, cidx, rs):
    """Code that computes the 16 values of chunk ``cidx`` of row ``m`` into ``dst`` (floats)."""
    if kind == "rms_norm":
        return f"""
      float xx[16], ww[16];
      load16(x + (size_t)m * K + ({cidx}) * 16, xx);
      load16(weight + ({cidx}) * 16, ww);
      {_UNROLL}
      for (int i = 0; i < 16; i++) {{
        ss += xx[i] * xx[i];
        {dst}[i] = xx[i] * ww[i] * {rs};
      }}"""
    if kind == "swiglu":
        return f"""
      float gg[16];
      load16(x + (size_t)m * 2 * K + ({cidx}) * 16, gg);
      load16(x + (size_t)m * 2 * K + K + ({cidx}) * 16, {dst});
      {_UNROLL}
      for (int i = 0; i < 16; i++) {dst}[i] *= gg[i] * sigmoid_f(gg[i]);"""
    if kind == "gate":
        return f"""
      float gg[16];
      load16(x + (size_t)m * K + ({cidx}) * 16, {dst});
      load16(gate + (size_t)m * K + ({cidx}) * 16, gg);
      {_UNROLL}
      for (int i = 0; i < 16; i++) {dst}[i] *= sigmoid_f(gg[i]);"""
    if kind == "gated_norm":
        return f"""
      float xx[16], gg[16], ww[16];
      load16(x + (size_t)m * K + ({cidx}) * 16, xx);
      load16(gate + (size_t)m * K + ({cidx}) * 16, gg);
      load16(weight + (({cidx}) * 16) % D, ww);
      float ssh = 0.0f;
      {_UNROLL}
      for (int i = 0; i < 16; i++) ssh += xx[i] * xx[i];
      // The TPH threads of one head are adjacent lanes of one simdgroup.
      {_UNROLL}
      for (int o = TPH / 2; o > 0; o >>= 1) ssh += simd_shuffle_xor(ssh, o);
      const float rsh = rsqrt(ssh / D + EPS);
      {_UNROLL}
      for (int i = 0; i < 16; i++) {dst}[i] = xx[i] * rsh * ww[i] * (gg[i] * sigmoid_f(gg[i]));"""
    return f"""
      load16(x + (size_t)m * K + ({cidx}) * 16, {dst});"""


def _scan(kind):
    """Scan-loop body over chunk ``c2``: a cheap upper bound of the row's max in ``amax``.

    rms_norm scans the exact max of |x * weight| (and the sum of squares); swiglu bounds
    |silu(g) * u| by max|g| * max|u| (into amax and amax2); gate bounds |x * sigmoid(g)|
    by max|x|; gated_norm bounds |x * rs * w * silu(z)| by sqrt(D) * max|w| * max|z|.
    """
    if kind == "rms_norm":
        return """
        float xx[16], ww[16];
        load16(x + (size_t)m * K + c2 * 16, xx);
        load16(weight + c2 * 16, ww);
        {_UNROLL}
        for (int i = 0; i < 16; i++) {
          ss += xx[i] * xx[i];
          amax = max(amax, fabs(xx[i] * ww[i]));
        }"""
    if kind == "swiglu":
        return """
        float gg[16], uu[16];
        load16(x + (size_t)m * 2 * K + c2 * 16, gg);
        load16(x + (size_t)m * 2 * K + K + c2 * 16, uu);
        {_UNROLL}
        for (int i = 0; i < 16; i++) {
          amax = max(amax, fabs(gg[i]));
          amax2 = max(amax2, fabs(uu[i]));
        }"""
    if kind == "gated_norm":
        return """
        float gg[16], ww[16];
        load16(gate + (size_t)m * K + c2 * 16, gg);
        load16(weight + (c2 * 16) % D, ww);
        {_UNROLL}
        for (int i = 0; i < 16; i++) {
          amax = max(amax, fabs(gg[i]));
          amax2 = max(amax2, fabs(ww[i]));
        }"""
    # copy and gate: max|x|
    return """
        float xx[16];
        load16(x + (size_t)m * K + c2 * 16, xx);
        {_UNROLL}
        for (int i = 0; i < 16; i++) amax = max(amax, fabs(xx[i]));"""


def _scan_threads(K):
    """Threads per prep threadgroup: all scan the row, the first 64 convert their segment."""
    return 256 if K <= 8192 else 512


def _prep_source(K, M, kind, eps=0.0, D=0):
    """One threadgroup per 1024 values of one row: producer op in fp32, row scale, fp16 store.

    Every threadgroup of a row first scans the whole row with all its threads for a bound
    of the row's max (and the RMS), so the fp16 scale is per row without a separate pass.
    kind: "copy" (x), "rms_norm" (x * rsqrt(mean(x^2) + eps) * weight), "swiglu"
    (silu(gate) * up with gate = x[:, :K] and up = x[:, K:]), "gate" (x * sigmoid(gate)),
    "gated_norm" (per-head RMSNorm over D values times silu(gate)).
    """
    Mp = _mp(M)
    NT = _scan_threads(K)
    NIT = -(-(K // 16) // NT)
    stores = "\n".join(
        f"      h[{c * 8 + j}] = half(v[{c * 8 + _ORDER[j]}] * (sc * {_SCALE[j]}f));"
        for c in range(2)
        for j in range(8)
    )
    reduce2 = """
    amax2 = simd_max(amax2);
    threadgroup float red3[NT / 32];
    if (thread_index_in_simdgroup == 0) red3[simdgroup_index_in_threadgroup] = amax2;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    amax2 = red3[0];
    for (int i = 1; i < NT / 32; i++) amax2 = max(amax2, red3[i]);"""
    if kind == "rms_norm":
        post = """
    ss = simd_sum(ss);
    threadgroup float red[NT / 32];
    if (thread_index_in_simdgroup == 0) red[simdgroup_index_in_threadgroup] = ss;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    ss = 0.0f;
    for (int i = 0; i < NT / 32; i++) ss += red[i];
    const float rs = rsqrt(ss / K + EPS);
    amax *= rs;"""
    elif kind == "swiglu":
        post = reduce2 + "\n    amax *= amax2;\n    const float rs = 1.0f;"
    elif kind == "gated_norm":
        post = (
            reduce2
            + "\n    amax *= amax2 * metal::sqrt(float(D));\n    const float rs = 1.0f;"
        )
    else:
        post = "\n    const float rs = 1.0f;"
    scan = _scan(kind).replace("{_UNROLL}", _UNROLL)
    return f"""
    constexpr int K = {K}, Mp = {Mp}, NT = {NT}, NIT = {NIT}, NSEG = K / {_SEG}, D = {D}, TPH = D / 16;
    constexpr float EPS = {float(eps)!r}f;
    const int m = threadgroup_position_in_grid.x / NSEG;
    const int seg = threadgroup_position_in_grid.x % NSEG;
    const int t = thread_position_in_threadgroup.x;
    float ss = 0.0f;
    float amax = 0.0f, amax2 = 0.0f;
    {_UNROLL}
    for (int j = 0; j < NIT; j++) {{
      const int c2 = t + j * NT;
      if (c2 < K / 16) {{{scan}
      }}
    }}
    (void)ss;
    (void)amax2;
    amax = simd_max(amax);
    threadgroup float red2[NT / 32];
    if (thread_index_in_simdgroup == 0) red2[simdgroup_index_in_threadgroup] = amax;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    amax = red2[0];
    for (int i = 1; i < NT / 32; i++) amax = max(amax, red2[i]);{post}
    (void)rs;
    // Scale the row so the bound of max |x| is in [4, 8): fp16 partials stay in range.
    int e = 0;
    float sc = 1.0f;
    if (amax > 0.0f) {{
      frexp(amax, e);
      sc = ldexp(1.0f, 3 - e);
    }}
    if (seg == 0 && t == 0) rscale[m] = 1.0f / sc;
    if (t < {_SEG} / 16) {{
      const int c = seg * ({_SEG} / 16) + t;
      float v[16];
      {{{_values(kind, "v", "c", "rs")}
      }}
      float s = 0.0f;
      {_UNROLL}
      for (int i = 0; i < 16; i++) s += v[i];
      xsum[c * Mp + m] = s * sc;
      half h[16];
{stores}
      device half4* o = (device half4*)(x16 + (size_t)m * K + c * 16);
      {_UNROLL}
      for (int i = 0; i < 4; i++) o[i] = half4(h[4 * i], h[4 * i + 1], h[4 * i + 2], h[4 * i + 3]);
    }}
"""


def _main_source(M, N, K, R, NSG, MC):
    Mp = _mp(M)
    NB = K // (32 * _VPL)
    wload = "\n".join(
        f"      {{dst}}[{r}] = *(const device uint2*)(wp + {r} * KW);" for r in range(R)
    )
    deq = []
    for r in range(R):
        for wi in range(2):
            wd = f"wv[{r}][{wi}]"
            deq.append(f"      {{ const uint lo = {wd}, hi = {wd} >> 8;")
            deq.append(
                f"        q2[{r}][{wi * 4}] = as_type<half2>((lo & 0x000F000Fu) | 0x64006400u) - half2(1024.0h);"
            )
            deq.append(
                f"        q2[{r}][{wi * 4 + 1}] = as_type<half2>((lo & 0x00F000F0u) | 0x64006400u) - half2(1024.0h);"
            )
            deq.append(
                f"        q2[{r}][{wi * 4 + 2}] = as_type<half2>((hi & 0x000F000Fu) | 0x64006400u) - half2(1024.0h);"
            )
            deq.append(
                f"        q2[{r}][{wi * 4 + 3}] = as_type<half2>((hi & 0x00F000F0u) | 0x64006400u) - half2(1024.0h); }}"
            )
    chunks = []
    for m0 in range(0, M, MC):
        rows = range(m0, min(m0 + MC, M))
        xl = "\n".join(
            f"      xv[{m - m0}][{c}] = *(const device uint4*)(xp + {m} * K + {c} * 8);"
            for m in rows
            for c in range(2)
        )
        xs = "\n".join(f"      xs[{m - m0}] = xsp[{m}];" for m in rows)
        fm = []
        for r in range(R):
            for m in rows:
                mm = m - m0
                fm.append(
                    f"      {{ half2 p = q2[{r}][0] * x2({mm}, 0);\n"
                    + "\n".join(
                        f"        p = fma(q2[{r}][{j}], x2({mm}, {j}), p);"
                        for j in range(1, 8)
                    )
                    + f"\n        acc[{r}][{m}] = fma(s[{r}], float(p.x + p.y), fma(bb[{r}], xs[{mm}], acc[{r}][{m}])); }}"
                )
        chunks.append(xl + "\n" + xs + "\n" + "\n".join(fm))
    return f"""
    constexpr int M = {M}, N = {N}, K = {K}, R = {R}, NSG = {NSG}, VPL = {_VPL}, Mp = {Mp}, MC = {MC};
    constexpr int KW = K / 8, KG = K / 64, NB = {NB};
    const int lane = thread_index_in_simdgroup;
    const int sg = simdgroup_index_in_threadgroup;
    const int row0 = (threadgroup_position_in_grid.x * NSG + sg) * R;
    const device uint32_t* wp = w + (size_t)row0 * KW + lane * 2;
    const device T* sp = scales + (size_t)row0 * KG + lane / 4;
    const device T* bp = biases + (size_t)row0 * KG + lane / 4;
    const device half* xp = x16 + lane * VPL;
    const device float* xsp = xsum + (size_t)lane * Mp;
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
{wload.format(dst="wv")}
    for (int b = 0; b < NB; b++) {{
      // Request the next step's weights before this step's math.
      wp += 32 * VPL / 8;
      if (b + 1 < NB) {{
{wload.format(dst="wn")}
      }}
{chr(10).join(f"      s[{r}] = float(sp[{r} * KG]); bb[{r}] = float(bp[{r} * KG]);" for r in range(R))}
{chr(10).join(deq)}
{chr(10).join(chunks)}
      sp += 32 * VPL / 64;
      bp += 32 * VPL / 64;
      xp += 32 * VPL;
      xsp += 32 * Mp;
{chr(10).join(f"      wv[{r}] = wn[{r}];" for r in range(R))}
    }}
{chr(10).join(f"    acc[{r}][{m}] = simd_sum(acc[{r}][{m}]);" for r in range(R) for m in range(M))}
    if (lane == 0) {{
{chr(10).join(f"      y[(size_t){m} * N + row0 + {r}] = T(acc[{r}][{m}] * rscale[{m}]);" for r in range(R) for m in range(M))}
    }}
    #undef x2
"""


def _kernel(kind, key, source, inputs, outputs, header=""):
    kern = _kernels.get((kind, key))
    if kern is None:
        name = kind + "_" + "_".join(str(v) for v in key)
        name = "".join(c if c.isalnum() else "_" for c in name)
        kern = mx.fast.metal_kernel(
            name=name,
            input_names=inputs,
            output_names=outputs,
            source=source(),
            header=header,
        )
        _kernels[(kind, key)] = kern
    return kern


class Prepped:
    """An (M, K) activation already converted for ``qmv_small``; mimics the array's shape and dtype."""

    def __init__(self, x16, xsum, rscale, shape, dtype):
        self.x16, self.xsum, self.rscale = x16, xsum, rscale
        self.shape, self.dtype = shape, dtype
        self.ndim = len(shape)


def prep(x, kind="copy", *extra, eps=0.0, d=0, shape=None):
    """Convert ``x`` (M, K) [(M, 2K) for swiglu] plus the ``extra`` inputs of ``kind`` into a ``Prepped``."""
    M, K = x.shape
    if kind == "swiglu":
        K //= 2
    Mp = _mp(M)
    kern = _kernel(
        "qmv_small_prep_" + kind,
        (K, M, eps, d, _tag(x.dtype)),
        lambda: _prep_source(K, M, kind, eps, d),
        _PREP_INPUTS[kind],
        ["x16", "xsum", "rscale"],
        _HEADER,
    )
    x16, xsum, rscale = kern(
        inputs=[x, *extra],
        template=[("T", x.dtype)],
        grid=(_scan_threads(K) * (K // _SEG) * M, 1, 1),
        threadgroup=(_scan_threads(K), 1, 1),
        output_shapes=[(M, K), (K // 16, Mp), (Mp,)],
        output_dtypes=[mx.float16, mx.float32, mx.float32],
    )
    return Prepped(x16, xsum, rscale, shape or (M, K), x.dtype)


def _config(m):
    # Two rows per simdgroup keeps registers low; at M > 4 four rows amortize the dequant.
    return (2, 2, m) if m <= 4 else (4, 2, 4)


def supported(x, w, scales, biases, group_size, bits):
    if x.ndim != 2 or bits != _BITS or group_size != _GROUP or biases is None:
        return False
    if (
        x.dtype not in (mx.bfloat16, mx.float16)
        or scales.dtype != x.dtype
        or biases.dtype != x.dtype
    ):
        return False
    m, k = x.shape
    n = w.shape[0]
    return (
        _MIN_M <= m <= _MAX_M and k % _SEG == 0 and n % 8 == 0 and w.shape[1] * 8 == k
    )


def qmv_main(p, w, scales, biases):
    """``x @ dequant(w).T`` from a ``Prepped`` x."""
    M, K = p.x16.shape
    N = w.shape[0]
    R, NSG, MC = _config(M)
    kern = _kernel(
        "qmv_small",
        (M, N, K, R, NSG, MC, _tag(p.dtype)),
        lambda: _main_source(M, N, K, R, NSG, MC),
        ["x16", "xsum", "rscale", "w", "scales", "biases"],
        ["y"],
    )
    ntg = N // (R * NSG)
    (y,) = kern(
        inputs=[p.x16, p.xsum, p.rscale, w, scales, biases],
        template=[("T", p.dtype)],
        grid=(32 * NSG * ntg, 1, 1),
        threadgroup=(32 * NSG, 1, 1),
        output_shapes=[(M, N)],
        output_dtypes=[p.dtype],
    )
    return y


def qmv_small(x, w, scales, biases, group_size=_GROUP, bits=_BITS):
    """``x @ dequant(w).T`` for ``x`` of shape (M, K); mx.quantized_matmul outside 3 <= M <= 8."""
    if not supported(x, w, scales, biases, group_size, bits):
        return mx.quantized_matmul(
            x, w, scales, biases, transpose=True, group_size=group_size, bits=bits
        )
    return qmv_main(prep(x), w, scales, biases)


def routes(module, shape, dtype):
    """True when ``qlinear`` uses the small-M kernel for ``module`` on an input of ``shape``."""
    *batch, k = shape
    m = 1
    for d in batch:
        m *= d
    return (
        isinstance(module, nn.QuantizedLinear)
        and module.bits == _BITS
        and module.group_size == _GROUP
        and getattr(module, "mode", "affine") == "affine"
        and "bias" not in module
        and dtype in (mx.bfloat16, mx.float16)
        and module.scales.dtype == dtype
        and module.biases.dtype == dtype
        and _MIN_M <= m <= _MAX_M
        and k % _SEG == 0
        and module.weight.shape[0] % 8 == 0
        and module.weight.shape[1] * 8 == k
    )


def prep_rms_norm(norm, x, module):
    """``norm(x)`` fused with the prep when the projection ``module`` routes; else ``norm(x)``."""
    if not routes(module, x.shape, x.dtype):
        return norm(x)
    k = x.shape[-1]
    return prep(
        x.reshape(-1, k), "rms_norm", norm.weight, eps=norm.eps, shape=tuple(x.shape)
    )


def prep_swiglu(gate_up, module):
    """``swiglu(gate, up)`` of the fused (.., 2K) projection output, prepped for ``module``."""
    *batch, k2 = gate_up.shape
    shape = (*batch, k2 // 2)
    if not routes(module, shape, gate_up.dtype):
        return None
    return prep(gate_up.reshape(-1, k2), "swiglu", shape=shape)


def prep_gate(x, gate, module):
    """``x * sigmoid(gate)`` prepped for ``module``, or None when it does not route."""
    if not routes(module, x.shape, x.dtype):
        return None
    k = x.shape[-1]
    return prep(x.reshape(-1, k), "gate", gate.reshape(-1, k), shape=tuple(x.shape))


def prep_gated_norm(norm, x, gate, module):
    """``norm(x, gate)`` (per-head RMSNorm times silu(gate)) prepped for ``module``, or None."""
    *batch, heads, d = x.shape
    shape = (*batch, heads * d)
    k = heads * d
    # The per-head reduction needs a power-of-two number of lanes per head (16 values each).
    tph = d // 16
    if (
        not routes(module, shape, x.dtype)
        or d % 16
        or tph > 32
        or tph & (tph - 1)
        or _SEG % d
    ):
        return None
    return prep(
        x.reshape(-1, k),
        "gated_norm",
        gate.reshape(-1, k),
        norm.weight,
        eps=norm.eps,
        d=d,
        shape=shape,
    )


def qlinear(module, x):
    """Apply a bias-free 4-bit g64 ``QuantizedLinear`` through ``qmv_small`` when 3 <= M <= 8."""
    if isinstance(x, Prepped):
        return qmv_main(x, module.weight, module.scales, module.biases).reshape(
            *x.shape[:-1], -1
        )
    if routes(module, x.shape, x.dtype):
        y = qmv_main(
            prep(x.reshape(-1, x.shape[-1])),
            module.weight,
            module.scales,
            module.biases,
        )
        return y.reshape(*x.shape[:-1], -1)
    return module(x)
