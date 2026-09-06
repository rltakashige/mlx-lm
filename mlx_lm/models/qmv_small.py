# Copyright © 2026 Apple Inc.

"""Small-M (2..8 rows) 4-bit affine g64 quantized matvec for the MTP verify pass.

``mx.quantized_matmul`` routes 2 <= M < 13 to ``qmv_wide``, which dequantizes
each weight in fp32 per input row and is ALU bound on M5. This kernel keeps the
``qmv_fast`` geometry (R rows per simdgroup, 16 K values per lane per step),
dequantizes each nibble once into fp16 and applies it to all M rows with half2
FMAs. Half partials are flushed into fp32 accumulators every step, so one group
of 64 per lane per step.

``x`` is first converted by a small prep kernel into fp16 (scaled per row by a
power of two so the fp16 partials stay in range), plus per-16-chunk sums for
the bias term. The kernel reads the nibble pairs (k, k+4) of each 8-value word
through the fp16 "magic number" trick, so the prep permutes x the same way and
pre-scales the odd pairs by 1/16.
"""

import mlx.core as mx
import mlx.nn as nn

_BITS = 4
_GROUP = 64
_VPL = 16  # K values per lane per step
# mx.quantized_matmul is already weight-bandwidth bound at M = 2.
_MIN_M, _MAX_M = 3, 8
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


def _prep_threads(K):
    """Chunks of 16 values per thread and threads per row."""
    chunks = K // 16
    cpt = 1 if chunks <= 1024 else 2
    nt = (-(-chunks // cpt) + 31) // 32 * 32
    return cpt, nt


def _prep_source(K, M, kind, eps=0.0):
    """One threadgroup per row: the values stay in registers between the row max and the store.

    kind: "copy" (x), "rms_norm" (x * rsqrt(mean(x^2) + eps) * weight), "swiglu"
    (silu(gate) * up with gate = x[:, :K] and up = x[:, K:]), "gate" (x * sigmoid(weight)).
    """
    Mp = _mp(M)
    C = K // 16
    CPT, NT = _prep_threads(K)
    stores = "\n".join(
        f"        h[{c * 8 + j}] = half(v[jj][{c * 8 + _ORDER[j]}] * (sc * {_SCALE[j]}f));"
        for c in range(2)
        for j in range(8)
    )
    if kind == "rms_norm":
        loads = """
      load16(x + (size_t)m * K + c * 16, v[jj]);
      {_UNROLL}
      for (int i = 0; i < 16; i++) ss += v[jj][i] * v[jj][i];"""
        norm = """
    ss = simd_sum(ss);
    if (thread_index_in_simdgroup == 0) red[simdgroup_index_in_threadgroup] = ss;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    ss = 0.0f;
    for (int i = 0; i < NT / 32; i++) ss += red[i];
    const float rs = rsqrt(ss / K + {eps}f);
    for (int jj = 0; jj < CPT; jj++) {{
      const int c = t + jj * NT;
      if (c < C) {{
        float wv[16];
        load16(weight + c * 16, wv);
        {_UNROLL}
        for (int i = 0; i < 16; i++) v[jj][i] *= rs * wv[i];
      }}
    }}"""
    elif kind == "swiglu":
        loads = """
      float g[16];
      load16(x + (size_t)m * 2 * K + c * 16, g);
      load16(x + (size_t)m * 2 * K + K + c * 16, v[jj]);
      {_UNROLL}
      for (int i = 0; i < 16; i++) v[jj][i] *= g[i] * sigmoid_f(g[i]);"""
        norm = ""
    elif kind == "gate":
        loads = """
      float g[16];
      load16(x + (size_t)m * K + c * 16, v[jj]);
      load16(weight + (size_t)m * K + c * 16, g);
      {_UNROLL}
      for (int i = 0; i < 16; i++) v[jj][i] *= sigmoid_f(g[i]);"""
        norm = ""
    else:
        loads = """
      load16(x + (size_t)m * K + c * 16, v[jj]);"""
        norm = ""
    loads = loads.replace("{_UNROLL}", _UNROLL)
    norm = norm.replace("{_UNROLL}", _UNROLL).replace("{eps}", repr(float(eps)))
    return f"""
    constexpr int K = {K}, C = {C}, CPT = {CPT}, NT = {NT}, Mp = {Mp};
    const int m = threadgroup_position_in_grid.x;
    const int t = thread_position_in_threadgroup.x;
    threadgroup float red[32];
    threadgroup float red2[32];
    float v[CPT][16];
    float ss = 0.0f;
    for (int jj = 0; jj < CPT; jj++) {{
      const int c = t + jj * NT;
      if (c < C) {{{loads}
      }} else {{
        {_UNROLL}
        for (int i = 0; i < 16; i++) v[jj][i] = 0.0f;
      }}
    }}
    (void)ss;{norm}
    float amax = 0.0f;
    for (int jj = 0; jj < CPT; jj++) {{
      {_UNROLL}
      for (int i = 0; i < 16; i++) amax = max(amax, fabs(v[jj][i]));
    }}
    amax = simd_max(amax);
    if (thread_index_in_simdgroup == 0) red2[simdgroup_index_in_threadgroup] = amax;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    amax = red2[0];
    for (int i = 1; i < NT / 32; i++) amax = max(amax, red2[i]);
    // Scale the row so max |x| is in [4, 8): fp16 partials stay far from overflow.
    int e = 0;
    float sc = 1.0f;
    if (amax > 0.0f) {{
      frexp(amax, e);
      sc = ldexp(1.0f, 3 - e);
    }}
    if (t == 0) rscale[m] = 1.0f / sc;
    for (int jj = 0; jj < CPT; jj++) {{
      const int c = t + jj * NT;
      if (c < C) {{
        float s = 0.0f;
        {_UNROLL}
        for (int i = 0; i < 16; i++) s += v[jj][i];
        xsum[c * Mp + m] = s * sc;
        half h[16];
{stores}
        device half4* o = (device half4*)(x16 + (size_t)m * K + c * 16);
        {_UNROLL}
        for (int i = 0; i < 4; i++) o[i] = half4(h[4 * i], h[4 * i + 1], h[4 * i + 2], h[4 * i + 3]);
      }}
    }}
"""


def _prep_gated_norm_source(K, M, D, eps):
    """Per-head RMSNorm (over D) times silu(gate), one thread per 16 values, K / 16 threads per row."""
    Mp = _mp(M)
    NT = K // 16
    assert NT % 32 == 0 and NT <= 1024 and D % 16 == 0 and D <= 16 * 32
    stores = "\n".join(
        f"    h[{c * 8 + j}] = half(v[{c * 8 + _ORDER[j]}] * (sc * {_SCALE[j]}f));" for c in range(2) for j in range(8)
    )
    return f"""
    constexpr int K = {K}, D = {D}, Mp = {Mp}, NT = {NT}, TPH = D / 16;
    const int m = threadgroup_position_in_grid.x;
    const int t = thread_position_in_threadgroup.x;
    const int k0 = t * 16;
    float v[16], g[16], wv[16];
    load16(x + (size_t)m * K + k0, v);
    load16(z + (size_t)m * K + k0, g);
    load16(weight + (k0 % D), wv);
    float ss = 0.0f;
    {_UNROLL}
    for (int i = 0; i < 16; i++) ss += v[i] * v[i];
    // The TPH threads of one head are adjacent lanes of one simdgroup.
    {_UNROLL}
    for (int o = TPH / 2; o > 0; o >>= 1) ss += simd_shuffle_xor(ss, o);
    const float rs = rsqrt(ss / D + {eps}f);
    float amax = 0.0f;
    {_UNROLL}
    for (int i = 0; i < 16; i++) {{
      v[i] = v[i] * rs * wv[i] * (g[i] * sigmoid_f(g[i]));
      amax = max(amax, fabs(v[i]));
    }}
    amax = simd_max(amax);
    threadgroup float red[NT / 32];
    if (thread_index_in_simdgroup == 0) red[simdgroup_index_in_threadgroup] = amax;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    amax = red[0];
    for (int i = 1; i < NT / 32; i++) amax = max(amax, red[i]);
    int e = 0;
    float sc = 1.0f;
    if (amax > 0.0f) {{
      frexp(amax, e);
      sc = ldexp(1.0f, 3 - e);
    }}
    if (t == 0) rscale[m] = 1.0f / sc;
    float s = 0.0f;
    {_UNROLL}
    for (int i = 0; i < 16; i++) s += v[i];
    xsum[t * Mp + m] = s * sc;
    half h[16];
{stores}
    device half4* o = (device half4*)(x16 + (size_t)m * K + k0);
    {_UNROLL}
    for (int i = 0; i < 4; i++) o[i] = half4(h[4 * i], h[4 * i + 1], h[4 * i + 2], h[4 * i + 3]);
"""


def _main_source(M, N, K, R, NSG, MC):
    Mp = _mp(M)
    NB = K // (32 * _VPL)
    wload = "\n".join(f"      {{dst}}[{r}] = *(const device uint2*)(wp + {r} * KW);" for r in range(R))
    deq = []
    for r in range(R):
        for wi in range(2):
            wd = f"wv[{r}][{wi}]"
            deq.append(f"      {{ const uint lo = {wd}, hi = {wd} >> 8;")
            deq.append(f"        q2[{r}][{wi * 4}] = as_type<half2>((lo & 0x000F000Fu) | 0x64006400u) - half2(1024.0h);")
            deq.append(f"        q2[{r}][{wi * 4 + 1}] = as_type<half2>((lo & 0x00F000F0u) | 0x64006400u) - half2(1024.0h);")
            deq.append(f"        q2[{r}][{wi * 4 + 2}] = as_type<half2>((hi & 0x000F000Fu) | 0x64006400u) - half2(1024.0h);")
            deq.append(f"        q2[{r}][{wi * 4 + 3}] = as_type<half2>((hi & 0x00F000F0u) | 0x64006400u) - half2(1024.0h); }}")
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
                    + "\n".join(f"        p = fma(q2[{r}][{j}], x2({mm}, {j}), p);" for j in range(1, 8))
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
            name=name, input_names=inputs, output_names=outputs, source=source(), header=header
        )
        _kernels[(kind, key)] = kern
    return kern


class Prepped:
    """An (M, K) activation already converted for ``qmv_small``; mimics the array's shape and dtype."""

    def __init__(self, x16, xsum, rscale, shape, dtype):
        self.x16, self.xsum, self.rscale = x16, xsum, rscale
        self.shape, self.dtype = shape, dtype
        self.ndim = len(shape)


def prep(x, kind="copy", weight=None, eps=0.0, shape=None):
    """Convert ``x`` (M, K) [or (M, 2K) for swiglu] into a ``Prepped``."""
    M, K = x.shape
    if kind == "swiglu":
        K //= 2
    Mp = _mp(M)
    inputs = [x] if weight is None else [x, weight]
    kern = _kernel(
        "qmv_small_prep_" + kind,
        (K, M, eps, _tag(x.dtype)),
        lambda: _prep_source(K, M, kind, eps),
        ["x"] if weight is None else ["x", "weight"],
        ["x16", "xsum", "rscale"],
        _HEADER,
    )
    _, nt = _prep_threads(K)
    x16, xsum, rscale = kern(
        inputs=inputs,
        template=[("T", x.dtype)],
        grid=(nt * M, 1, 1),
        threadgroup=(nt, 1, 1),
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
    if x.dtype not in (mx.bfloat16, mx.float16) or scales.dtype != x.dtype or biases.dtype != x.dtype:
        return False
    m, k = x.shape
    n = w.shape[0]
    return _MIN_M <= m <= _MAX_M and k % (32 * _VPL) == 0 and n % 8 == 0 and w.shape[1] * 8 == k


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
        return mx.quantized_matmul(x, w, scales, biases, transpose=True, group_size=group_size, bits=bits)
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
        and k % (32 * _VPL) == 0
        and module.weight.shape[0] % 8 == 0
        and module.weight.shape[1] * 8 == k
    )


def prep_rms_norm(norm, x, module):
    """``norm(x)`` fused with the prep when the projection ``module`` routes; else ``norm(x)``."""
    if not routes(module, x.shape, x.dtype):
        return norm(x)
    return prep(x.reshape(-1, x.shape[-1]), "rms_norm", norm.weight, norm.eps, shape=tuple(x.shape))


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
    if not routes(module, shape, x.dtype) or k // 16 > 1024 or d % 16 or tph > 32 or tph & (tph - 1):
        return None
    M = 1
    for b in batch:
        M *= b
    Mp = _mp(M)
    kern = _kernel(
        "qmv_small_prep_gated_norm",
        (k, M, d, norm.eps, _tag(x.dtype)),
        lambda: _prep_gated_norm_source(k, M, d, norm.eps),
        ["x", "z", "weight"],
        ["x16", "xsum", "rscale"],
        _HEADER,
    )
    x16, xsum, rscale = kern(
        inputs=[x.reshape(M, k), gate.reshape(M, k), norm.weight],
        template=[("T", x.dtype)],
        grid=(k // 16 * M, 1, 1),
        threadgroup=(k // 16, 1, 1),
        output_shapes=[(M, k), (k // 16, Mp), (Mp,)],
        output_dtypes=[mx.float16, mx.float32, mx.float32],
    )
    return Prepped(x16, xsum, rscale, shape, x.dtype)


def qlinear(module, x):
    """Apply a bias-free 4-bit g64 ``QuantizedLinear`` through ``qmv_small`` when 3 <= M <= 8."""
    if isinstance(x, Prepped):
        return qmv_main(x, module.weight, module.scales, module.biases).reshape(*x.shape[:-1], -1)
    if routes(module, x.shape, x.dtype):
        y = qmv_main(prep(x.reshape(-1, x.shape[-1])), module.weight, module.scales, module.biases)
        return y.reshape(*x.shape[:-1], -1)
    return module(x)
