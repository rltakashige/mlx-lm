# Copyright © 2026 Apple Inc.

"""Small-M 4-bit affine quantized matmul kernels for the MTP verify pass.

``mx.quantized_matmul`` only amortizes the weight read over the rows of ``x``
at M in {1, 2}; from M >= 3 it costs close to M GEMVs. These kernels stream
each weight word once, stage the K block of ``x`` in threadgroup memory and
accumulate all M rows in registers.
"""

import mlx.core as mx
import mlx.nn as nn

_BITS = 4
_GROUP = 64
_BLOCK = 512  # K values per block: 16 per lane, one uint2 of packed weights
_RPS = 4  # output rows per simdgroup
_MIN_M, _MAX_M = 2, 8
_MAX_PASS_M = 6  # larger M spills registers; run two passes instead
_MIN_N = 2048  # smaller N is launch-bound and mx.quantized_matmul wins
_NSG = 4  # simdgroups per threadgroup
_TG_MEM = 32768
_TARGET_SGS = 1280  # simdgroups needed to fill the GPU

_UNROLL = "#pragma clang loop unroll(full)"

_HEADER = """
template <typename T> struct Unpack;
template <> struct Unpack<bfloat16_t> {
  static inline float2 f(uint u) {
    return float2(as_type<float>(u << 16), as_type<float>(u & 0xffff0000u));
  }
};
template <> struct Unpack<half> {
  static inline float2 f(uint u) { return float2(as_type<half2>(u)); }
};

// Masked nibbles equal q * 16^j; x was pre-scaled by 16^-j at staging.
inline float qdot4(uint w, float4 xa) {
  return xa.x * (w & 0x000fu) + xa.y * (w & 0x00f0u) + xa.z * (w & 0x0f00u) +
      xa.w * (w & 0xf000u);
}

inline float qdot4_shift(uint w, float4 xa) {
  return xa.x * (w & 0xfu) + xa.y * ((w >> 4) & 0xfu) + xa.z * ((w >> 8) & 0xfu) +
      xa.w * ((w >> 12) & 0xfu);
}
"""


def _stage(xf, half):
    if xf:
        store = "for (int q = 0; q < 4; q++) xsm[sp][BUF][i][q][l] = v[q] * sc4;"
    elif half:
        # fp16 cannot hold the pre-scaled values; store them unscaled.
        store = """for (int q = 0; q < 4; q++)
          xsm[sp][BUF][i][q][l] = uint2(as_type<uint>(half2(v[q].x, v[q].y)), as_type<uint>(half2(v[q].z, v[q].w)));"""
    else:
        # bf16 storage is exact: the pre-scale only changes the exponent.
        store = """for (int q = 0; q < 4; q++) {
          const float4 f = v[q] * sc4;
          xsm[sp][BUF][i][q][l] = uint2((as_type<uint>(f.x) >> 16) | (as_type<uint>(f.y) & 0xffff0000u),
                                        (as_type<uint>(f.z) >> 16) | (as_type<uint>(f.w) & 0xffff0000u));
        }"""
    return f"""
      for (int c = tid; c < M * 32; c += NT) {{
        const int i = c / 32, l = c % 32;
        const device uint4* src = (const device uint4*)(x + i * K + KK + l * 16);
        const uint4 u0 = src[0], u1 = src[1];
        float4 v[4];
        {{ float2 a = Unpack<T>::f(u0.x), b2 = Unpack<T>::f(u0.y); v[0] = float4(a.x, a.y, b2.x, b2.y); }}
        {{ float2 a = Unpack<T>::f(u0.z), b2 = Unpack<T>::f(u0.w); v[1] = float4(a.x, a.y, b2.x, b2.y); }}
        {{ float2 a = Unpack<T>::f(u1.x), b2 = Unpack<T>::f(u1.y); v[2] = float4(a.x, a.y, b2.x, b2.y); }}
        {{ float2 a = Unpack<T>::f(u1.z), b2 = Unpack<T>::f(u1.w); v[3] = float4(a.x, a.y, b2.x, b2.y); }}
        xss[sp][BUF][i][l] = (v[0].x + v[0].y + v[0].z + v[0].w) + (v[1].x + v[1].y + v[1].z + v[1].w) +
            (v[2].x + v[2].y + v[2].z + v[2].w) + (v[3].x + v[3].y + v[3].z + v[3].w);
        const float4 sc4 = float4(1.0f, 1.0f / 16, 1.0f / 256, 1.0f / 4096);
        {store}
      }}"""


def _body(m, n, k, nsg, splits, xf, half):
    qdot = "qdot4"
    if xf:
        xdecl = "threadgroup float4 xsm[S][2][M][4][32];"
        xread = "const float4 xa = xsm[sp][buf][i][h][lane];"
    elif half:
        qdot = "qdot4_shift"
        xdecl = "threadgroup uint2 xsm[S][2][M][4][32];"
        xread = """const uint2 xu = xsm[sp][buf][i][h][lane];
          const half2 h0 = as_type<half2>(xu.x), h1 = as_type<half2>(xu.y);
          const float4 xa = float4(h0.x, h0.y, h1.x, h1.y);"""
    else:
        xdecl = "threadgroup uint2 xsm[S][2][M][4][32];"
        xread = """const uint2 xu = xsm[sp][buf][i][h][lane];
          const float4 xa = float4(as_type<float>(xu.x << 16), as_type<float>(xu.x & 0xffff0000u),
                                   as_type<float>(xu.y << 16), as_type<float>(xu.y & 0xffff0000u));"""
    return f"""
    constexpr int M = {m}, N = {n}, K = {k}, S = {splits}, NSG = {nsg};
    constexpr int RPS = {_RPS}, TILE = NSG * RPS, BLOCK = {_BLOCK}, NT = 32 * NSG;
    constexpr int KW = K / 8, KG = K / {_GROUP}, KS = K / S;
    const uint3 tp = thread_position_in_threadgroup;
    const int lane = tp.x, sg = tp.y, sp = tp.z, tid = sg * 32 + lane;
    const int out_row = threadgroup_position_in_grid.y * TILE + sg * RPS;
    const int row0 = min(out_row, N - RPS);
    const int k0 = sp * KS;

    const device uint32_t* wp = w + (long)row0 * KW + k0 / 8 + lane * 2;
    const device T* sc = scales + (long)row0 * KG + k0 / {_GROUP} + lane / 4;
    const device T* bs = biases + (long)row0 * KG + k0 / {_GROUP} + lane / 4;

    {xdecl}
    threadgroup float xss[S][2][M][32];

    float acc[M][RPS];
    {_UNROLL}
    for (int i = 0; i < M; i++)
      {_UNROLL}
      for (int r = 0; r < RPS; r++) acc[i][r] = 0;

    {{
      const int KK = k0; constexpr int BUF = 0;
      {_stage(xf, half)}
    }}
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (int kk = 0, blk = 0; kk < KS; kk += BLOCK, blk++) {{
      const int buf = blk & 1;
      if (kk + BLOCK < KS) {{
        const int KK = k0 + kk + BLOCK, BUF = buf ^ 1;
        {_stage(xf, half)}
      }}
      uint2 pk[RPS];
      float s[RPS], b[RPS];
      {_UNROLL}
      for (int r = 0; r < RPS; r++) {{
        pk[r] = *(const device uint2*)(wp + r * KW);
        s[r] = sc[r * KG];
        b[r] = bs[r * KG];
      }}
      float d[M][RPS];
      {_UNROLL}
      for (int h = 0; h < 4; h++) {{
        {_UNROLL}
        for (int i = 0; i < M; i++) {{
          {xread}
          {_UNROLL}
          for (int r = 0; r < RPS; r++) {{
            const uint wv = ((h & 2) ? pk[r].y : pk[r].x) >> ((h & 1) * 16);
            const float t = {qdot}(wv, xa);
            d[i][r] = (h == 0) ? t : d[i][r] + t;
          }}
        }}
      }}
      {_UNROLL}
      for (int i = 0; i < M; i++) {{
        const float xs = xss[sp][buf][i][lane];
        {_UNROLL}
        for (int r = 0; r < RPS; r++) acc[i][r] += s[r] * d[i][r] + xs * b[r];
      }}
      wp += BLOCK / 8;
      sc += BLOCK / {_GROUP};
      bs += BLOCK / {_GROUP};
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }}
    {_UNROLL}
    for (int i = 0; i < M; i++)
      {_UNROLL}
      for (int r = 0; r < RPS; r++) acc[i][r] = simd_sum(acc[i][r]);
"""


def _qmv_source(m, n, k, nsg, splits, xf, half):
    src = _body(m, n, k, nsg, splits, xf, half)
    if splits == 1:
        return (
            src
            + """
    if (lane == 0 && out_row < N) {
      for (int i = 0; i < M; i++)
        for (int r = 0; r < RPS; r++) y[(long)i * N + out_row + r] = T(acc[i][r]);
    }
"""
        )
    return (
        src
        + """
    threadgroup float part[S][TILE][M];
    if (lane == 0) {
      for (int i = 0; i < M; i++)
        for (int r = 0; r < RPS; r++) part[sp][sg * RPS + r][i] = acc[i][r];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sp == 0) {
      const int tile_row = out_row - sg * RPS;
      for (int t = tid; t < TILE * M; t += NT) {
        const int r = t / M, i = t % M;
        if (tile_row + r < N) {
          float v = 0;
          for (int j = 0; j < S; j++) v += part[j][r][i];
          y[(long)i * N + tile_row + r] = T(v);
        }
      }
    }
"""
    )


def _qargmax_source(m, n, k, nsg, xf, half):
    return (
        _body(m, n, k, nsg, 1, xf, half)
        + """
    threadgroup float bv[NSG][M];
    threadgroup uint bi[NSG][M];
    if (lane == 0) {
      for (int i = 0; i < M; i++) {
        float best = -INFINITY;
        uint idx = 0;
        for (int r = 0; r < RPS; r++) {
          // Compare after rounding to T so ties match mx.argmax on T logits.
          const float v = (out_row + r < N) ? float(T(acc[i][r])) : -INFINITY;
          if (v > best) { best = v; idx = out_row + r; }
        }
        bv[sg][i] = best;
        bi[sg][i] = idx;
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < M) {
      float best = bv[0][tid];
      uint idx = bi[0][tid];
      for (int g = 1; g < NSG; g++) {
        if (bv[g][tid] > best) { best = bv[g][tid]; idx = bi[g][tid]; }
      }
      const long o = (long)tid * ((N + TILE - 1) / TILE) + threadgroup_position_in_grid.y;
      vals[o] = T(best);
      idxs[o] = idx;
    }
"""
    )


_ARGMAX_REDUCE = """
    constexpr int NT = 1024;
    const int i = threadgroup_position_in_grid.y;
    const int t = thread_position_in_threadgroup.x;
    const int n = vals_shape[1];
    const device T* v = vals + (long)i * n;
    const device uint* ix = idxs + (long)i * n;
    float best = -INFINITY;
    uint bidx = 0xffffffffu;
    for (int j = t; j < n; j += NT) {
      const float c = v[j];
      const uint ci = ix[j];
      if (c > best || (c == best && ci < bidx)) { best = c; bidx = ci; }
    }
    for (int off = 16; off > 0; off >>= 1) {
      const float c = simd_shuffle_down(best, off);
      const uint ci = simd_shuffle_down(bidx, off);
      if (c > best || (c == best && ci < bidx)) { best = c; bidx = ci; }
    }
    threadgroup float sv[NT / 32];
    threadgroup uint si[NT / 32];
    if ((t & 31) == 0) { sv[t / 32] = best; si[t / 32] = bidx; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (t == 0) {
      for (int j = 1; j < NT / 32; j++) {
        if (sv[j] > best || (sv[j] == best && si[j] < bidx)) { best = sv[j]; bidx = si[j]; }
      }
      out[i] = bidx;
    }
"""

_kernels = {}


def _kernel(kind, dtype, m, n, k, nsg, splits, xf):
    key = (kind, dtype, m, n, k, nsg, splits, xf)
    kern = _kernels.get(key)
    if kern is None:
        half = dtype == mx.float16
        tag = "fp16" if half else "bf16"
        if kind == "qmv":
            source, inputs, outputs = (
                _qmv_source(m, n, k, nsg, splits, xf, half),
                ["x", "w", "scales", "biases"],
                ["y"],
            )
        elif kind == "qargmax":
            source, inputs, outputs = (
                _qargmax_source(m, n, k, nsg, xf, half),
                ["x", "w", "scales", "biases"],
                ["vals", "idxs"],
            )
        else:
            source, inputs, outputs = _ARGMAX_REDUCE, ["vals", "idxs"], ["out"]
        kern = mx.fast.metal_kernel(
            name=f"qmv4_{kind}_{tag}_m{m}_n{n}_k{k}_g{nsg}_s{splits}_x{int(xf)}",
            input_names=inputs,
            output_names=outputs,
            header=_HEADER if kind != "reduce" else "",
            source=source,
        )
        _kernels[key] = kern
    return kern


def _tg_bytes(m, nsg, splits, xf):
    x_bytes = splits * 2 * m * (2048 if xf else 1024)
    xs_bytes = splits * 2 * m * 128
    part = splits * nsg * _RPS * m * 4 if splits > 1 else 0
    return x_bytes + xs_bytes + part


def pick_splits(n, k):
    """Split K across simdgroups of a threadgroup until the GPU is full."""
    sgs = -(-n // _RPS)
    splits = 1
    for cand in (2, 4, 8):
        if k % (_BLOCK * cand) != 0 or sgs * splits >= _TARGET_SGS:
            break
        splits = cand
    return splits


def _config(m, n, k, splits=None):
    if splits is None:
        splits = pick_splits(n, k)
    while splits > 1 and (
        _NSG * splits * 32 > 1024 or _tg_bytes(m, _NSG, splits, False) > _TG_MEM
    ):
        splits //= 2
    # Float staging is exact and cheaper to read, but its footprint costs occupancy at M > 3.
    xf = m <= 3 and _tg_bytes(m, _NSG, splits, True) <= _TG_MEM
    return _NSG, splits, xf


def _supported(x, w, scales, biases, group_size, bits):
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
        k % _BLOCK == 0
        and n % 8 == 0
        and n >= _RPS
        and w.shape[1] * 8 == k
        and 1 <= m <= _MAX_M
    )


def _qmv(x, w, scales, biases, nsg, splits, xf):
    m, k = x.shape
    n = w.shape[0]
    tile = nsg * _RPS
    (y,) = _kernel("qmv", x.dtype, m, n, k, nsg, splits, xf)(
        inputs=[x, w, scales, biases],
        template=[("T", x.dtype)],
        grid=(32, nsg * (-(-n // tile)), splits),
        threadgroup=(32, nsg, splits),
        output_shapes=[(m, n)],
        output_dtypes=[x.dtype],
    )
    return y


def qmv_small_m(x, w, scales, biases, group_size=64, bits=4, splits=None):
    """``x @ dequant(w).T`` for ``x`` of shape (M, K) with 2 <= M <= 8."""
    if not _supported(x, w, scales, biases, group_size, bits) or x.shape[0] < _MIN_M:
        return mx.quantized_matmul(
            x, w, scales, biases, transpose=True, group_size=group_size, bits=bits
        )
    m, k = x.shape
    if m > _MAX_PASS_M:
        h = (m + 1) // 2
        return mx.concatenate(
            [
                qmv_small_m(x[:h], w, scales, biases, splits=splits),
                qmv_small_m(x[h:], w, scales, biases, splits=splits),
            ]
        )
    return _qmv(x, w, scales, biases, *_config(m, w.shape[0], k, splits))


def qlinear(module, x):
    """Apply a bias-free 4-bit g64 ``QuantizedLinear`` through ``qmv_small_m``."""
    *batch, k = x.shape
    m = 1
    for d in batch:
        m *= d
    if (
        isinstance(module, nn.QuantizedLinear)
        and module.bits == _BITS
        and module.group_size == _GROUP
        and "bias" not in module
        and _MIN_M <= m <= _MAX_M
        and module.weight.shape[0] >= _MIN_N
    ):
        x2 = x.reshape(m, k)
        if _supported(x2, module.weight, module.scales, module.biases, _GROUP, _BITS):
            y = qmv_small_m(x2, module.weight, module.scales, module.biases)
            return y.reshape(*batch, -1)
    return module(x)


def qargmax(x, w, scales, biases, group_size=64, bits=4):
    """``argmax(x @ dequant(w).T, axis=-1)`` as uint32 without materializing the logits."""
    if not _supported(x, w, scales, biases, group_size, bits):
        y = mx.quantized_matmul(
            x, w, scales, biases, transpose=True, group_size=group_size, bits=bits
        )
        return mx.argmax(y, axis=-1).astype(mx.uint32)
    m, k = x.shape
    n = w.shape[0]
    if m > _MAX_PASS_M:
        h = (m + 1) // 2
        return mx.concatenate(
            [qargmax(x[:h], w, scales, biases), qargmax(x[h:], w, scales, biases)]
        )
    nsg, _, xf = _config(m, n, k, 1)
    tile = nsg * _RPS
    tiles = -(-n // tile)
    vals, idxs = _kernel("qargmax", x.dtype, m, n, k, nsg, 1, xf)(
        inputs=[x, w, scales, biases],
        template=[("T", x.dtype)],
        grid=(32, nsg * tiles, 1),
        threadgroup=(32, nsg, 1),
        output_shapes=[(m, tiles), (m, tiles)],
        output_dtypes=[x.dtype, mx.uint32],
    )
    (out,) = _kernel("reduce", x.dtype, 0, 0, 0, 0, 1, True)(
        inputs=[vals, idxs],
        template=[("T", x.dtype)],
        grid=(1024, m, 1),
        threadgroup=(1024, 1, 1),
        output_shapes=[(m,)],
        output_dtypes=[mx.uint32],
    )
    return out
