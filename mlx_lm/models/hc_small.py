# Copyright © 2026 Apple Inc.

"""Small-M kernels for the hyper-connection sites of qwen4_exp (Flash-Next).

A site mixes the ``hc`` residual streams into one block input: a per-stream
RMSNorm, a low-rank down projection (silu), an up projection (sigmoid) and the
mean of the gated streams; its last ``hc`` down rows are the injection gates of
the block output. Two kernels replace the ~9 ops of a site:

``down``: the pending combine of the previous block (streams + x * inject), the
norm statistics, the down matvec and its activations. Every threadgroup owns R
down rows; simdgroup s reads stream s, so the norm of a stream stays inside one
simdgroup and the K split needs no second pass.
``up``: the up matvec with the down output in threadgroup memory, the sigmoid
gate and the mean over the streams; a threadgroup owns 32 dims of every stream.

The weights are 4-bit affine (group size 32 or 64). Rounding follows the ops
(bf16 at the op boundaries); the float accumulation order differs from
``mx.quantized_matmul``, so the results agree to bf16 rounding, not bitwise.
"""

import mlx.core as mx
import mlx.nn as nn

from .qmv_small import _HEADER, _UNROLL, _kernel, _tag

_MAX_M = 4
# Down projection: 8 rows per threadgroup and 4 simdgroups per stream (16 simdgroups), so the
# streams are read once per 8 rows and 41 threadgroups keep ~1.7 MB of weight loads in flight.
_R = 8
_KS = 4

from .fused_ops import _SIGMOID_T

_HC_HEADER = _HEADER + _SIGMOID_T + """
// The block output x: bf16, or the float32 atomic sum of the MoE rounded like the ops' sum
template <typename T>
inline void loadx(const device float* p, thread float* v) {
  const device float4* q = (const device float4*)p;
  #pragma clang loop unroll(full)
  for (int i = 0; i < 4; i++) { const float4 f = q[i]; v[4 * i] = float(T(f.x)); v[4 * i + 1] = float(T(f.y)); v[4 * i + 2] = float(T(f.z)); v[4 * i + 3] = float(T(f.w)); }
}
template <typename T>
inline void loadx(const device T* p, thread float* v) { load16(p, v); }
"""


def _down_source(M, HC, DIMS, LR, N, R, G, eps, KS):
    HCD = HC * DIMS
    NCH = DIMS // 512  # 16-value chunks per lane per stream
    NJ = -(-NCH // KS)  # chunks per simdgroup
    return f"""
    constexpr int M = {M}, HC = {HC}, DIMS = {DIMS}, HCD = {HCD}, LR = {LR}, N = {N}, R = {R}, KS = {KS};
    constexpr int G = {G}, KW = HCD / 8, KG = HCD / G, NCH = {NCH}, NJ = {NJ}, NSG = HC * KS;
    constexpr float EPS = {eps!r}f;
    const int lane = thread_index_in_simdgroup;
    const int sg = simdgroup_index_in_threadgroup;
    // Simdgroup sg: stream s, chunks i = part, part + KS, ...
    const int s = sg % HC, part = sg / HC;
    const int tg = threadgroup_position_in_grid.x;
    const int row0 = tg * R;
    // The R rows' nibbles of this simdgroup's chunks, requested before the norm pass
    uint2 wq[NJ][R];
    #pragma clang loop unroll(full)
    for (int j = 0; j < NJ; j++) {{
      const int i = part + j * KS;
      #pragma clang loop unroll(full)
      for (int r = 0; r < R; r++)
        wq[j][r] = i < NCH ? *(const device uint2*)(w + (size_t)min(row0 + r, N - 1) * KW + (s * DIMS) / 8 + 2 * lane + 64 * i) : uint2(0u);
    }}
    threadgroup float ssum[NSG][M];
    threadgroup float partial[NSG][R][M];
    // The row loops stay runtime loops: large unrolled bodies miscompile (M4 Max, Metal 4)
    for (int m = 0; m < M; m++) {{
      const float inj = has_combine ? float(inject[m * HC + s]) : 0.0f;
      // Pass 1: the combined stream and its partial sum of squares
      float ss = 0.0f;
      for (int i = part; i < NCH; i += KS) {{
        const int k = s * DIMS + 16 * (lane + 32 * i);
        float hv[16];
        load16(h + (size_t)m * HCD + k, hv);
        if (has_combine) {{
          float xv[16];
          loadx<T>(x + (size_t)m * DIMS + k - s * DIMS, xv);
          #pragma clang loop unroll(full)
          for (int e = 0; e < 16; e++) hv[e] = float(T(hv[e] + float(T(xv[e] * inj))));
          if (tg == 0) store16(h_out + (size_t)m * HCD + k, hv);
        }}
        #pragma clang loop unroll(full)
        for (int e = 0; e < 16; e++) ss += hv[e] * hv[e];
      }}
      ss = simd_sum(ss);
      if (lane == 0) ssum[sg][m] = ss;
      threadgroup_barrier(mem_flags::mem_threadgroup);
      float tot = 0.0f;
      for (int q = 0; q < KS; q++) tot += ssum[s + q * HC][m];
      const float rs = metal::precise::rsqrt(tot / DIMS + EPS);
      if (tg == 0 && sg == s && lane == 0) rs_out[m * HC + s] = rs;
      // Pass 2: the normed values times the nibbles
      float acc[R];
      #pragma clang loop unroll(full)
      for (int r = 0; r < R; r++) acc[r] = 0.0f;
      for (int j = 0; j < NJ; j++) {{
        const int i = part + j * KS;
        if (i >= NCH) break;
        const int k = s * DIMS + 16 * (lane + 32 * i);
        const int grp = k / G;
        float hv[16], gv[16];
        load16(gain + k, gv);
        load16(h + (size_t)m * HCD + k, hv);
        if (has_combine) {{
          float xv[16];
          loadx<T>(x + (size_t)m * DIMS + k - s * DIMS, xv);
          #pragma clang loop unroll(full)
          for (int e = 0; e < 16; e++) hv[e] = float(T(hv[e] + float(T(xv[e] * inj))));
        }}
        float xs = 0.0f;
        #pragma clang loop unroll(full)
        for (int e = 0; e < 16; e++) {{
          hv[e] = float(T(float(T(hv[e] * rs)) * gv[e]));
          xs += hv[e];
        }}
        #pragma clang loop unroll(full)
        for (int r = 0; r < R; r++) {{
          const size_t sb = (size_t)min(row0 + r, N - 1) * KG + grp;
          const float sc = float(scales[sb]), bi = float(biases[sb]);
          float d = 0.0f;
          #pragma clang loop unroll(full)
          for (int e = 0; e < 8; e++) {{
            d = fma(float((wq[j][r].x >> (4 * e)) & 0xFu), hv[e], d);
            d = fma(float((wq[j][r].y >> (4 * e)) & 0xFu), hv[8 + e], d);
          }}
          acc[r] = fma(sc, d, fma(bi, xs, acc[r]));
        }}
      }}
      #pragma clang loop unroll(full)
      for (int r = 0; r < R; r++) {{
        const float v = simd_sum(acc[r]);
        if (lane == 0) partial[sg][r][m] = v;
      }}
    }}
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0 && lane < R && row0 + lane < N) {{
      const int row = row0 + lane;
      for (int m = 0; m < M; m++) {{
        float tot = 0.0f;
        for (int q = 0; q < NSG; q++) tot += partial[q][lane][m];
        // The ops: bf16 matvec output, times 1 / hc (exact), silu or 2 * sigmoid
        const float d = float(T(tot)) * (1.0f / HC);
        if (row < LR) g_out[(size_t)m * LR + row] = T(d * sigmoid_t<T>(d));
        else inject_out[(size_t)m * HC + row - LR] = T(2.0f * sigmoid_t<T>(d));
      }}
    }}
"""


def _up_source(M, HC, DIMS, LR, G, LPR):
    HCD = HC * DIMS
    KG = LR // G
    KW = LR // 8
    RPS = 32 // LPR  # rows (dims) per simdgroup
    NW = KW // LPR  # nibble words per lane
    NH = NW // 2  # 16-value half groups per lane: a half group lies inside one scale group
    return f"""
    constexpr int M = {M}, HC = {HC}, DIMS = {DIMS}, HCD = {HCD}, LR = {LR};
    constexpr int G = {G}, KG = {KG}, KW = {KW}, LPR = {LPR}, RPS = {RPS}, NW = {NW}, NH = {NH}, NT = 32 * HC;
    const int lane = thread_index_in_simdgroup;
    const int s = simdgroup_index_in_threadgroup;
    const int t = thread_position_in_threadgroup.x;
    // Lane l of a row group holds the row's K values [l * NW * 8, (l + 1) * NW * 8)
    const int rg = lane / LPR, l = lane % LPR;
    const int d = threadgroup_position_in_grid.x * RPS + rg;
    const int row = s * DIMS + d;
    const int k0 = l * NW * 8;
    uint wq[NW];
    {_UNROLL}
    for (int i = 0; i < NW; i++) wq[i] = w[(size_t)row * KW + l * NW + i];
    // The down output of every row, shared by the threadgroup
    threadgroup float gsh[M][LR];
    for (int i = t; i < M * LR; i += NT) gsh[i / LR][i % LR] = float(g[i]);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float acc[M];
    {_UNROLL}
    for (int m = 0; m < M; m++) acc[m] = 0.0f;
    {_UNROLL}
    for (int hg = 0; hg < NH; hg++) {{
      const int grp = (k0 + 16 * hg) / G;
      const float sc = float(scales[(size_t)row * KG + grp]), bi = float(biases[(size_t)row * KG + grp]);
      float dd[M], xs[M];
      {_UNROLL}
      for (int m = 0; m < M; m++) {{ dd[m] = 0.0f; xs[m] = 0.0f; }}
      {_UNROLL}
      for (int e = 0; e < 16; e++) {{
        const float q = float((wq[2 * hg + e / 8] >> (4 * (e % 8))) & 0xFu);
        {_UNROLL}
        for (int m = 0; m < M; m++) {{
          const float gv = gsh[m][k0 + 16 * hg + e];
          dd[m] = fma(q, gv, dd[m]);
          xs[m] += gv;
        }}
      }}
      {_UNROLL}
      for (int m = 0; m < M; m++) acc[m] = fma(sc, dd[m], fma(bi, xs[m], acc[m]));
    }}
    {_UNROLL}
    for (int m = 0; m < M; m++) {{
      {_UNROLL}
      for (int o = LPR / 2; o > 0; o >>= 1) acc[m] += simd_shuffle_xor(acc[m], o);
    }}
    // sigmoid gate times the normed stream value, mean over the streams
    threadgroup float part[HC][RPS][M];
    if (l == 0) {{
      {_UNROLL}
      for (int m = 0; m < M; m++) {{
        const float wt = sigmoid_t<T>(float(T(acc[m])));
        const float hv = float(h[(size_t)m * HCD + row]);
        const float xn = float(T(float(T(hv * rs[m * HC + s])) * float(gain[row])));
        part[s][rg][m] = float(T(wt * xn));
      }}
    }}
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (s == 0 && l == 0) {{
      {_UNROLL}
      for (int m = 0; m < M; m++) {{
        float tot = 0.0f;
        for (int j = 0; j < HC; j++) tot += part[j][rg][m];
        mixed[(size_t)m * DIMS + d] = T(tot / HC);
      }}
    }}
"""


def _up_lanes(LR, G):
    """Lanes per up row: the most that leaves each lane whole 16-value half groups."""
    for lpr in (8, 4, 2):
        if (LR // 8) % lpr == 0 and (LR // lpr) % 16 == 0 and G % 16 == 0:
            return lpr
    return 1


def _quant(module):
    """(group_size, bits) of a 4-bit affine QuantizedLinear without bias, else None."""
    if (
        not isinstance(module, nn.QuantizedLinear)
        or module.bits != 4
        or getattr(module, "mode", "affine") != "affine"
        or "bias" in module
    ):
        return None
    return module.group_size, module.bits


def routes(site, hyper):
    """True when the kernels handle ``site`` on ``hyper`` (.., hc * dims)."""
    if not mx.metal.is_available() or hyper.dtype not in (mx.bfloat16, mx.float16):
        return False
    rows = hyper.size // hyper.shape[-1]
    down, up = site.input_mix_weight_down, site.input_mix_weight_up
    qd, qu = _quant(down), _quant(up)
    return (
        qd is not None
        and qu is not None
        and 1 <= rows <= _MAX_M
        and 1 <= site.hc * _KS <= 32
        and site.dims % 512 == 0
        and site.lowrank % 16 == 0
        and site.lowrank % qu[0] == 0
        and site.lowrank % 8 == 0
        and down.scales.dtype == hyper.dtype
        and up.scales.dtype == hyper.dtype
    )


def mix(site, hyper, pending=None):
    """The site on ``hyper`` (.., hc * dims) with the previous block's ``pending``
    (x, inject) combined first: (mixed, inject or None, combined streams)."""
    shape = hyper.shape
    HCD = shape[-1]
    HC, DIMS, LR = site.hc, site.dims, site.lowrank
    M = hyper.size // HCD
    down, up = site.input_mix_weight_down, site.input_mix_weight_up
    N = down.weight.shape[0]
    G, G2 = down.group_size, up.group_size
    T = hyper.dtype
    h = hyper.reshape(M, HCD)
    if pending is None:
        x, inject, has = h, h, 0
    else:
        x, inject, has = pending[0].reshape(M, DIMS), pending[1].reshape(M, HC), 1
        if x.dtype not in (T, mx.float32):
            x = x.astype(T)
    gain = site.hc_norm.gain()
    kern = _kernel(
        "hc_down",
        (M, HC, DIMS, LR, N, _R, G, site.hc_norm.eps, _KS, _tag(T), _tag(x.dtype)),
        lambda: _down_source(M, HC, DIMS, LR, N, _R, G, site.hc_norm.eps, _KS),
        ["h", "x", "inject", "gain", "w", "scales", "biases", "has_combine"],
        ["h_out", "rs_out", "g_out", "inject_out"],
        _HC_HEADER,
    )
    ntg = -(-N // _R)
    h_out, rs, g, inject_out = kern(
        inputs=[h, x, inject, gain, down.weight, down.scales, down.biases, has],
        template=[("T", T)],
        grid=(32 * HC * _KS * ntg, 1, 1),
        threadgroup=(32 * HC * _KS, 1, 1),
        output_shapes=[(M, HCD), (M, HC), (M, LR), (M, HC)],
        output_dtypes=[T, mx.float32, T, T],
    )
    combined = h_out if has else h
    LPR = _up_lanes(LR, G2)
    kern = _kernel(
        "hc_up",
        (M, HC, DIMS, LR, G2, LPR, _tag(T)),
        lambda: _up_source(M, HC, DIMS, LR, G2, LPR),
        ["h", "rs", "gain", "g", "w", "scales", "biases"],
        ["mixed"],
        _HC_HEADER,
    )
    (mixed,) = kern(
        inputs=[combined, rs, gain, g, up.weight, up.scales, up.biases],
        template=[("T", T)],
        grid=(32 * HC * (DIMS // (32 // LPR)), 1, 1),
        threadgroup=(32 * HC, 1, 1),
        output_shapes=[(M, DIMS)],
        output_dtypes=[T],
    )
    mixed = mixed.reshape(*shape[:-1], DIMS)
    inject_out = inject_out.reshape(*shape[:-1], HC) if N > LR else None
    return mixed, inject_out, combined.reshape(shape)
