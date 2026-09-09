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

_MAX_M = 8
_R = 4  # down rows per threadgroup

from .fused_ops import _SIGMOID_T

_HC_HEADER = _HEADER + _SIGMOID_T


def _down_source(M, HC, DIMS, LR, N, R, G, eps):
    HCD = HC * DIMS
    NCH = DIMS // 512  # 16-value chunks per lane per stream
    return f"""
    constexpr int M = {M}, HC = {HC}, DIMS = {DIMS}, HCD = {HCD}, LR = {LR}, N = {N}, R = {R};
    constexpr int G = {G}, KW = HCD / 8, KG = HCD / G, NCH = {NCH};
    constexpr float EPS = {eps!r}f;
    const int lane = thread_index_in_simdgroup;
    const int s = simdgroup_index_in_threadgroup;
    const int tg = threadgroup_position_in_grid.x;
    const int row0 = tg * R;
    const float inj[M] = {{ {", ".join(f"has_combine ? float(inject[{m} * HC + s]) : 0.0f" for m in range(M))} }};
    // Pass 1: the combined streams and the sum of squares of stream s
    float ss[M];
    {_UNROLL}
    for (int m = 0; m < M; m++) ss[m] = 0.0f;
    {_UNROLL}
    for (int i = 0; i < NCH; i++) {{
      const int k = s * DIMS + 16 * (lane + 32 * i);
      {_UNROLL}
      for (int m = 0; m < M; m++) {{
        float hv[16];
        load16(h + (size_t)m * HCD + k, hv);
        if (has_combine) {{
          float xv[16];
          load16(x + (size_t)m * DIMS + k - s * DIMS, xv);
          {_UNROLL}
          for (int e = 0; e < 16; e++) hv[e] = float(T(hv[e] + float(T(xv[e] * inj[m]))));
          if (tg == 0) store16(h_out + (size_t)m * HCD + k, hv);
        }}
        {_UNROLL}
        for (int e = 0; e < 16; e++) ss[m] += hv[e] * hv[e];
      }}
    }}
    float rs[M];
    {_UNROLL}
    for (int m = 0; m < M; m++) {{
      rs[m] = metal::precise::rsqrt(simd_sum(ss[m]) / DIMS + EPS);
      if (tg == 0 && lane == 0) rs_out[m * HC + s] = rs[m];
    }}
    // Pass 2: the normed values times the R rows' nibbles of stream s
    float acc[R][M];
    {_UNROLL}
    for (int r = 0; r < R; r++)
      {_UNROLL}
      for (int m = 0; m < M; m++) acc[r][m] = 0.0f;
    uint2 wq[NCH][R];
    {_UNROLL}
    for (int i = 0; i < NCH; i++)
      {_UNROLL}
      for (int r = 0; r < R; r++)
        wq[i][r] = *(const device uint2*)(w + (size_t)min(row0 + r, N - 1) * KW + (s * DIMS) / 8 + 2 * lane + 64 * i);
    {_UNROLL}
    for (int i = 0; i < NCH; i++) {{
      const int k = s * DIMS + 16 * (lane + 32 * i);
      const int grp = k / G;
      float gv[16];
      load16(gain + k, gv);
      float xn[M][16], xs[M];
      {_UNROLL}
      for (int m = 0; m < M; m++) {{
        float hv[16];
        load16(h + (size_t)m * HCD + k, hv);
        if (has_combine) {{
          float xv[16];
          load16(x + (size_t)m * DIMS + k - s * DIMS, xv);
          {_UNROLL}
          for (int e = 0; e < 16; e++) hv[e] = float(T(hv[e] + float(T(xv[e] * inj[m]))));
        }}
        xs[m] = 0.0f;
        {_UNROLL}
        for (int e = 0; e < 16; e++) {{
          xn[m][e] = float(T(float(T(hv[e] * rs[m])) * gv[e]));
          xs[m] += xn[m][e];
        }}
      }}
      {_UNROLL}
      for (int r = 0; r < R; r++) {{
        const size_t sb = (size_t)min(row0 + r, N - 1) * KG + grp;
        const float sc = float(scales[sb]), bi = float(biases[sb]);
        float q[16];
        {_UNROLL}
        for (int e = 0; e < 8; e++) {{
          q[e] = float((wq[i][r].x >> (4 * e)) & 0xFu);
          q[8 + e] = float((wq[i][r].y >> (4 * e)) & 0xFu);
        }}
        {_UNROLL}
        for (int m = 0; m < M; m++) {{
          float d = 0.0f;
          {_UNROLL}
          for (int e = 0; e < 16; e++) d = fma(q[e], xn[m][e], d);
          acc[r][m] = fma(sc, d, fma(bi, xs[m], acc[r][m]));
        }}
      }}
    }}
    threadgroup float part[HC][R][M];
    {_UNROLL}
    for (int r = 0; r < R; r++)
      {_UNROLL}
      for (int m = 0; m < M; m++) {{
        const float v = simd_sum(acc[r][m]);
        if (lane == 0) part[s][r][m] = v;
      }}
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (s == 0 && lane < R && row0 + lane < N) {{
      const int row = row0 + lane;
      {_UNROLL}
      for (int m = 0; m < M; m++) {{
        float tot = 0.0f;
        for (int j = 0; j < HC; j++) tot += part[j][lane][m];
        // The ops: bf16 matvec output, times 1 / hc (exact), silu or 2 * sigmoid
        const float d = float(T(tot)) * (1.0f / HC);
        if (row < LR) g_out[(size_t)m * LR + row] = T(d * sigmoid_t<T>(d));
        else inject_out[(size_t)m * HC + row - LR] = T(2.0f * sigmoid_t<T>(d));
      }}
    }}
"""


def _up_source(M, HC, DIMS, LR, G):
    HCD = HC * DIMS
    KG = LR // G
    return f"""
    constexpr int M = {M}, HC = {HC}, DIMS = {DIMS}, HCD = {HCD}, LR = {LR};
    constexpr int G = {G}, KG = {KG}, KW = LR / 8, NT = 32 * HC;
    const int lane = thread_index_in_simdgroup;
    const int s = simdgroup_index_in_threadgroup;
    const int t = thread_position_in_threadgroup.x;
    const int d = threadgroup_position_in_grid.x * 32 + lane;
    const int row = s * DIMS + d;
    // The down output of every row and its per-group sums, shared by the threadgroup
    threadgroup float gsh[M][LR];
    threadgroup float gsum[M][KG];
    for (int i = t; i < M * LR; i += NT) gsh[i / LR][i % LR] = float(g[i]);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (t < M * KG) {{
      float sum = 0.0f;
      for (int e = 0; e < G; e++) sum += gsh[t / KG][(t % KG) * G + e];
      gsum[t / KG][t % KG] = sum;
    }}
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint4 wq[KW / 4];
    {_UNROLL}
    for (int i = 0; i < KW / 4; i++) wq[i] = *(const device uint4*)(w + (size_t)row * KW + 4 * i);
    float acc[M];
    {_UNROLL}
    for (int m = 0; m < M; m++) acc[m] = 0.0f;
    {_UNROLL}
    for (int gi = 0; gi < KG; gi++) {{
      const float sc = float(scales[(size_t)row * KG + gi]), bi = float(biases[(size_t)row * KG + gi]);
      float dd[M];
      {_UNROLL}
      for (int m = 0; m < M; m++) dd[m] = 0.0f;
      {_UNROLL}
      for (int e = 0; e < G; e++) {{
        const int k = gi * G + e;
        const uint word = ((const thread uint*)wq)[k / 8];
        const float q = float((word >> (4 * (k % 8))) & 0xFu);
        {_UNROLL}
        for (int m = 0; m < M; m++) dd[m] = fma(q, gsh[m][k], dd[m]);
      }}
      {_UNROLL}
      for (int m = 0; m < M; m++) acc[m] = fma(sc, dd[m], fma(bi, gsum[m][gi], acc[m]));
    }}
    // sigmoid gate times the normed stream value, mean over the streams
    threadgroup float part[HC][32][M];
    {_UNROLL}
    for (int m = 0; m < M; m++) {{
      const float wt = sigmoid_t<T>(float(T(acc[m])));
      const float hv = float(h[(size_t)m * HCD + row]);
      const float xn = float(T(float(T(hv * rs[m * HC + s])) * float(gain[row])));
      part[s][lane][m] = float(T(wt * xn));
    }}
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (s == 0) {{
      {_UNROLL}
      for (int m = 0; m < M; m++) {{
        float tot = 0.0f;
        for (int j = 0; j < HC; j++) tot += part[j][lane][m];
        mixed[(size_t)m * DIMS + d] = T(tot / HC);
      }}
    }}
"""


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
        and 1 <= site.hc <= 8
        and site.dims % 512 == 0
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
    gain = site.hc_norm.gain()
    kern = _kernel(
        "hc_down",
        (M, HC, DIMS, LR, N, _R, G, site.hc_norm.eps, _tag(T)),
        lambda: _down_source(M, HC, DIMS, LR, N, _R, G, site.hc_norm.eps),
        ["h", "x", "inject", "gain", "w", "scales", "biases", "has_combine"],
        ["h_out", "rs_out", "g_out", "inject_out"],
        _HC_HEADER,
    )
    ntg = -(-N // _R)
    h_out, rs, g, inject_out = kern(
        inputs=[h, x, inject, gain, down.weight, down.scales, down.biases, has],
        template=[("T", T)],
        grid=(32 * HC * ntg, 1, 1),
        threadgroup=(32 * HC, 1, 1),
        output_shapes=[(M, HCD), (M, HC), (M, LR), (M, HC)],
        output_dtypes=[T, mx.float32, T, T],
    )
    combined = h_out if has else h
    kern = _kernel(
        "hc_up",
        (M, HC, DIMS, LR, G2, _tag(T)),
        lambda: _up_source(M, HC, DIMS, LR, G2),
        ["h", "rs", "gain", "g", "w", "scales", "biases"],
        ["mixed"],
        _HC_HEADER,
    )
    (mixed,) = kern(
        inputs=[combined, rs, gain, g, up.weight, up.scales, up.biases],
        template=[("T", T)],
        grid=(32 * HC * (DIMS // 32), 1, 1),
        threadgroup=(32 * HC, 1, 1),
        output_shapes=[(M, DIMS)],
        output_dtypes=[T],
    )
    mixed = mixed.reshape(*shape[:-1], DIMS)
    inject_out = inject_out.reshape(*shape[:-1], HC) if N > LR else None
    return mixed, inject_out, combined.reshape(shape)
