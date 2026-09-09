# Copyright © 2026 Apple Inc.

"""Merged small kernels of the Qwen3.5 decode step.

Each kernel replaces a chain of MLX ops by one dispatch and rounds where the ops
round (bf16 intermediates at the same places), so its output is bitwise the
output of the ops it replaces.
"""

import os

import mlx.core as mx

from .qmv_small import _HEADER as _SMALL_HEADER
from .qmv_small import _kernel, _nax_m, _rows, prep, prep_rms_norm, routes

_ENABLED = os.environ.get("MLX_QWEN_FUSED", "1") != "0"

# The MLX ops as the model compiles them: Sigmoid and LogAddExp in the input type,
# Exp in float. The eager (library) Sigmoid kernel uses the precise exp.
_HEADER = """
struct Sigmoid {
  template <typename T> T operator()(T x) {
    auto y = 1 / (1 + metal::exp(metal::abs(x)));
    return (x < 0) ? y : 1 - y;
  }
};
struct SigmoidLib {
  template <typename T> T operator()(T x) {
    auto y = 1 / (1 + metal::precise::exp(metal::abs(x)));
    return (x < 0) ? y : 1 - y;
  }
};
struct LogAddExp {
  template <typename T> T operator()(T x, T y) {
    if (metal::isnan(x) || metal::isnan(y)) return metal::numeric_limits<T>::quiet_NaN();
    constexpr T inf = metal::numeric_limits<T>::infinity();
    T maxval = metal::max(x, y);
    T minval = metal::min(x, y);
    return (minval == -inf || maxval == inf) ? maxval : (maxval + log1p(metal::exp(minval - maxval)));
  }
};

// mx.fast.rms_norm's reduction: 4 values per thread, simd sums, then the simd sums summed.
inline float rms_sum(float acc, threadgroup float* sums, uint lane, uint sg) {
  acc = simd_sum(acc);
  if (sg == 0) sums[lane] = 0.0f;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (lane == 0) sums[sg] = acc;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  return simd_sum(sums[lane]);
}
"""


# The compiled Sigmoid op in the activation type: every step rounds to T
_SIGMOID_T = """
template <typename T>
inline float sigmoid_t(float x) {
  const float a = float(T(metal::exp(metal::abs(x))));
  const float y = float(T(1.0f / float(T(1.0f + a))));
  return (x < 0.0f) ? y : float(T(1.0f - y));
}
"""


def enabled():
    return _ENABLED and mx.metal.is_available()


def _rms_threads(k):
    """Threads of mx.fast.rms_norm's threadgroup for a row of ``k`` values."""
    if k > 4096:
        return 1024
    return ((k + 3) // 4 + 31) // 32 * 32


def slot_sum(pending):
    """Sum the expert slot rows (.., S, K) of a MoE output; other pendings pass through."""
    if pending is not None and pending.ndim == 4:
        return pending.sum(axis=-2)
    return pending


def prep_add_rms_norm(norm, x, residual, module, fused=True):
    """``(x + residual, norm(x + residual))``, the add merged into the norm or its prep.

    The second value is a ``Prepped`` when the projection ``module`` routes, else an array.
    A residual of expert slot rows (.., S, K) is summed as ``mx.sum`` does before the add.
    """
    if residual is None:
        return x, prep_rms_norm(norm, x, module)
    k = x.shape[-1]
    rows = residual.shape[-2] if residual.ndim == 4 else 1
    if routes(module, x.shape, x.dtype):
        p = prep(
            x.reshape(-1, k),
            "rms_norm",
            norm.weight,
            eps=norm.eps,
            shape=tuple(x.shape),
            natural=_nax_m(_rows(x.shape)),
            residual=(residual.reshape(-1, k), rows),
        )
        return p.h.reshape(x.shape), p
    if fused and enabled():
        return add_rms_norm(norm, x, residual.reshape(-1, k), rows)
    h = x + slot_sum(residual)
    return h, norm(h)


def _gdn_in_source(Hk, Hv, Dk, Dv, PW, TG, eps, qscale, kscale):
    KD, VD = Hk * Dk, Hv * Dv
    return f"""
    constexpr int Hk = {Hk}, Hv = {Hv}, Dk = {Dk}, Dv = {Dv}, PW = {PW}, TG = {TG};
    constexpr int KD = {KD}, VD = {VD}, CD = 2 * KD + VD, KW = 4;
    constexpr float EPS = {eps!r}f;
    const int slot = threadgroup_position_in_grid.x;
    const int m = threadgroup_position_in_grid.y;
    const int b = m / S, t = m % S;
    const int lid = thread_position_in_threadgroup.x;
    int type, h, D, base;
    if (slot < Hk) {{ type = 0; h = slot; D = Dk; base = h * Dk; }}
    else if (slot < 2 * Hk) {{ type = 1; h = slot - Hk; D = Dk; base = KD + h * Dk; }}
    else {{ type = 2; h = slot - 2 * Hk; D = Dv; base = 2 * KD + h * Dv; }}
    const int c0 = lid * 4;
    const bool active = c0 < D;
    // Rows past the first C rows are siblings of chain rows 1..: their conv
    // window is the chain prefix of their position and the row itself.
    const int tc = t >= C ? t - C + 1 : t;
    // Depthwise conv over [state; qkv] then silu, rounded like conv1d and nn.silu.
    T sv[4];
    float acc2 = 0.0f;
    if (active) {{
      for (int i = 0; i < 4; i++) {{
        const int c = base + c0 + i;
        float acc = 0.0f;
        for (int j = 0; j < KW; j++) {{
          const int tt = j < KW - 1 ? tc + j : t + j;
          const T xv = tt < KW - 1 ? state_in[(size_t)(b * (KW - 1) + tt) * CD + c]
                                   : proj[(size_t)(b * S + tt - (KW - 1)) * PW + c];
          acc += static_cast<float>(xv) * w[c * KW + j];
        }}
        const T cb = static_cast<T>(acc);
        sv[i] = cb * Sigmoid{{}}(cb);
        const float f = static_cast<float>(sv[i]);
        acc2 += f * f;
      }}
    }}
    if (type < 2) {{
      threadgroup float sums[32];
      acc2 = rms_sum(acc2, sums, thread_index_in_simdgroup, simdgroup_index_in_threadgroup);
      if (active) {{
        const float inv = metal::precise::rsqrt(acc2 / D + EPS);
        const T sc = type == 0 ? T({qscale!r}f) : T({kscale!r}f);
        device T* o = (type == 0 ? q : k) + ((size_t)m * Hk + h) * Dk + c0;
        for (int i = 0; i < 4; i++) o[i] = static_cast<T>(static_cast<float>(sv[i]) * inv) * sc;
      }}
    }} else if (active) {{
      device T* o = v + ((size_t)m * Hv + h) * Dv + c0;
      for (int i = 0; i < 4; i++) o[i] = sv[i];
    }}
    // The next conv state is the last KW - 1 rows of [state; chain rows].
    if (active && t == 0) {{
      for (int j = 0; j < KW - 1; j++) {{
        const int tt = C + j;
        for (int i = 0; i < 4; i++) {{
          const int c = base + c0 + i;
          state_out[(size_t)(b * (KW - 1) + j) * CD + c] =
              tt < KW - 1 ? state_in[(size_t)(b * (KW - 1) + tt) * CD + c]
                          : proj[(size_t)(b * S + tt - (KW - 1)) * PW + c];
        }}
      }}
    }}
    // Gate values of head h: g = exp(-exp(A_log) * softplus(a + dt_bias)), beta = sigmoid(b).
    if (type == 2 && lid == 0) {{
      const device T* row = proj + (size_t)m * PW + CD + VD;
      const T sp = LogAddExp{{}}(row[Hv + h] + dt_bias[h], T(0.0f));
      const float e = metal::precise::exp(static_cast<float>(A_log[h]));
      g[(size_t)m * Hv + h] = metal::precise::exp((-e) * static_cast<float>(sp));
      beta[(size_t)m * Hv + h] = SigmoidLib{{}}(row[h]);
    }}
"""


def gdn_in_ok(net, proj, mask, cache):
    """True when ``gdn_in`` handles this call (a cache without padding, no mask)."""
    return (
        enabled()
        and cache is not None
        and mask is None
        and cache.lengths is None
        and proj.dtype in (mx.bfloat16, mx.float16)
        and net.dt_bias.dtype == proj.dtype
        and net.A_log.dtype in (mx.float32, proj.dtype)
        and net.conv1d.weight.dtype == proj.dtype
        and net.conv_kernel_size == 4
        and net.head_k_dim % 4 == 0
        and net.head_v_dim % 4 == 0
    )


def gdn_in(net, proj, conv_state, chain=None):
    """The GDN mixer inputs from the fused in_proj output ``proj`` (B, S, PW).

    Returns q, k (B, S, Hk, Dk) normalized and scaled, v (B, S, Hv, Dv), g (B, S, Hv)
    float32, beta (B, S, Hv) and the next conv state (B, KW - 1, conv_dim). With
    ``chain`` the rows past the first ``chain`` rows are siblings of chain rows 1..
    """
    B, S, PW = proj.shape
    Hk, Hv, Dk, Dv = net.num_k_heads, net.num_v_heads, net.head_k_dim, net.head_v_dim
    TG = -(-max(Dk, Dv) // 128) * 32
    eps = 1e-6 / Dk
    inv_scale = Dk**-0.5
    # The ops multiply by the scale rounded to the activation type.
    qscale = mx.array(inv_scale**2, proj.dtype).item()
    kscale = mx.array(inv_scale, proj.dtype).item()
    kern = _kernel(
        "gdn_in",
        (Hk, Hv, Dk, Dv, PW, TG, eps, qscale, kscale, str(proj.dtype), str(net.A_log.dtype)),
        lambda: _gdn_in_source(Hk, Hv, Dk, Dv, PW, TG, eps, qscale, kscale),
        ["proj", "state_in", "w", "A_log", "dt_bias", "S", "C"],
        ["q", "k", "v", "g", "beta", "state_out"],
        _HEADER,
    )
    return kern(
        inputs=[proj, conv_state, net.conv1d.weight, net.A_log, net.dt_bias, S, chain or S],
        template=[("T", proj.dtype)],
        grid=(TG * (2 * Hk + Hv), B * S, 1),
        threadgroup=(TG, 1, 1),
        output_shapes=[
            (B, S, Hk, Dk),
            (B, S, Hk, Dk),
            (B, S, Hv, Dv),
            (B, S, Hv),
            (B, S, Hv),
            conv_state.shape,
        ],
        output_dtypes=[proj.dtype] * 3 + [mx.float32, proj.dtype, proj.dtype],
    )


def _gated_norm_source(D, GS, GO, eps, act):
    # silu: the compiled precise swiglu in float; sigmoid: the eager op in T, then a T multiply
    gate = (
        "const float s = gf * Sigmoid{}(gf);\n      o[i] = static_cast<T>(s * static_cast<float>(n));"
        if act == "silu"
        else "o[i] = static_cast<T>(static_cast<float>(n) * static_cast<float>(SigmoidLib{}(zr[i])));"
    )
    return f"""
    constexpr int D = {D}, GS = {GS}, GO = {GO};
    constexpr float EPS = {eps!r}f;
    const int row = threadgroup_position_in_grid.x;
    const int lid = thread_position_in_threadgroup.x;
    const int c0 = lid * 4;
    const bool active = c0 < D;
    float xv[4];
    float acc = 0.0f;
    if (active) {{
      const device T* xr = x + (size_t)row * D + c0;
      for (int i = 0; i < 4; i++) {{ xv[i] = static_cast<float>(xr[i]); acc += xv[i] * xv[i]; }}
    }}
    threadgroup float sums[32];
    acc = rms_sum(acc, sums, thread_index_in_simdgroup, simdgroup_index_in_threadgroup);
    if (!active) return;
    const float inv = metal::precise::rsqrt(acc / D + EPS);
    const device T* zr = gate + (size_t)(row / HEADS) * GS + GO + (size_t)(row % HEADS) * D + c0;
    device T* o = out + (size_t)row * D + c0;
    for (int i = 0; i < 4; i++) {{
      // mx.fast.rms_norm with weight, then the gate
      const T n = weight[c0 + i] * static_cast<T>(xv[i] * inv);
      const float gf = static_cast<float>(zr[i]);
      (void)gf;
      {gate}
    }}
"""


def gated_norm(norm, x, gate, gate_offset=0):
    """``norm(x, z)`` (per-head RMSNorm times silu(z), or sigmoid(z) when
    ``norm.activation`` says so) as one kernel; z is read in place.

    ``x`` is (.., heads, D); ``gate`` (rows, GS) holds z of row r at column ``gate_offset``.
    """
    *batch, heads, D = x.shape
    rows = x.size // D
    TG = -(-D // 128) * 32
    GS = gate.shape[-1]
    act = getattr(norm, "activation", "silu")
    kern = _kernel(
        "gated_norm",
        (D, GS, gate_offset, norm.eps, heads, act, str(x.dtype)),
        lambda: _gated_norm_source(D, GS, gate_offset, norm.eps, act),
        ["x", "gate", "weight"],
        ["out"],
        _HEADER,
    )
    (out,) = kern(
        inputs=[x, gate, norm.weight],
        template=[("T", x.dtype), ("HEADS", heads)],
        grid=(TG * rows, 1, 1),
        threadgroup=(TG, 1, 1),
        output_shapes=[(rows, D)],
        output_dtypes=[x.dtype],
    )
    return out.reshape(*batch, heads * D)


def _add_rms_norm_source(K, NT, eps):
    return f"""
    constexpr int K = {K}, NT = {NT};
    constexpr float EPS = {eps!r}f;
    const int row = threadgroup_position_in_grid.x;
    const int lid = thread_position_in_threadgroup.x;
    const device T* xr = x + (size_t)row * K;
    const device T* rr = res + (size_t)row * res_rows * K;
    constexpr int NCH = (K + NT * 4 - 1) / (NT * 4);
    // The residual summed over its slot rows as mx.sum does, then the add of the ops;
    // the norm reads the rounded sum
    float xv[NCH][4];
    float acc = 0.0f;
    for (int c = 0; c < NCH; c++) {{
      const int j0 = c * NT * 4 + lid * 4;
      if (j0 + 4 <= K) {{
        float4 xf = float4(*(const device vec<T, 4>*)(xr + j0));
        if (has_res) {{
          // rows y, y + 8, ... per lane row in the input type, then the lane rows in order
          const int L = min(8, res_rows);
          vec<T, 4> tot = vec<T, 4>(T(0.0f));
          for (int y = 0; y < L; y++) {{
            vec<T, 4> t = vec<T, 4>(T(0.0f));
            for (int r = y; r < res_rows; r += L) t = *(const device vec<T, 4>*)(rr + (size_t)r * K + j0) + t;
            tot = (y == 0) ? t : t + tot;
          }}
          xf = float4(vec<T, 4>(xf) + tot);
        }}
        for (int i = 0; i < 4; i++) {{ xv[c][i] = xf[i]; acc += xf[i] * xf[i]; }}
      }} else {{
        for (int i = 0; i < 4; i++) xv[c][i] = 0.0f;
      }}
    }}
    threadgroup float sums[32];
    acc = rms_sum(acc, sums, thread_index_in_simdgroup, simdgroup_index_in_threadgroup);
    const float inv = metal::precise::rsqrt(acc / K + EPS);
    for (int c = 0; c < NCH; c++) {{
      const int j0 = c * NT * 4 + lid * 4;
      if (j0 + 4 <= K) {{
        for (int i = 0; i < 4; i++) {{
          if (has_res) h[(size_t)row * K + j0 + i] = static_cast<T>(xv[c][i]);
          out[(size_t)row * K + j0 + i] = weight[j0 + i] * static_cast<T>(xv[c][i] * inv);
        }}
      }}
    }}
"""


def add_rms_norm(norm, x, residual=None, res_rows=1):
    """``norm(x + residual)`` as one kernel; returns (x + residual, normed).

    ``residual`` holds ``res_rows`` slot rows per row of x, summed as ``mx.sum`` does.
    """
    k = x.shape[-1]
    rows = x.size // k
    NT = _rms_threads(k)
    kern = _kernel(
        "add_rms_norm",
        (k, NT, norm.eps, str(x.dtype)),
        lambda: _add_rms_norm_source(k, NT, norm.eps),
        ["x", "res", "weight", "has_res", "res_rows"],
        ["h", "out"],
        _HEADER + _SMALL_HEADER,
    )
    with_res = residual is not None
    h, out = kern(
        inputs=[x, residual if with_res else x, norm.weight, int(with_res), res_rows],
        template=[("T", x.dtype)],
        grid=(NT * rows, 1, 1),
        threadgroup=(NT, 1, 1),
        output_shapes=[x.shape] * 2,
        output_dtypes=[x.dtype] * 2,
    )
    return (h, out) if with_res else (x, out)


def _attn_qkv_source(H, Hkv, Dh, RD, QW, TG, eps, log2base, scale):
    return f"""
    constexpr int H = {H}, Hkv = {Hkv}, Dh = {Dh}, RD = {RD}, QW = {QW}, TG = {TG};
    constexpr float EPS = {eps!r}f, LOG2BASE = {log2base!r}f, SCALE = {scale!r}f;
    const int slot = threadgroup_position_in_grid.x;
    const int m = threadgroup_position_in_grid.y;
    const int b = m / L, l = m % L;
    const int lid = thread_position_in_threadgroup.x;
    const int c0 = lid * 4;
    const bool active = c0 < Dh;
    int type, h;
    const device T* in;
    if (slot < H) {{ type = 0; h = slot; in = qkv + (size_t)m * QW + h * 2 * Dh; }}
    else if (slot < H + Hkv) {{ type = 1; h = slot - H; in = qkv + (size_t)m * QW + 2 * H * Dh + h * Dh; }}
    else {{ type = 2; h = slot - H - Hkv; in = qkv + (size_t)m * QW + (2 * H + Hkv) * Dh + h * Dh; }}
    const int heads = type == 0 ? H : Hkv;
    device T* o = (type == 0 ? q : type == 1 ? k : v) + (((size_t)b * heads + h) * L + l) * Dh + c0;
    if (type == 2) {{
      if (active) for (int i = 0; i < 4; i++) o[i] = in[c0 + i];
      return;
    }}
    // mx.fast.rms_norm with weight, then mx.fast.rope on the first RD dims
    const device T* w = type == 0 ? q_w : k_w;
    float acc = 0.0f;
    if (active) for (int i = 0; i < 4; i++) {{ const float f = static_cast<float>(in[c0 + i]); acc += f * f; }}
    threadgroup float sums[32];
    acc = rms_sum(acc, sums, thread_index_in_simdgroup, simdgroup_index_in_threadgroup);
    if (!active) return;
    const float inv = metal::precise::rsqrt(acc / Dh + EPS);
    // Rows past the first C rows are siblings of chain rows 1.. and share their positions
    const int pos = l >= C ? l - C + 1 : l;
    const float Lp = SCALE * static_cast<float>(pos + offset);
    for (int i = 0; i < 4; i++) {{
      const int d = c0 + i;
      const T n = w[d] * static_cast<T>(static_cast<float>(in[d]) * inv);
      if (d >= RD) {{ o[i] = n; continue; }}
      const int d1 = d < RD / 2 ? d : d - RD / 2;
      const int d2 = d1 + RD / 2;
      const float theta = Lp * metal::exp2(-(static_cast<float>(d1) / static_cast<float>(RD / 2)) * LOG2BASE);
      const float c = metal::fast::cos(theta), s = metal::fast::sin(theta);
      const int dp = d < RD / 2 ? d2 : d1;
      const float np = static_cast<float>(w[dp] * static_cast<T>(static_cast<float>(in[dp]) * inv));
      const float x1 = d < RD / 2 ? static_cast<float>(n) : np;
      const float x2 = d < RD / 2 ? np : static_cast<float>(n);
      o[i] = static_cast<T>(d < RD / 2 ? x1 * c - x2 * s : x1 * s + x2 * c);
    }}
"""


def attn_ok(attn, qkv, cache):
    """True when ``attn_qkv`` handles this call: a plain RoPE and an integer cache offset."""
    rope = attn.rope
    return (
        enabled()
        and cache is not None
        and isinstance(getattr(cache, "offset", None), int)
        and type(rope).__name__ == "RoPE"
        and not rope.traditional
        and 0 < rope.dims <= attn.head_dim
        and rope.dims % 2 == 0
        and qkv.dtype in (mx.bfloat16, mx.float16)
        and attn.q_norm.weight.dtype == qkv.dtype
        and attn.k_norm.weight.dtype == qkv.dtype
        and attn.head_dim % 4 == 0
    )


def attn_qkv(attn, qkv, offset, chain=None):
    """q (B, H, L, Dh) and k (B, Hkv, L, Dh) normalized and rotated, v (B, Hkv, L, Dh) from ``qkv``.

    With ``chain`` the rows past the first ``chain`` rows are siblings of chain rows 1..
    """
    import math

    B, L, QW = qkv.shape
    H, Hkv, Dh = attn.num_attention_heads, attn.num_key_value_heads, attn.head_dim
    TG = -(-Dh // 128) * 32
    rope = attn.rope
    log2base = float(mx.array(math.log2(rope.base), mx.float32).item())
    kern = _kernel(
        "attn_qkv",
        (H, Hkv, Dh, rope.dims, QW, TG, attn.q_norm.eps, log2base, rope.scale, str(qkv.dtype)),
        lambda: _attn_qkv_source(H, Hkv, Dh, rope.dims, QW, TG, attn.q_norm.eps, log2base, rope.scale),
        ["qkv", "q_w", "k_w", "offset", "L", "C"],
        ["q", "k", "v"],
        _HEADER,
    )
    return kern(
        inputs=[qkv, attn.q_norm.weight, attn.k_norm.weight, offset, L, chain or L],
        template=[("T", qkv.dtype)],
        grid=(TG * (H + 2 * Hkv), B * L, 1),
        threadgroup=(TG, 1, 1),
        output_shapes=[(B, H, L, Dh), (B, Hkv, L, Dh), (B, Hkv, L, Dh)],
        output_dtypes=[qkv.dtype] * 3,
    )


def _attn_gate_source(H, Dh, QW):
    return f"""
    constexpr int H = {H}, Dh = {Dh}, QW = {QW};
    const int m = thread_position_in_grid.y;
    const int j = thread_position_in_grid.x;
    if (j >= H * Dh) return;
    const int h = j / Dh, d = j % Dh;
    const int b = m / L, l = m % L;
    const T xv = x[(((size_t)b * H + h) * L + l) * Dh + d];
    const T g = qkv[(size_t)m * QW + h * 2 * Dh + Dh + d];
    out[(size_t)m * H * Dh + j] = xv * SigmoidLib{{}}(g);
"""


def attn_gate(x, qkv):
    """``x * sigmoid(gate)`` for the attention output x (B, H, L, Dh); the gate is read in ``qkv``."""
    B, H, L, Dh = x.shape
    QW = qkv.shape[-1]
    kern = _kernel(
        "attn_gate",
        (H, Dh, QW, str(x.dtype)),
        lambda: _attn_gate_source(H, Dh, QW),
        ["x", "qkv", "L"],
        ["out"],
        _HEADER,
    )
    (out,) = kern(
        inputs=[x, qkv, L],
        template=[("T", x.dtype)],
        grid=(H * Dh, B * L, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[(B * L, H * Dh)],
        output_dtypes=[x.dtype],
    )
    return out.reshape(B, L, H * Dh)


def _ple_conv_source(KW, DIL):
    return f"""
    constexpr int KW = {KW}, DIL = {DIL}, NS = (KW - 1) * DIL;
    const int c = thread_position_in_grid.x;
    const int row = thread_position_in_grid.y;
    const int b = row / (L + NS), t = row % (L + NS);
    if (c >= C) return;
    // [state; x] row r of batch b
    auto in = [&](int r) -> T {{
      return r < NS ? state[((size_t)b * NS + r) * C + c] : x[((size_t)b * L + r - NS) * C + c];
    }};
    if (t < L) {{
      float acc = 0.0f;
      for (int j = 0; j < KW; j++) acc += float(in(t + j * DIL)) * float(w[c * KW + j]);
      // conv1d rounds to T, then the compiled silu in T
      const float cv = float(T(acc));
      out[((size_t)b * L + t) * C + c] = T(cv * sigmoid_t<T>(cv));
    }} else {{
      // The next state: the last NS rows of [state; x]
      state_out[((size_t)b * NS + t - L) * C + c] = in(t - L + L);
    }}
"""


def ple_conv(x, state, weight):
    """``silu(depthwise conv1d([state; x]))`` of the PLE layer and the next state, one kernel.

    ``x`` (B, L, C), ``state`` (B, (KW - 1) * dilation, C), ``weight`` (C, KW, 1).
    """
    B, L, C = x.shape
    KW = weight.shape[1]
    NS = state.shape[1]
    DIL = NS // (KW - 1)
    kern = _kernel(
        "ple_conv",
        (KW, DIL, str(x.dtype)),
        lambda: _ple_conv_source(KW, DIL),
        ["x", "state", "w", "L", "C"],
        ["out", "state_out"],
        _HEADER + _SIGMOID_T,
    )
    return kern(
        inputs=[x, state, weight, L, C],
        template=[("T", x.dtype)],
        grid=(C, B * (L + NS), 1),
        threadgroup=(256, 1, 1),
        output_shapes=[(B, L, C), (B, NS, C)],
        output_dtypes=[x.dtype, x.dtype],
    )
