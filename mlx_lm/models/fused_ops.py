# Copyright © 2026 Apple Inc.

"""Merged small kernels of the Qwen3.5 decode step.

Each kernel replaces a chain of MLX ops by one dispatch and rounds where the ops
round (bf16 intermediates at the same places), so its output is bitwise the
output of the ops it replaces.
"""

import os

import mlx.core as mx

from .qmv_small import _HEADER as _SMALL_HEADER
from .qmv_small import _kernel, _target, prep, prep_rms_norm, routes

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
            residual=(residual.reshape(-1, k), rows),
            **_target(module, x.shape),
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


def _gated_norm_source(D, GS, GO, eps):
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
      // mx.fast.rms_norm with weight, then silu(gate) * x in float (_precise_swiglu)
      const T n = weight[c0 + i] * static_cast<T>(xv[i] * inv);
      const float gf = static_cast<float>(zr[i]);
      const float s = gf * Sigmoid{{}}(gf);
      o[i] = static_cast<T>(s * static_cast<float>(n));
    }}
"""


def gated_norm(norm, x, gate, gate_offset=0):
    """``norm(x, z)`` (per-head RMSNorm times silu(z)) as one kernel; z is read in place.

    ``x`` is (.., heads, D); ``gate`` (rows, GS) holds z of row r at column ``gate_offset``.
    """
    *batch, heads, D = x.shape
    rows = x.size // D
    TG = -(-D // 128) * 32
    GS = gate.shape[-1]
    kern = _kernel(
        "gated_norm",
        (D, GS, gate_offset, norm.eps, heads, str(x.dtype)),
        lambda: _gated_norm_source(D, GS, gate_offset, norm.eps),
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


# The sdpa_vector kernel of MLX (sdpa_vector.h) runs one 1024-thread threadgroup per
# query head: 16 threadgroups on a 40-core GPU, each walking its keys in 32 simdgroups.
# Below 1024 keys MLX picks that kernel. These two kernels split the same work over
# (kv head, key block) threadgroups and combine the blocks as the kernel combines its
# simdgroups: block b holds the keys b, b + 32, ..., summed in the same order with the
# same exp, so the output is bitwise the one-pass output. The partials stay in float.
_SDPA_BLOCKS = 32
_SDPA_MAX_KEYS = 1024


def _sdpa_pass1_source(D, V, GQA):
    """The block's keys and values are staged in threadgroup memory in chunks of CH (16-byte loads
    shared by the GQA query heads of the group); each simdgroup then runs the one-pass kernel's
    loop over them in order, so the partials are the one-pass kernel's."""
    CH = 16
    return f"""
    constexpr int D = {D}, V = {V}, GQA = {GQA}, NBLK = {_SDPA_BLOCKS}, BD = 32, CH = {CH};
    constexpr int qk_per_thread = D / BD, v_per_thread = V / BD;
    constexpr int KW4 = D / 8, VW4 = V / 8;  // uint4 words per key / value row (T = 2 bytes)
    constexpr int NT = 32 * GQA;
    const int lane = thread_index_in_simdgroup;
    const int tid = thread_position_in_threadgroup.x;
    const int blk = threadgroup_position_in_grid.x;
    const int kvh = threadgroup_position_in_grid.y;
    const int h = kvh * GQA + int(simdgroup_index_in_threadgroup);
    const int N = keys_shape[2];
    const size_t k_head_stride = keys_strides[1], k_seq_stride = keys_strides[2];
    const size_t v_head_stride = values_strides[1], v_seq_stride = values_strides[2];
    threadgroup uint4 kbuf[CH][KW4];
    threadgroup uint4 vbuf[CH][VW4];
    const device T* qp = queries + (size_t)h * D + lane * qk_per_thread;
    const device T* kbase = keys + (size_t)kvh * k_head_stride;
    const device T* vbase = values + (size_t)kvh * v_head_stride;
    float q[qk_per_thread], k[qk_per_thread], o[v_per_thread];
    for (int i = 0; i < qk_per_thread; i++) q[i] = static_cast<float>(scale) * qp[i];
    for (int i = 0; i < v_per_thread; i++) o[i] = 0;
    float max_score = -metal::numeric_limits<float>::max();
    float sum_exp_score = 0;
    const int nk = blk < N ? (N - blk + NBLK - 1) / NBLK : 0;
    for (int c0 = 0; c0 < nk; c0 += CH) {{
      const int nc = min(CH, nk - c0);
      threadgroup_barrier(mem_flags::mem_threadgroup);
      for (int j = tid; j < nc * KW4; j += NT) {{
        const int kk = j / KW4, wd = j % KW4;
        kbuf[kk][wd] = ((const device uint4*)(kbase + (size_t)(blk + (c0 + kk) * NBLK) * k_seq_stride))[wd];
      }}
      for (int j = tid; j < nc * VW4; j += NT) {{
        const int kk = j / VW4, wd = j % VW4;
        vbuf[kk][wd] = ((const device uint4*)(vbase + (size_t)(blk + (c0 + kk) * NBLK) * v_seq_stride))[wd];
      }}
      threadgroup_barrier(mem_flags::mem_threadgroup);
      for (int kk = 0; kk < nc; kk++) {{
        const threadgroup T* kp = (const threadgroup T*)kbuf[kk] + lane * qk_per_thread;
        const threadgroup T* vp = (const threadgroup T*)vbuf[kk] + lane * v_per_thread;
        for (int j = 0; j < qk_per_thread; j++) k[j] = kp[j];
        float score = 0;
        for (int j = 0; j < qk_per_thread; j++) score += q[j] * k[j];
        score = simd_sum(score);
        float new_max = max(max_score, score);
        float factor = metal::fast::exp(max_score - new_max);
        float exp_score = metal::fast::exp(score - new_max);
        max_score = new_max;
        sum_exp_score = sum_exp_score * factor + exp_score;
        for (int j = 0; j < v_per_thread; j++) o[j] = o[j] * factor + exp_score * vp[j];
      }}
    }}
    device float4* po = (device float4*)(part + ((size_t)h * NBLK + blk) * V + lane * v_per_thread);
    #pragma clang loop unroll(full)
    for (int i = 0; i < v_per_thread; i += 4) po[i / 4] = float4(o[i], o[i + 1], o[i + 2], o[i + 3]);
    if (lane == 0) {{
      pmax[h * NBLK + blk] = max_score;
      psum[h * NBLK + blk] = sum_exp_score;
    }}
"""


def _sdpa_pass2_source(V):
    """The one-pass kernel's combine over the 32 blocks on NSG simdgroups: the head's partials are
    staged in threadgroup memory in slices (16-byte loads), lane b of a simdgroup holds block b
    and the ``simd_max`` / ``simd_sum`` trees are the ones of the one-pass kernel."""
    NSG = 8
    return f"""
    constexpr int V = {V}, BN = {_SDPA_BLOCKS}, BD = 32, v_per_thread = V / BD, NSG = {NSG}, NT = 32 * NSG;
    constexpr int SW = NSG * v_per_thread;  // floats of a block per round (NSG slices)
    constexpr int SW4 = SW / 4;
    const int h = threadgroup_position_in_grid.x;
    const int lane = thread_index_in_simdgroup;
    const int sg = simdgroup_index_in_threadgroup;
    const int tid = thread_position_in_threadgroup.x;
    threadgroup float4 pbuf[BN][SW4];
    threadgroup float max_scores[BN];
    threadgroup float sum_exp_scores[BN];
    const device float4* src = (const device float4*)(part + (size_t)h * BN * V);
    if (tid < BN) {{
      max_scores[tid] = pmax[h * BN + tid];
      sum_exp_scores[tid] = psum[h * BN + tid];
    }}
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float max_score = max_scores[lane];
    const float new_max = simd_max(max_score);
    const float factor = metal::fast::exp(max_score - new_max);
    const float sum_exp_score = simd_sum(sum_exp_scores[lane] * factor);
    const threadgroup float* pb = (const threadgroup float*)pbuf[lane];
    for (int r = 0; r < BD / NSG; r++) {{
      // slices r * NSG .. + NSG of every block: block b's floats [r * SW, (r + 1) * SW)
      threadgroup_barrier(mem_flags::mem_threadgroup);
      for (int j = tid; j < BN * SW4; j += NT) pbuf[j / SW4][j % SW4] = src[(j / SW4) * (V / 4) + r * SW4 + j % SW4];
      threadgroup_barrier(mem_flags::mem_threadgroup);
      const int g = r * NSG + sg;
      float o[v_per_thread];
      for (int i = 0; i < v_per_thread; i++) {{
        o[i] = simd_sum(pb[sg * v_per_thread + i] * factor);
        o[i] = sum_exp_score == 0 ? o[i] : (o[i] / sum_exp_score);
      }}
      if (lane == 0) {{
        device T* op = out + (size_t)h * V + g * v_per_thread;
        for (int i = 0; i < v_per_thread; i++) op[i] = static_cast<T>(o[i]);
      }}
    }}
"""


def sdpa_ok(queries, keys, cache):
    """True when ``sdpa_two_pass`` handles this call: one query row of one sequence, a
    plain KV cache and fewer than 1024 keys (where MLX runs its one-pass kernel)."""
    B, H, L, D = queries.shape
    Bk, Hkv, N, Dk = keys.shape
    return (
        enabled()
        and not hasattr(cache, "bits")
        and B == 1
        and L == 1
        and D == Dk
        and D % 32 == 0
        and H % Hkv == 0
        and 1 <= N < _SDPA_MAX_KEYS
        and queries.dtype in (mx.bfloat16, mx.float16)
        and keys.dtype == queries.dtype
    )


def sdpa_two_pass(queries, keys, values, scale):
    """``mx.fast.scaled_dot_product_attention`` of one query row, bitwise, as two kernels.

    ``queries`` (1, H, 1, D) contiguous; ``keys`` and ``values`` (1, Hkv, N, D) may be
    views into the cache (their strides are read in the kernel).
    """
    B, H, L, D = queries.shape
    Hkv, N = keys.shape[1], keys.shape[2]
    V = values.shape[-1]
    GQA = H // Hkv
    NBLK = _SDPA_BLOCKS
    kern1 = _kernel(
        "sdpa_pass1",
        (D, V, GQA, str(queries.dtype)),
        lambda: _sdpa_pass1_source(D, V, GQA),
        ["queries", "keys", "values", "scale"],
        ["part", "pmax", "psum"],
        ensure_row_contiguous=False,
    )
    kern2 = _kernel(
        "sdpa_pass2",
        (V, str(queries.dtype)),
        lambda: _sdpa_pass2_source(V),
        ["part", "pmax", "psum"],
        ["out"],
    )
    part, pmax, psum = kern1(
        inputs=[queries, keys, values, mx.array(scale, mx.float32)],
        template=[("T", queries.dtype)],
        grid=(32 * GQA * NBLK, Hkv, 1),
        threadgroup=(32 * GQA, 1, 1),
        output_shapes=[(H, NBLK, V), (H, NBLK), (H, NBLK)],
        output_dtypes=[mx.float32] * 3,
    )
    (out,) = kern2(
        inputs=[part, pmax, psum],
        template=[("T", queries.dtype)],
        grid=(256 * H, 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[(B, H, L, V)],
        output_dtypes=[queries.dtype],
    )
    return out


def _mtp_in_source(D, NT, eps_e, eps_h, bits, GS, WPR):
    if bits:
        # The library's dequantize: scale * value + bias in float, rounded to T once
        load = f"""
        constexpr int BITS = {bits}, GS = {GS}, WPR = {WPR}, VPW = 32 / BITS;
        constexpr uint MASK = (1u << BITS) - 1;
        const uint word = ew[(size_t)tok * WPR + j0 / VPW];
        const size_t g = (size_t)tok * (D / GS) + j0 / GS;
        const float s = static_cast<float>(es[g]), b = static_cast<float>(eb[g]);
        for (int i = 0; i < 4; i++) {{
          const uint d = (word >> (BITS * ((j0 + i) % VPW))) & MASK;
          xv[c][i] = static_cast<float>(static_cast<T>(metal::fma(s, static_cast<float>(d), b)));
        }}"""
    else:
        load = """
        const float4 xf = float4(*(const device vec<T, 4>*)(ew + (size_t)tok * D + j0));
        for (int i = 0; i < 4; i++) xv[c][i] = xf[i];"""
    return f"""
    constexpr int D = {D}, NT = {NT}, NCH = (D + NT * 4 - 1) / (NT * 4);
    constexpr float EPS_E = {eps_e!r}f, EPS_H = {eps_h!r}f;
    const int part = threadgroup_position_in_grid.x;
    const int m = threadgroup_position_in_grid.y;
    const int lid = thread_position_in_threadgroup.x;
    const uint tok = tokens[m];
    float xv[NCH][4];
    float acc = 0.0f;
    for (int c = 0; c < NCH; c++) {{
      const int j0 = c * NT * 4 + lid * 4;
      if (j0 + 4 > D) {{
        for (int i = 0; i < 4; i++) xv[c][i] = 0.0f;
        continue;
      }}
      if (part) {{
        const float4 xf = float4(*(const device vec<T, 4>*)(hidden + (size_t)m * D + j0));
        for (int i = 0; i < 4; i++) xv[c][i] = xf[i];
      }} else {{{load}
      }}
      for (int i = 0; i < 4; i++) acc += xv[c][i] * xv[c][i];
    }}
    threadgroup float sums[32];
    acc = rms_sum(acc, sums, thread_index_in_simdgroup, simdgroup_index_in_threadgroup);
    const float inv = metal::precise::rsqrt(acc / D + (part ? EPS_H : EPS_E));
    const device T* w = part ? h_w : e_w;
    device T* o = out + ((size_t)m * 2 + part) * D;
    for (int c = 0; c < NCH; c++) {{
      const int j0 = c * NT * 4 + lid * 4;
      if (j0 + 4 <= D) {{
        for (int i = 0; i < 4; i++) o[j0 + i] = w[j0 + i] * static_cast<T>(xv[c][i] * inv);
      }}
    }}
"""


def mtp_in(embed, norm_e, norm_h, tokens, hidden):
    """``[norm_e(embed(tokens)); norm_h(hidden)]`` (.., 2D) as one kernel, or None when
    the embedding is not plain or affine-quantized. The rows are dequantized in place."""
    D = hidden.shape[-1]
    quantized = hasattr(embed, "scales")
    weight, scales, biases = embed.weight, embed.get("scales"), embed.get("biases")
    if quantized:
        bits, GS = embed.bits, embed.group_size
        ok = (
            embed.mode == "affine"
            and bits in (2, 4, 8)
            and GS % 4 == 0
            and biases is not None
            and scales.dtype == hidden.dtype
        )
    else:
        bits, GS = 0, 0
        ok = weight.dtype == hidden.dtype
    if not (
        enabled()
        and ok
        and hidden.dtype in (mx.bfloat16, mx.float16, mx.float32)
        and D % 4 == 0
        and norm_e.weight.dtype == hidden.dtype
        and norm_h.weight.dtype == hidden.dtype
    ):
        return None
    M = hidden.size // D
    NT = _rms_threads(D)
    WPR = weight.shape[-1]
    kern = _kernel(
        "mtp_in",
        (D, NT, norm_e.eps, norm_h.eps, bits, GS, WPR, str(hidden.dtype)),
        lambda: _mtp_in_source(D, NT, norm_e.eps, norm_h.eps, bits, GS, WPR),
        ["tokens", "ew", "es", "eb", "e_w", "hidden", "h_w"],
        ["out"],
        _HEADER,
    )
    aux = scales if quantized else norm_e.weight
    (out,) = kern(
        inputs=[
            tokens.reshape(-1),
            weight,
            aux,
            biases if quantized else aux,
            norm_e.weight,
            hidden.reshape(M, D),
            norm_h.weight,
        ],
        template=[("T", hidden.dtype)],
        grid=(NT * 2, M, 1),
        threadgroup=(NT, 1, 1),
        output_shapes=[(M, 2 * D)],
        output_dtypes=[hidden.dtype],
    )
    return out.reshape(*hidden.shape[:-1], 2 * D)


def _swiglu_source(K):
    return f"""
    constexpr int K = {K};
    const int m = thread_position_in_grid.y;
    const int j = thread_position_in_grid.x * 4;
    if (j >= K) return;
    const vec<T, 4> g = *(const device vec<T, 4>*)(x + (size_t)m * 2 * K + j);
    const vec<T, 4> u = *(const device vec<T, 4>*)(x + (size_t)m * 2 * K + K + j);
    vec<T, 4> o;
    for (int i = 0; i < 4; i++) {{
      // nn.silu(gate) * up as the compiled ops round it: every step in T
      const T s = Sigmoid{{}}(g[i]);
      const T a = g[i] * s;
      o[i] = a * u[i];
    }}
    *(device vec<T, 4>*)(out + (size_t)m * K + j) = o;
"""


def swiglu(gate_up):
    """``silu(gate) * up`` of the fused (.., 2K) projection output, read in place; bitwise
    the compiled ``activations.swiglu``. None when the shape is not handled."""
    *batch, K2 = gate_up.shape
    K = K2 // 2
    M = gate_up.size // K2
    if not (enabled() and K % 4 == 0 and gate_up.dtype in (mx.bfloat16, mx.float16)):
        return None
    kern = _kernel(
        "swiglu",
        (K, str(gate_up.dtype)),
        lambda: _swiglu_source(K),
        ["x"],
        ["out"],
        _HEADER,
    )
    (out,) = kern(
        inputs=[gate_up.reshape(M, K2)],
        template=[("T", gate_up.dtype)],
        grid=(K // 4, M, 1),
        threadgroup=(min(256, K // 4), 1, 1),
        output_shapes=[(M, K)],
        output_dtypes=[gate_up.dtype],
    )
    return out.reshape(*batch, K)


# mx.logsumexp runs its one-block kernel up to this many values, its looped kernel above.
_LSE_LOOPED_LIMIT = 4096


def _draft_sample_source(looped, has_fixed, has_first, has_ids):
    """One threadgroup over the virtual row [fixed; rows masked by first]: the top-2 (value,
    index) with the smaller index winning ties (mx.argmax's rule), and the log-sum-exp as
    mx.logsumexp computes it (its one-block or looped kernel)."""
    value = "i < F ? static_cast<float>(fixed[i]) : " if has_fixed else ""
    masked = "(!first[i - F]) ? -INFINITY : " if has_first else ""
    if looped:
        lse = """
    float prevmax;
    float maxval = -FLT_MAX;
    float normalizer = 0.0f;
    for (int r = 0; r < (n + NR * lsize - 1) / (NR * lsize); r++) {
      const int offset = r * lsize * NR + lid * NR;
      float vals[NR];
      for (int i = 0; i < NR; i++) vals[i] = offset + i < n ? value(offset + i) : -INFINITY;
      prevmax = maxval;
      for (int i = 0; i < NR; i++) maxval = (maxval < vals[i]) ? vals[i] : maxval;
      normalizer *= metal::fast::exp(prevmax - maxval);
      for (int i = 0; i < NR; i++) {
        normalizer += metal::fast::exp(vals[i] - maxval);
        if (offset + i < n) insert(vals[i], offset + i, v1, i1, v2, i2);
      }
    }
    prevmax = maxval;
    maxval = simd_max(maxval);
    normalizer *= metal::fast::exp(prevmax - maxval);
    normalizer = simd_sum(normalizer);
    prevmax = maxval;
    if (lane == 0) local_max[sg] = maxval;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    maxval = simd_max(local_max[lane]);
    normalizer *= metal::fast::exp(prevmax - maxval);
    if (lane == 0) local_normalizer[sg] = normalizer;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    normalizer = simd_sum(local_normalizer[lane]);"""
    else:
        lse = """
    float ld[NR];
    for (int i = 0; i < NR; i++) {
      const int j = lid * NR + i;
      ld[i] = j < n ? value(j) : -INFINITY;
      if (j < n) insert(ld[i], j, v1, i1, v2, i2);
    }
    if (sg == 0) {
      local_max[lane] = -INFINITY;
      local_normalizer[lane] = 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float maxval = -FLT_MAX;
    for (int i = 0; i < NR; i++) maxval = (maxval < ld[i]) ? ld[i] : maxval;
    maxval = simd_max(maxval);
    if (lane == 0) local_max[sg] = maxval;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0) {
      maxval = simd_max(local_max[lane]);
      if (lane == 0) local_max[0] = maxval;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    maxval = local_max[0];
    float normalizer = 0.0f;
    for (int i = 0; i < NR; i++) normalizer += metal::fast::exp(ld[i] - maxval);
    normalizer = simd_sum(normalizer);
    if (lane == 0) local_normalizer[sg] = normalizer;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0) normalizer = simd_sum(local_normalizer[lane]);"""
    return f"""
    constexpr int NR = 4;
    const int lid = thread_position_in_threadgroup.x;
    const int lsize = threads_per_threadgroup.x;
    const int lane = thread_index_in_simdgroup;
    const int sg = simdgroup_index_in_threadgroup;
    const int n = count;
    const int F = fixed_count;
    (void)F;
    #define value(i) ({value}{masked}static_cast<float>(rows[(i) - F]))
    threadgroup float local_max[32];
    threadgroup float local_normalizer[32];
    threadgroup float tv1[32], tv2[32];
    threadgroup int ti1[32], ti2[32];
    float v1 = -INFINITY, v2 = -INFINITY;
    int i1 = INT_MAX, i2 = INT_MAX;
    {lse}
    // Top-2 across the lanes, then across the simdgroups
    for (int off = 16; off > 0; off >>= 1) {{
      const float ov1 = simd_shuffle_xor(v1, off), ov2 = simd_shuffle_xor(v2, off);
      const int oi1 = simd_shuffle_xor(i1, off), oi2 = simd_shuffle_xor(i2, off);
      insert(ov1, oi1, v1, i1, v2, i2);
      insert(ov2, oi2, v1, i1, v2, i2);
    }}
    if (lane == 0) {{ tv1[sg] = v1; ti1[sg] = i1; tv2[sg] = v2; ti2[sg] = i2; }}
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lid == 0) {{
      for (int s = 1; s < (lsize + 31) / 32; s++) {{
        insert(tv1[s], ti1[s], v1, i1, v2, i2);
        insert(tv2[s], ti2[s], v1, i1, v2, i2);
      }}
      const float lse = metal::isinf(maxval) ? maxval : metal::precise::log(normalizer) + maxval;
      // logprobs = logits - lse; p = exp(max), margin = |top-1 - top-2| of the logprobs
      const float lp1 = v1 - lse, lp2 = v2 - lse;
      stats[0] = metal::precise::exp(lp1);
      stats[1] = metal::abs(lp2 - lp1);
      tok[0] = {"ids[i1]" if has_ids else "uint(i1)"};
      tok[1] = {"ids[i2]" if has_ids else "uint(i2)"};
    }}
"""


_DRAFT_HEADER = """
// Keep the two best (value, index) pairs; on equal values the smaller index wins.
inline bool better(float v, int i, float bv, int bi) {
  return v > bv || (v == bv && i < bi);
}
inline void insert(float v, int i, thread float& v1, thread int& i1, thread float& v2, thread int& i2) {
  if (better(v, i, v1, i1)) {
    v2 = v1; i2 = i1; v1 = v; i1 = i;
  } else if (i != i1 && better(v, i, v2, i2)) {
    v2 = v; i2 = i;
  }
}
"""


def draft_sample(rows, first=None, ids=None, fixed=None):
    """The greedy draft from the logits of one row, as one kernel.

    The row is ``[fixed; rows]`` with the ``rows`` entries whose ``first`` is False scored
    -inf. Returns ``tok`` (2,) uint32 = the best and the second best entry (mapped through
    ``ids`` when given) and ``stats`` (2,) float32 = the probability of the best under the
    softmax of the row and the log-probability margin of the two best; bitwise the ops
    ``argmax``, ``exp(max(logprobs))`` and ``abs(diff(topk(logprobs, 2)))``.
    """
    n = rows.size + (fixed.size if fixed is not None else 0)
    looped = n > _LSE_LOOPED_LIMIT
    threads = 1024 if looped else min(1024, (-(-n // 4) + 31) // 32 * 32)
    key = (looped, fixed is not None, first is not None, ids is not None, str(rows.dtype))
    kern = _kernel(
        "draft_sample",
        key,
        lambda: _draft_sample_source(*key[:4]),
        ["rows", "fixed", "first", "ids", "count", "fixed_count"],
        ["tok", "stats"],
        _HEADER + _DRAFT_HEADER,
    )
    tok, stats = kern(
        inputs=[
            rows,
            rows if fixed is None else fixed,
            rows if first is None else first,
            rows if ids is None else ids,
            n,
            0 if fixed is None else fixed.size,
        ],
        template=[("T", rows.dtype)],
        grid=(threads, 1, 1),
        threadgroup=(threads, 1, 1),
        output_shapes=[(2,), (2,)],
        output_dtypes=[mx.uint32, mx.float32],
    )
    return tok, stats
