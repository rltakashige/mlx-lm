# Copyright © 2026 Apple Inc.

"""Small-M (3..8 rows) quantized matvec for the MTP verify pass.

``mx.quantized_matmul`` routes 2 <= M < 13 to ``qmv_wide``, which dequantizes
each weight in fp32 per input row and is ALU bound on M5. This kernel keeps the
``qmv_fast`` geometry (R rows per simdgroup, P K values per lane per step),
dequantizes each weight once into fp16 and applies it to all M rows with half2
FMAs. Half partials are flushed into fp32 accumulators every 16 values.

``x`` is first converted by a prep kernel into fp16 scaled per row by a power
of two (so the fp16 partials stay in range), plus per-step sums for the bias
term. The weights are unpacked through the fp16 "magic number" trick: a field
of a packed word is masked into the mantissa of 1024.0h, both halves of the
word at once, so x is stored in the pair order of its format (``_Format``).
The 4-bit affine kernel pre-scales the odd pairs of x by 1/16; the other
formats normalize each pair with one fma (3-bit affine) or decode the e2m1
nibbles as MLX does (mxfp4, nvfp4, with the 2^14 factor folded into the
scale). The prep can fuse the producer of x (RMSNorm, swiglu, the attention
output gate, the GDN gated norm) so no extra kernel is launched.
"""

import mlx.core as mx
import mlx.nn as nn

_VPL = 16  # K values per lane per step of the 4-bit kernels
# At M = 2 mx.quantized_matmul is already weight-bound and the prep launch costs more than it saves.
_MIN_M, _MAX_M = 3, 8
# From 6 rows the tensor-op kernel (qmv_nax) is faster than the SIMD kernel, also under sustained load.
_NAX_MIN_M, _NAX_MAX_M = 6, 32
_NAX_KSTEP = 1024  # K values per split-K slice at the largest split
# Below 8 MB of weights the prep launch and the low threadgroup count cost more than the kernel saves.
_MIN_BYTES = 8 << 20
_KSTEP = 32 * _VPL  # K values per main-kernel step
_UNROLL = "#pragma clang loop unroll(full)"
_kernels = {}


def _tag(dtype):
    return {mx.bfloat16: "bf16", mx.float16: "f16", mx.float32: "f32"}[dtype]


def _mp(m):
    """Rows of the per-chunk sums: M rounded up to a multiple of 4."""
    return -(-m // 4) * 4


# One half2 pair of the unpack: ("m", a, b, src, mask, p, q) = values a and b as the fields
# at bit p of the low half and bit q of the high half of ``src & mask``; ("c", a, b, src, p, q)
# the same for an expression that is already masked; ("f", a, b, src, e) = the e2m1 nibbles
# e and e + 4 of ``src``.
_PAIRS4 = [
    pair
    for o in range(2)
    for pair in (
        ("m", 8 * o, 8 * o + 4, f"w[{o}]", 0x000F000F, 0, 0),
        ("m", 8 * o + 1, 8 * o + 5, f"w[{o}]", 0x00F000F0, 4, 4),
        ("m", 8 * o + 2, 8 * o + 6, f"w[{o}] >> 8", 0x000F000F, 0, 0),
        ("m", 8 * o + 3, 8 * o + 7, f"w[{o}] >> 8", 0x00F000F0, 4, 4),
    )
]
# 3-bit: 32 values in 3 words, value i at bit 3i. The words shifted by 6, 9 and 6 bring the
# fields past bit 7 of a half into the mantissa; values 5, 10, 21 and 26 cross a half or a
# word boundary and are assembled with insert_bits.
_PAIRS3 = [
    ("m", 0, 6, "w[0]", 0x001C0007, 0, 2),
    ("m", 1, 7, "w[0]", 0x00E00038, 3, 5),
    ("m", 2, 8, "w[0] >> 6", 0x001C0007, 0, 2),
    ("m", 3, 9, "w[0] >> 6", 0x00E00038, 3, 5),
    ("m", 11, 16, "w[1]", 0x0007000E, 1, 0),
    ("m", 12, 17, "w[1]", 0x00380070, 4, 3),
    ("m", 13, 18, "w[1]", 0x01C00380, 7, 6),
    ("m", 14, 19, "w[1] >> 9", 0x0007000E, 1, 0),
    ("m", 15, 20, "w[1] >> 9", 0x00380070, 4, 3),
    ("m", 22, 27, "w[2]", 0x000E001C, 2, 1),
    ("m", 23, 28, "w[2]", 0x007000E0, 5, 4),
    ("m", 24, 29, "w[2] >> 6", 0x000E001C, 2, 1),
    ("m", 25, 30, "w[2] >> 6", 0x007000E0, 5, 4),
    ("c", 4, 5, "((w[0] >> 6) & 0x1C0u) | ((w[0] << 1) & 0x70000u)", 6, 0),
    ("c", 26, 31, "((w[2] >> 14) & 0x7u) | ((w[2] >> 6) & 0x03800000u)", 0, 7),
    (
        "c",
        10,
        21,
        "insert_bits(w[0] >> 30, w[1], 2u, 1u)"
        " | insert_bits((w[1] >> 15) & 0x10000u, w[2], 17u, 2u)",
        0,
        0,
    ),
]
_PAIRS_FP4 = [
    ("f", 8 * o + e, 8 * o + e + 4, f"w[{o}]", e) for o in range(2) for e in range(4)
]
# e2m1 nibbles e and e + 4 of a word as half2: the magnitude bits at 9..11 (2^-14 times the
# value, as MLX's fp4_e2m1 decodes it) and the sign at bit 15.
_FP4 = (
    "(({w} & 0x00070007u) << 9) | (({w} & 0x00080008u) << 12)",
    "(({w} & 0x00700070u) << 5) | (({w} & 0x00800080u) << 8)",
    "(({w} & 0x07000700u) << 1) | (({w} & 0x08000800u) << 4)",
    "(({w} & 0x70007000u) >> 3) | ({w} & 0x80008000u)",
)


class _Format:
    """A weight format: P values per lane per step in WPL words, the scale group and the
    pair order of the fp16 unpack. ``exact`` pairs are normalized to the values; the 4-bit
    affine SIMD kernel instead pre-scales x by 2^-p per position. ``xscale`` is the x
    pre-scale of the exact SIMD kernels (2^12 keeps the e2m1 products normal)."""

    def __init__(self, mode, bits, group, pairs, exact, xscale=1.0):
        self.mode, self.bits, self.group, self.pairs = mode, bits, group, pairs
        self.exact, self.key = exact, (mode, bits, group)
        self.order = tuple(v for pr in pairs for v in pr[1:3])
        self.P = len(self.order)
        self.WPL = self.P * bits // 32
        self.affine = mode == "affine"
        if exact:
            self.xscale = (xscale,) * self.P
        else:
            self.xscale = tuple(2.0**-o for pr in pairs for o in pr[5:7])

    def pair(self, j, w, exact=None):
        """Half2 pair ``j`` from the words ``w`` (an expression; ``w[i]`` is word i)."""
        exact = self.exact if exact is None else exact
        kind, a, b, src, *rest = self.pairs[j]
        src = src.replace("w[", w + "[")
        if kind == "f":
            return f"as_type<half2>({_FP4[rest[0]].format(w=f'({src})')})"
        if kind == "m":
            mask, p, q = rest
            m = f"(({src}) & {mask:#010x}u) | 0x64006400u"
        else:
            p, q = rest
            m = f"({src}) | 0x64006400u"
        if not exact or p == q == 0:
            return f"as_type<half2>({m}) - half2(1024.0h)"
        return (
            f"fma(as_type<half2>({m}), half2({2.0**-p!r}h, {2.0**-q!r}h),"
            f" half2({-(2.0 ** (10 - p))!r}h, {-(2.0 ** (10 - q))!r}h))"
        )

    def scale(self, expr, sh):
        """The scale ``expr`` as a float; times 2^sh for the e2m1 formats."""
        if self.affine:
            return f"float({expr})"
        if self.mode == "mxfp4":
            return f"e8m0_scale({expr}, {sh})"
        return f"e4m3_scale({expr}, {sh})"


_FORMATS = {
    f.key: f
    for f in (
        _Format("affine", 4, 64, _PAIRS4, False),
        _Format("affine", 3, 64, _PAIRS3, True),
        _Format("mxfp4", 4, 32, _PAIRS_FP4, True, 4096.0),
        _Format("nvfp4", 4, 16, _PAIRS_FP4, True, 4096.0),
    )
}
_AFFINE4 = _FORMATS[("affine", 4, 64)]


def _fmt_of(module):
    """The format of a quantized module, or None when no kernel handles it."""
    return _FORMATS.get(
        (getattr(module, "mode", "affine"), module.bits, module.group_size)
    )


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

// The e8m0 (2^(e - 127)) and e4m3 block scales as floats, times 2^sh.
inline float e8m0_scale(uint8_t e, int sh) { return ldexp(1.0f, int(e) - 127 + sh); }
inline float e4m3_scale(uint8_t b, int sh) {
  return ldexp(float(as_type<half>(ushort((b & 127) << 7))), 8 + sh);
}

template <typename T>
inline void store16(device T* p, const thread float* v) {
  #pragma clang loop unroll(full)
  for (int i = 0; i < 16; i++) p[i] = static_cast<T>(v[i]);
}

// 16 values of the residual summed over its slot rows as mx.sum does (col_reduce_small: rows
// y, y + 8, ... per lane row in the input type, then the lane rows in order), into floats.
template <typename T>
inline void load16_res(const device T* rr, int K, int rows, thread float* out) {
  float v[16];
  T tot[16], t[16];
  const int L = min(8, rows);
  for (int y = 0; y < L; y++) {
    for (int i = 0; i < 16; i++) t[i] = T(0.0f);
    for (int r = y; r < rows; r += L) {
      load16(rr + (size_t)r * K, v);
      for (int i = 0; i < 16; i++) t[i] = T(v[i]) + t[i];
    }
    for (int i = 0; i < 16; i++) tot[i] = (y == 0) ? t[i] : t[i] + tot[i];
  }
  for (int i = 0; i < 16; i++) out[i] = float(tot[i]);
}
"""

_PREP_INPUTS = {
    "copy": ["x"],
    "rms_norm": ["x", "res", "weight", "has_res", "res_rows"],
    "swiglu": ["x"],
    "gate": ["x", "gate"],
    "gated_norm": ["x", "gate", "weight"],
}


def _load_x(cidx, store=False):
    """Load chunk ``cidx`` of row ``m`` of x into ``xx``; with ``has_res`` add the residual
    (its ``res_rows`` slot rows summed as mx.sum does) as the ops do."""
    return f"""
      float xx[16];
      load16(x + (size_t)m * K + ({cidx}) * 16, xx);
      if (has_res) {{
        float rr[16];
        load16_res(res + (size_t)m * res_rows * K + ({cidx}) * 16, K, res_rows, rr);
        {_UNROLL}
        for (int i = 0; i < 16; i++) xx[i] = float(T(xx[i] + rr[i]));
        {"store16(h + (size_t)m * K + (" + cidx + ") * 16, xx);" if store else ""}
      }}"""


def _values(kind, dst, cidx, rs):
    """Code that computes the 16 values of chunk ``cidx`` of row ``m`` into ``dst`` (floats)."""
    if kind == "rms_norm":
        return f"""{_load_x(cidx, store=True)}
      float ww[16];
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
      load16(x + xoff(m, {cidx}), {dst});
      load16(gate + goff(m, {cidx}), gg);
      {_UNROLL}
      for (int i = 0; i < 16; i++) {dst}[i] *= sigmoid_f(gg[i]);"""
    if kind == "gated_norm":
        return f"""
      float xx[16], gg[16], ww[16];
      load16(x + (size_t)m * K + ({cidx}) * 16, xx);
      load16(gate + (size_t)m * GS + GO + ({cidx}) * 16, gg);
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
        return _load_x("c2") + """
        float ww[16];
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
        load16(gate + (size_t)m * GS + GO + c2 * 16, gg);
        load16(weight + (c2 * 16) % D, ww);
        {_UNROLL}
        for (int i = 0; i < 16; i++) {
          amax = max(amax, fabs(gg[i]));
          amax2 = max(amax2, fabs(ww[i]));
        }"""
    # copy and gate: max|x|
    return """
        float xx[16];
        load16(x + xoff(m, c2), xx);
        {_UNROLL}
        for (int i = 0; i < 16; i++) amax = max(amax, fabs(xx[i]));"""


def _scan_threads(K):
    """Threads per prep threadgroup; each converts P values after the row scan."""
    return 256 if K <= 8192 else 512


def _prep_segments(K, P=16):
    """Threadgroups per row of a prep of P values per thread."""
    return -(-(K // P) // _scan_threads(K))


def _values_gated_norm(dst, cidx, CH):
    """The 16 * CH values of chunks ``cidx`` .. of row ``m`` of the gated norm into ``dst``."""
    P = 16 * CH
    loads = "\n".join(
        f"""      load16(x + (size_t)m * K + ({cidx} + {i}) * 16, xx + {16 * i});
      load16(gate + (size_t)m * GS + GO + ({cidx} + {i}) * 16, gg + {16 * i});
      load16(weight + (({cidx} + {i}) * 16) % D, ww + {16 * i});"""
        for i in range(CH)
    )
    return f"""
      float xx[{P}], gg[{P}], ww[{P}];
{loads}
      float ssh = 0.0f;
      {_UNROLL}
      for (int i = 0; i < {P}; i++) ssh += xx[i] * xx[i];
      // The TPH threads of one head are adjacent lanes of one simdgroup.
      {_UNROLL}
      for (int o = TPH / 2; o > 0; o >>= 1) ssh += simd_shuffle_xor(ssh, o);
      const float rsh = rsqrt(ssh / D + EPS);
      {_UNROLL}
      for (int i = 0; i < {P}; i++) {dst}[i] = xx[i] * rsh * ww[i] * (gg[i] * sigmoid_f(gg[i]));"""


def _prep_source(K, M, kind, eps=0.0, D=0, fmt=None, nax=False, gs=0, go=0, attn=None):
    """NT threads per P * NT values of one row: producer op in fp32, row scale, fp16 store.

    Every threadgroup of a row first scans the whole row with all its threads for a bound
    of the row's max (and the RMS), so the fp16 scale is per row without a separate pass.
    kind: "copy" (x), "rms_norm" (x * rsqrt(mean(x^2) + eps) * weight), "swiglu"
    (silu(gate) * up with gate = x[:, :K] and up = x[:, K:]), "gate" (x * sigmoid(gate)),
    "gated_norm" (per-head RMSNorm over D values times silu(gate)).
    Each thread stores P = fmt.P values in the pair order of ``fmt``, pre-scaled for the
    SIMD kernel, plus their sum for the bias term; ``nax`` stores the plain pair order and
    the sums per row, so the source is the same for every M. rms_norm adds ``res`` to x
    first when ``has_res`` is set (and stores the sum in ``h``); ``gs`` and ``go`` locate
    ``gate`` inside a wider row. ``attn`` = (H, Dh, QW) reads x as the attention output
    (B, H, L, Dh), L from its shape, and the gate of head h at column 2 * h * Dh + Dh of
    the (rows, QW) projection.
    """
    fmt = fmt or _AFFINE4
    P, CH = fmt.P, fmt.P // 16
    scale = (1.0,) * P if nax else fmt.xscale
    Mp = 0 if nax else _mp(M)
    NT = _scan_threads(K)
    NIT = _prep_segments(K)
    NSEG = _prep_segments(K, P)
    stores = "\n".join(
        f"      h[{t}] = half(v[{fmt.order[t]}] * (sc * {scale[t]}f));" for t in range(P)
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
    if kind == "gated_norm":
        values = _values_gated_norm("v", f"c * {CH}", CH)
    else:
        values = "\n".join(
            f"      {{{_values(kind, f'(v + {16 * i})', f'(c * {CH} + {i})', 'rs')}\n      }}"
            for i in range(CH)
        )
    if nax:
        xsum_store = "xsum[m * (K / P) + c] = s * sc;"
    else:
        xsum_store = "xsum[c * Mp + m] = s * sc;"
    if attn:
        H, Dh, QW = attn
        offs = f"""
    const int AL = x_shape[2];
    constexpr int AH = {H}, ADH = {Dh}, AQW = {QW};
    #define xoff(m, c) ((((size_t)((m) / AL) * AH + ((c) * 16) / ADH) * AL + (m) % AL) * ADH + ((c) * 16) % ADH)
    #define goff(m, c) ((size_t)(m) * AQW + (((c) * 16) / ADH) * 2 * ADH + ADH + ((c) * 16) % ADH)"""
    else:
        offs = """
    #define xoff(m, c) ((size_t)(m) * K + (size_t)(c) * 16)
    #define goff(m, c) ((size_t)(m) * K + (size_t)(c) * 16)"""
    return f"""
    constexpr int K = {K}, P = {P}, Mp = {Mp}, NT = {NT}, NIT = {NIT}, NSEG = {NSEG}, D = {D}, TPH = D / P;
    constexpr int GS = {gs or K}, GO = {go};{offs}
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
    const int c = seg * NT + t;
    if (c < K / P) {{
      float v[P];
{values}
      float s = 0.0f;
      {_UNROLL}
      for (int i = 0; i < P; i++) s += v[i];
      {xsum_store}
      half h[P];
{stores}
      device half4* o = (device half4*)(x16 + (size_t)m * K + c * P);
      {_UNROLL}
      for (int i = 0; i < P / 4; i++) o[i] = half4(h[4 * i], h[4 * i + 1], h[4 * i + 2], h[4 * i + 3]);
    }}
"""


def _main_source(M, N, K, R, NSG, MC, fmt):
    P, WPL, H = fmt.P, fmt.WPL, fmt.P // 16
    Mp = _mp(M)
    NB = K // (32 * P)
    wt = "uint2" if WPL == 2 else "packed_uint3"
    st = "T" if fmt.affine else "uint8_t"
    wload = "\n".join(
        f"      {{dst}}[{r}] = *(const device {wt}*)(wp + {r} * KW);" for r in range(R)
    )
    sload = "\n".join(
        f"      s[{r}] = {fmt.scale(f'sp[{r} * KG]', 2)};"
        + (f" bb[{r}] = float(bp[{r} * KG]);" if fmt.affine else "")
        for r in range(R)
    )
    body = []
    for h in range(H):
        # The 8 pairs of this half of the step, then the rows in chunks of MC.
        body += [f"      q2[{r}][{j}] = {fmt.pair(8 * h + j, f'wv[{r}]')};" for r in range(R) for j in range(8)]
        for m0 in range(0, M, MC):
            rows = range(m0, min(m0 + MC, M))
            body += [
                f"      xv[{m - m0}][{c}] = *(const device uint4*)(xp + {m} * K + {2 * h + c} * 8);"
                for m in rows
                for c in range(2)
            ]
            if h == 0 and fmt.affine:
                body += [f"      xs[{m - m0}] = xsp[{m}];" for m in rows]
            for r in range(R):
                for m in rows:
                    mm = m - m0
                    acc = f"acc[{r}][{m}]"
                    if h == 0 and fmt.affine:
                        acc = f"fma(bb[{r}], xs[{mm}], {acc})"
                    body.append(
                        f"      {{ half2 p = q2[{r}][0] * x2({mm}, 0);\n"
                        + "\n".join(
                            f"        p = fma(q2[{r}][{j}], x2({mm}, {j}), p);"
                            for j in range(1, 8)
                        )
                        + f"\n        acc[{r}][{m}] = fma(s[{r}], float(p.x + p.y), {acc}); }}"
                    )
    return f"""
    constexpr int M = {M}, N = {N}, K = {K}, R = {R}, NSG = {NSG}, P = {P}, WPL = {WPL}, Mp = {Mp}, MC = {MC};
    constexpr int KW = K * {fmt.bits} / 32, KG = K / {fmt.group}, NB = {NB};
    const int lane = thread_index_in_simdgroup;
    const int sg = simdgroup_index_in_threadgroup;
    const int row0 = (threadgroup_position_in_grid.x * NSG + sg) * R;
    const device uint32_t* wp = w + (size_t)row0 * KW + lane * WPL;
    const device {st}* sp = scales + (size_t)row0 * KG + lane * P / {fmt.group};
    {f"const device T* bp = biases + (size_t)row0 * KG + lane * P / {fmt.group};" if fmt.affine else ""}
    const device half* xp = x16 + lane * P;
    const device float* xsp = xsum + (size_t)lane * Mp;
    #define x2(m, j) as_type<half2>(xv[m][(j) / 4][(j) % 4])
    float acc[R][M];
    {_UNROLL}
    for (int r = 0; r < R; r++)
      {_UNROLL}
      for (int m = 0; m < M; m++) acc[r][m] = 0.0f;
    {wt} wv[R], wn[R];
    uint4 xv[MC][2];
    half2 q2[R][8];
    float s[R], bb[R], xs[MC];
    (void)bb;
    (void)xs;
    (void)xsp;
{wload.format(dst="wv")}
    for (int b = 0; b < NB; b++) {{
      // Request the next step's weights before this step's math.
      wp += 32 * WPL;
      if (b + 1 < NB) {{
{wload.format(dst="wn")}
      }}
{sload}
{chr(10).join(body)}
      sp += 32 * P / {fmt.group};
      {f"bp += 32 * P / {fmt.group};" if fmt.affine else ""}
      xp += 32 * P;
      xsp += 32 * Mp;
{chr(10).join(f"      wv[{r}] = wn[{r}];" for r in range(R))}
    }}
{chr(10).join(f"    acc[{r}][{m}] = simd_sum(acc[{r}][{m}]);" for r in range(R) for m in range(M))}
    if (lane == 0) {{
{chr(10).join(f"      y[(size_t){m} * N + row0 + {r}] = T(acc[{r}][{m}] * rscale[{m}]);" for r in range(R) for m in range(M))}
    }}
    #undef x2
"""


def _kernel(kind, key, source, inputs, outputs, header="", **kwargs):
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
            **kwargs,
        )
        _kernels[(kind, key)] = kern
    return kern


class Prepped:
    """An (M, K) activation already converted for ``qmv_small``; mimics the array's shape and dtype."""

    def __init__(self, x16, xsum, rscale, shape, dtype):
        self.x16, self.xsum, self.rscale = x16, xsum, rscale
        self.shape, self.dtype = shape, dtype
        self.ndim = len(shape)


def prep(
    x,
    kind="copy",
    *extra,
    eps=0.0,
    d=0,
    shape=None,
    fmt=None,
    nax=False,
    residual=None,
    gs=0,
    go=0,
    attn=None,
):
    """Convert ``x`` (M, K) [(M, 2K) for swiglu] plus the ``extra`` inputs of ``kind`` into a ``Prepped``.

    ``fmt`` is the weight format the result is for (4-bit affine by default) and ``nax``
    selects the tensor-op kernel's order. With ``residual`` (rms_norm only) x + residual
    is normed and returned as ``p.h``. ``gs`` and ``go`` locate the gate of gated_norm
    inside a wider array; ``attn`` = (L, H, Dh, QW) is the attention layout of the gate
    prep (see ``_prep_source``).
    """
    fmt = fmt or _AFFINE4
    if attn:
        M, K = x.size // (attn[1] * attn[2]), attn[1] * attn[2]
        # The kernel reads L from the shape of x: one kernel for every row count
        attn = attn[1:]
    else:
        M, K = x.shape
    if kind == "swiglu":
        K //= 2
    Mp = _mp(M)
    P = fmt.P
    outputs = ["x16", "xsum", "rscale"]
    if kind == "rms_norm":
        # One kernel serves both cases: the add is switched at run time
        res, rows = residual if isinstance(residual, tuple) else (residual, 1)
        extra = (x if res is None else res, *extra, int(res is not None), rows)
        outputs.append("h")
    # The tensor-op prep does not depend on M: one kernel per shape and kind.
    kern = _kernel(
        "qmv_small_prep_" + kind + ("_nax" if nax else ""),
        (K, 0 if nax else M, eps, d, _tag(x.dtype), fmt.key, gs, go, attn),
        lambda: _prep_source(K, M, kind, eps, d, fmt, nax, gs, go, attn),
        _PREP_INPUTS[kind],
        outputs,
        _HEADER,
    )
    x16, xsum, rscale, *h = kern(
        inputs=[x, *extra],
        template=[("T", x.dtype)],
        grid=(_scan_threads(K) * _prep_segments(K, P) * M, 1, 1),
        threadgroup=(_scan_threads(K), 1, 1),
        output_shapes=[(M, K), (M, K // P) if nax else (K // P, Mp), (Mp,)]
        + ([(M, K)] if kind == "rms_norm" else []),
        output_dtypes=[mx.float16, mx.float32, mx.float32]
        + ([x.dtype] if kind == "rms_norm" else []),
    )
    p = Prepped(x16, xsum, rscale, shape or (M, K), x.dtype)
    p.fmt, p.nax = fmt, nax
    p.h = h[0] if residual is not None else None
    return p


def _config(m):
    # Two rows per simdgroup keeps registers low; at M > 4 four rows amortize the dequant.
    return (2, 2, m) if m <= 4 else (4, 2, 4)


def _m5():
    """The kernel was tuned and measured on M5-class GPUs only (fp16 FMA rate, occupancy manager)."""
    if "m5" not in _kernels:
        info = mx.device_info() if mx.metal.is_available() else {}
        _kernels["m5"] = str(info.get("architecture", "")).startswith("applegpu_g17")
    return _kernels["m5"]


def _nax_m(m):
    """True when rows ``m`` go to the tensor-op kernel."""
    return _NAX_MIN_M <= m <= _NAX_MAX_M


def _shape_ok(m, n, k, fmt):
    if _nax_m(m):
        return k % _NAX_KSTEP == 0 and n % 32 == 0
    return _MIN_M <= m <= _MAX_M and k % (32 * fmt.P) == 0 and n % 8 == 0


def _params_ok(fmt, dtype, scales, biases):
    """The activation and scale dtypes the kernels of ``fmt`` take."""
    if dtype not in (mx.bfloat16, mx.float16):
        return False
    if fmt.affine:
        return scales.dtype == dtype and biases is not None and biases.dtype == dtype
    return scales.dtype == mx.uint8 and biases is None


def supported(x, w, scales, biases, group_size, bits, mode="affine"):
    fmt = _FORMATS.get((mode, bits, group_size))
    if not _m5() or fmt is None or x.ndim != 2:
        return False
    if not _params_ok(fmt, x.dtype, scales, biases):
        return False
    m, k = x.shape
    n = w.shape[0]
    return (
        _shape_ok(m, n, k, fmt)
        and w.shape[1] * 32 == k * bits
        and w.nbytes >= _MIN_BYTES
    )


def qmv_main(p, w, scales, biases):
    """``x @ dequant(w).T`` from a ``Prepped`` x."""
    M, K = p.x16.shape
    N = w.shape[0]
    R, NSG, MC = _config(M)
    fmt = p.fmt
    kern = _kernel(
        "qmv_small",
        (M, N, K, R, NSG, MC, _tag(p.dtype), fmt.key),
        lambda: _main_source(M, N, K, R, NSG, MC, fmt),
        ["x16", "xsum", "rscale", "w", "scales"] + (["biases"] if fmt.affine else []),
        ["y"],
        _HEADER,
    )
    ntg = N // (R * NSG)
    (y,) = kern(
        inputs=[p.x16, p.xsum, p.rscale, w, scales] + ([biases] if fmt.affine else []),
        template=[("T", p.dtype)],
        grid=(32 * NSG * ntg, 1, 1),
        threadgroup=(32 * NSG, 1, 1),
        output_shapes=[(M, N)],
        output_dtypes=[p.dtype],
    )
    return y


def _main(p, w, scales, biases):
    """The kernel matching the prep order of ``p``."""
    if p.nax:
        from .qmv_nax import nax_main

        return nax_main(p, w, scales, biases)
    return qmv_main(p, w, scales, biases)


def qmv_small(x, w, scales, biases, group_size=64, bits=4, mode="affine"):
    """``x @ dequant(w).T`` for ``x`` of shape (M, K); mx.quantized_matmul outside 3 <= M <= 32."""
    if not supported(x, w, scales, biases, group_size, bits, mode):
        return mx.quantized_matmul(
            x, w, scales, biases, transpose=True, group_size=group_size, bits=bits, mode=mode
        )
    fmt = _FORMATS[(mode, bits, group_size)]
    return _main(prep(x, fmt=fmt, nax=_nax_m(x.shape[0])), w, scales, biases)


def routes(module, shape, dtype):
    """True when ``qlinear`` uses the small-M kernel for ``module`` on an input of ``shape``."""
    *batch, k = shape
    m = 1
    for d in batch:
        m *= d
    if not (_m5() and isinstance(module, nn.QuantizedLinear) and "bias" not in module):
        return False
    fmt = _fmt_of(module)
    return (
        fmt is not None
        and _params_ok(fmt, dtype, module.scales, module.get("biases"))
        and _shape_ok(m, module.weight.shape[0], k, fmt)
        and module.weight.shape[1] * 32 == k * fmt.bits
        and module.weight.nbytes >= _MIN_BYTES
    )


def _rows(shape):
    m = 1
    for d in shape[:-1]:
        m *= d
    return m


def _target(module, shape):
    """The prep arguments for the projection ``module`` on ``shape`` rows."""
    return dict(fmt=_fmt_of(module), nax=_nax_m(_rows(shape)))


def prep_rms_norm(norm, x, module) -> "Prepped | mx.array":
    """``norm(x)`` fused with the prep when the projection ``module`` routes; else ``norm(x)``."""
    if not routes(module, x.shape, x.dtype):
        return norm(x)
    k = x.shape[-1]
    return prep(
        x.reshape(-1, k),
        "rms_norm",
        norm.weight,
        eps=norm.eps,
        shape=tuple(x.shape),
        **_target(module, x.shape),
    )


def prep_swiglu(gate_up, module) -> "Prepped | None":
    """``swiglu(gate, up)`` of the fused (.., 2K) projection output, prepped for ``module``."""
    *batch, k2 = gate_up.shape
    shape = (*batch, k2 // 2)
    if not routes(module, shape, gate_up.dtype):
        return None
    return prep(gate_up.reshape(-1, k2), "swiglu", shape=shape, **_target(module, shape))


def prep_gate(x, gate, module, attn=None) -> "Prepped | None":
    """``x * sigmoid(gate)`` prepped for ``module``, or None when it does not route.

    With ``attn`` = (L, H, Dh, QW), ``x`` is the attention output (B, H, L, Dh) and ``gate``
    the (rows, QW) projection holding the gate of head h at column 2 * h * Dh + Dh.
    """
    if attn:
        B, H, L, Dh = x.shape
        shape = (B, L, H * Dh)
    else:
        shape = tuple(x.shape)
        gate = gate.reshape(-1, shape[-1])
    if not routes(module, shape, x.dtype):
        return None
    if attn:
        return prep(x, "gate", gate, shape=shape, attn=attn, **_target(module, shape))
    return prep(x.reshape(-1, shape[-1]), "gate", gate, shape=shape, **_target(module, shape))


def prep_gated_norm(norm, x, gate, module, gate_offset=0) -> "Prepped | None":
    """``norm(x, gate)`` (per-head RMSNorm times silu(gate)) prepped for ``module``, or None.

    ``gate`` is (rows, width) with the gate of a row at column ``gate_offset``.
    """
    *batch, heads, d = x.shape
    shape = (*batch, heads * d)
    k = heads * d
    if not routes(module, shape, x.dtype):
        return None
    # The per-head reduction needs a power-of-two number of lanes per head (P values each).
    P = _fmt_of(module).P
    tph = d // P
    if d % P or tph > 32 or tph & (tph - 1):
        return None
    return prep(
        x.reshape(-1, k),
        "gated_norm",
        gate,
        norm.weight,
        eps=norm.eps,
        d=d,
        shape=shape,
        gs=gate.shape[-1],
        go=gate_offset,
        **_target(module, shape),
    )


def qlinear(module, x):
    """Apply a bias-free ``QuantizedLinear`` through the small-M kernels when 3 <= M <= 32."""
    if isinstance(x, Prepped):
        return _main(x, module.weight, module.scales, module.get("biases")).reshape(
            *x.shape[:-1], -1
        )
    if routes(module, x.shape, x.dtype):
        p = prep(x.reshape(-1, x.shape[-1]), **_target(module, x.shape))
        y = _main(p, module.weight, module.scales, module.get("biases"))
        return y.reshape(*x.shape[:-1], -1)
    return module(x)
