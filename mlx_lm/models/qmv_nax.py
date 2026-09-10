# Copyright © 2026 Apple Inc.

"""Small-M (5..32 rows) quantized matmul on the M5 Neural Accelerators.

Each simdgroup runs ``matmul2d<MT x 32 x 32, execution_simdgroup>`` (Metal 4 tensor ops)
over one 32-column tile and a slice of K, with both inputs in cooperative tensors: the x
rows in fp16 and the weights dequantized to fp16 (the magic-number pairs of ``_Format``).
The lane layout of the cooperative tensors is fixed at compile time from the M5 probe: lane
``l`` holds rows ``fm + 8r`` and 4 consecutive k values of every 16-wide fragment, with
``fm = ((l>>2)&4) | ((l>>1)&3)`` and ``cls = ((l>>2)&2) | (l&1)``. A lane reads P
consecutive values per row and block (a block is 4 * P k values, P / 8 tensor ops); the
pairs of the unpack land at the tensor positions of the format's order, so x is prepped in
that order and both tensors are filled with packed writes. The K split is interleaved:
simdgroup ``sg`` takes the blocks ``sg``, ``sg + NSG``, ..., so one step of a threadgroup
reads NSG consecutive pieces of every weight row (whole cache lines). The partials of a
tile are summed through threadgroup memory in simdgroup order.
"""

import mlx.core as mx

from .qmv_small import _AFFINE4, _FORMATS
from .qmv_small import _HEADER as _SMALL_HEADER
from .qmv_small import _kernel, _params_ok, _tag, prep

_MAX_M = 32
_HEADER = _SMALL_HEADER + """
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace mpp::tensor_ops;
"""
_UNROLL = "#pragma clang loop unroll(full)"


def _source(MB, N, K, NSG, fmt):
    """Kernel for up to MB (8, 16 or 32) rows; the row count M is read from the x16 shape at
    run time. NSG simdgroups split K (interleaved); each holds one 32-column tile."""
    MT = 16 if MB <= 16 else 32
    NH = MB // 8  # x row sets fm + 8h read per lane
    P, WPL, OPS = fmt.P, fmt.WPL, fmt.P // 8
    KB = 4 * P  # k values per block
    NB = K // KB
    GS = -(-NB // NSG)
    guard = NB % NSG != 0
    wt = "uint2" if WPL == 2 else "packed_uint3"
    st = "T" if fmt.affine else "uint8_t"
    L = []
    add = L.append
    add(f"""
    constexpr int N = {N}, K = {K}, NSG = {NSG}, MT = {MT}, NH = {NH}, NB = {NB}, GS = {GS};
    constexpr int KW = K * {fmt.bits} / 32, G = K / {fmt.group}, KB = {KB}, P = {P}, WPL = {WPL};
    const int M = x16_shape[0];
    const int lane = thread_index_in_simdgroup;
    const int sg = simdgroup_index_in_threadgroup;
    const int n0 = threadgroup_position_in_grid.x * 32;
    const int fm = ((lane >> 2) & 4) | ((lane >> 1) & 3);
    const int cls = ((lane >> 2) & 2) | (lane & 1);
    const int g0 = sg;  // blocks g0, g0 + NSG, ...
    constexpr auto desc = matmul2d_descriptor(MT, 32, 32, false, true, false,
                                              matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<desc, metal::execution_simdgroup> op;
    auto ta = op.get_left_input_cooperative_tensor<half, half, float>();
    auto tb = op.get_right_input_cooperative_tensor<half, half, float>();
    auto tc = op.get_destination_cooperative_tensor<metal::remove_addrspace_t<decltype(ta)>,
                                                    metal::remove_addrspace_t<decltype(tb)>, float>();
    const device uint32_t* wp = w + (size_t)(n0 + fm) * KW + g0 * 4 * WPL + cls * WPL;
    const device {st}* sp = scales + (size_t)(n0 + fm) * G + g0 * (KB / {fmt.group}) + cls * P / {fmt.group};""")
    if fmt.affine:
        add(f"    const device T* bp = biases + (size_t)(n0 + fm) * G + g0 * (KB / {fmt.group}) + cls * P / {fmt.group};")
    add(f"""    threadgroup float red[(NSG - 1) * MT * 32];
    {_UNROLL}
    for (int i = 0; i < MT; i++) tc[i] = 0.0f;""")
    for h in range(NH):
        add(f"    const device half* xp{h} = x16 + (size_t)min(fm + 8 * {h}, M - 1) * K + g0 * KB + cls * P;")
    add("    for (int u = 0; u < GS; u++) {")
    if guard:
        add("      if (g0 + u * NSG < NB) {")
    add(f"      {wt} wq[4];")
    add("      half sv[4], bv[4];" if fmt.affine else "      half sv[4];")
    add(f"      uint4 xv[NH][{OPS}];")
    for r in range(4):
        add(f"      wq[{r}] = *(const device {wt}*)(wp + {8 * r} * KW);")
        if fmt.affine:
            add(f"      sv[{r}] = half(sp[{8 * r} * G]); bv[{r}] = half(bp[{8 * r} * G]);")
        else:
            add(f"      sv[{r}] = half({fmt.scale(f'sp[{8 * r} * G]', 14)});")
    for h in range(NH):
        for o in range(OPS):
            add(f"      xv[{h}][{o}] = *(const device uint4*)(xp{h} + {8 * o});")
    for o in range(OPS):
        # Left input element 8*f + 4*h + e, fragment f = (mf, kf): x row fm + 8*(2*mf + h) and the
        # 4 stored values of k fragment kf of op o, one 8-byte write per (f, h).
        for f in range(MT // 8):
            mf, kf = divmod(f, 2)
            for h in range(2):
                hh = min(2 * mf + h, NH - 1)
                add(f"      *(thread uint2*)&ta[{8 * f + 4 * h}] = xv[{hh}][{o}].{'xy' if kf == 0 else 'zw'};")
        # Right input: pair e of op o (tensor positions 8o + 2e, + 1), row fm + 8r, lands at
        # elements 16*(e>>1) + 4*r + 2*(e&1) and + 1 with one 4-byte write.
        for r in range(4):
            for e in range(4):
                q = fmt.pair(4 * o + e, f"wq[{r}]", exact=True)
                val = f"fma(q, half2(sv[{r}]), half2(bv[{r}]))" if fmt.affine else f"q * half2(sv[{r}])"
                add(f"      {{ half2 q = {q};")
                add(f"        *(thread half2*)&tb[{16 * (e >> 1) + 4 * r + 2 * (e & 1)}] = {val}; }}")
        add("      op.run(ta, tb, tc);")
    if guard:
        add("      }")
    add(f"      wp += NSG * 4 * WPL; sp += NSG * (KB / {fmt.group});" + (f" bp += NSG * (KB / {fmt.group});" if fmt.affine else ""))
    for h in range(NH):
        add(f"      xp{h} += NSG * KB;")
    add("    }")
    # Split-K partials: simdgroup 0 adds the others in order, then stores.
    add("    if (sg > 0) {")
    add(f"      {_UNROLL}")
    add("      for (int i = 0; i < MT; i++) red[((sg - 1) * MT + i) * 32 + lane] = tc[i];")
    add("    }")
    add("    threadgroup_barrier(mem_flags::mem_threadgroup);")
    add("    if (sg == 0) {")
    for h in range(NH):
        add(f"      const float rs{h} = rscale[min(fm + 8 * {h}, M - 1)];")
    for i in range(MT):
        # Destination element 8*f + 4*h + e, f = (mf, nf): row fm + 8*(2*mf + h), column 16*nf + 4*cls + e.
        f, rem = divmod(i, 8)
        h, e = divmod(rem, 4)
        mf, nf = divmod(f, 2) if MT == 32 else (0, f)
        hh = 2 * mf + h
        if hh >= NH:
            continue
        val = f"(tc[{i}]" + "".join(f" + red[(({s} - 1) * MT + {i}) * 32 + lane]" for s in range(1, NSG)) + ")"
        add(f"      if (fm + {8 * hh} < M) y[(size_t)(fm + {8 * hh}) * N + n0 + {16 * nf} + 4 * cls + {e}] = TO({val} * rs{hh});")
    add("    }")
    return "\n".join(L)


def _config(M, N, K, fmt):
    """Simdgroups per tile: split K over 8, 16 for the small-N projections where the grid is short
    (their 32-row reduction does not fit threadgroup memory) when each simdgroup keeps at
    least 4 blocks (at 3-bit the blocks are 128 wide: 16 would leave out_proj 3 blocks each)."""
    NB = K // (4 * fmt.P)
    if M > 16:
        return 4 if N >= 32768 else 8
    return 16 if N <= 8192 and NB % 16 == 0 and NB >= 64 else 8


def supported(x, w, scales, biases, group_size, bits, mode="affine"):
    fmt = _FORMATS.get((mode, bits, group_size))
    if fmt is None or x.ndim != 2 or not _params_ok(fmt, x.dtype, scales, biases):
        return False
    m, k = x.shape
    n = w.shape[0]
    return 1 <= m <= _MAX_M and k % (16 * fmt.P) == 0 and n % 32 == 0 and w.shape[1] * 32 == k * bits


def nax_main(p, w, scales, biases, cfg=None, out_dtype=None):
    """``x @ dequant(w).T`` from a ``Prepped`` x stored in the pair order (``prep(nax=True)``)."""
    M, K = p.x16.shape
    N = w.shape[0]
    fmt = p.fmt
    out_dtype = out_dtype or p.dtype
    MB = 8 if M <= 8 else (16 if M <= 16 else 32)  # one kernel per row bucket
    NSG = cfg[0] if cfg else _config(MB, N, K, fmt)
    kern = _kernel(
        "qmv_nax",
        (MB, N, K, NSG, _tag(p.dtype), _tag(out_dtype), fmt.key),
        lambda: _source(MB, N, K, NSG, fmt),
        ["x16", "xsum", "rscale", "w", "scales"] + (["biases"] if fmt.affine else []),
        ["y"],
        _HEADER,
    )
    (y,) = kern(
        inputs=[p.x16, p.xsum, p.rscale, w, scales] + ([biases] if fmt.affine else []),
        template=[("T", p.dtype), ("TO", out_dtype)],
        grid=(32 * NSG * (N // 32), 1, 1),
        threadgroup=(32 * NSG, 1, 1),
        output_shapes=[(M, N)],
        output_dtypes=[out_dtype],
    )
    return y


def qmv_nax(x, w, scales, biases, group_size=64, bits=4, mode="affine", cfg=None):
    """``x @ dequant(w).T`` for ``x`` of shape (M, K), M <= 32, through the tensor-op kernel."""
    if not supported(x, w, scales, biases, group_size, bits, mode):
        return mx.quantized_matmul(
            x, w, scales, biases, transpose=True, group_size=group_size, bits=bits, mode=mode
        )
    fmt = _FORMATS[(mode, bits, group_size)]
    return nax_main(prep(x, fmt=fmt, nax=True), w, scales, biases, cfg)
