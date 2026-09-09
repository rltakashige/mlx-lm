# Copyright © 2026 Apple Inc.

"""Small-M (6..32 rows) 4-bit affine g64 matmul on the M5 Neural Accelerators.

Each simdgroup runs ``matmul2d<MT x 32 x 32, execution_simdgroup>`` (Metal 4 tensor ops)
over one 32-column tile and a slice of K, with both inputs in cooperative tensors: the x
rows in fp16 and the weight nibbles dequantized to fp16 (magic-number pair trick). The
lane layout of the cooperative tensors is fixed at compile time from the M5 probe: lane
``l`` holds rows ``fm + 8r`` and 4 consecutive k values of every 16-wide fragment, with
``fm = ((l>>2)&4) | ((l>>1)&3)`` and ``cls = ((l>>2)&2) | (l&1)``. A lane reads 16
consecutive nibbles per row and step; the pair trick yields the nibbles (e, e + 4) of a
word together, so x is prepped in the pair order (0, 4, 1, 5, 2, 6, 3, 7) and both
tensors are filled with packed writes. The split-K partials of a tile are summed through
threadgroup memory in simdgroup order.
"""

import mlx.core as mx

from .qmv_small import _HEADER as _SMALL_HEADER
from .qmv_small import _kernel, _tag, prep

_MAX_M = 32
_HEADER = _SMALL_HEADER + """
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace mpp::tensor_ops;
"""
_UNROLL = "#pragma clang loop unroll(full)"


def _source(MB, N, K, NSG):
    """Kernel for up to MB (8, 16 or 32) rows; the row count M is read from the x16 shape at
    run time. NSG simdgroups split K; each holds one 32-column tile."""
    MT = 16 if MB <= 16 else 32
    NH = MB // 8  # x row sets fm + 8h read per lane
    G = K // 64
    GS = G // NSG
    L = []
    add = L.append
    add(f"""
    constexpr int N = {N}, K = {K}, NSG = {NSG}, MT = {MT}, NH = {NH}, G = {G}, GS = {GS}, KW = K / 8;
    const int M = x16_shape[0];
    const int lane = thread_index_in_simdgroup;
    const int sg = simdgroup_index_in_threadgroup;
    const int n0 = threadgroup_position_in_grid.x * 32;
    const int fm = ((lane >> 2) & 4) | ((lane >> 1) & 3);
    const int cls = ((lane >> 2) & 2) | (lane & 1);
    const int g0 = sg * GS;
    constexpr auto desc = matmul2d_descriptor(MT, 32, 32, false, true, false,
                                              matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<desc, metal::execution_simdgroup> op;
    auto ta = op.get_left_input_cooperative_tensor<half, half, float>();
    auto tb = op.get_right_input_cooperative_tensor<half, half, float>();
    auto tc = op.get_destination_cooperative_tensor<metal::remove_addrspace_t<decltype(ta)>,
                                                    metal::remove_addrspace_t<decltype(tb)>, float>();
    const device uint32_t* wp = w + (size_t)(n0 + fm) * KW + g0 * 8 + cls * 2;
    const device T* sp = scales + (size_t)(n0 + fm) * G + g0;
    const device T* bp = biases + (size_t)(n0 + fm) * G + g0;
    threadgroup float red[(NSG - 1) * MT * 32];
    {_UNROLL}
    for (int i = 0; i < MT; i++) tc[i] = 0.0f;""")
    for h in range(NH):
        add(f"    const device half* xp{h} = x16 + (size_t)min(fm + 8 * {h}, M - 1) * K + g0 * 64 + cls * 16;")
    # x, scale and bias of a step are loaded one step ahead (L2 latency); the weights stream from DRAM.
    add("    uint4 xn[NH][2];")
    add("    half sn[4], bn[4];")
    for h in range(NH):
        add(f"    xn[{h}][0] = *(const device uint4*)(xp{h}); xn[{h}][1] = *(const device uint4*)(xp{h} + 8);")
    for r in range(4):
        add(f"    sn[{r}] = half(sp[{8 * r} * G]); bn[{r}] = half(bp[{8 * r} * G]);")
    add("    for (int u = 0; u < GS; u++) {")
    add("      uint4 xv[NH][2];")
    add("      half sv[4], bv[4];")
    for h in range(NH):
        add(f"      xv[{h}][0] = xn[{h}][0]; xv[{h}][1] = xn[{h}][1];")
    for r in range(4):
        add(f"      sv[{r}] = sn[{r}]; bv[{r}] = bn[{r}];")
    add("      if (u + 1 < GS) {")
    for h in range(NH):
        add(f"        xn[{h}][0] = *(const device uint4*)(xp{h} + u * 64 + 64); xn[{h}][1] = *(const device uint4*)(xp{h} + u * 64 + 72);")
    for r in range(4):
        add(f"        sn[{r}] = half(sp[{8 * r} * G + 1]); bn[{r}] = half(bp[{8 * r} * G + 1]);")
    add("      }")
    add("      uint2 wq[4];")
    for r in range(4):
        add(f"      wq[{r}] = *(const device uint2*)(wp + {8 * r} * KW);")
    for o in range(2):
        # Left input element 8*f + 4*h + e, fragment f = (mf, kf): x row fm + 8*(2*mf + h) and the
        # 4 pair-order values of k fragment kf of op o, one 8-byte write per (f, h).
        for f in range(MT // 8):
            mf, kf = divmod(f, 2)
            for h in range(2):
                hh = min(2 * mf + h, NH - 1)
                add(f"      *(thread uint2*)&ta[{8 * f + 4 * h}] = xv[{hh}][{o}].{'xy' if kf == 0 else 'zw'};")
        # Right input: the pair (nibble e, nibble e + 4) of word o, row fm + 8r, lands at elements
        # 16*(e>>1) + 4*r + 2*(e&1) and + 1 with one 4-byte write.
        for r in range(4):
            for e in range(4):
                sh = f"(wq[{r}].{'xy'[o]} >> {4 * e})" if e else f"wq[{r}].{'xy'[o]}"
                add(f"      {{ half2 q = as_type<half2>(({sh} & 0x000F000Fu) | 0x64006400u) - half2(1024.0h);")
                add(f"        *(thread half2*)&tb[{16 * (e >> 1) + 4 * r + 2 * (e & 1)}] = fma(q, half2(sv[{r}]), half2(bv[{r}])); }}")
        add("      op.run(ta, tb, tc);")
    add("      wp += 8; sp += 1; bp += 1;")
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


def _config(M, N, K):
    """Simdgroups per tile: split K over 8, 16 for the small-N projections where the grid is short
    (their 32-row reduction does not fit threadgroup memory)."""
    G = K // 64
    if M > 16:
        return 4 if N >= 32768 else 8
    return 16 if N <= 8192 and G % 16 == 0 else 8


def supported(x, w, scales, biases, group_size, bits):
    if x.ndim != 2 or bits != 4 or group_size != 64 or biases is None:
        return False
    if x.dtype not in (mx.bfloat16, mx.float16) or scales.dtype != x.dtype or biases.dtype != x.dtype:
        return False
    m, k = x.shape
    n = w.shape[0]
    return 1 <= m <= _MAX_M and k % 256 == 0 and n % 32 == 0 and w.shape[1] * 8 == k


def nax_main(p, w, scales, biases, cfg=None, out_dtype=None):
    """``x @ dequant(w).T`` from a ``Prepped`` x stored in the pair order (``prep(natural=2)``)."""
    M, K = p.x16.shape
    N = w.shape[0]
    out_dtype = out_dtype or p.dtype
    MB = 8 if M <= 8 else (16 if M <= 16 else 32)  # one kernel per row bucket
    NSG = cfg[0] if cfg else _config(MB, N, K)
    kern = _kernel(
        "qmv_nax",
        (MB, N, K, NSG, _tag(p.dtype), _tag(out_dtype)),
        lambda: _source(MB, N, K, NSG),
        ["x16", "xsum", "rscale", "w", "scales", "biases"],
        ["y"],
        _HEADER,
    )
    (y,) = kern(
        inputs=[p.x16, p.xsum, p.rscale, w, scales, biases],
        template=[("T", p.dtype), ("TO", out_dtype)],
        grid=(32 * NSG * (N // 32), 1, 1),
        threadgroup=(32 * NSG, 1, 1),
        output_shapes=[(M, N)],
        output_dtypes=[out_dtype],
    )
    return y


def qmv_nax(x, w, scales, biases, group_size=64, bits=4, cfg=None):
    """``x @ dequant(w).T`` for ``x`` of shape (M, K), M <= 32, through the tensor-op kernel."""
    if not supported(x, w, scales, biases, group_size, bits):
        return mx.quantized_matmul(
            x, w, scales, biases, transpose=True, group_size=group_size, bits=bits
        )
    return nax_main(prep(x, natural=2), w, scales, biases, cfg)
