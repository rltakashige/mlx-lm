# Copyright © 2026 Apple Inc.

"""Small-M (4..32 rows) 4-bit affine g64 matmul on the M5 Neural Accelerators.

Each simdgroup runs ``matmul2d<MT x 32 x KOP, execution_simdgroup>`` (Metal 4 tensor ops)
with both inputs in cooperative tensors: the x rows in fp16 and the weight nibbles either
dequantized to fp16 (mode ``f16``) or fed as int8 with the group scale applied to the fp32
output of each group (modes ``i8``, ``i8p``). The lane layout of the cooperative tensors is
fixed at compile time from the M5 probe: lane ``l`` holds rows ``fm + 8r`` and 4 consecutive
k values ``4*cls..4*cls+3`` of every 16-wide fragment, with ``fm = ((l>>2)&4) | ((l>>1)&3)``
and ``cls = ((l>>2)&2) | (l&1)``. Weights are read as 16 consecutive nibbles per row per
step and x as 16 consecutive fp16 values; the k order inside a group is permuted the same
way on both sides, so it needs no memory shuffle.
"""

import mlx.core as mx

from .qmv_small import _HEADER as _SMALL_HEADER
from .qmv_small import Prepped, _kernel, _tag, prep

_MAX_M = 32
_HEADER = _SMALL_HEADER + """
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace mpp::tensor_ops;
"""
_UNROLL = "#pragma clang loop unroll(full)"


def _source(M, N, K, NSG, NT, KOP, mode, NTS, GU, PF):
    """NSG simdgroups split K; NT tiles of 32 columns per simdgroup at once; NTS tiles in sequence;
    GU groups (64 k) per load step (GU = 2: 16-byte weight loads); PF steps requested ahead."""
    i8 = mode != "f16"
    BT = "int8_t" if i8 else "half"
    MT = 16 if M <= 16 else 32
    NH = 1 if M <= 8 else (2 if M <= 16 else 4)  # x row sets fm + 8h read per lane
    G = K // 64
    GS = G // NSG
    NU = GS // GU  # load steps per simdgroup
    WT = "uint4" if GU == 2 else "uint2"
    if i8:
        assert GU == 1, "the int8 modes need one group per step"
    CA, CC = MT * KOP // 32, MT  # cooperative tensor elements per lane
    NOP = 64 // KOP  # ops per group and tile
    L = []
    add = L.append
    add(f"""
    constexpr int M = {M}, N = {N}, K = {K}, NSG = {NSG}, NT = {NT}, NTS = {NTS}, MT = {MT}, NH = {NH};
    constexpr int G = {G}, GS = {GS}, NU = {NU}, GU = {GU}, KW = K / 8, CC = {CC};
    const int lane = thread_index_in_simdgroup;
    const int sg = simdgroup_index_in_threadgroup;
    const int tile0 = threadgroup_position_in_grid.x * (NT * NTS);
    const int fm = ((lane >> 2) & 4) | ((lane >> 1) & 3);
    const int cls = ((lane >> 2) & 2) | (lane & 1);
    const int g0 = sg * GS;
    constexpr auto desc = matmul2d_descriptor(MT, 32, {KOP}, false, true, false,
                                              matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<desc, metal::execution_simdgroup> op;
    auto ta = op.get_left_input_cooperative_tensor<half, {BT}, float>();
    auto tb = op.get_right_input_cooperative_tensor<half, {BT}, float>();""")
    for t in range(NT):
        add(f"""    auto tc{t} = op.get_destination_cooperative_tensor<metal::remove_addrspace_t<decltype(ta)>,
                                                      metal::remove_addrspace_t<decltype(tb)>, float>();""")
        if i8:
            add(f"    float acc{t}[CC];")
    if i8:
        add(f"""    constexpr auto descb = matmul2d_descriptor(MT, 32, 16, false, true, false,
                                               matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<descb, metal::execution_simdgroup> opb;
    auto tab = opb.get_left_input_cooperative_tensor<half, half, float>();
    auto tbb = opb.get_right_input_cooperative_tensor<half, half, float>();
    auto tcb = opb.get_destination_cooperative_tensor<metal::remove_addrspace_t<decltype(tab)>,
                                                      metal::remove_addrspace_t<decltype(tbb)>, float>();""")
    for h in range(NH):
        add(f"    const device half* xp{h} = x16 + (size_t)min(fm + 8 * {h}, M - 1) * K + g0 * 64 + cls * {16 * GU};")
        if i8:
            add(f"    const device float* xq{h} = xsum + (size_t)min(fm + 8 * {h}, M - 1) * G + cls * 4;")
        add(f"    const float rs{h} = rscale[min(fm + 8 * {h}, M - 1)];")
    if NSG > 1:
        add("    threadgroup float red[(NSG - 1) * NT * CC * 32];")
    add(f"    {WT} wq[{PF}][NT][4], wn[NT][4];")
    add(f"    uint4 xv[NH][{2 * GU}];")
    add("""    for (int tt = 0; tt < NTS; tt++) {
      // The last threadgroup may own fewer tiles; the break is uniform over the threadgroup.
      if (tile0 + tt * NT >= N / 32) break;
      const int n0 = (tile0 + tt * NT) * 32;
      const device uint32_t* wp = w + (size_t)(n0 + fm) * KW + g0 * 8 + cls * 2 * GU;
      const device T* sp = scales + (size_t)(n0 + fm) * G + g0 + (GU == 2 ? cls >> 1 : 0);
      const device T* bp = biases + (size_t)(n0 + fm) * G + g0 + (GU == 2 ? cls >> 1 : 0);
      const device T* sq = scales + (size_t)(n0 + 4 * cls) * G + g0;
      (void)sp; (void)bp; (void)sq;""")
    for t in range(NT):
        add(f"      {_UNROLL}")
        add(f"      for (int i = 0; i < CC; i++) tc{t}[i] = 0.0f;")
        if i8:
            add(f"      {_UNROLL}")
            add(f"      for (int i = 0; i < CC; i++) acc{t}[i] = 0.0f;")
    if i8:
        # Bias term sum_g biases[n, g] * xsum[m, g]: a matmul over G with xsum split into hi + lo halves.
        add("      if (sg == 0) {")
        add(f"        {_UNROLL}")
        add("        for (int i = 0; i < CC; i++) tcb[i] = 0.0f;")
        add("        const device T* bq = biases + (size_t)(n0 + fm) * G + cls * 4;")
        add("        for (int j = 0; j < G / 16; j++) {")
        for h in range(NH):
            add(f"          const float4 v{h} = *(const device float4*)(xq{h} + j * 16);")
            add(f"          const half4 hi{h} = half4(v{h}); const half4 lo{h} = half4(v{h} - float4(hi{h}));")
        for t in range(NT):
            for r in range(4):
                add("          " + " ".join(
                    f"tbb[{4 * r + e}] = half(bq[({t} * 32 + {r} * 8) * G + j * 16 + {e}]);" for e in range(4)
                ))
            for part in ("hi", "lo"):
                for i in range(MT // 2):
                    f, rem = divmod(i, 8)
                    h, e = divmod(rem, 4)
                    add(f"          tab[{i}] = {part}{min(2 * f + h, NH - 1)}[{e}];")
                add("          opb.run(tab, tbb, tcb);")
            add(f"          {_UNROLL}")
            add(f"          for (int i = 0; i < CC; i++) {{ acc{t}[i] += tcb[i]; tcb[i] = 0.0f; }}")
        add("        }")
        add("      }")
    for u in range(PF):
        for t in range(NT):
            for r in range(4):
                add(f"      wq[{u}][{t}][{r}] = *(const device {WT}*)(wp + ({t} * 32 + {r} * 8) * KW + {u * 8 * GU});")
    add("      for (int u = 0; u < NU; u++) {")
    add(f"        // Request the weights {PF} step(s) ahead before this step's math.")
    add(f"        if (u + {PF} < NU) {{")
    for t in range(NT):
        for r in range(4):
            add(f"          wn[{t}][{r}] = *(const device {WT}*)(wp + ({t} * 32 + {r} * 8) * KW + {PF * 8 * GU});")
    add("        }")
    if not i8:
        add("        half sv[NT][4], bv[NT][4];")
        for t in range(NT):
            for r in range(4):
                add(f"        sv[{t}][{r}] = half(sp[({t} * 32 + {r} * 8) * G]); bv[{t}][{r}] = half(bp[({t} * 32 + {r} * 8) * G]);")
    for h in range(NH):
        for c in range(2 * GU):
            add(f"        xv[{h}][{c}] = *(const device uint4*)(xp{h} + u * {64 * GU} + {8 * c});")
        add(f"        const thread half* xh{h} = (const thread half*)&xv[{h}];")
    for o in range(NOP * GU):
        # Left input element 8*f + 4*h + e, fragment f = (mf, kf): x row fm + 8*(2*mf + h), logical
        # k = 16*kf + 4*cls + e of op o, which is one of the lane's 16*GU physical values (see the B fill).
        for i in range(CA):
            f, rem = divmod(i, 8)
            h, e = divmod(rem, 4)
            mf, kf = divmod(f, 2) if KOP == 32 else (f, 0)
            j = o * (KOP // 16) + kf
            phys = (8 * (j >> 1) + 2 * e + (j & 1)) if i8 else (4 * j + e)
            add(f"        ta[{i}] = xh{min(2 * mf + h, NH - 1)}[{phys}];")
        for t in range(NT):
            # Right input element 16*kf + 4*r + e: weight row fm + 8r, k fragment j = o*(KOP/16) + kf,
            # value e. f16: nibble 4j + e of the lane's 16*GU nibbles. i8: nibbles 0,2,4,6 of word (j>>1)
            # are kf = 0 and nibbles 1,3,5,7 are kf = 1 (one mask per 4 values).
            for r in range(4):
                word = f"wq[0][{t}][{r}].{'xyzw'[o if KOP == 32 else o // 2]}"
                if i8:
                    for kf in ((0, 1) if KOP == 32 else (o % 2,)):
                        src = f"(({word} >> {4 * kf}) & 0x0F0F0F0Fu)" if kf else f"({word} & 0x0F0F0F0Fu)"
                        base = (16 * kf if KOP == 32 else 0) + 4 * r
                        if mode == "i8p":
                            add(f"        *(thread uint32_t*)&tb[{base}] = {src};")
                        else:
                            add(f"        {{ const uint32_t u8 = {src};")
                            add("          " + " ".join(f"tb[{base + e}] = int8_t((u8 >> {8 * e}) & 0xFu);" for e in range(4)) + " }")
                    continue
                for e in range(4):
                    sh = f"({word} >> {4 * e})" if e else word
                    add(f"        {{ half2 q = as_type<half2>(({sh} & 0x000F000Fu) | 0x64006400u) - half2(1024.0h);")
                    add(f"          q = fma(q, half2(sv[{t}][{r}]), half2(bv[{t}][{r}]));")
                    if KOP == 32:
                        add(f"          tb[{4 * r + e}] = q.x; tb[{16 + 4 * r + e}] = q.y; }}")
                    else:
                        add(f"          tb[{4 * r + e}] = q.{'x' if o % 2 == 0 else 'y'}; }}")
            add(f"        op.run(ta, tb, tc{t});")
    if i8:
        # Scale of the lane's output columns 16*nf + 4*cls + e for this group.
        for t in range(NT):
            for nf in range(2):
                for e in range(4):
                    add(f"        const float s{t}_{nf}{e} = float(sq[({t} * 32 + {16 * nf} + {e}) * G]);")
            for i in range(CC):
                f, rem = divmod(i, 8)
                nf = f % 2 if MT == 32 else f
                add(f"        acc{t}[{i}] = fma(tc{t}[{i}], s{t}_{nf}{rem % 4}, acc{t}[{i}]); tc{t}[{i}] = 0.0f;")
    add(f"        wp += {8 * GU}; sp += {GU}; bp += {GU}; sq += 1;")
    for u in range(PF - 1):
        for t in range(NT):
            for r in range(4):
                add(f"        wq[{u}][{t}][{r}] = wq[{u + 1}][{t}][{r}];")
    for t in range(NT):
        for r in range(4):
            add(f"        wq[{PF - 1}][{t}][{r}] = wn[{t}][{r}];")
    add("      }")
    # Cross-simdgroup reduction of the split-K partials, then the store by simdgroup 0.
    src = "acc" if i8 else "tc"
    if NSG > 1:
        add("      if (sg > 0) {")
        for t in range(NT):
            add(f"        {_UNROLL}")
            add(f"        for (int i = 0; i < CC; i++) red[(((sg - 1) * NT + {t}) * CC + i) * 32 + lane] = {src}{t}[i];")
        add("      }")
        add("      threadgroup_barrier(mem_flags::mem_threadgroup);")
    add("      if (sg == 0) {")
    for t in range(NT):
        # Destination element 8*f + 4*h + e, f = (mf, nf): row fm + 8*(2*mf + h), column 16*nf + 4*cls + e.
        for i in range(CC):
            f, rem = divmod(i, 8)
            h, e = divmod(rem, 4)
            mf, nf = divmod(f, 2) if MT == 32 else (0, f)
            hh = 2 * mf + h
            if hh >= NH:
                continue
            val = f"{src}{t}[{i}]"
            if NSG > 1:
                val = f"({val}" + "".join(
                    f" + red[((({s} - 1) * NT + {t}) * CC + {i}) * 32 + lane]" for s in range(1, NSG)
                ) + ")"
            add(f"        if (fm + {8 * hh} < M) y[(size_t)(fm + {8 * hh}) * N + n0 + {32 * t + 16 * nf} + 4 * cls + {e}] = TO({val} * rs{hh});")
    add("      }")
    if NSG > 1 and NTS > 1:
        add("      threadgroup_barrier(mem_flags::mem_threadgroup);")
    add("    }")
    return "\n".join(L)


def _config(M, N, K):
    """(NSG, NT, KOP, mode, NTS, GU, PF) for a shape: split K over 8 simdgroups, 16 for the small-N
    projections where the grid is short (their 32-row reduction does not fit threadgroup memory)."""
    G = K // 64
    if M > 16:
        # 32-row tiles: 4 simdgroups on the wide projections, 8 elsewhere.
        NSG = 4 if N >= 32768 else 8
    else:
        NSG = 16 if N <= 8192 and G % 16 == 0 else 8
    return NSG, 1, 32, "f16", 1, 1, 1


def supported(x, w, scales, biases, group_size, bits):
    if x.ndim != 2 or bits != 4 or group_size != 64 or biases is None:
        return False
    if x.dtype not in (mx.bfloat16, mx.float16) or scales.dtype != x.dtype or biases.dtype != x.dtype:
        return False
    m, k = x.shape
    n = w.shape[0]
    return 1 <= m <= _MAX_M and k % 256 == 0 and n % 32 == 0 and w.shape[1] * 8 == k


def nax_main(p, w, scales, biases, cfg=None, out_dtype=None):
    """``x @ dequant(w).T`` from a ``Prepped`` x stored in natural k order."""
    M, K = p.x16.shape
    N = w.shape[0]
    out_dtype = out_dtype or p.dtype
    NSG, NT, KOP, mode, NTS, GU, PF = (tuple(cfg) + ("f16", 1, 1, 1))[:7] if cfg else _config(M, N, K)
    kern = _kernel(
        "qmv_nax_" + mode,
        (M, N, K, NSG, NT, KOP, mode, NTS, GU, PF, _tag(p.dtype), _tag(out_dtype)),
        lambda: _source(M, N, K, NSG, NT, KOP, mode, NTS, GU, PF),
        ["x16", "xsum", "rscale", "w", "scales", "biases"],
        ["y"],
        _HEADER,
    )
    ntg = -(-N // (32 * NT * NTS))
    (y,) = kern(
        inputs=[p.x16, p.xsum, p.rscale, w, scales, biases],
        template=[("T", p.dtype), ("TO", out_dtype)],
        grid=(32 * NSG * ntg, 1, 1),
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
    return nax_main(prep(x, natural=True), w, scales, biases, cfg)
