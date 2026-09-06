# Small-M 4-bit qmv kernel log (M5 Max, troy)

Branch: leo/qmv-small-m (from leo/qwen-mtp-opt). mlx 0.32.2, macOS 26.6.2 on troy.

## Step 1: baseline (qmm_curve.py, 64 reps, GB/s)

| shape | N x K | M=1 | M=2 | M=3 | M=4 | M=6 | M=8 |
|---|---|---|---|---|---|---|---|
| gdn.in_proj | 16480x5120 | 525 | 567 | 550 | 503 | 260 | 182 |
| gdn.out_proj | 5120x6144 | 520 | 512 | 505 | 451 | 225 | 190 |
| attn.qkv_proj | 14336x5120 | 566 | 563 | 545 | 502 | 263 | 190 |
| attn.o_proj | 5120x6144 | 518 | 526 | 460 | 451 | 221 | 194 |
| mlp.gate_up | 34816x5120 | 560 | 578 | 551 | 479 | 219 | 145 |
| mlp.down | 5120x17408 | 564 | 534 | 483 | 388 | 253 | 204 |
| lm_head | 248320x5120 | 584 | 513 | 426 | 252 | 149 | 102 |

Sum over model (ms): M=1 26.1, M=2 25.9, M=3 27.6, M=4 32.9, M=6 63.2, M=8 88.8.
(Task numbers: 25.6 / 32.3 / 88.8. Confirmed.)

## ALU probe (local M4 Max, for reference)

fp32 FMA 7.4 Tops/s, fp16 scalar 7.5, half2 7.6 (no packed gain), fp32+fp16 1:1 mix 12.0
(separate fp16 pipe co-issues with fp32). shift+and = 2 int ops, and = 1, cvt ~1-1.7.

## ALU probe on troy (M5 Max), Gop/s (probe counts ops per unrolled step)

fp32 FMA 7.7 T; fp16 scalar FMA 12.5 T (1.6x); half2 FMA 13.0 T (no packed gain, 2 ops);
half4 14.0 T. fp32+half2 1:1 mix 7.6 T (no co-issue). shift+and = 1 op (bfe fusion; same
cost as and). and+cvt+fma per nibble ~ 2 slots. Slot model: fp32 1.0, fp16 0.6, int 1, cvt ~1.

## Noise on troy

WallpaperAerialsExtension + VTDecoderXPCService are always running (aerial wallpaper); GPU
timings swing 20-40% for seconds at a time. run1/run2 numbers are unreliable between variants;
bench_small.py now interleaves variants over rounds and reports best [median].

## run2 (noisy, best seen per shape, GB/s) — kernel = R,NSG,VPL,deq,PF,MC

in_proj: M=2 517 (2,2,16,magic16,1) M=3 517 M=4 509 M=8 392; mlx 544/540/478/179
mlp.down: M=2 515 M=3 517 M=4 505 M=8 366; mlx 523/461/392/209
out_proj: all variants 250-290 (mlx M=1 486, M=2 474) -> suspicious, recheck with interleaving

## run3 (interleaved best-of-5, GB/s; prep + main kernel; cfg R,NSG,VPL,deq,PF,MC)

| shape | M | mlx | 4,2,16,cvt,0 | 4,2,16,magic16,1 | 2,2,16,magic16,1 | 4,2,16,magic16,1,4 |
|---|---|---|---|---|---|---|
| in_proj (M=1 mlx 546) | 2 | 539 | 514 | 513 | 516 | 513 |
| | 3 | 534 | 503 | 511 | 515 | 511 |
| | 4 | 487 | 491 | 506 | 514 | 506 |
| | 8 | 180 | 389 | 399 | 394 | 411 |
| out_proj (M=1 mlx 483) | 2 | 476 | 430 | 427 | 431 | 432 |
| | 3 | 460 | 412 | 421 | 427 | 420 |
| | 4 | 414 | 404 | 412 | 422 | 413 |
| | 8 | 199 | 318 | 313 | 334 | 339 |
| mlp.down (M=1 mlx 546) | 2 | 526 | 518 | 517 | 522 | 516 |
| | 3 | 467 | 502 | 509 | 514 | 511 |
| | 4 | 398 | 491 | 504 | 511 | 504 |
| | 8 | 197 | 325 | 343 | 255 | 376 |

Sum over the 3 shapes (ms): mlx M=4 14.8, ours 12.7 (2,2,16,magic16,1); M=8 mlx 33.2, ours 16.6.
Errors: ours 1.0-1.2x of mlx's max abs error (bf16 output rounding dominates).

## prep kernel cost (prep_cost.py, cfg 2,2,16,magic16,1)

prep alone ~9.5 us (launch bound); in-stream it adds 4-5 us per call.
main kernel alone: in_proj M=2/4/8 547/546/383 GB/s; out_proj 471/471/350; mlp.down 546/530/249.
=> main kernel at M=4 is within 1.03x of mlx M=1 on all three shapes; the separate prep launch
   is what costs 5-10%. Plan: fuse prep into the x producers (rms_norm, swiglu, gates) in the model.

## tensor-op probe (step 3)

mx.fast.metal_kernel compiles `#include <metal_tensor>` + `<MetalPerformancePrimitives/...>`
and runs mpp::tensor_ops::matmul2d (half x half -> float, 32x32x32) correctly on troy (M5) and
locally (M4). Inputs must be cast to non-const `device half*`. troy's runtime header (macOS 26.6)
lists half x uint4b_format -> float among supported combos (int4 weights consumed directly).
