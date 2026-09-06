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

## tensor-op qmv (scratch k/tensor_qmv.py)

matmul2d(half x uint4b_format -> float, 16 x 32 x 64 per group) compiles through mx.fast.metal_kernel
and reads MLX's packed 4-bit weight rows directly (tensor<device metal::uint4b_format> over the
uint32 buffer, transpose_right=true). Per-group scale/bias applied on the cooperative tensor
elements (is_valid_element / get_multidimensional_index). Local M4 check: err == mlx err.
API notes: get_mask does not exist (use is_valid_element); `#pragma unroll full` is not accepted
(use `#pragma clang loop unroll(full)`); get_capacity() is not constexpr.

## integration (commit d848f02)

mlx_lm/models/qmv_small.py (prep + main kernels, qlinear wrapper); qwen3_5.py routes qkv/o/in/out/
gate_up/down/lm_head through qlinear when 2 <= M <= 8. Local tests: 10 passed, 1 skipped.

## verify S-curve (NO_CAPTURE=1 verify_capture.py, 27B, 512-token prompt), median ms

| S | before (leo/qwen-mtp-opt) | after d848f02 (standalone prep, M>=2) |
|---|---|---|
| 1 | 31.01 | 31.13 |
| 2 | 31.64 | 33.40 |
| 3 | 33.95 | 34.55 |
| 4 | 36.47 | 35.33 |
| 6 | 53.14 | 41.76 |
| 8 | 62.70 | 46.24 |

S=2 regresses (mlx qmv_wide is already bandwidth bound at M=2; the extra prep launch costs
~5 us x 257 matmuls): route only M >= 3. Next: fuse the prep into rms_norm / swiglu.

## tensor-op path timing on troy (M padded to 16, per-group epilogue), GB/s

| shape | NT=32 | NT=64 | NT=128 | mlx M=1 | mlx M=4 | mlx M=8 |
|---|---|---|---|---|---|---|
| out_proj | 161 | 162 | 308 | 477 | 409 | 195 |
| in_proj | 152 | - | - | 548 | 487 | 180 |
| lm_head | 106 | 116 | 404-411 | 585 | 259 | 105 |

lm_head M=16-padded at NT=128: 1.75 ms vs qmv_fast M=1 1.22 ms, mlx M=4 2.77 ms, M=8 6.84 ms.

## final sweep, all 7 shapes (interleaved best-of-5, GB/s), kernel incl. standalone prep launch

cfg A = 2,2,16,magic16,1 (used for M <= 4), cfg B = 4,2,16,magic16,1,4 (used for M >= 5)

| shape | mlx M=1 | M=2 mlx/A | M=3 mlx/A | M=4 mlx/A | M=6 mlx/B | M=8 mlx/B |
|---|---|---|---|---|---|---|
| gdn.in_proj | 549 | 535/519 | 534/517 | 488/513 | 256/451 | 181/410 |
| gdn.out_proj | 487 | 479/430 | 467/429 | 419/421 | 231/350 | 201/339 |
| attn.qkv_proj | 543 | 530/512 | 528/503 | 482/501 | 264/444 | 186/401 |
| attn.o_proj | 492 | 495/434 | 473/432 | 421/427 | 229/351 | 200/339 |
| mlp.gate_up | 566 | 565/556 | 555/556 | 489/553 | 224/524 | 147/433 |
| mlp.down | 546 | 524/524 | 482/517 | 403/508 | 244/395 | 203/384 |
| lm_head | 581 | 518/578 | 429/575 | 256/566 | 151/531 | 99/432 |

Sum over model (ms): mlx M=1 26.1, M=2 26.7, M=3 27.8, M=4 32.6, M=6 62.9, M=8 88.5
ours (A for M<=4, B for M>=6): M=2 27.2, M=3 27.3, M=4 27.6, M=6 31.5, M=8 35.4
The "/main" (prep excluded) columns of this run are invalid: one shared prepared x for all
calls made the main kernel slow (see rerun with rotating prepared copies).

## sweep_v4: main kernel only (8 rotating prepared x copies) vs mlx, GB/s

| shape | mlx M=1 | M=3 main | M=4 main | M=6 main (R=4) | M=8 main (R=4) |
|---|---|---|---|---|---|
| gdn.in_proj | 548 | 544 | 537 | 512 | 406 |
| gdn.out_proj | 482 | 463 | 460 | 412 | 364 |
| mlp.gate_up | 571 | 569 | 567 | 538 | 427 |
| mlp.down | 546 | 539 | 532 | 429 | 415 |
| lm_head | 583 | 576 | 573 | 531 | 423 |

M=4 within 1.05x of M=1 on all shapes once the prep launch is fused away; M=8 at 1.32-1.38x.
half4 FMA variant (V4): no gain (equal at M=4, 3-8% worse at M=8) -> dropped.

## S-curve after2 (0cb8c69: fused prep v1, per-thread scalar loops, 256 threads/row)

S=1 31.25, S=2 31.91, S=3 37.14, S=4 37.72, S=6 43.99, S=8 48.27 -> worse than the standalone
prep (S=4 35.33). The fused prep kernels were slow (M threadgroups, scalar bf16 loads, 2-3
passes). Rewrote them: 16-byte loads, values kept in registers, K/16 threads per row, one pass
(commit "Vectorize the prep kernels ..."). Local fp32-reference check: fused paths are more
accurate than mlx's bf16 intermediates (rms_norm 0.0207 vs 0.0248, swiglu 0.0222 vs 0.0245).

## S-curve after3 (ad4beee: vectorized fused preps, gate + gated-norm preps, M >= 3), median ms

| S | before | after3 |
|---|---|---|
| 1 | 31.01 | 31.91 (min 30.99; M=1 untouched, noise) |
| 2 | 31.64 | 32.57 (min 31.90; M=2 routed to mlx) |
| 3 | 33.95 | 33.26 |
| 4 | 36.47 | 34.12 |
| 6 | 53.14 | 40.49 |
| 8 | 62.70 | 44.86 |

S=4 verify forward 34.1 ms < 36.3 ms target. Non-matmul time at S=4 is ~7.5 ms vs ~4.9 ms at
S=1 (main kernels sum to ~26.6 ms): the prep launches still cost ~1-2 ms; measuring in-stream
prep cost with chain_bench.py.

## final_sweep2 (venv-small-m, ad4beee module): mlx vs integrated (standalone vectorized prep,
## 1 TG per row) vs main kernel only, GB/s, best of 5 interleaved rounds

| shape | M | mlx | integrated | main only (cfg used) |
|---|---|---|---|---|
| gdn.in_proj (M=1 548) | 3 | 526 | 512 | 541 |
| | 4 | 481 | 507 | 541 |
| | 6 | 254 | 465 | 513 |
| | 8 | 180 | 406 | 411 |
| gdn.out_proj (M=1 487) | 3 | 459 | 418 | 469 |
| | 4 | 412 | 412 | 453 |
| | 6 | 211 | 363 | 397 |
| | 8 | 195 | 333 | 321 |
| attn.qkv_proj (M=1 541) | 3 | 518 | 497 | 533 |
| | 4 | 482 | 499 | 532 |
| | 6 | 261 | 452 | 501 |
| | 8 | 189 | 399 | 413 |
| attn.o_proj (M=1 484) | 3 | 463 | 419 | 461 |
| | 4 | 422 | 415 | 462 |
| | 6 | 230 | 370 | 406 |
| | 8 | 201 | 336 | 361 |
| mlp.gate_up (M=1 572) | 3 | 553 | 551 | 559 |
| | 4 | 489 | 549 | 567 |
| | 6 | 226 | 530 | 535 |
| | 8 | 150 | 417 | 427 |
| mlp.down (M=1 550) | 3 | 472 | 512 | 532 |
| | 4 | 389 | 505 | 530 |
| | 6 | 244 | 416 | 416 |
| | 8 | 209 | 370 | 406 |
| lm_head (M=1 584) | 3 | 434 | 575 | 578 |
| | 4 | 260 | 569 | 573 |
| | 6 | 153 | 556 | 533 |
| | 8 | 103 | 431 | 430 |

Sum over model (ms): mlx M=1 26.0 M=2 26.7 M=3 28.1 M=4 33.0 M=6 62.9 M=8 87.0;
integrated M=3 27.6 M=4 27.9 M=6 30.6 M=8 36.4; main only M=3 26.6 M=4 26.6 M=6 29.6 M=8 35.2.

## chain_bench (ad4beee): in-stream cost of the fused preps

prep_rms_norm 3-7 us, prep_swiglu 7-13 us, prep_gated_norm/gate ~5 us; mlx's own norm/swiglu
in the chain cost 3-4 us. Prep kernels ran M threadgroups only. Rewrote once more: 64-thread
threadgroups per 1024 values, fixed x scale 1/256 (no row max pass; rms_norm threadgroups reduce
the whole row redundantly).

## S-curve after4 (f67e8fd: 64-thread prep segments, fixed 1/256 scale), median ms

S=1 31.47, S=2 32.62, S=3 33.11, S=4 33.28, S=6 40.03, S=8 44.46 (before: 31.0/31.6/34.0/36.5/53.1/62.7)

chain_bench2: in-stream prep cost unchanged (4-6 us; down M=8 14 us): the cost is the dependent
kernel launch bubble (mlx's own rms_norm adds 3-4 us in the same chain), not the prep math.
final_sweep3 (integrated, f67e8fd): sum M=3 27.4, M=4 27.7, M=6 30.8, M=8 36.1 ms.
Accuracy with the fixed 1/256 scale on tiny activations (max|x| 0.04): 5x mlx's error (fp16
subnormals) -> back to a per-row power-of-two scale, computed redundantly by every segment
threadgroup from a full-row scan (no extra launch).

## per-row scale restored (commit "Restore the per-row fp16 scale ..."), local checks

Robustness (5120x6144, M=4, max abs err ours / mlx): unit 0.0252/0.0202, tiny (max|x| 0.04)
0.000197/0.000172, x100 2.01/2.01, huge (2e4 entries) 54.6/48.5; all finite. Fused paths vs a
full-fp32 reference: rms_norm 0.0207 (mlx 0.0248), swiglu 0.0222 (0.0245), gate 0.0119 (0.0142),
gated_norm 0.0116 (0.0168). Tests: 10 passed, 1 skipped. Queued: S-curve after5, final_sweep4.

## S-curve after5 (80fe538: per-row scale via redundant row scan), median ms

S=1 31.50, S=2 32.70, S=3 34.62, S=4 35.36, S=6 41.69, S=8 45.86 -> ~2 ms slower than after4
(33.1/33.3/40.0/44.5) although chain_bench3 shows the prep only +0.5-2 us per call. Queued A/B
reruns (after5, after4 via PYTHONPATH to an alt checkout, after5 again) to separate noise.
final_sweep4 (integrated, 80fe538): sum M=3 27.4, M=4 27.7, M=6 30.9, M=8 36.4 ms (same as before).

## A/B S-curve reruns (median ms, sizes 1/3/4/8)

after5b (row scan, 80fe538): 31.86 / 34.48 / 35.18 / 46.02
after4b (fixed 1/256, f67e8fd): 31.29 / 33.06 / 33.74 / 44.42
after5c (row scan again): 31.30 / 34.52 / 35.20 / 45.94
=> the redundant row scan costs ~1.4 ms at S=3/4 in the model (dependent chain: the scan loop's
serialized load latency, 5-17 iterations per thread). Fix: scan with 256/512 threads per
threadgroup and an unrolled compile-time loop; only the first 64 threads convert.

## S-curve after6 (2aaee95: 256/512-thread row scan), median ms

S=1 31.32, S=2 32.08, S=3 33.59, S=4 34.85, S=6 41.16, S=8 45.57 -> recovers 0.3-0.5 ms of the
scan cost, still ~1 ms behind the fixed scale at S=4. chain_bench4: swiglu prep at M=8 25 us
(exp-heavy scan). Next: scans without transcendentals, using bounds: swiglu max|g|*max|u|,
gate max|x|, gated_norm sqrt(D)*max|w|*max|z|; rms_norm keeps the exact max|x*w| (its loop
already exists for the RMS).

## final_sweep5 (integrated, 2aaee95 wide-scan prep), sums (ms)

mlx M=1 26.2 M=2 26.5 M=3 28.2 M=4 32.5 M=6 63.2 M=8 86.9; integrated M=3 27.6 M=4 27.9 M=6 31.3 M=8 36.8.
Commit 2c8b5a2 (bound scans): accuracy unchanged (rms_norm 0.0207 vs mlx 0.0248, tiny rows
0.000197 vs 0.000172, huge 54.6 vs 48.5, all finite). Queued S-curve after7, final_sweep6.
