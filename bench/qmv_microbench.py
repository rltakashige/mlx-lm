# Copyright © 2026 Apple Inc.

"""Microbenchmark of the small-M 4-bit qmv kernels on the Qwen3.8-27B-4bit shapes."""

import argparse
import json
import subprocess
import time

import mlx.core as mx

from mlx_lm.models import qmv

# name: (K, N)
SHAPES = {
    "attn.q_proj": (5120, 16384),
    "attn.k_proj": (5120, 1024),
    "attn.o_proj": (8192, 5120),
    "gdn.in_proj_qkv": (5120, 10240),
    "gdn.in_proj_z": (5120, 6144),
    "gdn.in_proj_fused": (5120, 16480),
    "gdn.out_proj": (6144, 5120),
    "mlp.gate_proj": (5120, 17408),
    "mlp.gate_up_fused": (5120, 34816),
    "mlp.down_proj": (17408, 5120),
    "lm_head": (5120, 248320),
}
MS = (1, 2, 3, 4, 8)


def chip_name():
    try:
        return subprocess.check_output(
            ["sysctl", "-n", "machdep.cpu.brand_string"], text=True
        ).strip()
    except Exception:
        return mx.device_info().get("device_name", "unknown")


def timeit(fn, inner, reps):
    for _ in range(3):
        mx.eval(fn())
    mx.synchronize()
    best = float("inf")
    for _ in range(reps):
        tic = time.perf_counter()
        outs = [fn() for _ in range(inner)]
        mx.eval(outs)
        mx.synchronize()
        best = min(best, (time.perf_counter() - tic) / inner)
    return best


def warm_up():
    a = mx.random.normal((4096, 4096))
    for _ in range(20):
        a = a @ a * 1e-3
    mx.eval(a)
    mx.synchronize()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--json", type=str, default=None, help="Write results to this file"
    )
    parser.add_argument("--inner", type=int, default=10)
    parser.add_argument("--reps", type=int, default=7)
    parser.add_argument(
        "--shapes", type=str, default=None, help="Comma separated subset of shape names"
    )
    parser.add_argument(
        "--dtype", type=str, default="bfloat16", choices=["bfloat16", "float16"]
    )
    args = parser.parse_args()
    dtype = getattr(mx, args.dtype)
    shapes = (
        SHAPES
        if args.shapes is None
        else {k: SHAPES[k] for k in args.shapes.split(",")}
    )

    print(f"chip: {chip_name()}  mlx: {mx.__version__}  dtype: {args.dtype}")
    warm_up()
    rows = []
    header = f"{'shape':20s} {'K':>6s} {'N':>7s} {'M':>2s} | {'mlx':>13s} | {'qmv s=1':>13s} | {'qmv auto':>16s} | winner"
    print(header)
    print("-" * len(header))
    for name, (k, n) in shapes.items():
        mx.random.seed(0)
        w = mx.random.normal((n, k), scale=0.05).astype(dtype)
        wq, scales, biases = mx.quantize(w, 64, 4)
        mx.eval(wq, scales, biases)
        del w
        nbytes = n * k // 2 + 2 * n * (k // 64) * 2
        for m in MS:
            x = mx.random.normal((m, k)).astype(dtype)
            mx.eval(x)
            res = {}
            res["mlx"] = timeit(
                lambda: mx.quantized_matmul(
                    x, wq, scales, biases, transpose=True, group_size=64, bits=4
                ),
                args.inner,
                args.reps,
            )
            if m >= 2:
                res["qmv_s1"] = timeit(
                    lambda: qmv.qmv_small_m(x, wq, scales, biases, splits=1),
                    args.inner,
                    args.reps,
                )
                res["qmv_auto"] = timeit(
                    lambda: qmv.qmv_small_m(x, wq, scales, biases),
                    args.inner,
                    args.reps,
                )
            if name == "lm_head":
                res["mlx_argmax"] = timeit(
                    lambda: mx.argmax(
                        mx.quantized_matmul(
                            x, wq, scales, biases, transpose=True, group_size=64, bits=4
                        ),
                        axis=-1,
                    ),
                    args.inner,
                    args.reps,
                )
                if m >= 2:
                    res["qmv_argmax"] = timeit(
                        lambda: mx.argmax(
                            qmv.qmv_small_m(x, wq, scales, biases), axis=-1
                        ),
                        args.inner,
                        args.reps,
                    )
                res["qargmax"] = timeit(
                    lambda: qmv.qargmax(x, wq, scales, biases), args.inner, args.reps
                )
            gbs = {kk: nbytes / v / 1e9 for kk, v in res.items()}
            matmul_keys = [kk for kk in ("mlx", "qmv_s1", "qmv_auto") if kk in res]
            winner = min(matmul_keys, key=res.get)
            splits = qmv.pick_splits(n, k)

            def cell(key):
                return (
                    f"{res[key] * 1e6:7.1f}us {gbs[key]:4.0f}"
                    if key in res
                    else " " * 13
                )

            auto_cell = (
                f"{cell('qmv_auto')} s={splits:<2d}" if "qmv_auto" in res else " " * 17
            )
            line = f"{name:20s} {k:6d} {n:7d} {m:2d} | {cell('mlx')} | {cell('qmv_s1')} | {auto_cell} | {winner}"
            if name == "lm_head":
                amk = [
                    kk for kk in ("mlx_argmax", "qmv_argmax", "qargmax") if kk in res
                ]
                aw = min(amk, key=res.get)
                line += (
                    "  argmax: "
                    + " ".join(f"{kk}={res[kk] * 1e6:.1f}us" for kk in amk)
                    + f" -> {aw}"
                )
            print(line, flush=True)
            rows.append(
                {
                    "shape": name,
                    "K": k,
                    "N": n,
                    "M": m,
                    "bytes": nbytes,
                    "splits": splits,
                    "us": {kk: v * 1e6 for kk, v in res.items()},
                    "GBps": gbs,
                    "winner": winner,
                }
            )
    if args.json:
        with open(args.json, "w") as f:
            json.dump(
                {
                    "chip": chip_name(),
                    "mlx": mx.__version__,
                    "dtype": args.dtype,
                    "rows": rows,
                },
                f,
                indent=1,
            )
        print(f"wrote {args.json}")


if __name__ == "__main__":
    main()
