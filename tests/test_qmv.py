# Copyright © 2026 Apple Inc.

import unittest

import mlx.core as mx
import mlx.nn as nn

from mlx_lm.models import qmv


def _quantized(n, k, dtype, seed=0):
    mx.random.seed(seed)
    w = mx.random.normal((n, k), scale=0.05).astype(dtype)
    wq, scales, biases = mx.quantize(w, 64, 4)
    return wq, scales, biases


def _reference(x, wq, scales, biases, group_size=64, bits=4):
    return mx.quantized_matmul(
        x, wq, scales, biases, transpose=True, group_size=group_size, bits=bits
    )


def _cosine(a, b):
    a, b = a.astype(mx.float32).flatten(), b.astype(mx.float32).flatten()
    return (mx.sum(a * b) / (mx.linalg.norm(a) * mx.linalg.norm(b))).item()


@unittest.skipUnless(mx.metal.is_available(), "Metal is required")
class TestQmvSmallM(unittest.TestCase):
    def _check(self, out, ref):
        self.assertEqual(out.shape, ref.shape)
        self.assertEqual(out.dtype, ref.dtype)
        self.assertGreaterEqual(_cosine(out, ref), 0.9999)
        o, r = out.astype(mx.float32), ref.astype(mx.float32)
        rel = mx.abs(o - r) / (mx.abs(r) + 1e-2)
        self.assertLessEqual(rel.max().item(), 1e-2)

    def test_matches_quantized_matmul(self):
        for dtype in (mx.bfloat16, mx.float16):
            for n, k in ((64, 1024), (48, 512), (256, 1536)):
                wq, s, b = _quantized(n, k, dtype)
                for m in range(2, 9):
                    x = mx.random.normal((m, k)).astype(dtype)
                    ref = _reference(x, wq, s, b)
                    for splits in (1, 2, None):
                        if splits and k % (512 * splits):
                            continue
                        with self.subTest(dtype=dtype, n=n, k=k, m=m, splits=splits):
                            out = qmv.qmv_small_m(x, wq, s, b, splits=splits)
                            self._check(out, ref)

    def test_pick_splits(self):
        self.assertEqual(qmv.pick_splits(248320, 5120), 1)
        self.assertGreater(qmv.pick_splits(1024, 5120), 1)
        self.assertEqual(qmv.pick_splits(1024, 4864), 1)

    def test_fallbacks(self):
        x = mx.random.normal((3, 1024)).astype(mx.bfloat16)
        cases = []
        wq, s, b = _quantized(64, 1024, mx.bfloat16)
        cases.append((x[:1], wq, s, b, 64, 4))  # M = 1
        cases.append((mx.concatenate([x] * 3), wq, s, b, 64, 4))  # M = 9
        cases.append(
            (
                x.astype(mx.float32),
                wq,
                s.astype(mx.float32),
                b.astype(mx.float32),
                64,
                4,
            )
        )
        w = mx.random.normal((64, 1024)).astype(mx.bfloat16)
        wq8, s8, b8 = mx.quantize(w, 64, 8)
        cases.append((x, wq8, s8, b8, 64, 8))
        wq32, s32, b32 = mx.quantize(w, 32, 4)
        cases.append((x, wq32, s32, b32, 32, 4))
        wqk, sk, bk = mx.quantize(w[:, :768], 64, 4)
        cases.append((x[:, :768], wqk, sk, bk, 64, 4))  # K % 512 != 0
        wqn, sn, bn = mx.quantize(w[:60], 64, 4)
        cases.append((x, wqn, sn, bn, 64, 4))  # N % 8 != 0
        for xc, wc, sc, bc, g, bits in cases:
            out = qmv.qmv_small_m(xc, wc, sc, bc, group_size=g, bits=bits)
            ref = _reference(xc, wc, sc, bc, g, bits)
            self.assertTrue(mx.array_equal(out, ref))

    def test_qlinear(self):
        mod = nn.QuantizedLinear(1024, 2048, bias=False, group_size=64, bits=4)
        x = mx.random.normal((1, 3, 1024)).astype(mx.bfloat16)
        mod.scales = mod.scales.astype(mx.bfloat16)
        mod.biases = mod.biases.astype(mx.bfloat16)
        out = qmv.qlinear(mod, x)
        self.assertEqual(out.shape, (1, 3, 2048))
        self._check(out, mod(x))
        self.assertTrue(mx.array_equal(qmv.qlinear(mod, x[:, :1]), mod(x[:, :1])))
        mod_b = nn.QuantizedLinear(1024, 2048, bias=True, group_size=64, bits=4)
        self.assertTrue(mx.array_equal(qmv.qlinear(mod_b, x), mod_b(x)))
        mod_s = nn.QuantizedLinear(1024, 512, bias=False, group_size=64, bits=4)
        self.assertTrue(mx.array_equal(qmv.qlinear(mod_s, x), mod_s(x)))


@unittest.skipUnless(mx.metal.is_available(), "Metal is required")
class TestQargmax(unittest.TestCase):
    def test_random(self):
        agree, total = 0, 0
        for dtype in (mx.bfloat16, mx.float16):
            wq, s, b = _quantized(4096, 1024, dtype, seed=1)
            for m in range(1, 9):
                x = mx.random.normal((m, 1024)).astype(dtype)
                out = qmv.qargmax(x, wq, s, b)
                ref = mx.argmax(_reference(x, wq, s, b), axis=-1)
                self.assertEqual(out.dtype, mx.uint32)
                self.assertEqual(out.shape, (m,))
                agree += mx.sum(out == ref.astype(mx.uint32)).item()
                total += m
        # Near-ties can legitimately differ by accumulation order.
        rate = agree / total
        print(f"qargmax agreement {agree}/{total} ({rate:.3f})")
        self.assertGreaterEqual(rate, 0.98)

    def test_exact_ties_pick_lowest_index(self):
        wq, s, b = _quantized(2048, 512, mx.bfloat16, seed=2)
        x = mx.random.normal((4, 512)).astype(mx.bfloat16)
        best = mx.argmax(_reference(x, wq, s, b), axis=-1)
        rows = [5, 700, 1300, 2047]
        for i, r in enumerate(rows):
            src = best[i].item()
            wq[r] = wq[src]
            s[r] = s[src]
            b[r] = b[src]
        out = qmv.qargmax(x, wq, s, b)
        ref = mx.argmax(_reference(x, wq, s, b), axis=-1)
        for i, r in enumerate(rows):
            self.assertEqual(out[i].item(), min(r, best[i].item()))
        self.assertTrue(mx.array_equal(out, ref.astype(mx.uint32)))

    def test_fallback(self):
        x = mx.random.normal((3, 768)).astype(mx.bfloat16)
        wq, s, b = _quantized(64, 768, mx.bfloat16)
        out = qmv.qargmax(x, wq, s, b)
        ref = mx.argmax(_reference(x, wq, s, b), axis=-1)
        self.assertTrue(mx.array_equal(out, ref.astype(mx.uint32)))


if __name__ == "__main__":
    unittest.main()
