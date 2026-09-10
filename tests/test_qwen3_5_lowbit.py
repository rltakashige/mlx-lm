# Copyright © 2026 Apple Inc.

import contextlib
import os
import unittest
from unittest import mock

os.environ.setdefault("MLX_ENABLE_TF32", "0")

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from mlx_lm.models import fused_ops, qmv_small, qwen3_5, qwen3_5_mtp
from mlx_lm.models.cache import make_prompt_cache

# (mode, bits, group_size) of every format the small-M kernels handle
FORMATS = [("affine", 4, 64), ("affine", 3, 64), ("mxfp4", 4, 32), ("nvfp4", 4, 16)]
CONFIG = {
    "model_type": "qwen3_5",
    "hidden_size": 1024,
    "intermediate_size": 1024,
    "num_hidden_layers": 2,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "head_dim": 32,
    "vocab_size": 256,
    "linear_num_value_heads": 4,
    "linear_num_key_heads": 2,
    "linear_key_head_dim": 32,
    "linear_value_head_dim": 32,
    "linear_conv_kernel_dim": 4,
    "rms_norm_eps": 1e-5,
    "full_attention_interval": 2,
    "tie_word_embeddings": False,
    "max_position_embeddings": 512,
}


def _quantize(w, mode, bits, group):
    out = mx.quantize(w, group, bits, mode=mode)
    return out[0], out[1], out[2] if len(out) == 3 else None


def _ulp(amax):
    """The bf16 ulp at magnitude ``amax``."""
    return 2.0 ** (np.floor(np.log2(amax)) - 7)


def _model(mode, bits, group, seed=0):
    mx.random.seed(seed)
    args = qwen3_5.ModelArgs.from_dict({"model_type": "qwen3_5", "text_config": CONFIG})
    model = qwen3_5.Model(args)
    model.eval()
    model.set_dtype(mx.bfloat16)
    nn.quantize(model, group, bits, mode=mode)
    mx.eval(model.parameters())
    return model


@unittest.skipUnless(mx.metal.is_available(), "Metal only")
class TestLowBitKernels(unittest.TestCase):
    def test_kernels_match_quantized_matmul(self):
        """Every format's kernels are within 2 bf16 ulps (of the output magnitude) of the
        fp32 product and as close to it as mx.quantized_matmul; the tensor-op kernel on M5."""
        rows = [3, 4, 5, 8] + ([6, 12, 16, 20, 32] if qmv_small._m5() else [])
        for mode, bits, group in FORMATS:
            fmt = qmv_small._FORMATS[(mode, bits, group)]
            for m in rows:
                for n, k in ((256, 2048), (96, 1024), (32, 3072)):
                    mx.random.seed(m * k)
                    w = (mx.random.normal((n, k)) * 0.05).astype(mx.bfloat16)
                    x = (mx.random.normal((m, k)) * 1.5).astype(mx.bfloat16)
                    w_q, s, b = _quantize(w, mode, bits, group)
                    deq = mx.dequantize(w_q, s, b, group, bits, mode=mode).astype(mx.float32)
                    ref = x.astype(mx.float32) @ deq.T
                    mlx = mx.quantized_matmul(
                        x, w_q, s, b, transpose=True, group_size=group, bits=bits, mode=mode
                    )
                    nax = qmv_small._nax_m(m) and qmv_small._m5()
                    y = qmv_small._main(qmv_small.prep(x, fmt=fmt, nax=nax), w_q, s, b)
                    mx.eval(y, mlx, ref)
                    ulp = _ulp(mx.abs(ref).max().item())
                    err = mx.abs(y.astype(mx.float32) - ref).max().item()
                    err_mlx = mx.abs(mlx.astype(mx.float32) - ref).max().item()
                    tag = (mode, bits, m, n, k, nax)
                    self.assertLess(err, 2 * ulp, tag)
                    self.assertLess(err, 2 * err_mlx + 0.25 * ulp, tag)

    def test_model_routes_every_format(self):
        """The verify forwards through the kernels match the stock path per format."""
        prompt = mx.random.randint(0, CONFIG["vocab_size"], (1, 12))
        toks = mx.random.randint(0, CONFIG["vocab_size"], (1, 16))
        sizes = [1, 3, 4, 5, 6, 8, 16]
        for mode, bits, group in FORMATS:
            model = _model(mode, bits, group)
            outs = []
            for kernels in (True, False):
                with contextlib.ExitStack() as stack:
                    stack.enter_context(mock.patch.object(qmv_small, "_MIN_BYTES", 0))
                    if not kernels:
                        for module in (qmv_small, fused_ops):
                            stack.enter_context(mock.patch.object(module, "routes", return_value=False))
                    cache = make_prompt_cache(model)
                    mx.eval(model(prompt, cache=cache))
                    res = []
                    for s in sizes:
                        logits = model(toks[:, :s], cache=cache)
                        mx.eval(logits)
                        res.append(logits.astype(mx.float32))
                    outs.append(res)
            for s, a, b in zip(sizes, *outs):
                d = mx.abs(a - b).max().item()
                self.assertLess(d, 0.05 * mx.abs(b).max().item(), (mode, bits, s))
                # The argmax agrees wherever the stock path's top-2 margin exceeds the difference
                top2 = mx.sort(b, axis=-1)[..., -2:]
                clear = (top2[..., 1] - top2[..., 0]) > 2 * d
                same = mx.argmax(a, -1) == mx.argmax(b, -1)
                self.assertTrue(mx.all(same | ~clear).item(), (mode, bits, s))

    def test_routes_by_format(self):
        """Only the formats of the table route, with their own scale dtypes."""
        for mode, bits, group in FORMATS + [("affine", 4, 32), ("affine", 8, 64), ("mxfp8", 8, 32)]:
            lin = nn.Linear(2048, 256, bias=False)
            q = nn.QuantizedLinear.from_linear(lin, group, bits, mode=mode)
            q.update(
                {k: v.astype(mx.bfloat16) for k, v in q.parameters().items() if v.dtype == mx.float32}
            )
            expected = (mode, bits, group) in FORMATS and qmv_small._m5()
            with mock.patch.object(qmv_small, "_MIN_BYTES", 0):
                self.assertEqual(qmv_small.routes(q, (4, 2048), mx.bfloat16), expected, (mode, bits))
                self.assertEqual(qmv_small.routes(q, (8, 2048), mx.bfloat16), expected, (mode, bits))
                self.assertFalse(qmv_small.routes(q, (2, 2048), mx.bfloat16))
                self.assertFalse(qmv_small.routes(q, (33, 2048), mx.bfloat16))


class TestLowBitMTP(unittest.TestCase):
    def test_speculative_matches_greedy_per_format(self):
        """MTP drafts through a target and head quantized in every format, with the
        candidate-set head (no biases in the e2m1 formats), decode greedily."""
        import sys

        sys.path.insert(0, os.path.dirname(__file__))
        from test_qwen3_5_mtp import PROMPT, _greedy, _models, _speculative

        for mode, bits, group in FORMATS:
            target, head = _models()
            # Layers whose input width is not a multiple of the group stay fp32
            pred = lambda _, m: hasattr(m, "to_quantized") and m.weight.shape[-1] % group == 0
            nn.quantize(target, group, bits, mode=mode, class_predicate=pred)
            nn.quantize(head, group, bits, mode=mode, class_predicate=pred)
            head.bind(target)
            expected = _greedy(PROMPT, target, 32)
            self.assertGreater(len(set(expected)), 8, mode)
            for cand, stop in ((0, 0.0), (16, 0.5), (128, 0.0)):
                tokens, _ = _speculative(
                    PROMPT,
                    target,
                    head,
                    32,
                    num_draft_tokens=3,
                    draft_stop_prob=stop,
                    draft_candidates=cand,
                )
                self.assertEqual(tokens, expected, (mode, bits, cand, stop))
            # The candidate head over the quantized output head
            h = mx.random.normal((1, 1, target.args.text_config["hidden_size"]))
            module = head.candidates(0, size=4).head
            cand = qwen3_5_mtp.CandidateHead(module, 4, mx.array([5, 7, 100], mx.uint32))
            full = head.lm_head(h)
            out = cand(h)
            for j, i in enumerate(cand.ids.tolist()):
                self.assertTrue(mx.allclose(out[..., j], full[..., i], atol=1e-2), (mode, j))


if __name__ == "__main__":
    unittest.main()
