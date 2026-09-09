# Copyright © 2026 Apple Inc.

import json
import os
import tempfile
import unittest

# The tiny fp32 models need exact GEMMs: M5 uses tf32 for float32 matmuls by default
os.environ.setdefault("MLX_ENABLE_TF32", "0")
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx.utils import tree_flatten

from mlx_lm.models import qwen4_exp
from mlx_lm.models.cache import make_prompt_cache, trim_prompt_cache
from mlx_lm.utils import load_model

TEXT_CONFIG = {
    "hidden_size": 64,
    "num_hidden_layers": 4,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "head_dim": 32,
    "vocab_size": 1000,
    "rms_norm_eps": 1e-6,
    "full_attention_interval": 4,
    "num_experts": 8,
    "num_experts_per_tok": 2,
    "moe_intermediate_size": 32,
    "shared_expert_intermediate_size": 32,
    "linear_num_key_heads": 2,
    "linear_num_value_heads": 6,
    "linear_key_head_dim": 32,
    "linear_value_head_dim": 32,
    "linear_conv_kernel_dim": 4,
    "hc_count": 4,
    "hc_lowrank": 16,
    "indexer_n_heads": 2,
    "indexer_kv_heads": 1,
    "indexer_head_dim": 16,
    "indexer_budget": 8,
    "indexer_compress_ratio": 4,
    "ngram_size": 3,
    "heads_per_ngram": 2,
    "ngram_vocab_size_base": 101,
    "split_ngram_parts": 4,
    "ple_embed_dim": 64,
    "ple_layer_ids": [2],
    "eos_token_id": 1,
    "rope_parameters": {"rope_theta": 10000000, "partial_rotary_factor": 0.25},
}
_MASK64 = (1 << 64) - 1


def reference_ngram_ids(history, eos, mults, sizes, offsets, ngram_size, heads_per_ngram):
    """The HF Qwen4ExpTextNGramEmbedding hash of one token history in Python ints."""
    T = len(history)

    def shift(s):
        out, last_eos = [], -1
        for t in range(T):
            ok = s == 0 or (t - (last_eos + 1) >= s and t - s >= 0)
            out.append(history[t - s] if ok else eos)
            if history[t] == eos:
                last_eos = t
        return out

    shifted = [shift(s) for s in range(ngram_size)]
    ids = [[] for _ in range(T)]
    for ngram in range(2, ngram_size + 1):
        lo = (ngram - 2) * heads_per_ngram
        for t in range(T):
            mixed = (shifted[0][t] * mults[0]) & _MASK64
            for p in range(1, ngram):
                mixed ^= (shifted[p][t] * mults[p]) & _MASK64
            # torch multiplies in int64 and takes a Python-style remainder
            if mixed >= 1 << 63:
                mixed -= 1 << 64
            for h in range(lo, lo + heads_per_ngram):
                ids[t].append(mixed % sizes[h] + offsets[h])
    return ids


def tiny_model(seed=0):
    mx.random.seed(seed)
    args = qwen4_exp.ModelArgs(model_type="qwen4_exp", text_config=dict(TEXT_CONFIG))
    model = qwen4_exp.Model(args)
    emb = model.layers[1].ple.ple_embedding
    emb._resident = mx.random.normal((emb.hasher.rows, emb.width)) * 0.1
    return model


class TestNGramHash(unittest.TestCase):
    def test_multipliers_and_primes_match_the_checkpoint(self):
        # The buffers stored in mlx-community/Qwen3.8-Flash-Next-4bit
        self.assertEqual(
            qwen4_exp.layer_multipliers(248320, 3, 0, 1234),
            [23703573157769, 20109073645365, 8052911324071],
        )
        hasher = qwen4_exp.NGramHasher(qwen4_exp.TextArgs(), 0)
        self.assertEqual(hasher.sizes[:4].tolist(), [20000003, 20000023, 20000033, 20000047])
        self.assertEqual(hasher.offsets[:3].tolist(), [0, 20000003, 40000026])
        self.assertEqual(hasher.rows, 320001536)

    def test_hash_matches_the_reference_formula(self):
        for vocab, base, seed in [(1000, 101, 7), (248320, 20_000_000, 1234)]:
            args = qwen4_exp.TextArgs(vocab_size=vocab, ngram_vocab_size_base=base, seed=seed, eos_token_id=1)
            hasher = qwen4_exp.NGramHasher(args, 0)
            mults = qwen4_exp.layer_multipliers(vocab, 3, 0, seed)
            rng = np.random.default_rng(seed)
            history = rng.integers(0, vocab, size=(3, 40))
            history[:, :2] = 1
            history[0, 10] = history[0, 11] = 1
            history[1, 20] = 1
            history[2, 39] = 1
            got = hasher(history)
            for b in range(3):
                ref = reference_ngram_ids(
                    history[b].tolist(), 1, mults, hasher.sizes.tolist(), hasher.offsets.tolist(), 3, 8
                )
                self.assertEqual(got[b].tolist(), ref)

    def test_host_dequantization(self):
        x = mx.random.normal((7, 160)).astype(mx.bfloat16)
        w, s, b = mx.quantize(x, group_size=32, bits=4)
        ref = np.array(mx.dequantize(w, s, b, group_size=32, bits=4).astype(mx.float32))
        u16 = lambda a: np.array(a.view(mx.uint16))
        rows = qwen4_exp.dequantize_rows(np.array(w), u16(s), u16(b))
        self.assertTrue(np.allclose(rows, ref, atol=1e-2))


class TestHyperConnection(unittest.TestCase):
    def test_gated_residual_matches_the_reference(self):
        args = qwen4_exp.TextArgs(hidden_size=8, hc_count=4, hc_lowrank=6, rms_norm_eps=1e-6)
        block = qwen4_exp.GatedResidual(args)
        rng = np.random.default_rng(0)
        params = {k: mx.array(rng.standard_normal(v.shape).astype(np.float32)) for k, v in tree_flatten(block.parameters())}
        block.load_weights(list(params.items()))
        h = rng.standard_normal((2, 3, 32)).astype(np.float32)
        x = rng.standard_normal((2, 3, 8)).astype(np.float32)
        mixed, inject = block(mx.array(h))
        out = block.combine(mx.array(h), mx.array(x), inject)

        w = {k: np.array(v) for k, v in params.items()}
        g = h.reshape(2, 3, 4, 8)
        normed = (g / np.sqrt((g**2).mean(-1, keepdims=True) + 1e-6)).reshape(2, 3, 32) * (1 + w["hc_norm.weight"])
        # The last hc rows of the down projection are the reference's block_inject_weight
        w_down, w_inject = w["input_mix_weight_down.weight"][:6], w["input_mix_weight_down.weight"][6:]
        d = normed @ w_down.T / 4
        d = d / (1 + np.exp(-d))
        mix = 1 / (1 + np.exp(-(d @ w["input_mix_weight_up.weight"].T)))
        ref_mixed = (mix.reshape(2, 3, 4, 8) * normed.reshape(2, 3, 4, 8)).mean(-2)
        ref_inject = 2 / (1 + np.exp(-(normed @ w_inject.T) / 4))
        ref_out = h + (x[:, :, None, :] * ref_inject[..., None]).reshape(2, 3, 32)
        self.assertTrue(np.allclose(np.array(mixed), ref_mixed, atol=1e-5))
        self.assertTrue(np.allclose(np.array(inject), ref_inject, atol=1e-5))
        self.assertTrue(np.allclose(np.array(out), ref_out, atol=1e-5))


class TestTinyModel(unittest.TestCase):
    def test_generation_is_deterministic_and_consistent(self):
        model = tiny_model()
        inputs = mx.array([list(range(2, 24))])
        prefill = model(inputs)
        self.assertEqual(prefill.shape, (1, 22, 1000))
        self.assertTrue(mx.allclose(prefill, tiny_model()(inputs)))

        # 22 tokens exceed the budget of 8: the sparse path stays causal and
        # matches a token by token decode
        other = mx.concatenate([inputs[:, :-1], mx.array([[997]])], axis=-1)
        self.assertTrue(mx.allclose(prefill[:, :-1], model(other)[:, :-1]))
        cache = make_prompt_cache(model)
        steps = [model(inputs[:, i : i + 1], cache=cache) for i in range(22)]
        self.assertTrue(mx.allclose(prefill, mx.concatenate(steps, axis=1), atol=1e-4))
        # ...and a chunked prefill
        cache = make_prompt_cache(model)
        chunks = [model(inputs[:, :9], cache=cache), model(inputs[:, 9:], cache=cache)]
        self.assertTrue(mx.allclose(prefill, mx.concatenate(chunks, axis=1), atol=1e-4))

        logits, hidden = model(inputs, return_hidden=True)
        self.assertEqual(hidden.shape, (1, 22, 4 * 64))
        self.assertTrue(mx.allclose(logits, prefill))

        # EOS tokens (id 1) reset the n-gram context: the cached decode must agree
        with_eos = mx.array([[3, 8, 12, 5, 9, 1, 1, 30, 31, 32, 7, 1, 40, 41, 42, 43]])
        prefill = model(with_eos)
        cache = make_prompt_cache(model)
        steps = [model(with_eos[:, i : i + 1], cache=cache) for i in range(16)]
        self.assertTrue(mx.allclose(prefill, mx.concatenate(steps, axis=1), atol=1e-4))

    def test_mtp_head_reproduces_greedy_decoding(self):
        from mlx_lm.generate import generate_step, speculative_generate_step
        from mlx_lm.models import qwen4_exp_mtp

        model = tiny_model()
        mx.random.seed(1)
        draft = qwen4_exp_mtp.Model(
            qwen4_exp_mtp.ModelArgs(model_type="qwen4_exp_mtp", text_config=dict(TEXT_CONFIG))
        )
        prompt = mx.array([3, 17, 42, 7, 99, 5, 61, 8, 23, 44, 12, 13])
        greedy = [t for t, _ in generate_step(prompt, model, max_tokens=24)]
        # A random head drafts wrong tokens: every cycle rolls the caches back
        for k in (1, 3):
            out = speculative_generate_step(
                prompt, model, draft, max_tokens=24, num_draft_tokens=k, draft_stop_prob=0.0, draft_candidates=0
            )
            self.assertEqual([t for t, _, _ in out], greedy)

    def test_trim_rolls_the_states_back(self):
        model = tiny_model()
        inputs = mx.array([list(range(2, 16))])
        short = make_prompt_cache(model)
        model(inputs[:, :9], cache=short)
        expected = model(inputs[:, 9:10], cache=short)

        cache = make_prompt_cache(model)
        model(inputs[:, :6], cache=cache)
        for c in cache:
            if hasattr(c, "keep_states"):
                c.keep_states = True
        model(inputs[:, 6:14], cache=cache)
        self.assertEqual(trim_prompt_cache(cache, 5), 5)
        self.assertTrue(mx.allclose(model(inputs[:, 9:10], cache=cache), expected, atol=1e-4))

    def test_memory_mapped_table_matches_the_resident_one(self):
        mx.random.seed(0)
        table = mx.random.normal((512, 32)).astype(mx.bfloat16)
        rows = table.shape[0] // 4
        weights = {}
        prefix = "language_model.model.layers.1.ple.ple_embedding.ngram_embedding.shards"
        for i in range(4):
            w, s, b = mx.quantize(table[i * rows : (i + 1) * rows], group_size=32, bits=4)
            weights.update({f"{prefix}.{i}.weight": w, f"{prefix}.{i}.scales": s, f"{prefix}.{i}.biases": b})
        with tempfile.TemporaryDirectory() as d:
            files = {"a": [f"{prefix}.0", f"{prefix}.1"], "b": [f"{prefix}.2", f"{prefix}.3"]}
            weight_map = {}
            for name, prefixes in files.items():
                part = {k: v for k, v in weights.items() if k.rsplit(".", 1)[0] in prefixes}
                mx.save_safetensors(str(Path(d) / f"model-{name}.safetensors"), part)
                weight_map.update({k: f"model-{name}.safetensors" for k in part})
            with open(Path(d) / "model.safetensors.index.json", "w") as f:
                json.dump({"weight_map": weight_map}, f)
            disk = qwen4_exp.NGramTable(d, "language_model.model.layers.1")
            self.assertEqual(disk.rows_total, table.shape[0])
            ids = np.array([0, 5, rows - 1, rows, 2 * rows + 3, table.shape[0] - 1, 5])
            got = disk(ids)
            ref = np.array(mx.dequantize(*mx.quantize(table, group_size=32, bits=4), group_size=32, bits=4)[mx.array(ids)].astype(mx.float32))
            self.assertTrue(np.allclose(got, ref, atol=1e-2))
            self.assertEqual(disk.stats()["hits"], 1)


if __name__ == "__main__":
    unittest.main()


class TestHyperConnectionKernels(unittest.TestCase):
    """hc_small against the compiled ops of a quantized bf16 site (bf16 rounding noise)."""

    def _site(self, inject, group_size, seed=0):
        args = qwen4_exp.TextArgs(hidden_size=512, hc_count=4, hc_lowrank=64, rms_norm_eps=1e-6)
        block = qwen4_exp.GatedResidual(args, use_combine=inject)
        mx.random.seed(seed)
        params = {k: mx.random.normal(v.shape) * 0.2 for k, v in tree_flatten(block.parameters())}
        block.load_weights(list(params.items()))
        nn.quantize(block, group_size=group_size, bits=4)
        block.set_dtype(mx.bfloat16)
        return block

    @staticmethod
    def _reference(block, h):
        """The site in float32 from the dequantized weights, rounded to bf16 where the ops round."""
        f = lambda a: a.astype(mx.bfloat16).astype(mx.float32)
        hc, dims, lr = block.hc, block.dims, block.lowrank
        down, up = block.input_mix_weight_down, block.input_mix_weight_up
        deq = lambda m: mx.dequantize(m.weight, m.scales, m.biases, group_size=m.group_size, bits=m.bits).astype(mx.float32)
        g = h.astype(mx.float32).reshape(1, -1, hc, dims)
        normed = f(g * mx.rsqrt((g * g).mean(-1, keepdims=True) + block.hc_norm.eps))
        normed = f(normed.reshape(1, -1, hc * dims) * block.hc_norm.gain().astype(mx.float32))
        d = f(normed @ deq(down).T) * (1 / hc)
        gate = f(d[..., :lr] * mx.sigmoid(d[..., :lr]))
        w = f(mx.sigmoid(f(gate @ deq(up).T))).reshape(1, -1, hc, dims)
        mixed = f(f(w * normed.reshape(1, -1, hc, dims)).mean(-2))
        inject = 2 * f(mx.sigmoid(d[..., lr:])) if block.inject else None
        return mixed, inject

    def _check(self, block, rows, pending, group_size):
        from mlx_lm.models import hc_small

        mx.random.seed(rows)
        h = mx.random.normal((1, rows, 4 * 512)).astype(mx.bfloat16)
        self.assertTrue(hc_small.routes(block, h))
        if pending:
            x = mx.random.normal((1, rows, 512)).astype(mx.bfloat16)
            inject = (2 * mx.sigmoid(mx.random.normal((1, rows, 4)))).astype(mx.bfloat16)
            pending = (x, inject)
            ref_h = block.combine(h, x, inject)
        else:
            pending, ref_h = None, h
        mixed, inject_out, combined = hc_small.mix(block, h, pending)
        self.assertTrue(mx.array_equal(combined, ref_h))
        ops = block(ref_h)
        ops = ops if block.inject else (ops, None)
        for got, op, want in zip((mixed, inject_out), ops, self._reference(block, ref_h)):
            if want is None:
                self.assertIsNone(got)
                continue
            self.assertEqual(got.shape, want.shape)
            err = (got.astype(mx.float32) - want).abs()
            err_ops = (op.astype(mx.float32) - want).abs()
            # bf16 rounding noise: no worse than the ops path, at most two ulps
            self.assertLessEqual(err.mean().item(), 1.5 * err_ops.mean().item() + 1e-4)
            self.assertLess((err / (1 + want.abs())).max().item(), 2e-2)
        if rows > 1:
            # The kernels give the same rows at every M
            one = hc_small.mix(block, h[:, :1], None if pending is None else (pending[0][:, :1], pending[1][:, :1]))
            self.assertTrue(mx.array_equal(one[0], mixed[:, :1]))

    def test_sites_match_the_compiled_ops(self):
        for group_size in (32, 64):
            block = self._site(True, group_size)
            for rows in (1, 3, 4):
                self._check(block, rows, True, group_size)
            self._check(block, 2, False, group_size)
            self._check(self._site(False, group_size), 1, True, group_size)

    def test_unsupported_shapes_use_the_ops(self):
        from mlx_lm.models import hc_small

        block = self._site(True, 32)
        h = mx.random.normal((1, 5, 4 * 512)).astype(mx.bfloat16)
        self.assertFalse(hc_small.routes(block, h))
        self.assertFalse(hc_small.routes(block, h[:, :1].astype(mx.float32)))
        args = qwen4_exp.TextArgs(hidden_size=64, hc_count=4, hc_lowrank=16)
        small = qwen4_exp.GatedResidual(args)
        self.assertFalse(hc_small.routes(small, mx.zeros((1, 1, 256), mx.bfloat16)))


@unittest.skipUnless(mx.metal.is_available(), "Metal only")
class TestFlashNextKernels(unittest.TestCase):
    def test_g32_moe_gather_matches_ops(self):
        """The expert-grouped gather at group size 32 and K = 640 (8-lane row groups)."""
        import os
        import sys
        from unittest import mock

        from mlx.utils import tree_map

        from mlx_lm.models import moe_small

        sys.path.insert(0, os.path.dirname(__file__))
        from test_qwen3_5_fusion import MOE_CONFIG, _moe_block

        for hidden, inter, group in ((2560, 640, 32), (1024, 512, 64)):
            config = {**MOE_CONFIG, "hidden_size": hidden, "moe_intermediate_size": inter,
                      "shared_expert_intermediate_size": inter, "num_experts": 16, "num_experts_per_tok": 8}
            block = _moe_block(config, scale=0.05)
            nn.quantize(block, group, 4)
            block.update(tree_map(lambda p: p.astype(mx.bfloat16) if p.dtype == mx.float32 else p, block.parameters()))
            for mm in (1, 3, 8):
                xx = mx.random.normal((1, mm, hidden)).astype(mx.bfloat16)
                lg = block.gate(xx)
                inds = mx.argpartition(lg[..., :16], kth=-8, axis=-1)[..., -8:]
                y = moe_small.experts(block, xx, lg, inds).astype(mx.float32)
                with mock.patch.object(moe_small, "routes", return_value=False):
                    expected = block(xx).astype(mx.float32)
                tol = 0.03 * mx.abs(expected).max().item()
                self.assertLess(mx.abs(y - expected).max().item(), tol, (hidden, group, mm))

    def test_sigmoid_gated_norm_matches_ops(self):
        from mlx_lm.models import fused_ops

        heads, D, GS, GO = 6, 128, 2000, 800
        norm = qwen4_exp.RMSNormGated(D, eps=1e-6, activation="sigmoid")
        norm.weight = (mx.random.normal((D,)) * 0.2 + 1).astype(mx.bfloat16)
        for rows in (1, 4):
            x = (mx.random.normal((1, rows, heads, D)) * 2).astype(mx.bfloat16)
            wide = (mx.random.normal((rows, GS)) * 2).astype(mx.bfloat16)
            z = wide[:, GO : GO + heads * D].reshape(1, rows, heads, D)
            got = fused_ops.gated_norm(norm, x, wide, GO)
            want = norm(x, z).reshape(1, rows, heads * D)
            self.assertTrue(mx.array_equal(got, want).item(), rows)

    def test_ple_conv_matches_ops(self):
        from mlx_lm.models import fused_ops

        C, KW, DIL = 512, 4, 3
        conv = nn.Conv1d(C, C, kernel_size=KW, dilation=DIL, groups=C, bias=False)
        conv.weight = (mx.random.normal(conv.weight.shape) * 0.3).astype(mx.bfloat16)
        for B, L in ((1, 1), (2, 5), (1, 12)):
            x = mx.random.normal((B, L, C)).astype(mx.bfloat16)
            state = mx.random.normal((B, (KW - 1) * DIL, C)).astype(mx.bfloat16)
            out, new_state = fused_ops.ple_conv(x, state, conv.weight)
            conv_input = mx.concatenate([state, x], axis=1)
            want = nn.silu(conv(conv_input))
            self.assertTrue(mx.array_equal(new_state, conv_input[:, -(KW - 1) * DIL :]).item())
            diff = (out.astype(mx.float32) - want.astype(mx.float32)).abs()
            self.assertLess(diff.max().item(), 1e-2, (B, L))
            self.assertGreater((diff == 0).mean().item(), 0.9, (B, L))

