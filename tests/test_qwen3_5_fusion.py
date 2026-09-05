# Copyright © 2026 Apple Inc.

import os
import tempfile
import unittest
from contextlib import contextmanager
from itertools import accumulate
from pathlib import Path
from unittest import mock

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten, tree_map

from mlx_lm.models import qwen3_5, qwen3_next
from mlx_lm.models.cache import make_prompt_cache
from mlx_lm.utils import (
    dequantize_model,
    load_model,
    quantize_model,
    save_config,
    save_model,
)

TEXT_CONFIG = {
    "model_type": "qwen3_5",
    "hidden_size": 64,
    "intermediate_size": 96,
    "num_hidden_layers": 2,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "head_dim": 16,
    "vocab_size": 128,
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
PROMPT = mx.array([3, 17, 42, 7, 99, 5, 61, 8, 23, 44])


def _model(seed=0):
    mx.random.seed(seed)
    args = qwen3_5.ModelArgs.from_dict(
        {"model_type": "qwen3_5", "text_config": TEXT_CONFIG}
    )
    model = qwen3_5.Model(args)
    model.eval()
    mx.eval(model.parameters())
    return model


def _fused_projections(layer):
    """(module, fused name, part names, part rows) of a decoder layer."""
    if layer.is_linear:
        m = layer.linear_attn
        yield (
            "linear_attn",
            "in_proj",
            ("in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a"),
            (m.conv_dim, m.value_dim, m.num_v_heads, m.num_v_heads),
        )
    else:
        m = layer.self_attn
        kv_dim = m.num_key_value_heads * m.head_dim
        yield (
            "self_attn",
            "qkv_proj",
            ("q_proj", "k_proj", "v_proj"),
            (2 * m.num_attention_heads * m.head_dim, kv_dim, kv_dim),
        )
    hidden_dim = layer.mlp.gate_up_proj.weight.shape[0] // 2
    yield "mlp", "gate_up_proj", ("gate_proj", "up_proj"), (hidden_dim, hidden_dim)


def _unfuse(model):
    """The model's weights with the projections split as in a checkpoint."""
    weights = dict(tree_flatten(model.parameters()))
    for i, layer in enumerate(model.layers):
        for module, fused, parts, rows in _fused_projections(layer):
            prefix = f"language_model.model.layers.{i}.{module}"
            for param in ("weight", "scales", "biases"):
                if (key := f"{prefix}.{fused}.{param}") not in weights:
                    continue
                splits = list(accumulate(rows))[:-1]
                for part, w in zip(parts, mx.split(weights.pop(key), splits, axis=0)):
                    weights[f"{prefix}.{part}.{param}"] = w
    return weights


def _qwen3_next_twin(model):
    """The same network built from the unfused qwen3_next modules."""
    args = model.language_model.args
    twin = qwen3_next.Model(
        qwen3_next.ModelArgs(
            model_type="qwen3_next",
            hidden_size=args.hidden_size,
            num_hidden_layers=args.num_hidden_layers,
            intermediate_size=args.intermediate_size,
            num_attention_heads=args.num_attention_heads,
            linear_num_value_heads=args.linear_num_value_heads,
            linear_num_key_heads=args.linear_num_key_heads,
            linear_key_head_dim=args.linear_key_head_dim,
            linear_value_head_dim=args.linear_value_head_dim,
            linear_conv_kernel_dim=args.linear_conv_kernel_dim,
            num_experts=0,
            num_experts_per_tok=0,
            decoder_sparse_step=1,
            shared_expert_intermediate_size=0,
            mlp_only_layers=[],
            moe_intermediate_size=0,
            rms_norm_eps=args.rms_norm_eps,
            vocab_size=args.vocab_size,
            num_key_value_heads=args.num_key_value_heads,
            rope_theta=args.rope_theta,
            partial_rotary_factor=args.partial_rotary_factor,
            max_position_embeddings=args.max_position_embeddings,
            head_dim=args.head_dim,
            tie_word_embeddings=args.tie_word_embeddings,
            attention_bias=args.attention_bias,
            rope_scaling=args.rope_scaling,
            full_attention_interval=args.full_attention_interval,
        )
    )
    weights = {
        k.replace("language_model.model.", "model.").replace(
            "language_model.lm_head.", "lm_head."
        ): v
        for k, v in _unfuse(model).items()
    }
    # qwen3_next interleaves the rows per key head: q | k | v | z and b | a
    for i, layer in enumerate(model.layers):
        if not layer.is_linear:
            continue
        m = layer.linear_attn
        prefix = f"model.layers.{i}.linear_attn"
        for param in ("weight", "scales", "biases"):
            if f"{prefix}.in_proj_qkv.{param}" not in weights:
                continue
            q, k, v = mx.split(
                weights.pop(f"{prefix}.in_proj_qkv.{param}"),
                [m.key_dim, 2 * m.key_dim],
                axis=0,
            )
            z, b, a = (
                weights.pop(f"{prefix}.in_proj_{n}.{param}") for n in ("z", "b", "a")
            )
            heads = lambda x: mx.split(x, m.num_k_heads, axis=0)
            weights[f"{prefix}.in_proj_qkvz.{param}"] = mx.concatenate(
                [t for ts in zip(*map(heads, (q, k, v, z))) for t in ts], axis=0
            )
            weights[f"{prefix}.in_proj_ba.{param}"] = mx.concatenate(
                [t for ts in zip(*map(heads, (b, a))) for t in ts], axis=0
            )
    quantized = [
        m for _, m in model.named_modules() if isinstance(m, nn.QuantizedLinear)
    ]
    if quantized:
        nn.quantize(
            twin,
            quantized[0].group_size,
            quantized[0].bits,
            class_predicate=lambda p, _: f"{p}.scales" in weights,
        )
    twin.load_weights(list(weights.items()))
    twin.eval()
    mx.eval(twin.parameters())
    return twin


@contextmanager
def _l2norm_eps():
    """Give qwen3_next's q/k norm the eps on sum(x^2) that qwen3_5 uses."""
    rms_norm = mx.fast.rms_norm

    def patched(x, weight, eps):
        return rms_norm(x, weight, eps / x.shape[-1] if weight is None else eps)

    with mock.patch.object(mx.fast, "rms_norm", patched):
        yield


def _logits(model, prompt, steps=4):
    """Prefill logits followed by cached single token decode logits."""
    out = [model(prompt[None])]
    cache = make_prompt_cache(model)
    model(prompt[:-1][None], cache=cache)
    y = prompt[-1:]
    for _ in range(steps):
        logits = model(y[None], cache=cache)
        out.append(logits)
        y = logits[0, -1].argmax(keepdims=True)
    return mx.concatenate(out, axis=1).astype(mx.float32)


class _Group:
    """Stands in for a distributed group to shard on one process."""

    def __init__(self, size, rank):
        self._size, self._rank = size, rank

    def size(self):
        return self._size

    def rank(self):
        return self._rank


class TestQwen3_5Fusion(unittest.TestCase):
    def assert_same_weights(self, a, b):
        self.assertEqual(set(a), set(b))
        for k in a:
            self.assertTrue(mx.array_equal(a[k], b[k]), k)

    def test_sanitize_fuses_projections(self):
        model = _model()
        fused = dict(tree_flatten(model.parameters()))
        unfused = _unfuse(model)
        self.assertFalse(any(".in_proj." in k or ".qkv_proj." in k for k in unfused))

        # Both checkpoint layouts, and fused keys pass through unchanged
        for prefix in ("language_model.model.", "model.language_model."):
            weights = {
                k.replace("language_model.model.", prefix): v
                for k, v in unfused.items()
            }
            self.assert_same_weights(model.sanitize(weights), fused)
        self.assert_same_weights(model.sanitize(dict(fused)), fused)

    def test_quantized_fusion_is_exact(self):
        model = _model()
        unfused = _unfuse(model)
        qmodel = _model()
        nn.quantize(qmodel, 32, 4)
        # Quantizing the parts on their own gives the rows of the fused module
        weights = {}
        for k, v in unfused.items():
            if v.ndim == 2 and any(p in k for p in ("proj", "lm_head", "embed")):
                w, s, b = mx.quantize(v, 32, 4)
                weights.update({k: w, k[:-6] + "scales": s, k[:-6] + "biases": b})
            else:
                weights[k] = v
        self.assert_same_weights(
            qmodel.sanitize(weights), dict(tree_flatten(qmodel.parameters()))
        )

    def test_matches_qwen3_next(self):
        model = _model()
        with _l2norm_eps():
            expected = _logits(_qwen3_next_twin(model), PROMPT)
        self.assertTrue(mx.allclose(_logits(model, PROMPT), expected, atol=1e-5))

    def test_checkpoint_matches_qwen3_next(self):
        path = os.environ.get("QWEN3_5_MODEL_PATH")
        if path is None:
            raise unittest.SkipTest("set QWEN3_5_MODEL_PATH to a Qwen3.5 checkpoint")
        model, _ = load_model(Path(path))
        twin = _qwen3_next_twin(model)
        prompt = mx.arange(1000, 1024)

        def f32_logits(m):
            m = dequantize_model(m)
            m.update(tree_map(lambda p: p.astype(mx.float32), m.parameters()))
            return _logits(m, prompt)

        expected = f32_logits(model)
        with _l2norm_eps():
            diff = mx.abs(f32_logits(twin) - expected).max().item()
        self.assertLess(diff, 1e-2, f"max abs logit diff {diff}")

    def test_load_unfused_and_fused_checkpoints(self):
        model = _model()
        config = {"model_type": "qwen3_5", "text_config": TEXT_CONFIG}
        with tempfile.TemporaryDirectory() as d:
            unfused = Path(d) / "unfused"
            unfused.mkdir()
            mx.save_safetensors(str(unfused / "model.safetensors"), _unfuse(model))
            save_config(dict(config), unfused / "config.json")
            loaded, _ = load_model(unfused)
            self.assert_same_weights(
                dict(tree_flatten(loaded.parameters())),
                dict(tree_flatten(model.parameters())),
            )

            # Quantize, save fused, and load again
            qmodel, qconfig = quantize_model(loaded, config, 32, 4)
            fused_q = Path(d) / "fused_q"
            save_model(fused_q, qmodel)
            save_config(dict(qconfig), fused_q / "config.json")
            loaded_q, _ = load_model(fused_q)
            expected = dict(tree_flatten(qmodel.parameters()))
            self.assertIn(
                "language_model.model.layers.0.linear_attn.in_proj.scales", expected
            )
            self.assert_same_weights(
                dict(tree_flatten(loaded_q.parameters())), expected
            )

            # An already converted checkpoint with separate quantized projections
            unfused_q = Path(d) / "unfused_q"
            unfused_q.mkdir()
            mx.save_safetensors(str(unfused_q / "model.safetensors"), _unfuse(qmodel))
            save_config(dict(qconfig), unfused_q / "config.json")
            loaded_q, _ = load_model(unfused_q)
            self.assert_same_weights(
                dict(tree_flatten(loaded_q.parameters())), expected
            )

    def test_shard_splits_per_head(self):
        model = _model()
        x = mx.random.normal((1, 3, TEXT_CONFIG["hidden_size"]))
        h = TEXT_CONFIG["num_key_value_heads"]

        def outputs(model):
            outs = []
            for layer in model.layers:
                if layer.is_linear:
                    m = layer.linear_attn
                    kd, vd, nv = m.key_dim, m.value_dim, m.num_v_heads
                    splits = [
                        kd,
                        2 * kd,
                        2 * kd + vd,
                        2 * kd + 2 * vd,
                        2 * kd + 2 * vd + nv,
                    ]
                    outs.append(mx.split(m.in_proj(x), splits, axis=-1))
                else:
                    m = layer.self_attn
                    q_dim = 2 * m.num_attention_heads * m.head_dim
                    kv_dim = m.num_key_value_heads * m.head_dim
                    outs.append(
                        mx.split(m.qkv_proj(x), [q_dim, q_dim + kv_dim], axis=-1)
                    )
                outs[-1] += mx.split(layer.mlp.gate_up_proj(x), 2, axis=-1)
            return outs

        reference = outputs(model)
        for N in (2, 4):
            for rank in range(N):
                sharded = _model()
                sharded.shard(_Group(N, rank))
                for layer, got, ref in zip(sharded.layers, outputs(sharded), reference):
                    expected = [mx.split(r, N, axis=-1)[rank] for r in ref]
                    if not layer.is_linear and N > h:
                        # Repeated kv heads: each node gets one whole head
                        expected[1:3] = [
                            mx.split(r, h, axis=-1)[rank // (N // h)] for r in ref[1:3]
                        ]
                    for g, e in zip(got, expected):
                        self.assertEqual(g.shape, e.shape)
                        self.assertTrue(mx.allclose(g, e, atol=1e-6))


if __name__ == "__main__":
    unittest.main()
