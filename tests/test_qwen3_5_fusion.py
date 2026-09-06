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

from mlx_lm.models import moe_small, qwen3_5, qwen3_next
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
MOE_CONFIG = {
    **TEXT_CONFIG,
    "model_type": "qwen3_5_moe",
    "num_experts": 4,
    "num_experts_per_tok": 2,
    "moe_intermediate_size": 32,
    "shared_expert_intermediate_size": 32,
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


def _moe_block(config=MOE_CONFIG, seed=0, scale=0.3):
    mx.random.seed(seed)
    block = qwen3_5.SparseMoeBlock(qwen3_5.TextModelArgs.from_dict(config))
    block.update(
        tree_map(lambda p: mx.random.normal(p.shape) * scale, block.parameters())
    )
    mx.eval(block.parameters())
    return block


def _unfuse_experts(weights, num_experts, hidden_dim, prefix=""):
    """The checkpoint layout of a fused MoE block: separate gate, up and shared expert."""
    E, I = num_experts, hidden_dim
    out = {}
    for key, v in weights.items():
        name, param = key.rsplit(".", 1)
        if name == "gate":
            out[f"{prefix}gate.{param}"] = v[:E]
            out[f"{prefix}shared_expert_gate.{param}"] = v[E:]
        elif name == "switch_mlp.gate_up_proj":
            out[f"{prefix}switch_mlp.gate_proj.{param}"] = v[:E, :I]
            out[f"{prefix}switch_mlp.up_proj.{param}"] = v[:E, I:]
            out[f"{prefix}shared_expert.gate_proj.{param}"] = v[E, :I]
            out[f"{prefix}shared_expert.up_proj.{param}"] = v[E, I:]
        elif name == "switch_mlp.down_proj":
            out[f"{prefix}switch_mlp.down_proj.{param}"] = v[:E]
            out[f"{prefix}shared_expert.down_proj.{param}"] = v[E]
        else:
            out[prefix + key] = v
    return out


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


class TestQwen3_5MoeFusion(unittest.TestCase):
    def test_block_matches_qwen3_next(self):
        block = _moe_block()
        args = qwen3_5.TextModelArgs.from_dict(MOE_CONFIG)
        twin = qwen3_next.Qwen3NextSparseMoeBlock(args)
        weights = _unfuse_experts(
            dict(tree_flatten(block.parameters())),
            args.num_experts,
            args.moe_intermediate_size,
        )
        twin.load_weights(list(weights.items()))
        mx.eval(twin.parameters())
        # 70 tokens takes the sorted path of both blocks
        for length in (1, 3, 70):
            x = mx.random.normal((1, length, args.hidden_size))
            self.assertTrue(mx.allclose(block(x), twin(x), atol=1e-5), length)

    def test_fusion_is_exact(self):
        prefix = "language_model.model.layers.0.mlp."
        for quantize in (False, True):
            block = _moe_block()
            if quantize:
                nn.quantize(block, 32, 4)
            fused = {prefix + k: v for k, v in tree_flatten(block.parameters())}
            unfused = _unfuse_experts(
                dict(tree_flatten(block.parameters())),
                block.num_experts,
                MOE_CONFIG["moe_intermediate_size"],
                prefix,
            )
            for weights in (unfused, dict(fused)):
                got = qwen3_5.fuse_projections(weights)
                self.assertEqual(set(got), set(fused))
                for k, v in fused.items():
                    self.assertTrue(mx.array_equal(got[k], v), k)

    def test_kernel_matches_fallback(self):
        if not moe_small._m5():
            raise unittest.SkipTest("the MoE decode kernel runs on M5 GPUs")
        config = {
            **MOE_CONFIG,
            "hidden_size": 2048,
            "moe_intermediate_size": 512,
            "shared_expert_intermediate_size": 512,
            "num_experts": 16,
            "num_experts_per_tok": 8,
        }
        block = _moe_block(config, scale=0.05)
        nn.quantize(block, 64, 4)
        block.update(
            tree_map(
                lambda p: p.astype(mx.bfloat16) if p.dtype == mx.float32 else p,
                block.parameters(),
            )
        )
        for m in (1, 2, 3, 5, 8):
            x = mx.random.normal((1, m, 2048)).astype(mx.bfloat16)
            self.assertTrue(moe_small.routes(block, x))
            y = block(x).astype(mx.float32)
            with mock.patch.object(moe_small, "routes", return_value=False):
                expected = block(x).astype(mx.float32)
            tol = 0.03 * mx.abs(expected).max().item()
            self.assertLess(mx.abs(y - expected).max().item(), tol, m)
        self.assertFalse(
            moe_small.routes(block, mx.random.normal((1, 9, 2048)).astype(mx.bfloat16))
        )
