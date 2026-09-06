# Copyright © 2026 Apple Inc.

import json
import tempfile
import unittest
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten, tree_map

from mlx_lm.generate import generate_step, speculative_generate_step
from mlx_lm.models import qwen3_5, qwen3_5_moe, qwen3_5_mtp
from mlx_lm.models.cache import (
    ArraysCache,
    KVCache,
    make_prompt_cache,
    trim_prompt_cache,
)
from mlx_lm.models.gated_delta import gated_delta_kernel, gated_delta_ops
from mlx_lm.models.switch_layers import QuantizedSwitchLinear
from mlx_lm.utils import load_model

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


def _reinit(model, scale=0.3):
    # With the default init greedy decoding collapses to a single token
    weights = tree_map(
        lambda p: mx.random.normal(p.shape) * scale if p.ndim >= 2 else p,
        model.parameters(),
    )
    model.update(weights)
    mx.eval(model.parameters())
    return model


def _models(tie=False, moe=False):
    mx.random.seed(0)
    module = qwen3_5_moe if moe else qwen3_5
    text_config = {**(MOE_CONFIG if moe else TEXT_CONFIG), "tie_word_embeddings": tie}
    args = module.ModelArgs.from_dict(
        {"model_type": text_config["model_type"], "text_config": text_config}
    )
    target = _reinit(module.Model(args))
    head = _reinit(
        qwen3_5_mtp.Model(qwen3_5_mtp.ModelArgs("qwen3_5_mtp", text_config, 3))
    )
    return target, head


def _unfused(weights):
    """The head's weights with q, k and v split as in the checkpoints."""
    weights = dict(weights)
    q_dim = 2 * TEXT_CONFIG["num_attention_heads"] * TEXT_CONFIG["head_dim"]
    kv_dim = TEXT_CONFIG["num_key_value_heads"] * TEXT_CONFIG["head_dim"]
    for key in [k for k in weights if ".qkv_proj." in k]:
        prefix, param = key.split("qkv_proj.")
        parts = mx.split(weights.pop(key), [q_dim, q_dim + kv_dim], axis=0)
        for name, w in zip(("q_proj", "k_proj", "v_proj"), parts):
            weights[f"{prefix}{name}.{param}"] = w
    return weights


def _hf_layout(weights):
    """Bundled HF layout: "mtp." prefix, norms as (actual - 1), stacked experts."""
    hf = {
        "mtp." + k: v - 1.0 if "norm" in k else v for k, v in _unfused(weights).items()
    }
    for key in [k for k in hf if k.endswith(".switch_mlp.gate_proj.weight")]:
        prefix = key[: -len("switch_mlp.gate_proj.weight")]
        hf[f"{prefix}experts.gate_up_proj"] = mx.concatenate(
            [hf.pop(key), hf.pop(f"{prefix}switch_mlp.up_proj.weight")], axis=-2
        )
        hf[f"{prefix}experts.down_proj"] = hf.pop(
            f"{prefix}switch_mlp.down_proj.weight"
        )
    return hf


def _dequantized(module):
    return mx.dequantize(
        module.weight, module.scales, module.biases, module.group_size, module.bits
    )


def _greedy(prompt, model, max_tokens):
    return [t for t, _ in generate_step(prompt, model, max_tokens=max_tokens)]


def _speculative(prompt, model, draft, max_tokens, **kwargs):
    out = list(
        speculative_generate_step(prompt, model, draft, max_tokens=max_tokens, **kwargs)
    )
    return [t for t, _, _ in out], [d for _, _, d in out]


class _OracleDrafter:
    """Drafts the expected token, except at every third position."""

    needs_hidden = True

    def __init__(self, expected, vocab_size):
        self.expected = expected
        self.vocab_size = vocab_size

    def bind(self, target):
        pass

    def make_cache(self):
        return [KVCache()]

    def __call__(self, inputs, hidden, cache):
        assert hidden.shape[:2] == inputs.shape
        S = inputs.shape[1]
        # Position q pairs token q + 1 with hidden state q and predicts token q + 2
        start = cache[0].offset
        cache[0].update_and_fetch(mx.zeros((1, 1, S, 1)), mx.zeros((1, 1, S, 1)))
        tokens = [
            (self.expected[q + 2] + (q % 3 == 2)) % self.vocab_size
            for q in range(start, start + S)
        ]
        logits = (
            mx.arange(self.vocab_size)[None, None] == mx.array(tokens)[None, :, None]
        )
        return logits.astype(mx.float32), hidden


class TestQwen3_5MTP(unittest.TestCase):
    def test_gated_delta_states(self):
        if mx.default_device() != mx.gpu:
            raise unittest.SkipTest("gated delta kernels are GPU only")
        mx.random.seed(3)
        B, T, Hk, Hv, Dk, Dv = 2, 7, 2, 4, 128, 128

        def normed(shape):
            x = mx.fast.rms_norm(mx.random.normal(shape), None, 1e-6)
            return (x * Dk**-0.5).astype(mx.bfloat16)

        q, k = normed((B, T, Hk, Dk)), normed((B, T, Hk, Dk))
        v = mx.random.normal((B, T, Hv, Dv)).astype(mx.bfloat16)
        g = mx.exp(-mx.random.uniform(shape=(B, T, Hv)) * 0.2)
        beta = mx.random.uniform(shape=(B, T, Hv)).astype(mx.bfloat16)
        state = mx.random.normal((B, Hv, Dv, Dk)) * 0.3
        for mask in (None, mx.arange(T)[None] < mx.array([[5], [3]])):
            y, s = gated_delta_kernel(q, k, v, g, beta, state, mask)
            y2, s2, states = gated_delta_kernel(
                q, k, v, g, beta, state, mask, return_states=True
            )
            self.assertEqual(states.shape, (B, T, Hv, Dv, Dk))
            self.assertTrue(mx.array_equal(y, y2))
            self.assertTrue(mx.array_equal(s, s2))
            for t in range(T):
                prefix = [x[:, : t + 1] for x in (q, k, v, g, beta)]
                m = None if mask is None else mask[:, : t + 1]
                _, s_t = gated_delta_kernel(*prefix, state, m)
                self.assertTrue(mx.array_equal(states[:, t], s_t))
        _, s, states = gated_delta_ops(q, k, v, g, beta, state, return_states=True)
        self.assertEqual(states.shape, (B, T, Hv, Dv, Dk))
        self.assertTrue(mx.array_equal(states[:, -1], s))

    def test_arrays_cache_trim(self):
        for moe in (False, True):
            target, _ = _models(moe=moe)
            tokens = PROMPT[None, :5]
            self.assertEqual(trim_prompt_cache(make_prompt_cache(target), 1), 0)

            cache = make_prompt_cache(target)
            for c in cache:
                if isinstance(c, ArraysCache):
                    c.keep_states = True
            target(tokens, cache=cache)
            for c in cache:
                if isinstance(c, ArraysCache):
                    c.stage(mx.array(3))
            self.assertEqual(trim_prompt_cache(cache, 2), 2)

            expected = make_prompt_cache(target)
            target(tokens[:, :3], cache=expected)
            for c, e in zip(cache, expected):
                if isinstance(c, ArraysCache):
                    self.assertIsNone(c.rollback)
                    self.assertTrue(mx.allclose(c[0], e[0], atol=1e-6))
                    self.assertTrue(mx.allclose(c[1], e[1], atol=1e-6))
                else:
                    self.assertEqual(c.offset, e.offset)
            out = target(tokens[:, 3:4], cache=cache)
            self.assertTrue(mx.allclose(out, target(tokens[:, 3:4], cache=expected)))

    def test_speculative_matches_greedy(self):
        for moe in (False, True):
            for tie in (False, True):
                target, head = _models(tie, moe)
                expected = _greedy(PROMPT, target, 32)
                self.assertGreater(len(set(expected)), 8)
                for k in (1, 2, 3):
                    for step in (512, 3):
                        tokens, _ = _speculative(
                            PROMPT,
                            target,
                            head,
                            32,
                            num_draft_tokens=k,
                            prefill_step_size=step,
                        )
                        self.assertEqual(tokens, expected)
                for stop in (0.0, 0.9):
                    tokens, _ = _speculative(
                        PROMPT,
                        target,
                        head,
                        32,
                        num_draft_tokens=3,
                        draft_stop_prob=stop,
                    )
                    self.assertEqual(tokens, expected)
            # A one token prompt has no hidden state before the first target step
            tokens, _ = _speculative(PROMPT[:1], target, head, 16, num_draft_tokens=2)
            self.assertEqual(tokens, _greedy(PROMPT[:1], target, 16))

    def test_partial_acceptance(self):
        for moe in (False, True):
            target, _ = _models(moe=moe)
            expected = _greedy(PROMPT, target, 32)
            drafter = _OracleDrafter(
                PROMPT.tolist() + expected + [0] * 8, TEXT_CONFIG["vocab_size"]
            )
            for k in (1, 2, 3):
                tokens, from_draft = _speculative(
                    PROMPT, target, drafter, 32, num_draft_tokens=k, prefill_step_size=4
                )
                self.assertEqual(tokens, expected)
                self.assertTrue(0 < sum(from_draft) < len(from_draft))

    def test_load_sidecar_and_bundled(self):
        for moe in (False, True):
            self._check_load(moe)

    def _check_load(self, moe):
        _, head = _models(moe=moe)
        text_config = MOE_CONFIG if moe else TEXT_CONFIG
        weights = dict(tree_flatten(head.parameters()))
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            # Sidecar: no prefix, separate q, k, v and unshifted norm weights
            config = {
                "model_type": "qwen3_5_mtp",
                "block_size": 3,
                "text_config": text_config,
            }
            json.dump(config, open(d / "config.json", "w"))
            mx.save_safetensors(str(d / "model.safetensors"), _unfused(weights))
            model, _ = load_model(d)
            self.assertEqual(type(model.layers[0].mlp), type(head.layers[0].mlp))
            loaded = dict(tree_flatten(model.parameters()))
            self.assertEqual(set(loaded), set(weights))
            for k, v in loaded.items():
                self.assertTrue(mx.array_equal(v, weights[k]), k)

            # Bundled HF layout with a quantized target
            hf = _hf_layout(weights)
            hf["model.language_model.norm.weight"] = mx.zeros(4)
            mx.save_safetensors(str(d / "model.safetensors"), hf)
            index = {"weight_map": {k: "model.safetensors" for k in hf}}
            json.dump(index, open(d / "model.safetensors.index.json", "w"))
            config = {
                "model_type": text_config["model_type"],
                "text_config": text_config,
                "quantization": {"group_size": 32, "bits": 4},
            }
            json.dump(config, open(d / "config.json", "w"))
            model = qwen3_5_mtp.load_bundled(d)
            self.assertIsInstance(model.fc, nn.QuantizedLinear)
            self.assertTrue(mx.array_equal(model.norm.weight, head.norm.weight))
            q_norm = model.layers[0].self_attn.q_norm.weight
            self.assertTrue(
                mx.array_equal(q_norm, head.layers[0].self_attn.q_norm.weight)
            )
            if moe:
                for name in ("gate_up_proj", "down_proj"):
                    got = getattr(model.layers[0].mlp.switch_mlp, name)
                    self.assertIsInstance(got, QuantizedSwitchLinear)
                    expected = getattr(head.layers[0].mlp.switch_mlp, name).weight
                    self.assertTrue(
                        mx.allclose(_dequantized(got), expected, atol=0.1), name
                    )


if __name__ == "__main__":
    unittest.main()
