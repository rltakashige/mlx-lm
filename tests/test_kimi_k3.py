import json
import math
import tempfile
import unittest
import weakref
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx.utils import tree_flatten, tree_map_with_path

from mlx_lm.models.cache import ArraysCache, KVCache, QuantizedKVCache
from mlx_lm.models.gated_delta import (
    compute_g_lower_bound,
    gated_delta_update,
)
from mlx_lm.models.kimi_k3 import Model as KimiK3
from mlx_lm.models.kimi_k3 import ModelArgs as KimiK3Args
from mlx_lm.models.kimi_linear import (
    KimiDeltaAttention,
    KimiMLAAttention,
    KimiSparseMoE,
    _apply_attention_residual,
    _attention_residual_streaming,
    _group_expert_select,
    _kda_norm_gate,
    _kda_normalize_qk,
    situ,
)
from mlx_lm.models.kimi_linear import ModelArgs as KimiLinearArgs
from mlx_lm.models.switch_layers import QuantizedSwitchLinear
from mlx_lm.utils import _compressed_tensors_quantization, load_model, quantize_model


class Identity(nn.Module):
    def __call__(self, x, *args, **kwargs):
        return x


def stacked_attention_residual_reference(sources, score_weight, eps):
    values = mx.stack(sources, axis=-2).astype(mx.float32)
    keys = values * mx.rsqrt(mx.mean(mx.square(values), axis=-1, keepdims=True) + eps)
    probabilities = mx.softmax(
        mx.sum(keys * score_weight.astype(mx.float32), axis=-1),
        axis=-1,
        precise=True,
    )
    output = mx.sum(probabilities[..., None] * values, axis=-2)
    return output.astype(sources[-1].dtype)


def tiny_text_config():
    return {
        "model_type": "kimi_linear",
        "vocab_size": 32,
        "hidden_size": 16,
        "num_hidden_layers": 4,
        "num_attention_heads": 2,
        "num_key_value_heads": 2,
        "intermediate_size": 32,
        "head_dim": 4,
        "rms_norm_eps": 1e-5,
        "linear_attn_config": {
            "num_heads": 2,
            "head_dim": 4,
            "kda_layers": [1, 2, 3],
            "full_attn_layers": [4],
            "short_conv_kernel_size": 4,
            "use_full_rank_gate": True,
            "gate_lower_bound": -5.0,
        },
        "num_experts": 2,
        "num_experts_per_token": 1,
        "num_shared_experts": 1,
        "moe_intermediate_size": 32,
        "first_k_dense_replace": 1,
        "kv_lora_rank": 4,
        "q_lora_rank": 4,
        "qk_nope_head_dim": 4,
        "qk_rope_head_dim": 2,
        "v_head_dim": 4,
        "mla_use_nope": True,
        "mla_use_output_gate": True,
        "hidden_act": "situ",
        "activation_situ_beta": 4.0,
        "activation_situ_linear_beta": 25.0,
        "routed_expert_hidden_size": 32,
        "latent_moe_use_norm": True,
        "attn_res_block_size": 2,
        "max_position_embeddings": 128,
    }


class TestKimiK3(unittest.TestCase):
    def setUp(self):
        mx.random.seed(7)
        self.args = KimiK3Args.from_dict(
            {"model_type": "kimi_k3", "text_config": tiny_text_config()}
        )
        self.model = KimiK3(self.args)

    def test_nested_config_and_k3_modules(self):
        text_args = self.args.text_config
        self.assertEqual(text_args.model_max_length, 128)
        self.assertEqual(text_args.rope_theta, 10000.0)

        self.assertEqual(
            [layer.is_linear for layer in self.model.layers],
            [True, True, True, False],
        )
        self.assertNotIsInstance(self.model.layers[0].mlp, KimiSparseMoE)
        self.assertTrue(
            all(isinstance(layer.mlp, KimiSparseMoE) for layer in self.model.layers[1:])
        )

        kda = self.model.layers[0].self_attn
        self.assertIsInstance(kda, KimiDeltaAttention)
        self.assertTrue(kda.use_full_rank_gate)
        self.assertTrue(hasattr(kda, "g_proj"))
        self.assertFalse(hasattr(kda, "g_a_proj"))

        mla = self.model.layers[-1].self_attn
        self.assertIsInstance(mla, KimiMLAAttention)
        self.assertEqual(mla.q_lora_rank, 4)
        self.assertTrue(mla.use_output_gate)
        self.assertTrue(hasattr(mla, "q_a_proj"))
        self.assertTrue(hasattr(mla, "g_proj"))

        moe = self.model.layers[1].mlp
        self.assertEqual(moe.routed_expert_down_proj.weight.shape, (32, 16))
        self.assertEqual(moe.switch_mlp.gate_proj.weight.shape, (2, 32, 32))
        self.assertEqual(moe.routed_expert_up_proj.weight.shape, (16, 32))

    def test_kimi_k3_requires_nope_mla(self):
        config = tiny_text_config()
        config["mla_use_nope"] = False
        self.assertFalse(KimiLinearArgs.from_dict(config).mla_use_nope)
        with self.assertRaisesRegex(
            ValueError,
            "Kimi-K3 requires mla_use_nope=True",
        ):
            KimiK3Args.from_dict({"model_type": "kimi_k3", "text_config": config})

    def test_router_logits_are_computed_in_float32(self):
        moe = self.model.layers[1].mlp
        weight = mx.zeros((2, 16), dtype=mx.bfloat16)
        weight[0, 0] = 1.0
        weight[1, 0] = 1.0
        weight[1, 1] = 1.0 / 256.0
        moe.gate.weight = weight
        logits = moe._router_logits(mx.ones((1, 1, 16), dtype=mx.bfloat16))
        mx.eval(logits)
        self.assertEqual(logits.dtype, mx.float32)
        self.assertEqual(mx.argmax(logits, axis=-1).item(), 1)
        self.assertAlmostEqual(logits[0, 0, 1].item(), 1.00390625)

    def test_conversion_keeps_router_dense_and_forward_compatible(self):
        config = tiny_text_config()
        config["hidden_size"] = 32
        model = KimiK3(
            KimiK3Args.from_dict({"model_type": "kimi_k3", "text_config": config})
        )
        moe = model.layers[1].mlp
        predicate = model.quant_predicate
        prefix = "language_model.model.layers.1.mlp"

        self.assertFalse(predicate(f"{prefix}.gate", moe.gate))
        quantize_model(
            model,
            {"model_type": "kimi_k3"},
            group_size=32,
            bits=4,
        )

        self.assertIsInstance(moe.gate, nn.Linear)
        self.assertIsInstance(moe.routed_expert_down_proj, nn.QuantizedLinear)
        self.assertIsInstance(moe.switch_mlp.gate_proj, QuantizedSwitchLinear)
        output = moe(mx.ones((1, 1, 32), dtype=mx.bfloat16))
        mx.eval(output)
        self.assertTrue(mx.all(mx.isfinite(output)).item())

    def test_custom_conversion_quantized_router_remains_forward_compatible(self):
        config = tiny_text_config()
        config["hidden_size"] = 32
        model = KimiK3(
            KimiK3Args.from_dict({"model_type": "kimi_k3", "text_config": config})
        )
        moe = model.layers[1].mlp
        quantize_model(
            model,
            {"model_type": "kimi_k3"},
            group_size=32,
            bits=4,
            quant_predicate=lambda _path, _module: True,
        )

        self.assertIsInstance(moe.gate, nn.QuantizedLinear)
        output = moe(mx.ones((1, 1, 32), dtype=mx.bfloat16))
        mx.eval(output)
        self.assertTrue(mx.all(mx.isfinite(output)).item())

    def test_activation_quantized_router_has_actionable_error(self):
        config = tiny_text_config()
        config["hidden_size"] = 32
        model = KimiK3(
            KimiK3Args.from_dict({"model_type": "kimi_k3", "text_config": config})
        )
        moe = model.layers[1].mlp
        moe.gate = nn.QQLinear.from_linear(
            moe.gate,
            group_size=32,
            bits=8,
            mode="mxfp8",
        )
        moe.gate.eval()

        with self.assertRaisesRegex(
            TypeError,
            "Kimi-K3 routers do not support activation quantization",
        ):
            moe._router_logits(mx.ones((1, 3, 32), dtype=mx.bfloat16))

    def test_grouped_router_masks_excluded_negative_scores(self):
        indices, weights = _group_expert_select(
            mx.zeros((1, 4), dtype=mx.float32),
            mx.array([-0.6, -0.7, -0.8, -0.9], dtype=mx.float32),
            top_k=1,
            n_group=2,
            topk_group=1,
            routed_scaling_factor=1.0,
            renormalize=True,
            score_function="sigmoid",
        )
        mx.eval(indices, weights)

        self.assertEqual(indices.item(), 0)
        self.assertEqual(weights.item(), 0.5)

    def test_grouped_router_skips_mask_when_all_groups_are_selected(self):
        gates = mx.array([[0.1, 0.2, 0.3, 0.4]], dtype=mx.float32)
        bias = mx.array([0.4, -0.3, 0.2, -0.1], dtype=mx.float32)
        grouped_indices, grouped_weights = _group_expert_select(
            gates,
            bias,
            top_k=2,
            n_group=2,
            topk_group=2,
            routed_scaling_factor=1.0,
            renormalize=True,
            score_function="sigmoid",
        )
        plain_indices, plain_weights = _group_expert_select(
            gates,
            bias,
            top_k=2,
            n_group=1,
            topk_group=1,
            routed_scaling_factor=1.0,
            renormalize=True,
            score_function="sigmoid",
        )
        mx.eval(grouped_indices, grouped_weights, plain_indices, plain_weights)

        np.testing.assert_array_equal(grouped_indices, plain_indices)
        np.testing.assert_allclose(grouped_weights, plain_weights, rtol=0, atol=0)

    def test_situ_matches_reference(self):
        gate = mx.array([[-8.0, -1.0, 0.5, 9.0]], dtype=mx.float16)
        up = mx.array([[40.0, -2.0, 3.0, -50.0]], dtype=mx.float16)
        actual = situ(gate, up, beta=4.0, linear_beta=25.0)
        gate32 = gate.astype(mx.float32)
        up32 = up.astype(mx.float32)
        expected = (
            4.0
            * mx.tanh(gate32 / 4.0)
            * mx.sigmoid(gate32)
            * 25.0
            * mx.tanh(up32 / 25.0)
        ).astype(mx.float16)
        np.testing.assert_allclose(actual, expected, rtol=0, atol=0)

        unbounded_actual = situ(gate, up, beta=4.0)
        unbounded_expected = (
            4.0 * mx.tanh(gate32 / 4.0) * mx.sigmoid(gate32) * up32
        ).astype(mx.float16)
        np.testing.assert_allclose(
            unbounded_actual,
            unbounded_expected,
            rtol=0,
            atol=0,
        )

    def test_kda_lower_bound_decay(self):
        a_log = mx.array([[math.log(2.0)]], dtype=mx.float32)
        raw_gate = mx.array([[[[-0.5], [0.75]]]], dtype=mx.float32)
        dt_bias = mx.array([[0.25]], dtype=mx.float32)

        actual = compute_g_lower_bound(a_log, raw_gate, dt_bias, -5.0)
        expected = mx.exp(-5.0 * mx.sigmoid(mx.exp(a_log) * (raw_gate + dt_bias)))
        np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-7)

    def test_kda_gate_lower_bound_validation(self):
        for lower_bound in (-5.0001, 0.0, 1.0):
            with self.subTest(lower_bound=lower_bound):
                config = tiny_text_config()
                config["linear_attn_config"]["gate_lower_bound"] = lower_bound
                args = KimiK3Args.from_dict(
                    {"model_type": "kimi_k3", "text_config": config}
                )
                with self.assertRaisesRegex(
                    ValueError,
                    r"KDA gate lower bound must be in \[-5.0, 0.0\)",
                ):
                    KimiK3(args)

        config = tiny_text_config()
        config["linear_attn_config"]["gate_lower_bound"] = -1e-6
        KimiK3(KimiK3Args.from_dict({"model_type": "kimi_k3", "text_config": config}))

    def test_kda_qk_normalization_matches_explicit_l2(self):
        head_dim = 4
        q = mx.array(
            [
                [
                    [[1.0, -2.0, 3.0, -4.0]],
                    [[1e-5, -2e-5, 3e-5, -4e-5]],
                ]
            ],
            dtype=mx.float32,
        )
        k = mx.array(
            [
                [
                    [[-0.5, 1.5, -2.5, 3.5]],
                    [[-4e-5, 3e-5, -2e-5, 1e-5]],
                ]
            ],
            dtype=mx.float32,
        )

        actual_q, actual_k = _kda_normalize_qk(q, k, head_dim)
        q_l2 = q * mx.rsqrt(mx.sum(mx.square(q), axis=-1, keepdims=True) + 1e-6)
        k_l2 = k * mx.rsqrt(mx.sum(mx.square(k), axis=-1, keepdims=True) + 1e-6)
        expected_q = q_l2 / math.sqrt(head_dim)
        expected_k = k_l2
        mx.eval(actual_q, actual_k, expected_q, expected_k)

        np.testing.assert_allclose(actual_q, expected_q, rtol=1e-6, atol=1e-7)
        np.testing.assert_allclose(actual_k, expected_k, rtol=1e-6, atol=1e-7)

        old_q = (head_dim**-1) * mx.fast.rms_norm(q, None, 1e-6)
        self.assertGreater(
            mx.max(mx.abs(actual_q[:, 1] - old_q[:, 1])).item(),
            1e-3,
        )

    def test_mla_quantized_kv_cache_has_actionable_error(self):
        mla = self.model.layers[-1].self_attn
        with self.assertRaisesRegex(
            TypeError,
            "Kimi-K3 MLA does not support quantized KV caches; omit --kv-bits",
        ):
            mla(
                mx.zeros((1, 1, self.args.text_config.hidden_size)),
                cache=QuantizedKVCache(group_size=2, bits=8),
            )

    def test_mla_additive_scores_match_vllm_concatenated_reference(self):
        heads, rope_dim = 2, 64
        cases = (
            # vLLM's expanded MHA prefill, both cold and cache-offset.
            (4, 4, 128, 128, heads),
            (4, 7, 128, 128, heads),
            # vLLM's absorbed latent-MQA decode.
            (1, 5, 512, 512, 1),
        )

        for query_length, key_length, nope_dim, value_dim, kv_heads in cases:
            # Absorbed decode still uses K3's original 128+64 Q/K scale.
            scale = (128 + rope_dim) ** -0.5
            q_nope = mx.random.normal((1, heads, query_length, nope_dim)).astype(
                mx.bfloat16
            )
            q_pe = mx.random.normal((1, heads, query_length, rope_dim)).astype(
                mx.bfloat16
            )
            k_nope = mx.random.normal((1, kv_heads, key_length, nope_dim)).astype(
                mx.bfloat16
            )
            k_pe = mx.random.normal((1, 1, key_length, rope_dim)).astype(mx.bfloat16)
            values = mx.random.normal((1, kv_heads, key_length, value_dim)).astype(
                mx.bfloat16
            )

            additive_logits = (q_nope * scale) @ k_nope.swapaxes(-1, -2)
            additive_logits += (q_pe * scale) @ k_pe.swapaxes(-1, -2)
            concatenated_logits = (
                mx.concatenate([q_nope, q_pe], axis=-1) * scale
            ) @ mx.concatenate(
                [
                    k_nope,
                    mx.broadcast_to(k_pe, k_nope.shape[:-1] + (rope_dim,)),
                ],
                axis=-1,
            ).swapaxes(-1, -2)

            if query_length > 1:
                query_positions = mx.arange(
                    key_length - query_length,
                    key_length,
                )[:, None]
                mask = mx.arange(key_length)[None, :] <= query_positions
                minimum = mx.array(
                    mx.finfo(additive_logits.dtype).min,
                    additive_logits.dtype,
                )
                additive_logits = mx.where(mask, additive_logits, minimum)
                concatenated_logits = mx.where(
                    mask,
                    concatenated_logits,
                    minimum,
                )

            additive_output = mx.softmax(
                additive_logits.astype(mx.float32),
                axis=-1,
                precise=True,
            ) @ values.astype(mx.float32)
            reference_output = mx.softmax(
                concatenated_logits.astype(mx.float32),
                axis=-1,
                precise=True,
            ) @ values.astype(mx.float32)
            mx.eval(additive_output, reference_output)
            np.testing.assert_allclose(
                additive_output,
                reference_output,
                rtol=0,
                atol=1e-2,
            )

    def test_kda_output_gate_uses_float32_intermediates(self):
        norm = nn.RMSNorm(4, eps=1e-6)
        norm.weight = mx.array([0.75, 1.25, 0.5, 1.75], dtype=mx.float32)
        out = mx.array(
            [[[[0.25, -1.5, 3.0, -0.125]]]],
            dtype=mx.bfloat16,
        )
        gate = mx.array(
            [[[[0.59765625, -1.1015625, 2.203125, -3.296875]]]],
            dtype=mx.bfloat16,
        )

        actual = _kda_norm_gate(out, gate, norm, mx.bfloat16)
        out32 = out.astype(mx.float32)
        variance = mx.mean(mx.square(out32), axis=-1, keepdims=True)
        normalized = out32 * mx.rsqrt(variance + norm.eps) * norm.weight
        expected = (normalized * mx.sigmoid(gate.astype(mx.float32))).astype(
            mx.bfloat16
        )
        low_precision = (normalized * mx.sigmoid(gate).astype(mx.float32)).astype(
            mx.bfloat16
        )
        mx.eval(actual, expected, low_precision)
        np.testing.assert_array_equal(
            np.array(actual.astype(mx.float32)),
            np.array(expected.astype(mx.float32)),
        )
        self.assertFalse(mx.array_equal(actual, low_precision).item())

    def test_kda_metal_vector_gate_matches_ops(self):
        if not mx.metal.is_available():
            self.skipTest("Metal is required to validate the fused KDA path")

        batch, length, heads, head_dim = 1, 3, 2, 32
        q = mx.random.normal((batch, length, heads, head_dim)).astype(mx.float16)
        k = mx.random.normal((batch, length, heads, head_dim)).astype(mx.float16)
        v = mx.random.normal((batch, length, heads, head_dim)).astype(mx.float16)
        raw_gate = mx.random.normal((batch, length, heads, head_dim)).astype(mx.float16)
        raw_beta = mx.random.normal((batch, length, heads)).astype(mx.float16)
        a_log = mx.log(mx.random.uniform(low=1.0, high=16.0, shape=(heads, 1)))
        dt_bias = mx.arange(heads * head_dim, dtype=mx.float32).reshape(
            heads, head_dim
        ) / (heads * head_dim)

        q = mx.fast.rms_norm(q, None, 1e-6) / head_dim
        k = mx.fast.rms_norm(k, None, 1e-6) / math.sqrt(head_dim)
        expected_y, expected_state = gated_delta_update(
            q,
            k,
            v,
            raw_gate,
            raw_beta,
            a_log,
            dt_bias,
            use_kernel=False,
            gate_lower_bound=-5.0,
        )
        actual_y, actual_state = gated_delta_update(
            q,
            k,
            v,
            raw_gate,
            raw_beta,
            a_log,
            dt_bias,
            use_kernel=True,
            gate_lower_bound=-5.0,
        )
        mx.eval(expected_y, expected_state, actual_y, actual_state)
        np.testing.assert_allclose(actual_y, expected_y, rtol=5e-4, atol=5e-4)
        np.testing.assert_allclose(actual_state, expected_state, rtol=5e-4, atol=5e-4)

    def test_kda_metal_vector_gate_mask_matches_ops(self):
        if not mx.metal.is_available():
            self.skipTest("Metal is required to validate the fused KDA path")

        batch, length, heads, head_dim = 2, 3, 2, 32
        q = mx.random.normal((batch, length, heads, head_dim)).astype(mx.float16)
        k = mx.random.normal((batch, length, heads, head_dim)).astype(mx.float16)
        v = mx.random.normal((batch, length, heads, head_dim)).astype(mx.float16)
        raw_gate = mx.random.normal((batch, length, heads, head_dim)).astype(mx.float16)
        raw_beta = mx.random.normal((batch, length, heads)).astype(mx.float16)
        a_log = mx.log(mx.random.uniform(low=1.0, high=16.0, shape=(heads, 1)))
        dt_bias = mx.arange(heads * head_dim, dtype=mx.float32).reshape(
            heads, head_dim
        ) / (heads * head_dim)
        mask = mx.array([[True, False, True], [False, True, True]])

        q = mx.fast.rms_norm(q, None, 1e-6) / head_dim
        k = mx.fast.rms_norm(k, None, 1e-6) / math.sqrt(head_dim)
        expected_y, expected_state = gated_delta_update(
            q,
            k,
            v,
            raw_gate,
            raw_beta,
            a_log,
            dt_bias,
            mask=mask,
            use_kernel=False,
            gate_lower_bound=-5.0,
        )
        actual_y, actual_state = gated_delta_update(
            q,
            k,
            v,
            raw_gate,
            raw_beta,
            a_log,
            dt_bias,
            mask=mask,
            use_kernel=True,
            gate_lower_bound=-5.0,
        )
        mx.eval(expected_y, expected_state, actual_y, actual_state)
        np.testing.assert_allclose(actual_y, expected_y, rtol=5e-4, atol=5e-4)
        np.testing.assert_allclose(actual_state, expected_state, rtol=5e-4, atol=5e-4)
        np.testing.assert_array_equal(
            np.array(actual_y)[np.logical_not(np.array(mask))],
            0,
        )

    def test_attention_residual_matches_reference(self):
        layer = self.model.layers[0]
        norm = layer.self_attention_res_norm
        projection = layer.self_attention_res_proj
        norm.weight = mx.array([1.0, 0.5, 1.5, 2.0])
        projection.weight = mx.array([[0.25, -0.5, 1.0, 0.75]])

        prefix = mx.array([[[0.5, -1.0, 2.0, 0.25]]])
        blocks = [
            mx.array([[[1.0, 0.0, -0.5, 2.0]]]),
            mx.array([[[-1.0, 1.5, 0.5, -0.25]]]),
        ]
        actual = _apply_attention_residual(prefix, blocks, projection, norm)

        expected = stacked_attention_residual_reference(
            [*blocks, prefix],
            norm.weight * projection.weight.squeeze(0),
            norm.eps,
        )
        np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)

    def test_attention_residual_bfloat16_matches_stacked_reference(self):
        layer = self.model.layers[0]
        norm = layer.self_attention_res_norm
        projection = layer.self_attention_res_proj
        mx.random.seed(7)

        for num_sources in (2, 5, 9):
            sources = [
                mx.random.normal((2, 17, 16)).astype(mx.bfloat16)
                for _ in range(num_sources)
            ]
            score_weight = norm.weight.astype(mx.float32) * projection.weight.squeeze(
                0
            ).astype(mx.float32)
            actual = _attention_residual_streaming(
                sources,
                score_weight,
                mx.array(norm.eps, dtype=mx.float32),
            )
            expected = stacked_attention_residual_reference(
                sources, score_weight, norm.eps
            )
            mx.eval(actual, expected)
            np.testing.assert_allclose(
                actual.astype(mx.float32),
                expected.astype(mx.float32),
                rtol=2e-3,
                atol=1.6e-2,
            )

    def test_attention_residual_streaming_gradients_match_stacked_reference(self):
        mx.random.seed(11)
        sources = [mx.random.normal((2, 3, 16)) for _ in range(5)]
        score_weight = mx.random.normal((16,))
        eps = mx.array(1e-6, dtype=mx.float32)

        def streaming_loss(source_values, weight):
            output = _attention_residual_streaming(source_values, weight, eps)
            return mx.sum(mx.square(output))

        def reference_loss(source_values, weight):
            output = stacked_attention_residual_reference(source_values, weight, eps)
            return mx.sum(mx.square(output))

        actual_sources, actual_weight = mx.grad(streaming_loss, argnums=(0, 1))(
            sources, score_weight
        )
        expected_sources, expected_weight = mx.grad(reference_loss, argnums=(0, 1))(
            sources, score_weight
        )
        mx.eval(
            *actual_sources,
            actual_weight,
            *expected_sources,
            expected_weight,
        )

        for actual, expected in zip(actual_sources, expected_sources):
            np.testing.assert_allclose(actual, expected, rtol=2e-4, atol=1e-4)
        np.testing.assert_allclose(actual_weight, expected_weight, rtol=2e-4, atol=1e-4)

    def test_attention_residual_block_write_order(self):
        layers = self.model.layers[:3]
        for layer in layers:
            layer.input_layernorm = Identity()
            layer.post_attention_layernorm = Identity()
            layer.self_attn = Identity()
            layer.mlp = Identity()
            layer.self_attention_res_proj.weight = mx.zeros((1, 16))
            layer.mlp_res_proj.weight = mx.zeros((1, 16))

        residual_bank = []
        hidden = mx.full((1, 1, 16), 10.0)
        expected_outputs = (20.0, 57.5, 67.5)
        for layer, expected in zip(layers, expected_outputs):
            hidden = layer(hidden, block_residual=residual_bank)
            np.testing.assert_allclose(hidden, expected, rtol=0, atol=1e-5)

        self.assertEqual(len(residual_bank), 2)
        np.testing.assert_allclose(residual_bank[0], 10.0, rtol=0, atol=1e-5)
        np.testing.assert_allclose(residual_bank[1], 57.5, rtol=0, atol=1e-5)

    def test_checkpoint_sanitize(self):
        weights = {
            "vision_tower.block.weight": mx.ones((1,)),
            "mm_projector.proj.weight": mx.ones((1,)),
            "language_model.model.layers.0.self_attn.A_log": mx.arange(
                4, dtype=mx.float32
            ),
            "language_model.model.layers.0.self_attn.q_conv1d.weight": mx.ones(
                (8, 1, 4)
            ),
        }
        for expert in range(2):
            for name in ("w1", "w2", "w3"):
                prefix = (
                    "language_model.model.layers.1.block_sparse_moe."
                    f"experts.{expert}.{name}"
                )
                weights[f"{prefix}.weight_packed"] = mx.arange(
                    32 * 16, dtype=mx.uint8
                ).reshape(32, 16)
                weights[f"{prefix}.weight_scale"] = mx.ones((32, 1), dtype=mx.uint8)
        routed_prefix = "language_model.model.layers.1.block_sparse_moe.routed_expert"
        weights[f"{routed_prefix}_down_proj.weight"] = mx.ones((32, 16))
        weights[f"{routed_prefix}_up_proj.weight"] = mx.ones((16, 32))
        weights[f"{routed_prefix}_norm.weight"] = mx.ones((32,))

        sanitized = self.model.sanitize(weights)
        self.assertNotIn("vision_tower.block.weight", sanitized)
        self.assertNotIn("mm_projector.proj.weight", sanitized)

        expert_prefix = "language_model.model.layers.1.mlp.switch_mlp.gate_proj"
        self.assertEqual(
            sanitized[f"{expert_prefix}.weight"].shape,
            (2, 32, 4),
        )
        self.assertEqual(
            sanitized[f"{expert_prefix}.weight"].dtype,
            mx.uint32,
        )
        self.assertEqual(
            sanitized[f"{expert_prefix}.scales"].shape,
            (2, 32, 1),
        )
        latent_prefix = "language_model.model.layers.1.mlp.routed_expert"
        self.assertEqual(
            sanitized[f"{latent_prefix}_down_proj.weight"].shape,
            (32, 16),
        )
        self.assertEqual(
            sanitized[f"{latent_prefix}_up_proj.weight"].shape,
            (16, 32),
        )
        self.assertEqual(
            sanitized[f"{latent_prefix}_norm.weight"].shape,
            (32,),
        )
        self.assertEqual(
            sanitized["language_model.model.layers.0.self_attn.A_log"].shape,
            (1, 1, 2, 1),
        )
        self.assertEqual(
            sanitized[
                "language_model.model.layers.0.self_attn.q_conv.conv.weight"
            ].shape,
            (8, 4, 1),
        )

        expert_module_prefix = "language_model.model.layers.1.mlp.switch_mlp."
        nn.quantize(
            self.model,
            group_size=32,
            bits=4,
            mode="mxfp4",
            class_predicate=lambda path, module: path.startswith(expert_module_prefix)
            and hasattr(module, "to_quantized"),
        )
        self.model.load_weights(list(sanitized.items()), strict=False)
        switch_mlp = self.model.layers[1].mlp.switch_mlp
        self.assertEqual(switch_mlp.gate_proj.mode, "mxfp4")
        self.assertEqual(switch_mlp.down_proj.mode, "mxfp4")
        output = self.model.layers[1].mlp(mx.ones((1, 1, 16), dtype=mx.bfloat16))
        mx.eval(output)
        self.assertTrue(mx.all(mx.isfinite(output)).item())

    def test_load_model_accepts_base_checkpoint_layout_without_convert(self):
        config = tiny_text_config()
        config["num_hidden_layers"] = 1
        args = KimiK3Args.from_dict({"model_type": "kimi_k3", "text_config": config})
        expected_model = KimiK3(args)
        expected_weights = dict(tree_flatten(expected_model.parameters()))
        base_weights = {}
        for key, value in expected_weights.items():
            if ".self_attn." in key and ".q_conv.conv.weight" in key:
                key = key.replace(".q_conv.conv.weight", ".q_conv1d.weight")
                value = value.moveaxis(1, 2)
            elif ".self_attn." in key and ".k_conv.conv.weight" in key:
                key = key.replace(".k_conv.conv.weight", ".k_conv1d.weight")
                value = value.moveaxis(1, 2)
            elif ".self_attn." in key and ".v_conv.conv.weight" in key:
                key = key.replace(".v_conv.conv.weight", ".v_conv1d.weight")
                value = value.moveaxis(1, 2)
            elif key.endswith(".self_attn.A_log"):
                value = value.reshape(-1)
            base_weights[key] = value

        with tempfile.TemporaryDirectory() as directory:
            model_path = Path(directory)
            (model_path / "config.json").write_text(
                json.dumps(
                    {
                        "model_type": "kimi_k3",
                        "text_config": config,
                    }
                )
            )
            mx.save_safetensors(
                str(model_path / "model.safetensors"),
                base_weights,
            )
            loaded_model, loaded_config = load_model(
                model_path,
                lazy=True,
                strict=True,
            )

        # The production K3 head dimension is 128; this tiny fixture uses 4,
        # so keep both sides on the shape-generic ops recurrence.
        expected_model.train()
        loaded_model.train()
        tokens = mx.array([[1, 2, 3]])
        expected = expected_model(tokens)
        actual = loaded_model(tokens)
        mx.eval(expected, actual)
        self.assertEqual(loaded_config["model_type"], "kimi_k3")
        np.testing.assert_allclose(actual, expected, rtol=0, atol=0)

    def test_load_model_strict_base_checkpoint_with_native_mxfp4(self):
        config = tiny_text_config()
        config["num_hidden_layers"] = 2
        config["moe_intermediate_size"] = 64
        config["linear_attn_config"]["kda_layers"] = [1]
        config["linear_attn_config"]["full_attn_layers"] = [2]
        config["quantization_config"] = {
            "quant_method": "compressed-tensors",
            "format": "mxfp4-pack-quantized",
            "config_groups": {
                "group_0": {
                    "weights": {
                        "group_size": 32,
                        "num_bits": 4,
                    }
                }
            },
        }

        args = KimiK3Args.from_dict({"model_type": "kimi_k3", "text_config": config})
        expected_model = KimiK3(args)
        nn.quantize(
            expected_model,
            group_size=32,
            bits=4,
            mode="mxfp4",
            class_predicate=lambda path, module: ".switch_mlp." in path
            and hasattr(module, "to_quantized"),
        )
        base_weights = dict(tree_flatten(expected_model.parameters()))

        layer_prefix = "language_model.model.layers.1"
        source_prefix = f"{layer_prefix}.block_sparse_moe"
        for source, target in (
            ("w1", "gate_proj"),
            ("w2", "down_proj"),
            ("w3", "up_proj"),
        ):
            target_prefix = f"{layer_prefix}.mlp.switch_mlp.{target}"
            packed = base_weights.pop(f"{target_prefix}.weight")
            scales = base_weights.pop(f"{target_prefix}.scales")
            for expert in range(config["num_experts"]):
                expert_prefix = f"{source_prefix}.experts.{expert}.{source}"
                base_weights[f"{expert_prefix}.weight_packed"] = mx.contiguous(
                    packed[expert]
                ).view(mx.uint8)
                base_weights[f"{expert_prefix}.weight_scale"] = scales[expert]

        for name in ("gate_proj", "up_proj", "down_proj"):
            base_weights[f"{source_prefix}.shared_experts.{name}.weight"] = (
                base_weights.pop(f"{layer_prefix}.mlp.shared_experts.{name}.weight")
            )
        base_weights[f"{source_prefix}.gate.weight"] = base_weights.pop(
            f"{layer_prefix}.mlp.gate.weight"
        )
        base_weights[f"{source_prefix}.gate.e_score_correction_bias"] = (
            base_weights.pop(f"{layer_prefix}.mlp.e_score_correction_bias")
        )
        for name in (
            "routed_expert_down_proj",
            "routed_expert_up_proj",
            "routed_expert_norm",
        ):
            base_weights[f"{source_prefix}.{name}.weight"] = base_weights.pop(
                f"{layer_prefix}.mlp.{name}.weight"
            )

        for layer_idx in range(config["num_hidden_layers"]):
            attention_prefix = f"language_model.model.layers.{layer_idx}.self_attn"
            for name in ("q", "k", "v"):
                key = f"{attention_prefix}.{name}_conv.conv.weight"
                if key in base_weights:
                    base_weights[f"{attention_prefix}.{name}_conv1d.weight"] = (
                        base_weights.pop(key).moveaxis(1, 2)
                    )
            a_log_key = f"{attention_prefix}.A_log"
            if a_log_key in base_weights:
                base_weights[a_log_key] = base_weights[a_log_key].reshape(-1)

        attention_prefix = "language_model.model.layers.1.self_attn"
        embed_q = base_weights.pop(f"{attention_prefix}.embed_q.weight")
        unembed_out = base_weights.pop(f"{attention_prefix}.unembed_out.weight")
        base_weights[f"{attention_prefix}.kv_b_proj.weight"] = mx.concatenate(
            [embed_q.swapaxes(-1, -2), unembed_out],
            axis=1,
        ).reshape(-1, config["kv_lora_rank"])

        with tempfile.TemporaryDirectory() as directory:
            model_path = Path(directory)
            (model_path / "config.json").write_text(
                json.dumps(
                    {
                        "model_type": "kimi_k3",
                        "text_config": config,
                    }
                )
            )
            mx.save_safetensors(
                str(model_path / "model.safetensors"),
                base_weights,
            )
            loaded_model, loaded_config = load_model(
                model_path,
                lazy=True,
                strict=True,
            )

        self.assertEqual(
            loaded_config["quantization"],
            {"group_size": 32, "bits": 4, "mode": "mxfp4"},
        )
        switch_mlp = loaded_model.layers[1].mlp.switch_mlp
        expected_shapes = {
            "gate_proj": ((2, 64, 4), (2, 64, 1)),
            "up_proj": ((2, 64, 4), (2, 64, 1)),
            "down_proj": ((2, 32, 8), (2, 32, 2)),
        }
        for name, (weight_shape, scales_shape) in expected_shapes.items():
            module = getattr(switch_mlp, name)
            self.assertIsInstance(module, QuantizedSwitchLinear)
            self.assertEqual(module.mode, "mxfp4")
            self.assertEqual(module.group_size, 32)
            self.assertEqual(module.bits, 4)
            self.assertEqual(module.weight.shape, weight_shape)
            self.assertEqual(module.weight.dtype, mx.uint32)
            self.assertEqual(module.scales.shape, scales_shape)
            self.assertEqual(module.scales.dtype, mx.uint8)

        # The tiny KDA head dimension uses the shape-generic training path.
        expected_model.train()
        loaded_model.train()
        tokens = mx.array([[1, 2, 3]])
        expected = expected_model(tokens)
        actual = loaded_model(tokens)
        mx.eval(expected, actual)
        np.testing.assert_allclose(actual, expected, rtol=0, atol=0)

    def test_compressed_tensors_selects_native_mxfp4(self):
        config = {
            "format": "mxfp4-pack-quantized",
            "config_groups": {
                "group_0": {
                    "weights": {
                        "group_size": 32,
                        "num_bits": 4,
                    }
                }
            },
        }
        self.assertEqual(
            _compressed_tensors_quantization(config),
            {"group_size": 32, "bits": 4, "mode": "mxfp4"},
        )

    def test_kda_float32_weights_are_not_downcast(self):
        predicate = self.model.cast_predicate
        for suffix in (
            "A_log",
            "dt_bias",
            "q_conv.conv.weight",
            "k_conv.conv.weight",
            "v_conv.conv.weight",
            "o_norm.weight",
            "e_score_correction_bias",
        ):
            self.assertFalse(
                predicate(f"language_model.model.layers.0.self_attn.{suffix}")
            )
        self.assertTrue(
            predicate("language_model.model.layers.0.self_attn.q_proj.weight")
        )

    def test_kda_float32_parameters_preserve_bfloat16_activations(self):
        def to_bfloat16(_, value):
            if mx.issubdtype(value.dtype, mx.floating):
                return value.astype(mx.bfloat16)
            return value

        self.model.update(
            tree_map_with_path(
                to_bfloat16,
                self.model.parameters(),
            )
        )
        for layer in self.model.layers:
            if not layer.is_linear:
                continue
            attention = layer.self_attn
            attention.A_log = attention.A_log.astype(mx.float32)
            attention.dt_bias = attention.dt_bias.astype(mx.float32)
            attention.o_norm.weight = attention.o_norm.weight.astype(mx.float32)
            for convolution in (
                attention.q_conv,
                attention.k_conv,
                attention.v_conv,
            ):
                convolution.conv.weight = convolution.conv.weight.astype(mx.float32)

        cache = self.model.make_cache()
        logits = self.model(mx.array([[1, 2, 3]]), cache)
        mx.eval(logits, cache)
        self.assertEqual(logits.dtype, mx.bfloat16)
        self.assertEqual(cache[0][0].dtype, mx.bfloat16)
        self.assertEqual(cache[0][3].dtype, mx.float32)

    def test_local_pipeline_slices_derive_masks_from_their_cache_types(self):
        inner = self.model.language_model.model
        all_layers = inner.layers

        try:
            inner.layers = all_layers[:3]
            kda_cache = self.model.make_cache()
            kda_output = self.model(mx.array([[1, 2]]), kda_cache)
            mx.eval(kda_output, kda_cache)
            self.assertEqual(len(kda_cache), 3)
            self.assertTrue(all(isinstance(entry, ArraysCache) for entry in kda_cache))

            inner.layers = all_layers[-1:]
            mla_cache = self.model.make_cache()
            mla_output = self.model(mx.array([[1, 2]]), mla_cache)
            mx.eval(mla_output, mla_cache)
            self.assertEqual(len(mla_cache), 1)
            self.assertIsInstance(mla_cache[0], KVCache)
        finally:
            inner.layers = all_layers

    def test_pipeline_runtime_can_finalize_with_call_local_residual_bank(self):
        class FinalizingLayer(nn.Module):
            def __init__(self, layer, finalizer):
                super().__init__()
                self.layer = layer
                self.finalizer_ref = weakref.WeakMethod(finalizer)
                self.is_linear = layer.is_linear
                self.bank_size = None
                self.finalize_calls = 0

            def __call__(self, *args, **kwargs):
                output = self.layer(*args, **kwargs)
                bank = kwargs.get("block_residual")
                if not isinstance(bank, list):
                    raise ValueError("Pipeline finalization requires a residual bank")
                finalizer = self.finalizer_ref()
                if finalizer is None:
                    raise RuntimeError("Pipeline finalizer is no longer available")
                self.bank_size = len(bank)
                self.finalize_calls += 1
                return finalizer(output, bank)

        tokens = mx.array([[1, 2, 3]])
        expected = self.model(tokens)
        inner = self.model.language_model.model
        original_layer = inner.layers[-1]
        finalizing_layer = FinalizingLayer(original_layer, inner.finalize_hidden)

        try:
            inner.layers[-1] = finalizing_layer
            inner.pipeline_managed_finalization = True
            actual = self.model(tokens)
            mx.eval(expected, actual)
        finally:
            inner.pipeline_managed_finalization = False
            inner.layers[-1] = original_layer

        self.assertEqual(finalizing_layer.bank_size, 2)
        self.assertEqual(finalizing_layer.finalize_calls, 1)
        self.assertFalse(hasattr(inner, "attn_residual_bank"))
        np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)

    def test_prefill_matches_cached_decode(self):
        tokens = mx.array([[1, 2, 3]])
        prefill = self.model(tokens)
        cache = self.model.make_cache()
        decode = mx.concatenate(
            [self.model(tokens[:, i : i + 1], cache) for i in range(3)],
            axis=1,
        )
        mx.eval(prefill, decode)
        self.assertEqual(cache[0][3].dtype, mx.float32)
        np.testing.assert_allclose(prefill, decode, rtol=2e-4, atol=2e-4)


if __name__ == "__main__":
    unittest.main()
