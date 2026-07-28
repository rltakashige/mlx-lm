import unittest

import mlx.core as mx
from mlx.utils import tree_flatten

from mlx_lm.models import k3_dspark


def tiny_args(*, confidence: bool) -> k3_dspark.ModelArgs:
    return k3_dspark.ModelArgs(
        vocab_size=11,
        draft_vocab_size=11,
        hidden_size=4,
        intermediate_size=8,
        num_hidden_layers=1,
        num_attention_heads=1,
        num_key_value_heads=1,
        q_lora_rank=4,
        kv_lora_rank=2,
        qk_nope_head_dim=2,
        qk_rope_head_dim=2,
        v_head_dim=2,
        max_position_embeddings=32,
        target_hidden_size=4,
        num_target_layers=1,
        target_layer_ids=[0],
        mask_token_id=10,
        markov_rank=2,
        enable_confidence_head=confidence,
        confidence_head_with_markov=confidence,
    )


class TestK3DSparkConfidenceHead(unittest.TestCase):
    def test_strict_load_and_previous_token_feature_order(self):
        model = k3_dspark.Model(tiny_args(confidence=True))
        weights = dict(tree_flatten(model.parameters()))
        weights["confidence_head.proj.weight"] = mx.array(
            [[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]],
            dtype=mx.float32,
        )
        weights["confidence_head.proj.bias"] = mx.array(
            [0.5],
            dtype=mx.float32,
        )
        markov_embeddings = mx.zeros((11, 2), dtype=mx.float32)
        markov_embeddings[3] = mx.array([7.0, 8.0])
        weights["markov_head.markov_w1.weight"] = markov_embeddings
        weights["embed_tokens.weight"] = mx.zeros((11, 4))

        sanitized = model.sanitize(weights)
        self.assertNotIn("embed_tokens.weight", sanitized)
        self.assertIn("confidence_head.proj.weight", sanitized)
        self.assertIn("confidence_head.proj.bias", sanitized)
        model.load_weights(list(sanitized.items()), strict=True)

        hidden = mx.array([[[1.0, 2.0, 3.0, 4.0]]])
        logits = model.predict_confidence_step(
            hidden,
            prev_token_ids=mx.array([[3]]),
        )
        self.assertIsNotNone(logits)
        mx.eval(logits)
        expected = 1.0 + 4.0 + 9.0 + 16.0 + 35.0 + 48.0 + 0.5
        self.assertAlmostEqual(float(logits.item()), expected, places=5)
        self.assertEqual(logits.dtype, mx.float32)

    def test_disabled_model_drops_unused_confidence_weights(self):
        model = k3_dspark.Model(tiny_args(confidence=False))
        sanitized = model.sanitize(
            {
                "confidence_head.proj.weight": mx.zeros((1, 4)),
                "confidence_head.proj.bias": mx.zeros((1,)),
                "embed_tokens.weight": mx.zeros((11, 4)),
            }
        )
        self.assertEqual(sanitized, {})


if __name__ == "__main__":
    unittest.main()
