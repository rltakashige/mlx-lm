# Copyright © 2026 Apple Inc.

"""Kimi-K3 DSpark companion model.

DSpark is a semi-autoregressive speculative decoder.  It consumes hidden
states captured from several layers of the target K3 model, stores their
projected MLA keys/values as context, and predicts a block of tokens from an
anchor followed by mask tokens in one non-causal forward pass.  A small
low-rank Markov head restores left-to-right dependence while sampling the
parallel block.

The checkpoint intentionally has no language-model head.  Callers share the
target model's embedding and output projection after loading this module.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import mlx.core as mx
import mlx.nn as nn

from .activations import swiglu
from .base import BaseModelArgs, scaled_dot_product_attention
from .cache import KVCache, QuantizedKVCache
from .kimi_linear import KimiMLAAttention, KimiMLP
from .mla import MultiLinear
from .rope_utils import initialize_rope


@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str = "k3_dspark"
    vocab_size: int = 163840
    draft_vocab_size: int = 163840
    hidden_size: int = 7168
    intermediate_size: int = 14336
    num_hidden_layers: int = 5
    num_attention_heads: int = 64
    num_key_value_heads: int = 64
    q_lora_rank: int = 1536
    kv_lora_rank: int = 512
    qk_nope_head_dim: int = 128
    qk_rope_head_dim: int = 64
    v_head_dim: int = 128
    max_position_embeddings: int = 1048576
    rms_norm_eps: float = 1e-5
    rope_theta: float = 50000.0
    rope_parameters: Optional[Dict[str, Any]] = None
    rope_scaling: Optional[Dict[str, Any]] = None
    attention_bias: bool = False
    target_hidden_size: int = 7168
    num_target_layers: int = 5
    target_layer_ids: Optional[List[int]] = None
    mask_token_id: int = 163837
    markov_rank: int = 256
    enable_confidence_head: bool = False
    confidence_head_with_markov: bool = False
    sample_from_anchor: bool = True
    dspark_bonus_anchor: bool = False
    mla_use_nope: bool = False
    mla_use_output_gate: bool = False
    mla_use_qk_norm: bool = False
    tie_word_embeddings: bool = False


class K3DSparkAttention(KimiMLAAttention):
    """Dense MLA with YaRN RoPE and non-causal multi-token query support."""

    def __init__(self, args: ModelArgs):
        # This class subclasses the target attention only so Exo's existing K3
        # tensor sharder recognizes its MLA geometry. It intentionally rebuilds
        # the dense draft modules instead of invoking the target constructor.
        nn.Module.__init__(self)
        self.args = args
        self.num_heads = args.num_attention_heads
        self.num_key_value_heads = args.num_key_value_heads
        self.qk_nope_head_dim = args.qk_nope_head_dim
        self.qk_rope_head_dim = args.qk_rope_head_dim
        self.q_head_dim = self.qk_nope_head_dim + self.qk_rope_head_dim
        self.v_head_dim = args.v_head_dim
        self.kv_lora_rank = args.kv_lora_rank
        self.q_lora_rank = args.q_lora_rank
        self.use_output_gate = False
        self.scale = self.q_head_dim**-0.5

        self.q_a_proj = nn.Linear(
            args.hidden_size, args.q_lora_rank, bias=args.attention_bias
        )
        self.q_a_layernorm = nn.RMSNorm(args.q_lora_rank, eps=args.rms_norm_eps)
        self.q_b_proj = nn.Linear(
            args.q_lora_rank,
            args.num_attention_heads * self.q_head_dim,
            bias=False,
        )
        self.kv_a_proj_with_mqa = nn.Linear(
            args.hidden_size,
            args.kv_lora_rank + args.qk_rope_head_dim,
            bias=args.attention_bias,
        )
        self.kv_a_layernorm = nn.RMSNorm(args.kv_lora_rank, eps=args.rms_norm_eps)
        self.embed_q = MultiLinear(
            args.qk_nope_head_dim, args.kv_lora_rank, args.num_attention_heads
        )
        self.unembed_out = MultiLinear(
            args.kv_lora_rank, args.v_head_dim, args.num_attention_heads
        )
        self.o_proj = nn.Linear(
            args.num_attention_heads * args.v_head_dim,
            args.hidden_size,
            bias=False,
        )

        scaling = args.rope_parameters or args.rope_scaling
        if scaling is not None and scaling.get("mscale_all_dim", 0):
            factor = scaling["factor"]
            if factor > 1:
                mscale = 0.1 * scaling["mscale_all_dim"] * math.log(factor) + 1.0
                self.scale *= mscale * mscale
        self.rope = initialize_rope(
            dims=args.qk_rope_head_dim,
            base=args.rope_theta,
            traditional=True,
            max_position_embeddings=args.max_position_embeddings,
            scaling_config=scaling,
        )

    def project_context_kv(
        self,
        context_states: mx.array,
        cache: KVCache,
    ) -> None:
        """Append target-derived latent K/V to this draft layer's cache."""
        if isinstance(cache, QuantizedKVCache):
            raise TypeError("K3 DSpark does not support quantized draft KV caches")
        compressed = self.kv_a_proj_with_mqa(context_states)
        kv_latent, k_pe = mx.split(compressed, [self.kv_lora_rank], axis=-1)
        kv_latent = self.kv_a_layernorm(kv_latent)
        k_pe = k_pe.reshape(
            context_states.shape[0],
            context_states.shape[1],
            1,
            self.qk_rope_head_dim,
        ).transpose(0, 2, 1, 3)
        k_pe = self.rope(k_pe, cache.offset)
        cache.update_and_fetch(mx.expand_dims(kv_latent, axis=1), k_pe)

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[KVCache] = None,
    ) -> mx.array:
        if isinstance(cache, QuantizedKVCache):
            raise TypeError("K3 DSpark does not support quantized draft KV caches")

        batch, length, _ = x.shape
        q = self.q_b_proj(self.q_a_layernorm(self.q_a_proj(x)))
        q = q.reshape(batch, length, self.num_heads, self.q_head_dim)
        q = q.transpose(0, 2, 1, 3)
        q_nope, q_pe = mx.split(q, [self.qk_nope_head_dim], axis=-1)

        compressed = self.kv_a_proj_with_mqa(x)
        kv_latent, k_pe = mx.split(compressed, [self.kv_lora_rank], axis=-1)
        kv_latent = self.kv_a_layernorm(kv_latent)
        k_pe = k_pe.reshape(batch, length, 1, self.qk_rope_head_dim).transpose(
            0, 2, 1, 3
        )

        offset = cache.offset if cache is not None else 0
        q_pe = self.rope(q_pe, offset)
        k_pe = self.rope(k_pe, offset)
        kv_latent = mx.expand_dims(kv_latent, axis=1)
        if cache is not None:
            kv_latent, k_pe = cache.update_and_fetch(kv_latent, k_pe)

        pe_scores = (q_pe * self.scale) @ k_pe.swapaxes(-1, -2)
        if mask is not None:
            pe_scores = mx.where(
                mask,
                pe_scores,
                mx.array(mx.finfo(pe_scores.dtype).min, pe_scores.dtype),
            )

        if length == 1:
            q_nope = self.embed_q(q_nope)
            k = v = kv_latent
        else:
            k = self.embed_q(kv_latent, transpose=False)
            v = self.unembed_out(kv_latent)
        output = scaled_dot_product_attention(
            q_nope,
            k,
            v,
            cache=cache,
            scale=self.scale,
            mask=pe_scores,
        )
        if length == 1:
            output = self.unembed_out(output)
        output = output.transpose(0, 2, 1, 3).reshape(batch, length, -1)
        return self.o_proj(output)


class K3DSparkMLP(KimiMLP):
    def __init__(self, args: ModelArgs):
        # See K3DSparkAttention: retain isinstance compatibility with Exo's K3
        # tensor sharder without constructing the target model's MLP first.
        nn.Module.__init__(self)
        self.gate_proj = nn.Linear(args.hidden_size, args.intermediate_size, bias=False)
        self.up_proj = nn.Linear(args.hidden_size, args.intermediate_size, bias=False)
        self.down_proj = nn.Linear(args.intermediate_size, args.hidden_size, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        return self.down_proj(swiglu(self.gate_proj(x), self.up_proj(x)))


class K3DSparkDecoderLayer(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.self_attn = K3DSparkAttention(args)
        self.mlp = K3DSparkMLP(args)
        self.input_layernorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(
            args.hidden_size, eps=args.rms_norm_eps
        )

    def __call__(
        self,
        x: mx.array,
        *,
        mask: Optional[mx.array] = None,
        cache: Optional[KVCache] = None,
    ) -> mx.array:
        h = x + self.self_attn(self.input_layernorm(x), mask=mask, cache=cache)
        return h + self.mlp(self.post_attention_layernorm(h))


class DSparkMarkovHead(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.markov_w1 = nn.Embedding(args.vocab_size, args.markov_rank)
        self.markov_w2 = nn.Linear(args.markov_rank, args.draft_vocab_size, bias=False)

    def embed(self, token_ids: mx.array) -> mx.array:
        return self.markov_w1(token_ids)

    def bias(self, markov_embed: mx.array) -> mx.array:
        return self.markov_w2(markov_embed)


class AcceptRatePredictor(nn.Module):
    """Predict the conditional acceptance probability for one draft slot."""

    def __init__(self, input_dim: int):
        super().__init__()
        self.proj = nn.Linear(input_dim, 1, bias=True)

    def __call__(self, features: mx.array) -> mx.array:
        return self.proj(features).squeeze(-1)


class Model(nn.Module):
    """The independently loadable DSpark companion.

    ``embed_tokens`` is deliberately assigned only after checkpoint loading,
    because the companion's frozen copy is redundant with the target K3
    embedding.  The target LM head is likewise owned by the Exo strategy.
    """

    def __init__(self, args: ModelArgs):
        super().__init__()
        if not args.sample_from_anchor:
            raise ValueError("K3 DSpark requires sample_from_anchor=true")
        unsupported = [
            name
            for name in (
                "mla_use_nope",
                "mla_use_output_gate",
                "mla_use_qk_norm",
                "dspark_bonus_anchor",
            )
            if getattr(args, name)
        ]
        if unsupported:
            raise ValueError("K3 DSpark does not support " + ", ".join(unsupported))
        if args.q_lora_rank is None:
            raise ValueError("K3 DSpark requires q_lora_rank")
        if args.draft_vocab_size != args.vocab_size:
            raise ValueError("K3 DSpark requires equal target and draft vocabularies")
        if not args.target_layer_ids:
            raise ValueError("K3 DSpark requires target_layer_ids")
        if len(args.target_layer_ids) != args.num_target_layers:
            raise ValueError("K3 DSpark num_target_layers must match target_layer_ids")
        self.args = args
        self.model_type = args.model_type
        self.context_proj = nn.Linear(
            args.target_hidden_size * args.num_target_layers,
            args.hidden_size,
            bias=False,
        )
        self.context_norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.layers = [
            K3DSparkDecoderLayer(args) for _ in range(args.num_hidden_layers)
        ]
        self.final_norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.markov_head = DSparkMarkovHead(args)
        if args.enable_confidence_head and args.confidence_head_with_markov:
            confidence_input_dim = args.hidden_size + args.markov_rank
        else:
            confidence_input_dim = args.hidden_size
        self.confidence_head = (
            AcceptRatePredictor(confidence_input_dim)
            if args.enable_confidence_head
            else None
        )
        self.embed_tokens: Optional[nn.Module] = None

    def make_cache(self) -> list[KVCache]:
        return [KVCache() for _ in self.layers]

    def combine_hidden_states(self, target_hiddens: mx.array) -> mx.array:
        expected = self.args.target_hidden_size * self.args.num_target_layers
        if target_hiddens.shape[-1] != expected:
            raise ValueError(
                f"Expected {expected} concatenated target features, "
                f"received {target_hiddens.shape[-1]}"
            )
        return self.context_norm(self.context_proj(target_hiddens))

    def precompute_context(
        self,
        target_hiddens: mx.array,
        cache: list[KVCache],
    ) -> None:
        if len(cache) != len(self.layers):
            raise ValueError(
                f"Expected {len(self.layers)} draft caches, received {len(cache)}"
            )
        context = self.combine_hidden_states(target_hiddens)
        for layer, layer_cache in zip(self.layers, cache, strict=True):
            layer.self_attn.project_context_kv(context, layer_cache)
            mx.async_eval(layer_cache.state)

    def __call__(
        self,
        input_ids: mx.array,
        cache: Optional[list[KVCache]] = None,
    ) -> mx.array:
        if self.embed_tokens is None:
            raise RuntimeError("K3 DSpark target embedding has not been attached")
        if cache is None:
            cache = self.make_cache()
        if len(cache) != len(self.layers):
            raise ValueError(
                f"Expected {len(self.layers)} draft caches, received {len(cache)}"
            )
        hidden = self.embed_tokens(input_ids)
        # Deliberately no causal mask: all anchor/mask query slots are processed
        # in parallel, exactly as the checkpoint was trained.
        for layer, layer_cache in zip(self.layers, cache, strict=True):
            hidden = layer(hidden, mask=None, cache=layer_cache)
            mx.async_eval(hidden)
        return self.final_norm(hidden)

    def predict_confidence_step(
        self,
        hidden_states: mx.array,
        prev_token_ids: Optional[mx.array] = None,
    ) -> Optional[mx.array]:
        """Return raw conditional-acceptance logits for each proposal slot."""
        if self.confidence_head is None:
            return None
        previous_embeddings = None
        if self.args.confidence_head_with_markov:
            if prev_token_ids is None:
                raise ValueError(
                    "K3 DSpark confidence head requires previous token ids"
                )
            previous_embeddings = self.markov_head.embed(prev_token_ids)
        return self.predict_confidence_from_markov(
            hidden_states,
            previous_embeddings=previous_embeddings,
        )

    def predict_confidence_from_markov(
        self,
        hidden_states: mx.array,
        previous_embeddings: Optional[mx.array] = None,
    ) -> Optional[mx.array]:
        """Predict confidence while reusing an already-looked-up Markov embedding.

        DSpark's causal sampler needs the same previous-token embedding for both
        its confidence and vocabulary heads. Accepting that embedding directly
        avoids a duplicate gather at every selected proposal position while the
        token-id API above remains available to ordinary callers.
        """
        if self.confidence_head is None:
            return None
        features = hidden_states
        if self.args.confidence_head_with_markov:
            if previous_embeddings is None:
                raise ValueError(
                    "K3 DSpark confidence head requires previous-token embeddings"
                )
            previous_embeddings = previous_embeddings.astype(hidden_states.dtype)
            features = mx.concatenate([hidden_states, previous_embeddings], axis=-1)
        return self.confidence_head(features).astype(mx.float32)

    def sanitize(self, weights: Dict[str, mx.array]) -> Dict[str, mx.array]:
        # The checkpoint embedding is an exact frozen target copy and is shared
        # instead of loaded a second time.
        weights = {
            key: value
            for key, value in weights.items()
            if not key.startswith("embed_tokens.")
            and (
                self.args.enable_confidence_head
                or not key.startswith("confidence_head.")
            )
        }
        # The checkpoint stores MLA's latent K/V expansion as one
        # [heads * (qk_nope + v), kv_rank] matrix. MLX absorbs its K half into
        # q (`embed_q`) and keeps the V half as `unembed_out`.
        for layer_idx in range(self.args.num_hidden_layers):
            prefix = f"layers.{layer_idx}.self_attn"
            kv_b_key = f"{prefix}.kv_b_proj.weight"
            if kv_b_key not in weights:
                continue
            value = weights.pop(kv_b_key).reshape(
                self.args.num_attention_heads,
                self.args.qk_nope_head_dim + self.args.v_head_dim,
                self.args.kv_lora_rank,
            )
            weights[f"{prefix}.embed_q.weight"] = mx.contiguous(
                value[:, : self.args.qk_nope_head_dim, :].swapaxes(-1, -2)
            )
            weights[f"{prefix}.unembed_out.weight"] = mx.contiguous(
                value[:, self.args.qk_nope_head_dim :, :]
            )
        return weights
