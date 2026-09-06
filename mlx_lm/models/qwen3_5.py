# Copyright © 2026 Apple Inc.

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

import mlx.core as mx
import mlx.nn as nn
from mlx.nn.layers.distributed import shard_inplace, shard_linear, sum_gradients
from mlx.utils import tree_map

from .activations import swiglu
from .base import (
    BaseModelArgs,
    create_attention_mask,
    create_ssm_mask,
    scaled_dot_product_attention,
)
from .cache import ArraysCache, KVCache
from .gated_delta import gated_delta_update
from .pipeline import PipelineMixin
from .qmv_small import prep_gate, prep_gated_norm, prep_rms_norm, prep_swiglu, qlinear
from .qwen3_next import Qwen3NextAttention, Qwen3NextMLP
from .qwen3_next import Qwen3NextRMSNormGated as RMSNormGated
from .qwen3_next import Qwen3NextSparseMoeBlock as SparseMoeBlock


@dataclass
class TextModelArgs(BaseModelArgs):
    model_type: str = ""
    hidden_size: int = 4096
    intermediate_size: int = 14336
    num_hidden_layers: int = 32
    num_attention_heads: int = 32
    rms_norm_eps: float = 1e-6
    vocab_size: int = 151936
    num_key_value_heads: int = 8
    max_position_embeddings: int = 131072
    linear_num_value_heads: int = 64
    linear_num_key_heads: int = 16
    linear_key_head_dim: int = 192
    linear_value_head_dim: int = 128
    linear_conv_kernel_dim: int = 4
    tie_word_embeddings: bool = False
    attention_bias: bool = False
    head_dim: Optional[int] = None
    full_attention_interval: int = 4

    # MoE fields (optional, for Qwen3_5MoeForConditionalGeneration)
    num_experts: int = 0
    num_experts_per_tok: int = 0
    decoder_sparse_step: int = 1
    shared_expert_intermediate_size: int = 0
    moe_intermediate_size: int = 0
    norm_topk_prob: bool = True

    # Rope parameters
    rope_parameters: Optional[Dict[str, Union[float, str, bool, List[int]]]] = field(
        default_factory=lambda: {
            "type": "default",
            "mrope_section": [11, 11, 10],
            "rope_theta": 100000,
            "partial_rotary_factor": 0.25,
        }
    )

    # Derived from rope_parameters (set in __post_init__)
    partial_rotary_factor: float = 0.25
    rope_theta: float = 100000.0
    rope_scaling: Optional[Dict[str, Union[float, str]]] = None

    def __post_init__(self):
        if self.head_dim is None:
            self.head_dim = self.hidden_size // self.num_attention_heads

        if self.rope_parameters:
            if (
                "type" not in self.rope_parameters
                and "rope_type" in self.rope_parameters
            ):
                self.rope_parameters["type"] = self.rope_parameters.pop("rope_type")

            self.partial_rotary_factor = self.rope_parameters.get(
                "partial_rotary_factor", 0.25
            )
            self.rope_theta = self.rope_parameters.get("rope_theta", 100000.0)
            self.rope_scaling = self.rope_parameters


class Attention(Qwen3NextAttention):
    """Qwen3Next attention with q (and its output gate), k and v in one projection."""

    def __init__(self, args: TextModelArgs):
        super().__init__(args)
        self.qkv_proj = nn.Linear(
            args.hidden_size,
            2 * (self.num_attention_heads + self.num_key_value_heads) * self.head_dim,
            bias=args.attention_bias,
        )
        # A module is a dict of its children; drop the separate projections
        for name in ("q_proj", "k_proj", "v_proj"):
            self.pop(name)

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        B, L, D = x.shape
        q_dim = 2 * self.num_attention_heads * self.head_dim
        kv_dim = self.num_key_value_heads * self.head_dim

        q_proj_output, keys, values = mx.split(
            qlinear(self.qkv_proj, x), [q_dim, q_dim + kv_dim], axis=-1
        )
        queries, gate = mx.split(
            q_proj_output.reshape(B, L, self.num_attention_heads, -1), 2, axis=-1
        )
        gate = gate.reshape(B, L, -1)

        queries = self.q_norm(queries).transpose(0, 2, 1, 3)
        keys = self.k_norm(keys.reshape(B, L, self.num_key_value_heads, -1)).transpose(
            0, 2, 1, 3
        )
        values = values.reshape(B, L, self.num_key_value_heads, -1).transpose(
            0, 2, 1, 3
        )

        if cache is not None:
            queries = self.rope(queries, offset=cache.offset)
            keys = self.rope(keys, offset=cache.offset)
            keys, values = cache.update_and_fetch(keys, values)
        else:
            queries = self.rope(queries)
            keys = self.rope(keys)

        output = scaled_dot_product_attention(
            queries, keys, values, cache=cache, scale=self.scale, mask=mask
        )
        output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)

        prepped = prep_gate(output, gate, self.o_proj)
        if prepped is not None:
            return qlinear(self.o_proj, prepped)
        return self.o_proj(output * mx.sigmoid(gate))


class MLP(Qwen3NextMLP):
    """Qwen3Next MLP with the gate and up projections fused."""

    def __init__(self, dim, hidden_dim):
        super().__init__(dim, hidden_dim)
        self.gate_up_proj = nn.Linear(dim, 2 * hidden_dim, bias=False)
        for name in ("gate_proj", "up_proj"):
            self.pop(name)

    def __call__(self, x) -> mx.array:
        gate_up = qlinear(self.gate_up_proj, x)
        prepped = prep_swiglu(gate_up, self.down_proj)
        if prepped is not None:
            return qlinear(self.down_proj, prepped)
        gate, up = mx.split(gate_up, 2, axis=-1)
        return self.down_proj(swiglu(gate, up))


class GatedDeltaNet(nn.Module):
    def __init__(self, config: TextModelArgs):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_v_heads = config.linear_num_value_heads
        self.num_k_heads = config.linear_num_key_heads
        self.head_k_dim = config.linear_key_head_dim
        self.head_v_dim = config.linear_value_head_dim
        self.key_dim = self.head_k_dim * self.num_k_heads
        self.value_dim = self.head_v_dim * self.num_v_heads
        if self.num_v_heads % self.num_k_heads != 0:
            raise ValueError(
                f"num_v_heads ({self.num_v_heads}) must be divisible by num_k_heads ({self.num_k_heads})"
            )

        self.conv_kernel_size = config.linear_conv_kernel_dim
        self.layer_norm_epsilon = config.rms_norm_eps

        self.conv_dim = self.key_dim * 2 + self.value_dim
        self.conv1d = nn.Conv1d(
            in_channels=self.conv_dim,
            out_channels=self.conv_dim,
            bias=False,
            kernel_size=self.conv_kernel_size,
            groups=self.conv_dim,
            padding=0,
        )

        # One projection for qkv | z | b | a
        self.in_proj = nn.Linear(
            self.hidden_size,
            2 * self.key_dim + 2 * self.value_dim + 2 * self.num_v_heads,
            bias=False,
        )

        self.dt_bias = mx.ones(self.num_v_heads)

        A = mx.random.uniform(low=0, high=16, shape=(self.num_v_heads,))
        self.A_log = mx.log(A)

        self.norm = RMSNormGated(self.head_v_dim, eps=self.layer_norm_epsilon)

        self.out_proj = nn.Linear(self.value_dim, self.hidden_size, bias=False)

        self.sharding_group = None

    def __call__(
        self,
        inputs: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        B, S, _ = inputs.shape

        if self.sharding_group is not None:
            inputs = sum_gradients(self.sharding_group)(inputs)

        qkv, z, b, a = mx.split(
            qlinear(self.in_proj, inputs),
            [
                self.conv_dim,
                self.conv_dim + self.value_dim,
                self.conv_dim + self.value_dim + self.num_v_heads,
            ],
            axis=-1,
        )
        z = z.reshape(B, S, self.num_v_heads, self.head_v_dim)

        if cache is not None and cache[0] is not None:
            conv_state = cache[0]
        else:
            conv_state = mx.zeros(
                (B, self.conv_kernel_size - 1, self.conv_dim),
                dtype=inputs.dtype,
            )

        if mask is not None:
            qkv = mx.where(mask[..., None], qkv, 0)
        conv_input = mx.concatenate([conv_state, qkv], axis=1)
        if cache is not None:
            n_keep = self.conv_kernel_size - 1
            if cache.lengths is not None:
                ends = mx.clip(cache.lengths, 0, S)
                positions = (ends[:, None] + mx.arange(n_keep))[..., None]
                cache[0] = mx.take_along_axis(conv_input, positions, axis=1)
            else:
                cache[0] = mx.contiguous(conv_input[:, -n_keep:, :])
        conv_out = nn.silu(self.conv1d(conv_input))

        q, k, v = [
            t.reshape(B, S, h, d)
            for t, h, d in zip(
                mx.split(conv_out, [self.key_dim, 2 * self.key_dim], -1),
                [self.num_k_heads, self.num_k_heads, self.num_v_heads],
                [self.head_k_dim, self.head_k_dim, self.head_v_dim],
            )
        ]

        state = cache[1] if cache else None
        # rms_norm adds eps to mean(x^2); FLA's l2norm adds 1e-6 to sum(x^2)
        eps = 1e-6 / self.head_k_dim
        inv_scale = k.shape[-1] ** -0.5
        q = (inv_scale**2) * mx.fast.rms_norm(q, None, eps)
        k = inv_scale * mx.fast.rms_norm(k, None, eps)

        out, state, *states = gated_delta_update(
            q,
            k,
            v,
            a,
            b,
            self.A_log,
            self.dt_bias,
            state,
            mask,
            use_kernel=not self.training,
            return_states=cache is not None and cache.keep_states and S > 1,
        )

        if cache is not None:
            cache[1] = state
            cache.advance(S)
            if states:
                cache.states, cache.conv_input = states[0], conv_input

        prepped = prep_gated_norm(self.norm, out, z, self.out_proj)
        if prepped is not None:
            out = qlinear(self.out_proj, prepped)
        else:
            out = self.out_proj(self.norm(out, z).reshape(B, S, -1))

        if self.sharding_group is not None:
            out = mx.distributed.all_sum(out, group=self.sharding_group)

        return out


class DecoderLayer(nn.Module):
    def __init__(self, args: TextModelArgs, layer_idx: int):
        super().__init__()
        self.is_linear = (layer_idx + 1) % args.full_attention_interval != 0
        if self.is_linear:
            self.linear_attn = GatedDeltaNet(args)
        else:
            self.self_attn = Attention(args)

        self.input_layernorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(
            args.hidden_size, eps=args.rms_norm_eps
        )

        if args.num_experts > 0:
            self.mlp = SparseMoeBlock(args)
        else:
            self.mlp = MLP(args.hidden_size, args.intermediate_size)

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        if self.is_linear:
            xn = prep_rms_norm(self.input_layernorm, x, self.linear_attn.in_proj)
            r = self.linear_attn(xn, mask, cache)
        else:
            xn = prep_rms_norm(self.input_layernorm, x, self.self_attn.qkv_proj)
            r = self.self_attn(xn, mask, cache)
        h = x + r
        hn = prep_rms_norm(
            self.post_attention_layernorm, h, getattr(self.mlp, "gate_up_proj", None)
        )
        return h + self.mlp(hn)


class Qwen3_5TextModel(PipelineMixin, nn.Module):
    def __init__(self, args: TextModelArgs):
        super().__init__()
        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [
            DecoderLayer(args=args, layer_idx=i) for i in range(args.num_hidden_layers)
        ]
        self.norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.ssm_idx = 0
        self.fa_idx = args.full_attention_interval - 1

    def pipeline(self, group):
        super().pipeline(group)
        self.ssm_idx = None
        self.fa_idx = None
        for e, l in enumerate(self.pipeline_layers):
            if self.ssm_idx is None and l.is_linear:
                self.ssm_idx = e
            elif self.fa_idx is None and not l.is_linear:
                self.fa_idx = e
            if self.ssm_idx is not None and self.fa_idx is not None:
                break

    def __call__(
        self,
        inputs: mx.array,
        cache: Optional[Any] = None,
        input_embeddings: Optional[mx.array] = None,
    ) -> mx.array:
        if input_embeddings is not None:
            hidden_states = input_embeddings
        else:
            hidden_states = self.embed_tokens(inputs)

        pipeline_rank = self.pipeline_rank
        pipeline_size = self.pipeline_size

        if cache is None:
            cache = [None] * len(self.pipeline_layers)

        fa_mask = None
        ssm_mask = None
        if self.fa_idx is not None:
            fa_mask = create_attention_mask(hidden_states, cache[self.fa_idx])
        if self.ssm_idx is not None:
            ssm_mask = create_ssm_mask(hidden_states, cache[self.ssm_idx])

        # Receive from the previous process in the pipeline
        if pipeline_rank < pipeline_size - 1:
            hidden_states = mx.distributed.recv_like(hidden_states, (pipeline_rank + 1))

        for layer, c in zip(self.pipeline_layers, cache):
            mask = ssm_mask if layer.is_linear else fa_mask
            hidden_states = layer(hidden_states, mask=mask, cache=c)

        # Send to the next process in the pipeline
        if pipeline_rank != 0:
            hidden_states = mx.distributed.send(
                hidden_states, (pipeline_rank - 1) % pipeline_size
            )
            if cache[-1] is not None:
                if hasattr(cache[-1], "keys"):
                    cache[-1].keys = mx.depends(cache[-1].keys, hidden_states)
                else:
                    cache[-1][0] = mx.depends(cache[-1][0], hidden_states)

        # Broadcast h while keeping it in the graph
        if pipeline_size > 1:
            hidden_states = mx.distributed.all_gather(hidden_states)[
                : hidden_states.shape[0]
            ]

        return self.norm(hidden_states)


# Same-input projections fused along the output rows: (module, fused, parts)
_FUSED_PROJECTIONS = (
    ("linear_attn", "in_proj", ("in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a")),
    ("self_attn", "qkv_proj", ("q_proj", "k_proj", "v_proj")),
    ("mlp", "gate_up_proj", ("gate_proj", "up_proj")),
)


def fuse_projections(weights):
    """Concatenate separate projections into the fused ones, in any key prefix.

    Exact for quantized weights: ``weight``, ``scales`` and ``biases`` are all
    indexed by output row on axis 0.
    """
    for module, fused, parts in _FUSED_PROJECTIONS:
        marker = f".{module}.{parts[0]}."
        for key in [k for k in weights if marker in k]:
            prefix, param = key.split(marker)
            weights[f"{prefix}.{module}.{fused}.{param}"] = mx.concatenate(
                [weights.pop(f"{prefix}.{module}.{part}.{param}") for part in parts],
                axis=0,
            )
    return weights


class TextModel(nn.Module):
    def __init__(self, args: TextModelArgs):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.model = Qwen3_5TextModel(args)
        if not args.tie_word_embeddings:
            self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)

    def __call__(
        self,
        inputs: mx.array,
        cache: Optional[Any] = None,
        input_embeddings: Optional[mx.array] = None,
        return_hidden: bool = False,
    ) -> mx.array:
        hidden = self.model(inputs, cache, input_embeddings=input_embeddings)
        if self.args.tie_word_embeddings:
            out = self.model.embed_tokens.as_linear(hidden)
        else:
            out = qlinear(self.lm_head, hidden)
        return (out, hidden) if return_hidden else out

    @property
    def layers(self):
        return self.model.pipeline_layers

    def make_cache(self):
        return [ArraysCache(size=2) if l.is_linear else KVCache() for l in self.layers]

    def sanitize(self, weights):
        has_unsanitized_conv1d = any(
            "conv1d.weight" in k and v.shape[-1] != 1 for k, v in weights.items()
        )
        weights = {k: v for k, v in weights.items() if "mtp." not in k}

        if self.args.tie_word_embeddings:
            weights.pop("lm_head.weight", None)

        weights = fuse_projections(weights)

        norm_keys = (
            ".input_layernorm.weight",
            ".post_attention_layernorm.weight",
            "model.norm.weight",
            ".q_norm.weight",
            ".k_norm.weight",
        )
        for k, v in weights.items():
            if "conv1d.weight" in k and v.shape[-1] != 1:
                weights[k] = v.moveaxis(2, 1)
            if has_unsanitized_conv1d and any(k.endswith(sfx) for sfx in norm_keys):
                if v.ndim == 1:
                    weights[k] = v + 1.0
        return weights

    @property
    def quant_predicate(self):
        if self.args.num_experts <= 0:
            return None

        def predicate(path, _):
            if path.endswith("mlp.gate") or path.endswith("shared_expert_gate"):
                return {"group_size": 64, "bits": 8}
            return True

        return predicate

    @property
    def cast_predicate(self):
        def predicate(path: str):
            if path.endswith("A_log"):
                return False
            return True

        return predicate


@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str
    text_config: dict

    @classmethod
    def from_dict(cls, params):
        if "text_config" not in params:
            return cls(model_type=params["model_type"], text_config=params)
        return super().from_dict(params)


class Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.language_model = TextModel(TextModelArgs.from_dict(args.text_config))

    def __call__(
        self,
        inputs: mx.array,
        cache=None,
        input_embeddings: Optional[mx.array] = None,
        return_hidden: bool = False,
    ):
        return self.language_model(
            inputs,
            cache=cache,
            input_embeddings=input_embeddings,
            return_hidden=return_hidden,
        )

    @property
    def model(self):
        return self.language_model.model

    def sanitize(self, weights):
        sanitized = {}
        for key, value in weights.items():
            if key.startswith("vision_tower") or key.startswith("model.visual"):
                continue
            if key.startswith("model.visual"):
                continue
            if key.startswith("model.language_model"):
                key = key.replace("model.language_model", "language_model.model")
            elif key.startswith("language_model."):
                pass
            else:
                key = "language_model." + key
            sanitized[key] = value
        return self.language_model.sanitize(sanitized)

    def shard(self, group=None):
        group = group or mx.distributed.init()
        N = group.size()
        rank = group.rank()

        # A sharding factory for the convolution in gated delta net
        def conv_sharding(key_dim):
            return lambda p, w: (0, [key_dim, 2 * key_dim])

        def repeat_kv_inplace(attn):
            # No repeat needed cause we have more heads than nodes
            h = attn.num_key_value_heads
            if N <= h:
                return
            q_dim = 2 * attn.num_attention_heads * attn.head_dim

            # Repeat the k and v rows of the fused projection
            def _repeat(p):
                q, k, v = mx.split(p, [q_dim, q_dim + h * attn.head_dim], axis=0)
                k, v = (
                    mx.repeat(t.reshape(h, -1, *t.shape[1:]), N // h, axis=0).reshape(
                        -1, *t.shape[1:]
                    )
                    for t in (k, v)
                )
                return mx.concatenate([q, k, v], axis=0)

            attn.qkv_proj.update(tree_map(_repeat, attn.qkv_proj.parameters()))

        for layer in self.layers:
            # Linear attention
            if layer.is_linear:
                kd = layer.linear_attn.key_dim
                vd = layer.linear_attn.value_dim
                nv = layer.linear_attn.num_v_heads
                layer.linear_attn.sharding_group = group
                shard_inplace(layer.linear_attn.conv1d, conv_sharding(kd), group=group)
                layer.linear_attn.conv1d.groups //= N
                shard_inplace(
                    layer.linear_attn.in_proj,
                    "all-to-sharded",
                    segments=[
                        kd,
                        2 * kd,
                        2 * kd + vd,
                        2 * kd + 2 * vd,
                        2 * kd + 2 * vd + nv,
                    ],
                    group=group,
                )
                layer.linear_attn.dt_bias = mx.contiguous(
                    mx.split(layer.linear_attn.dt_bias, N)[rank]
                )
                layer.linear_attn.A_log = mx.contiguous(
                    mx.split(layer.linear_attn.A_log, N)[rank]
                )
                shard_inplace(layer.linear_attn.out_proj, "sharded-to-all", group=group)
                layer.linear_attn.num_k_heads //= N
                layer.linear_attn.num_v_heads //= N
                layer.linear_attn.key_dim //= N
                layer.linear_attn.value_dim //= N
                layer.linear_attn.conv_dim //= N

            # Softmax attention
            else:
                attn = layer.self_attn
                attn.o_proj = shard_linear(attn.o_proj, "sharded-to-all", group=group)
                repeat_kv_inplace(attn)
                q_dim = 2 * attn.num_attention_heads * attn.head_dim
                kv_dim = max(attn.num_key_value_heads, N) * attn.head_dim
                shard_inplace(
                    attn.qkv_proj,
                    "all-to-sharded",
                    segments=[q_dim, q_dim + kv_dim],
                    group=group,
                )
                attn.num_attention_heads //= N
                attn.num_key_value_heads = max(1, attn.num_key_value_heads // N)

            # MLP
            if isinstance(layer.mlp, MLP):
                shard_inplace(
                    layer.mlp.gate_up_proj, "all-to-sharded", segments=2, group=group
                )
                layer.mlp.down_proj = shard_linear(
                    layer.mlp.down_proj, "sharded-to-all", group=group
                )

            # MoE
            else:
                layer.mlp.sharding_group = group
                shard_inplace(
                    layer.mlp.shared_expert.gate_proj, "all-to-sharded", group=group
                )
                shard_inplace(
                    layer.mlp.shared_expert.down_proj, "sharded-to-all", group=group
                )
                shard_inplace(
                    layer.mlp.shared_expert.up_proj, "all-to-sharded", group=group
                )
                shard_inplace(
                    layer.mlp.switch_mlp.gate_proj, "all-to-sharded", group=group
                )
                shard_inplace(
                    layer.mlp.switch_mlp.down_proj, "sharded-to-all", group=group
                )
                shard_inplace(
                    layer.mlp.switch_mlp.up_proj, "all-to-sharded", group=group
                )

    @property
    def layers(self):
        return self.language_model.model.pipeline_layers

    def make_cache(self):
        return self.language_model.make_cache()

    @property
    def quant_predicate(self):
        return self.language_model.quant_predicate

    @property
    def cast_predicate(self):
        return self.language_model.cast_predicate
