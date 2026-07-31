# Copyright © 2025 Apple Inc.

from dataclasses import dataclass
from functools import partial
from typing import Any, Dict, List, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn
from mlx.nn.layers.distributed import sum_gradients

from .activations import swiglu
from .base import (
    BaseModelArgs,
    create_attention_mask,
    create_ssm_mask,
    scaled_dot_product_attention,
)
from .cache import ArraysCache, KVCache, QuantizedKVCache
from .gated_delta import gated_delta_update
from .mla import MultiLinear
from .switch_layers import SwitchGLU


@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str
    vocab_size: int
    hidden_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    intermediate_size: int
    head_dim: int
    rope_theta: float
    rms_norm_eps: float
    linear_attn_config: Dict[str, Any]
    model_max_length: int
    num_experts: int
    moe_intermediate_size: int
    kv_lora_rank: int
    rope_scaling: Optional[Dict[str, Any]] = None
    tie_word_embeddings: bool = False
    qk_nope_head_dim: Optional[int] = None
    qk_rope_head_dim: Optional[int] = None
    v_head_dim: Optional[int] = None
    mla_use_nope: bool = False
    num_experts_per_token: int = 1
    num_shared_experts: int = 0
    moe_router_activation_func: str = "sigmoid"
    moe_renormalize: bool = True
    routed_scaling_factor: float = 1.0
    first_k_dense_replace: int = 0
    moe_layer_freq: int = 1
    use_grouped_topk: bool = True
    num_expert_group: int = 1
    topk_group: int = 1
    hidden_act: str = "silu"
    q_lora_rank: Optional[int] = None
    mla_use_output_gate: bool = False
    attn_res_block_size: Optional[int] = None
    routed_expert_hidden_size: Optional[int] = None
    latent_moe_use_norm: bool = False
    activation_situ_beta: Optional[float] = None
    activation_situ_linear_beta: Optional[float] = None

    @classmethod
    def from_dict(cls, params):
        params = dict(params)
        params.setdefault(
            "head_dim", params["hidden_size"] // params["num_attention_heads"]
        )
        params.setdefault("rope_theta", 10000.0)
        params.setdefault(
            "model_max_length", params.get("max_position_embeddings", 4096)
        )
        return super().from_dict(params)


@partial(mx.compile, shapeless=True)
def _situ(gate: mx.array, up: mx.array, beta: float) -> mx.array:
    dtype = gate.dtype
    gate = gate.astype(mx.float32)
    up = up.astype(mx.float32)
    gate = beta * mx.tanh(gate / beta) * mx.sigmoid(gate)
    return (gate * up).astype(dtype)


@partial(mx.compile, shapeless=True)
def _situ_linear(
    gate: mx.array,
    up: mx.array,
    beta: float,
    linear_beta: float,
) -> mx.array:
    dtype = gate.dtype
    gate = gate.astype(mx.float32)
    up = up.astype(mx.float32)
    gate = beta * mx.tanh(gate / beta) * mx.sigmoid(gate)
    up = linear_beta * mx.tanh(up / linear_beta)
    return (gate * up).astype(dtype)


def situ(
    gate: mx.array,
    up: mx.array,
    beta: float = 1.0,
    linear_beta: Optional[float] = None,
) -> mx.array:
    """Fused SiTU-and-multiply with float32 activation intermediates."""
    if linear_beta is None:
        return _situ(gate, up, beta)
    return _situ_linear(gate, up, beta, linear_beta)


class SituGLU(nn.Module):
    def __init__(self, beta: float, linear_beta: Optional[float]):
        super().__init__()
        self.beta = beta
        self.linear_beta = linear_beta

    def __call__(self, up: mx.array, gate: mx.array) -> mx.array:
        return situ(gate, up, self.beta, self.linear_beta)


class KimiMLP(nn.Module):
    def __init__(
        self,
        args: ModelArgs,
        hidden_size: Optional[int] = None,
        intermediate_size: Optional[int] = None,
    ):
        super().__init__()
        dim = hidden_size or args.hidden_size
        hidden = intermediate_size or args.intermediate_size
        self.gate_proj = nn.Linear(dim, hidden, bias=False)
        self.up_proj = nn.Linear(dim, hidden, bias=False)
        self.down_proj = nn.Linear(hidden, dim, bias=False)
        self.hidden_act = args.hidden_act
        self.situ_beta = args.activation_situ_beta or 1.0
        self.situ_linear_beta = args.activation_situ_linear_beta

    def __call__(self, x: mx.array) -> mx.array:
        gate = self.gate_proj(x)
        up = self.up_proj(x)
        if self.hidden_act == "situ":
            activation = situ(gate, up, self.situ_beta, self.situ_linear_beta)
        elif self.hidden_act == "silu":
            activation = swiglu(gate, up)
        else:
            raise ValueError(f"Unsupported Kimi MLP activation '{self.hidden_act}'")
        return self.down_proj(activation)


@mx.compile
def _group_expert_select(
    gates: mx.array,
    bias: Optional[mx.array],
    top_k: int,
    n_group: int,
    topk_group: int,
    routed_scaling_factor: float,
    renormalize: bool,
    score_function: str,
) -> Tuple[mx.array, mx.array]:
    if score_function == "sigmoid":
        scores = mx.sigmoid(gates)
    elif score_function == "softmax":
        scores = mx.softmax(gates, axis=-1, precise=True)
    else:
        raise ValueError(f"Unsupported MoE router activation '{score_function}'")

    orig_scores = scores
    if bias is not None:
        scores = scores + bias.astype(scores.dtype)

    if n_group > 1 and n_group > topk_group:
        scores = mx.unflatten(scores, axis=-1, shape=(n_group, -1))
        group_scores = mx.topk(scores, 2, axis=-1).sum(axis=-1, keepdims=True)
        k = n_group - topk_group
        group_idx = mx.argpartition(group_scores, kth=k - 1, axis=-2)[..., :k, :]
        scores = mx.put_along_axis(
            scores,
            mx.stop_gradient(group_idx),
            mx.array(-mx.inf, dtype=scores.dtype),
            axis=-2,
        )
        scores = mx.flatten(scores, -2, -1)

    inds = mx.argpartition(-scores, kth=top_k - 1, axis=-1)[..., :top_k]
    scores = mx.take_along_axis(orig_scores, inds, axis=-1)

    if top_k > 1 and renormalize:
        denominator = scores.sum(axis=-1, keepdims=True) + 1e-20
        scores = scores / denominator

    return inds, scores * routed_scaling_factor


class KimiSparseMoE(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        hidden = args.hidden_size
        routed_hidden = args.routed_expert_hidden_size or hidden
        experts = args.num_experts
        if experts is None:
            raise ValueError("num_experts must be specified for MoE layers")

        self.gate = nn.Linear(hidden, experts, bias=False)
        activation = None
        if args.hidden_act == "situ":
            activation = SituGLU(
                args.activation_situ_beta or 1.0,
                args.activation_situ_linear_beta,
            )
        elif args.hidden_act != "silu":
            raise ValueError(f"Unsupported Kimi MoE activation '{args.hidden_act}'")
        if activation is None:
            self.switch_mlp = SwitchGLU(
                routed_hidden, args.moe_intermediate_size, experts
            )
        else:
            self.switch_mlp = SwitchGLU(
                routed_hidden,
                args.moe_intermediate_size,
                experts,
                activation=activation,
            )
        self.e_score_correction_bias = mx.zeros((experts,), dtype=mx.float32)
        self.use_latent_moe = args.routed_expert_hidden_size is not None
        if self.use_latent_moe:
            self.routed_expert_down_proj = nn.Linear(hidden, routed_hidden, bias=False)
            self.routed_expert_up_proj = nn.Linear(routed_hidden, hidden, bias=False)
            self.routed_expert_norm = (
                nn.RMSNorm(routed_hidden, eps=args.rms_norm_eps)
                if args.latent_moe_use_norm
                else None
            )

        if args.num_shared_experts:
            shared_hidden = args.moe_intermediate_size * args.num_shared_experts
            self.shared_experts = KimiMLP(args, intermediate_size=shared_hidden)
        else:
            self.shared_experts = None
        self.sharding_group: Optional[mx.distributed.Group] = None

    def _router_logits(self, x: mx.array) -> mx.array:
        if isinstance(self.gate, nn.QQLinear):
            raise TypeError(
                "Kimi-K3 routers do not support activation quantization; "
                "keep the router dense."
            )
        if isinstance(self.gate, nn.QuantizedLinear):
            # The default K3 conversion keeps this accuracy-sensitive router
            # dense. Still support an explicit custom quantization predicate
            # without interpreting the packed integer weight as a dense matrix.
            return self.gate(x).astype(mx.float32)
        return x.astype(mx.float32) @ self.gate.weight.astype(mx.float32).swapaxes(
            -1, -2
        )

    def __call__(self, x: mx.array) -> mx.array:
        if self.sharding_group is not None:
            x = sum_gradients(self.sharding_group)(x)

        identity = x
        scores = self._router_logits(x)
        inds, weights = _group_expert_select(
            scores,
            self.e_score_correction_bias,
            self.args.num_experts_per_token,
            self.args.num_expert_group,
            self.args.topk_group,
            self.args.routed_scaling_factor,
            self.args.moe_renormalize,
            self.args.moe_router_activation_func,
        )
        if self.use_latent_moe:
            x = self.routed_expert_down_proj(x)
        out = self.switch_mlp(x, inds)
        out_dtype = out.dtype
        out = (
            (out.astype(mx.float32) * weights[..., None]).sum(axis=-2).astype(out_dtype)
        )

        shared_out = (
            self.shared_experts(identity) if self.shared_experts is not None else None
        )
        if self.sharding_group is not None:
            if self.use_latent_moe:
                # The routed expert output is normalized before its replicated
                # up projection. Reduce the tensor-parallel partials before
                # that nonlinear normalization. When shared experts are
                # present, reduce both partials with one collective.
                if shared_out is not None:
                    routed_size = out.shape[-1]
                    reduced = mx.distributed.all_sum(
                        mx.concatenate([out, shared_out], axis=-1),
                        group=self.sharding_group,
                    )
                    out, shared_out = mx.split(reduced, [routed_size], axis=-1)
                else:
                    out = mx.distributed.all_sum(
                        out,
                        group=self.sharding_group,
                    )
            else:
                if shared_out is not None:
                    out = out + shared_out
                    shared_out = None
                out = mx.distributed.all_sum(out, group=self.sharding_group)

        if self.use_latent_moe:
            if self.routed_expert_norm is not None:
                out = self.routed_expert_norm(out)
            out = self.routed_expert_up_proj(out)
        if shared_out is not None:
            out = out + shared_out
        return out


class KimiMLAAttention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.num_heads = args.num_attention_heads
        self.num_key_value_heads = args.num_key_value_heads
        self.qk_nope_head_dim = args.qk_nope_head_dim or args.head_dim
        self.qk_rope_head_dim = args.qk_rope_head_dim or 0
        self.q_head_dim = self.qk_nope_head_dim + self.qk_rope_head_dim
        self.v_head_dim = args.v_head_dim or args.head_dim
        self.kv_lora_rank = args.kv_lora_rank
        self.scale = self.q_head_dim**-0.5

        hidden = args.hidden_size
        self.q_lora_rank = args.q_lora_rank
        if self.q_lora_rank is None:
            self.q_proj = nn.Linear(
                hidden, self.num_heads * self.q_head_dim, bias=False
            )
        else:
            self.q_a_proj = nn.Linear(hidden, self.q_lora_rank, bias=False)
            self.q_a_layernorm = nn.RMSNorm(self.q_lora_rank, eps=args.rms_norm_eps)
            self.q_b_proj = nn.Linear(
                self.q_lora_rank,
                self.num_heads * self.q_head_dim,
                bias=False,
            )
        self.kv_a_proj_with_mqa = nn.Linear(
            hidden,
            args.kv_lora_rank + self.qk_rope_head_dim,
            bias=False,
        )
        self.kv_a_layernorm = nn.RMSNorm(args.kv_lora_rank, eps=args.rms_norm_eps)
        self.embed_q = MultiLinear(
            self.qk_nope_head_dim, args.kv_lora_rank, self.num_heads
        )
        self.unembed_out = MultiLinear(
            args.kv_lora_rank, self.v_head_dim, self.num_heads
        )
        self.o_proj = nn.Linear(self.num_heads * self.v_head_dim, hidden, bias=False)
        self.use_output_gate = args.mla_use_output_gate
        if self.use_output_gate:
            self.g_proj = nn.Linear(
                hidden, self.num_heads * self.v_head_dim, bias=False
            )

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        if isinstance(cache, QuantizedKVCache):
            raise TypeError(
                "Kimi-K3 MLA does not support quantized KV caches; omit --kv-bits."
            )

        B, L, _ = x.shape

        if self.q_lora_rank is None:
            q = self.q_proj(x)
        else:
            q = self.q_b_proj(self.q_a_layernorm(self.q_a_proj(x)))
        q = q.reshape(B, L, self.num_heads, self.q_head_dim)
        q = q.transpose(0, 2, 1, 3)
        q_nope, q_pe = mx.split(q, [self.qk_nope_head_dim], axis=-1)

        compressed_kv = self.kv_a_proj_with_mqa(x)
        compressed_kv, k_pe = mx.split(compressed_kv, [self.kv_lora_rank], axis=-1)
        k_pe = k_pe.reshape(B, L, 1, self.qk_rope_head_dim).transpose(0, 2, 1, 3)
        kv_latent = self.kv_a_layernorm(compressed_kv)

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

        if L == 1:
            q_nope = self.embed_q(q_nope)
            k = v = kv_latent
        else:
            k = self.embed_q(kv_latent, transpose=False)
            v = self.unembed_out(kv_latent)

        output = scaled_dot_product_attention(
            q_nope, k, v, cache=cache, scale=self.scale, mask=pe_scores
        )

        if L == 1:
            output = self.unembed_out(output)

        output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
        if self.use_output_gate:
            output = output * mx.sigmoid(self.g_proj(x))
        return self.o_proj(output)


class ShortConv1d(nn.Module):
    def __init__(self, channels: int, kernel_size: int):
        super().__init__()
        self.kernel_size = kernel_size
        self.conv = nn.Conv1d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=kernel_size,
            bias=False,
            groups=channels,
            padding=0,
        )

    def __call__(
        self,
        x: mx.array,
        state: Optional[mx.array],
        mask: Optional[mx.array],
        lengths: Optional[mx.array],
    ) -> Tuple[mx.array, mx.array]:
        if mask is not None:
            x = mx.where(mask[..., None], x, 0)

        if state is None:
            state = mx.zeros(
                (x.shape[0], self.kernel_size - 1, x.shape[-1]), dtype=x.dtype
            )
        conv_input = mx.concatenate([state, x], axis=1)
        out = nn.silu(self.conv(conv_input)).astype(x.dtype)
        n_keep = self.kernel_size - 1
        if lengths is not None:
            ends = mx.clip(lengths, 0, x.shape[1])
            positions = (ends[:, None] + mx.arange(n_keep))[..., None]
            new_state = mx.take_along_axis(conv_input, positions, axis=1)
        else:
            new_state = mx.contiguous(conv_input[:, -n_keep:, :])

        return out, new_state


def _kda_norm_gate(
    out: mx.array,
    gate: mx.array,
    norm: nn.RMSNorm,
    output_dtype: mx.Dtype,
) -> mx.array:
    """Apply K3's output RMSNorm and sigmoid gate with float32 intermediates."""
    normalized = norm(out.astype(mx.float32))
    return (normalized * mx.sigmoid(gate.astype(mx.float32))).astype(output_dtype)


def _kda_normalize_qk(
    q: mx.array,
    k: mx.array,
    head_dim: int,
) -> Tuple[mx.array, mx.array]:
    """Apply KDA's L2 normalization and query scale via RMSNorm."""
    scale = float(head_dim) ** -0.5
    rms_eps = 1e-6 / head_dim
    q = (scale**2) * mx.fast.rms_norm(q, None, rms_eps)
    k = scale * mx.fast.rms_norm(k, None, rms_eps)
    return q, k


class KimiDeltaAttention(nn.Module):
    def __init__(self, args: ModelArgs, layer_idx: int):
        super().__init__()
        cfg = args.linear_attn_config

        self.layer_idx = layer_idx
        self.num_heads = cfg["num_heads"]
        self.head_dim = cfg["head_dim"]
        self.conv_kernel = cfg.get("short_conv_kernel_size", 4)

        self.projection_dim = self.num_heads * self.head_dim
        hidden = args.hidden_size

        self.scale = float(self.head_dim) ** -0.5

        self.q_proj = nn.Linear(hidden, self.projection_dim, bias=False)
        self.k_proj = nn.Linear(hidden, self.projection_dim, bias=False)
        self.v_proj = nn.Linear(hidden, self.projection_dim, bias=False)

        self.q_conv = ShortConv1d(self.projection_dim, self.conv_kernel)
        self.k_conv = ShortConv1d(self.projection_dim, self.conv_kernel)
        self.v_conv = ShortConv1d(self.projection_dim, self.conv_kernel)

        self.f_a_proj = nn.Linear(hidden, self.head_dim, bias=False)
        self.f_b_proj = nn.Linear(self.head_dim, self.projection_dim, bias=False)
        self.b_proj = nn.Linear(hidden, self.num_heads, bias=False)

        self.use_full_rank_gate = cfg.get("use_full_rank_gate", False)
        self.gate_lower_bound = cfg.get("gate_lower_bound")
        if self.gate_lower_bound is not None and not (
            -5.0 <= self.gate_lower_bound < 0.0
        ):
            raise ValueError(
                "KDA gate lower bound must be in [-5.0, 0.0); "
                f"got {self.gate_lower_bound}."
            )
        if self.use_full_rank_gate:
            self.g_proj = nn.Linear(hidden, self.projection_dim, bias=False)
        else:
            self.g_a_proj = nn.Linear(hidden, self.head_dim, bias=False)
            self.g_b_proj = nn.Linear(self.head_dim, self.projection_dim, bias=False)

        self.A_log = mx.expand_dims(
            mx.log(mx.random.uniform(low=1.0, high=16.0, shape=(self.num_heads,))),
            (0, 1, 3),
        )
        self.dt_bias = mx.zeros((self.projection_dim,))

        self.o_norm = nn.RMSNorm(self.head_dim, eps=args.rms_norm_eps)
        self.o_proj = nn.Linear(self.projection_dim, hidden, bias=False)

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        B, T, _ = x.shape
        dtype = x.dtype

        if cache is not None:
            q_state, k_state, v_state, ssm_state = cache
            lengths = cache.lengths
        else:
            q_state = None
            k_state = None
            v_state = None
            ssm_state = None
            lengths = None

        if q_state is None:
            s = mx.zeros((B, self.conv_kernel - 1, self.projection_dim), dtype=dtype)
            q_state = s
            k_state = s
            v_state = s

        q_conv, q_state = self.q_conv(self.q_proj(x), q_state, mask, lengths)
        k_conv, k_state = self.k_conv(self.k_proj(x), k_state, mask, lengths)
        v_conv, v_state = self.v_conv(self.v_proj(x), v_state, mask, lengths)

        if cache is not None:
            cache[0] = q_state
            cache[1] = k_state
            cache[2] = v_state

        q = q_conv.reshape(B, T, self.num_heads, self.head_dim)
        k = k_conv.reshape(B, T, self.num_heads, self.head_dim)
        v = v_conv.reshape(B, T, self.num_heads, self.head_dim)

        q, k = _kda_normalize_qk(q, k, self.head_dim)

        a_logits = self.f_b_proj(self.f_a_proj(x)).reshape(
            B, T, self.num_heads, self.head_dim
        )
        b_logits = self.b_proj(x).reshape(B, T, self.num_heads)

        out, ssm_state = gated_delta_update(
            q,
            k,
            v,
            a_logits,
            b_logits,
            self.A_log.reshape(self.num_heads, 1),
            self.dt_bias.reshape(self.num_heads, self.head_dim),
            state=ssm_state,
            mask=mask,
            use_kernel=not self.training,
            gate_lower_bound=self.gate_lower_bound,
        )

        if cache is not None:
            cache[3] = ssm_state
            cache.advance(T)

        if self.use_full_rank_gate:
            gate = self.g_proj(x)
        else:
            gate = self.g_b_proj(self.g_a_proj(x))
        gate = gate.reshape(B, T, self.num_heads, self.head_dim)
        out = _kda_norm_gate(
            out.reshape(B, T, self.num_heads, self.head_dim),
            gate,
            self.o_norm,
            dtype,
        )
        out = out.reshape(B, T, -1)
        return self.o_proj(out)


@mx.compile
def _attention_residual_streaming(
    sources: List[mx.array],
    score_weight: mx.array,
    eps: mx.array,
) -> mx.array:
    """Apply AttnRes without materializing a stacked float32 residual bank.

    Scores, softmax probabilities, and the weighted sum all use float32. The
    result is cast once to the activation dtype after the accumulation.
    """
    scores = []
    for source in sources:
        source_float = source.astype(mx.float32)
        reciprocal_std = mx.rsqrt(mx.mean(mx.square(source_float), axis=-1) + eps)
        scores.append(mx.sum(source_float * score_weight, axis=-1) * reciprocal_std)

    probabilities = mx.softmax(mx.stack(scores, axis=-1), axis=-1, precise=True)
    output = mx.zeros(sources[0].shape, dtype=mx.float32)
    for index, source in enumerate(sources):
        output = output + (probabilities[..., index, None] * source.astype(mx.float32))
    return output.astype(sources[-1].dtype)


def _apply_attention_residual(
    prefix_sum: mx.array,
    block_residual: List[mx.array],
    projection: nn.Linear,
    norm: nn.RMSNorm,
) -> mx.array:
    """Mix the current residual with block residuals using depth-wise attention."""
    score_weight = norm.weight.astype(mx.float32) * projection.weight.squeeze(0).astype(
        mx.float32
    )
    return _attention_residual_streaming(
        [*block_residual, prefix_sum],
        score_weight,
        mx.array(norm.eps, dtype=mx.float32),
    )


class KimiDecoderLayer(nn.Module):
    def __init__(self, args: ModelArgs, layer_idx: int):
        super().__init__()
        kda_layers = args.linear_attn_config["kda_layers"]
        self.is_linear = (layer_idx + 1) in kda_layers

        if self.is_linear:
            self.self_attn = KimiDeltaAttention(args, layer_idx)
        else:
            self.self_attn = KimiMLAAttention(args)

        if (
            args.num_experts > 0
            and layer_idx >= args.first_k_dense_replace
            and layer_idx % args.moe_layer_freq == 0
        ):
            self.mlp = KimiSparseMoE(args)
        else:
            self.mlp = KimiMLP(args)

        self.input_layernorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(
            args.hidden_size, eps=args.rms_norm_eps
        )
        self.use_attn_residuals = args.attn_res_block_size is not None
        if self.use_attn_residuals:
            self.attn_res_block_size = args.attn_res_block_size
            self.is_block_write_layer = layer_idx % self.attn_res_block_size == 0
            self.self_attention_res_norm = nn.RMSNorm(
                args.hidden_size, eps=args.rms_norm_eps
            )
            self.self_attention_res_proj = nn.Linear(args.hidden_size, 1, bias=False)
            self.mlp_res_norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
            self.mlp_res_proj = nn.Linear(args.hidden_size, 1, bias=False)

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
        block_residual: Optional[List[mx.array]] = None,
    ) -> mx.array:
        attn_cache = None if cache is None else cache
        if self.use_attn_residuals:
            if block_residual is None:
                raise ValueError("Attention residual layers require a residual bank")
            prefix_sum = x
            if block_residual:
                x = _apply_attention_residual(
                    prefix_sum,
                    block_residual,
                    self.self_attention_res_proj,
                    self.self_attention_res_norm,
                )
            if self.is_block_write_layer:
                block_residual.append(prefix_sum)
                prefix_sum = None

            y = self.self_attn(self.input_layernorm(x), mask, attn_cache)
            prefix_sum = y if prefix_sum is None else prefix_sum + y
            x = _apply_attention_residual(
                prefix_sum,
                block_residual,
                self.mlp_res_proj,
                self.mlp_res_norm,
            )
            z = self.mlp(self.post_attention_layernorm(x))
            return prefix_sum + z

        y = self.self_attn(self.input_layernorm(x), mask, attn_cache)
        h = x + y
        z = self.mlp(self.post_attention_layernorm(h))
        return h + z


class KimiLinearModel(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [KimiDecoderLayer(args, i) for i in range(args.num_hidden_layers)]
        self.norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.use_attn_residuals = args.attn_res_block_size is not None
        self.attn_res_block_size = args.attn_res_block_size
        if self.use_attn_residuals:
            self.output_attn_res_norm = nn.RMSNorm(
                args.hidden_size, eps=args.rms_norm_eps
            )
            self.output_attn_res_proj = nn.Linear(args.hidden_size, 1, bias=False)
        # Pipeline runtimes may replace ``layers`` with a local contiguous
        # slice and finalize the hidden state at the terminal boundary.
        self.pipeline_managed_finalization = False

    def finalize_hidden(
        self,
        hidden: mx.array,
        block_residual: Optional[List[mx.array]],
    ) -> mx.array:
        """Apply the global K3 output AttnRes and final norm.

        Pipeline runtimes must call this exactly once, after the last model
        layer. They can set ``pipeline_managed_finalization`` to keep the
        regular model call from applying it on intermediate stages.
        """
        if block_residual is not None:
            hidden = _apply_attention_residual(
                hidden,
                block_residual,
                self.output_attn_res_proj,
                self.output_attn_res_norm,
            )
        return self.norm(hidden)

    def __call__(
        self,
        inputs: mx.array,
        cache: Optional[List[Any]] = None,
    ) -> mx.array:
        h = self.embed_tokens(inputs)
        if cache is None:
            cache = [None] * len(self.layers)

        if len(cache) != len(self.layers):
            raise ValueError(
                f"Expected {len(self.layers)} cache entries, received {len(cache)}"
            )

        layer_caches = list(zip(self.layers, cache))
        first_ssm_cache = next(
            (layer_cache for layer, layer_cache in layer_caches if layer.is_linear),
            None,
        )
        first_attn_cache = next(
            (layer_cache for layer, layer_cache in layer_caches if not layer.is_linear),
            None,
        )
        has_ssm = any(layer.is_linear for layer in self.layers)
        has_attention = any(not layer.is_linear for layer in self.layers)
        ssm_mask = create_ssm_mask(h, first_ssm_cache) if has_ssm else None
        attn_mask = (
            create_attention_mask(h, first_attn_cache, return_array=True)
            if has_attention
            else None
        )
        block_residual: Optional[List[mx.array]] = (
            [] if self.use_attn_residuals else None
        )

        for layer, layer_cache in layer_caches:
            mask = ssm_mask if layer.is_linear else attn_mask
            h = layer(
                h,
                mask=mask,
                cache=layer_cache,
                block_residual=block_residual,
            )

        if self.pipeline_managed_finalization:
            return h
        return self.finalize_hidden(h, block_residual)


class Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.model = KimiLinearModel(args)
        if args.tie_word_embeddings:
            self.lm_head = None
        else:
            self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)

    def __call__(
        self,
        inputs: mx.array,
        cache: Optional[List[Any]] = None,
    ) -> mx.array:
        out = self.model(inputs, cache)
        if self.lm_head is None:
            return self.model.embed_tokens.as_linear(out)
        return self.lm_head(out)

    @property
    def layers(self):
        return self.model.layers

    def make_cache(self):
        caches: List[Any] = []
        for layer in self.layers:
            if layer.is_linear:
                caches.append(ArraysCache(size=4))
            else:
                caches.append(KVCache())
        return caches

    def sanitize(self, weights: Dict[str, mx.array]) -> Dict[str, mx.array]:
        weights = {k: v for k, v in weights.items() if not k.startswith("model.mtp")}

        if self.args.tie_word_embeddings:
            weights.pop("lm_head.weight", None)

        for layer_idx, layer in enumerate(self.layers):
            prefix = f"model.layers.{layer_idx}"

            if isinstance(layer.mlp, KimiSparseMoE):
                src_prefix = f"{prefix}.block_sparse_moe"
                dst_prefix = f"{prefix}.mlp"
                for src, dst in [
                    ("w1", "gate_proj"),
                    ("w2", "down_proj"),
                    ("w3", "up_proj"),
                ]:
                    weight_key = f"{src_prefix}.experts.0.{src}.weight"
                    packed_key = f"{src_prefix}.experts.0.{src}.weight_packed"
                    if weight_key in weights:
                        stacked = [
                            weights.pop(f"{src_prefix}.experts.{i}.{src}.weight")
                            for i in range(self.args.num_experts)
                        ]
                        weights[f"{dst_prefix}.switch_mlp.{dst}.weight"] = mx.stack(
                            stacked
                        )
                    elif packed_key in weights:
                        packed_weights = []
                        scales = []
                        for i in range(self.args.num_experts):
                            expert_prefix = f"{src_prefix}.experts.{i}.{src}"
                            packed = mx.contiguous(
                                weights.pop(f"{expert_prefix}.weight_packed").astype(
                                    mx.uint8
                                )
                            )
                            if packed.shape[-1] % 4:
                                raise ValueError(
                                    f"Invalid MXFP4 packed width for {expert_prefix}"
                                )
                            packed_weights.append(
                                packed.view(mx.uint32).reshape(
                                    *packed.shape[:-1], packed.shape[-1] // 4
                                )
                            )
                            scales.append(
                                weights.pop(f"{expert_prefix}.weight_scale").astype(
                                    mx.uint8
                                )
                            )
                            weights.pop(f"{expert_prefix}.weight_shape", None)
                        weights[f"{dst_prefix}.switch_mlp.{dst}.weight"] = mx.stack(
                            packed_weights
                        )
                        weights[f"{dst_prefix}.switch_mlp.{dst}.scales"] = mx.stack(
                            scales
                        )

                for name in ("gate_proj", "up_proj", "down_proj"):
                    src_key = f"{src_prefix}.shared_experts.{name}.weight"
                    if src_key in weights:
                        weights[f"{dst_prefix}.shared_experts.{name}.weight"] = (
                            weights.pop(src_key)
                        )

                gate_key = f"{src_prefix}.gate.weight"
                if gate_key in weights:
                    weights[f"{dst_prefix}.gate.weight"] = weights.pop(gate_key)

                bias_key = f"{src_prefix}.gate.e_score_correction_bias"
                if bias_key in weights:
                    weights[f"{dst_prefix}.e_score_correction_bias"] = weights.pop(
                        bias_key
                    )

                for name in (
                    "routed_expert_down_proj",
                    "routed_expert_up_proj",
                    "routed_expert_norm",
                ):
                    src_key = f"{src_prefix}.{name}.weight"
                    if src_key in weights:
                        weights[f"{dst_prefix}.{name}.weight"] = weights.pop(src_key)

            attn = getattr(layer, "self_attn", None)
            if isinstance(attn, KimiDeltaAttention):
                attn_prefix = f"{prefix}.self_attn"
                for src_name, dst_name in (
                    ("q_conv1d", "q_conv"),
                    ("k_conv1d", "k_conv"),
                    ("v_conv1d", "v_conv"),
                ):
                    src_key = f"{attn_prefix}.{src_name}.weight"
                    if src_key in weights:
                        w = weights.pop(src_key)
                        if w.ndim == 3:
                            w = w.moveaxis(2, 1)
                        weights[f"{attn_prefix}.{dst_name}.conv.weight"] = w
                dt_key = f"{attn_prefix}.dt_bias"
                if dt_key in weights:
                    if weights[dt_key].ndim > 1:
                        weights[dt_key] = mx.reshape(weights[dt_key], (-1,))
                a_log_key = f"{attn_prefix}.A_log"
                if a_log_key in weights:
                    a_log = weights[a_log_key].reshape(-1)[: attn.num_heads]
                    weights[a_log_key] = a_log.reshape(1, 1, attn.num_heads, 1)

            attn_prefix = f"{prefix}.self_attn"
            kv_b_key = f"{attn_prefix}.kv_b_proj.weight"
            if kv_b_key in weights:
                qk_nope = self.args.qk_nope_head_dim or self.args.head_dim
                v_head = self.args.v_head_dim or self.args.head_dim
                head_dim = qk_nope + v_head
                num_heads = self.args.num_attention_heads

                quantized = f"{attn_prefix}.kv_b_proj.scales" in weights
                v = weights.pop(kv_b_key)

                if quantized:
                    dims = self.args.kv_lora_rank
                    scales = weights.pop(f"{attn_prefix}.kv_b_proj.scales")
                    biases = weights.pop(f"{attn_prefix}.kv_b_proj.biases")
                    bits = (v.shape[-1] * 32) // dims
                    group_size = dims // scales.shape[-1]
                    v = mx.dequantize(
                        v, scales, biases, bits=bits, group_size=group_size
                    )

                v = v.reshape(num_heads, head_dim, -1)
                wk = mx.contiguous(v[:, :qk_nope, :].swapaxes(-1, -2))
                wv = mx.contiguous(v[:, qk_nope:, :])

                if quantized:
                    wk, wk_s, wk_b = mx.quantize(wk, bits=bits, group_size=group_size)
                    wv, wv_s, wv_b = mx.quantize(wv, bits=bits, group_size=group_size)
                    weights[f"{attn_prefix}.embed_q.scales"] = wk_s
                    weights[f"{attn_prefix}.embed_q.biases"] = wk_b
                    weights[f"{attn_prefix}.unembed_out.scales"] = wv_s
                    weights[f"{attn_prefix}.unembed_out.biases"] = wv_b

                weights[f"{attn_prefix}.embed_q.weight"] = wk
                weights[f"{attn_prefix}.unembed_out.weight"] = wv

        return weights

    @property
    def cast_predicate(self):
        def predicate(path: str):
            if "e_score_correction_bias" in path:
                return False
            if path.endswith(
                (
                    "A_log",
                    "dt_bias",
                    "q_conv.conv.weight",
                    "k_conv.conv.weight",
                    "v_conv.conv.weight",
                    "o_norm.weight",
                )
            ):
                return False
            return True

        return predicate

    @property
    def quant_predicate(self):
        def predicate(path, _):
            if path.endswith("mlp.gate"):
                # Router logits are deliberately accumulated in float32 from
                # the dense BF16 weight. Quantizing this module would replace
                # ``weight`` with packed integers, while ``_router_logits``
                # reads that parameter directly instead of calling the module.
                return False
            return True

        return predicate
