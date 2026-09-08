# Copyright © 2026 Apple Inc.

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

import mlx.core as mx
import mlx.nn as nn
from mlx.nn.layers.distributed import shard_inplace, shard_linear, sum_gradients
from mlx.utils import tree_map

from . import fused_ops, moe_small
from .activations import swiglu
from .base import (
    BaseModelArgs,
    create_attention_mask,
    create_sibling_mask,
    create_ssm_mask,
    scaled_dot_product_attention,
)
from .cache import ArraysCache, KVCache
from .fused_ops import prep_add_rms_norm
from .gated_delta import gated_delta_kernel, gated_delta_update
from .pipeline import PipelineMixin
from .qmv_small import prep_gate, prep_gated_norm, prep_swiglu, qlinear
from .qwen3_next import Qwen3NextAttention, Qwen3NextMLP
from .qwen3_next import Qwen3NextRMSNormGated as RMSNormGated
from .qwen3_next import Qwen3NextSparseMoeBlock
from .switch_layers import SwitchLinear, _gather_sort, _scatter_unsort


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
        chain: Optional[int] = None,
    ) -> mx.array:
        """``chain``: the rows past the first ``chain`` rows are siblings of chain
        rows 1.. (same positions), see ``create_sibling_mask``."""
        B, L, D = x.shape
        q_dim = 2 * self.num_attention_heads * self.head_dim
        kv_dim = self.num_key_value_heads * self.head_dim

        qkv = qlinear(self.qkv_proj, x)
        fused = not self.training and fused_ops.attn_ok(self, qkv, cache)
        if fused:
            # One kernel: the q/k norms, the rope and the head layouts
            queries, keys, values = fused_ops.attn_qkv(self, qkv, cache.offset, chain)
            keys, values = cache.update_and_fetch(keys, values)
        else:
            q_proj_output, keys, values = mx.split(qkv, [q_dim, q_dim + kv_dim], axis=-1)
            queries, gate = mx.split(
                q_proj_output.reshape(B, L, self.num_attention_heads, -1), 2, axis=-1
            )
            gate = gate.reshape(B, L, -1)

            queries = self.q_norm(queries).transpose(0, 2, 1, 3)
            keys = self.k_norm(
                keys.reshape(B, L, self.num_key_value_heads, -1)
            ).transpose(0, 2, 1, 3)
            values = values.reshape(B, L, self.num_key_value_heads, -1).transpose(
                0, 2, 1, 3
            )

            offset = cache.offset if cache is not None else 0
            if chain is not None:
                # The sibling rows take the positions of chain rows 1..
                queries, keys = (
                    mx.concatenate(
                        [
                            self.rope(t[:, :, :chain], offset=offset),
                            self.rope(t[:, :, chain:], offset=offset + 1),
                        ],
                        axis=2,
                    )
                    for t in (queries, keys)
                )
            else:
                queries = self.rope(queries, offset=offset)
                keys = self.rope(keys, offset=offset)
            if cache is not None:
                keys, values = cache.update_and_fetch(keys, values)

        output = scaled_dot_product_attention(
            queries, keys, values, cache=cache, scale=self.scale, mask=mask
        )
        if fused:
            # The gate is read in place from the projection rows
            qkv2 = qkv.reshape(B * L, -1)
            layout = (L, self.num_attention_heads, self.head_dim, qkv2.shape[-1])
            prepped = prep_gate(output, qkv2, self.o_proj, layout)
            if prepped is not None:
                return qlinear(self.o_proj, prepped)
            return self.o_proj(fused_ops.attn_gate(output, qkv2))
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


class FusedSwitchGLU(nn.Module):
    """SwitchGLU with the gate and up projections fused along the output rows."""

    def __init__(self, input_dims, hidden_dims, num_experts):
        super().__init__()
        self.gate_up_proj = SwitchLinear(
            input_dims, 2 * hidden_dims, num_experts, bias=False
        )
        self.down_proj = SwitchLinear(hidden_dims, input_dims, num_experts, bias=False)

    def __call__(self, x, indices):
        x = mx.expand_dims(x, (-2, -3))
        # With many tokens, sort them so the experts are accessed in order
        do_sort = indices.size >= 64
        idx, inv_order = indices, None
        if do_sort:
            x, idx, inv_order = _gather_sort(x, indices)
        gate, up = mx.split(
            self.gate_up_proj(x, idx, sorted_indices=do_sort), 2, axis=-1
        )
        x = self.down_proj(swiglu(gate, up), idx, sorted_indices=do_sort)
        if do_sort:
            x = _scatter_unsort(x, inv_order, indices.shape)
        return x.squeeze(-2)


class SparseMoeBlock(nn.Module):
    """Qwen3Next MoE block with fused expert weights.

    The shared expert's gate is the last row of ``gate`` and the shared expert is the
    last expert of ``switch_mlp``, so one gather step computes every expert of a token.
    """

    def __init__(self, args: TextModelArgs):
        super().__init__()
        dim = args.hidden_size
        self.norm_topk_prob = args.norm_topk_prob
        self.num_experts = args.num_experts
        self.top_k = args.num_experts_per_tok
        self.gate = nn.Linear(dim, args.num_experts + 1, bias=False)
        self.switch_mlp = FusedSwitchGLU(
            dim, args.moe_intermediate_size, args.num_experts + 1
        )
        self.sharding_group = None

    def __call__(self, x: mx.array, slots: bool = False) -> mx.array:
        """With ``slots`` the top_k + 1 expert outputs of a token may come back unsummed
        (.., S, K); the consumer then sums them."""
        if self.sharding_group is not None:
            x = sum_gradients(self.sharding_group)(x)

        E, k = self.num_experts, self.top_k
        logits = self.gate(x)
        inds = mx.argpartition(logits[..., :E], kth=-k, axis=-1)[..., -k:]
        if moe_small.routes(self, x):
            y = moe_small.experts(
                self, x, logits, inds, slots and self.sharding_group is None
            )
        else:
            if self.norm_topk_prob:
                top = mx.take_along_axis(logits, inds, axis=-1)
                scores = mx.softmax(top, axis=-1, precise=True)
            else:
                probs = mx.softmax(logits[..., :E], axis=-1, precise=True)
                scores = mx.take_along_axis(probs, inds, axis=-1)
            scores = mx.concatenate([scores, mx.sigmoid(logits[..., E:])], axis=-1)
            shared = mx.full(inds.shape[:-1] + (1,), E, inds.dtype)
            y = self.switch_mlp(x, mx.concatenate([inds, shared], axis=-1))
            y = (y * scores[..., None]).sum(axis=-2)

        if self.sharding_group is not None:
            y = mx.distributed.all_sum(y, group=self.sharding_group)
        return y


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
        chain: Optional[int] = None,
    ) -> mx.array:
        """``chain``: the rows past the first ``chain`` rows are siblings of chain
        rows 1..: each is one step of the conv and the recurrence from the state
        before its chain row."""
        B, S, _ = inputs.shape

        if self.sharding_group is not None:
            inputs = sum_gradients(self.sharding_group)(inputs)

        proj = qlinear(self.in_proj, inputs)
        fused = not self.training and fused_ops.gdn_in_ok(self, proj, mask, cache)
        if fused:
            out, z, z_off = self._mixer_fused(proj, cache, chain)
        else:
            out, z = self._mixer(proj, mask, cache, chain)
            z_off = 0

        gate = z if fused else z.reshape(B * S, -1)
        prepped = prep_gated_norm(self.norm, out, gate, self.out_proj, z_off)
        if prepped is not None:
            out = qlinear(self.out_proj, prepped)
        elif fused:
            out = self.out_proj(fused_ops.gated_norm(self.norm, out, gate, z_off))
        else:
            out = self.out_proj(self.norm(out, z).reshape(B, S, -1))

        if self.sharding_group is not None:
            out = mx.distributed.all_sum(out, group=self.sharding_group)

        return out

    def _mixer(self, proj, mask, cache, chain=None):
        """conv, norms and the recurrence with MLX ops; returns (out, z (B, S, Hv, Dv))."""
        B, S, _ = proj.shape
        qkv, z, b, a = mx.split(
            proj,
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
                dtype=proj.dtype,
            )

        if mask is not None:
            qkv = mx.where(mask[..., None], qkv, 0)
        conv_input = mx.concatenate([conv_state, qkv], axis=1)
        n_keep = self.conv_kernel_size - 1
        if cache is not None:
            if cache.lengths is not None:
                ends = mx.clip(cache.lengths, 0, S)
                positions = (ends[:, None] + mx.arange(n_keep))[..., None]
                cache[0] = mx.take_along_axis(conv_input, positions, axis=1)
            else:
                rows = S if chain is None else chain
                cache[0] = mx.contiguous(conv_input[:, rows : rows + n_keep, :])
        if chain is None:
            conv_out = nn.silu(self.conv1d(conv_input))
        else:
            # A sibling's conv window is the chain prefix of its position and itself
            windows = [
                mx.concatenate(
                    [conv_input[:, i : i + n_keep], qkv[:, chain - 1 + i : chain + i]],
                    axis=1,
                )
                for i in range(1, S - chain + 1)
            ]
            sib = self.conv1d(mx.concatenate(windows, axis=0)).reshape(B, -1, qkv.shape[-1])
            chain_out = self.conv1d(conv_input[:, : chain + n_keep])
            conv_out = nn.silu(mx.concatenate([chain_out, sib], axis=1))

        q, k, v = [
            t.reshape(B, S, h, d)
            for t, h, d in zip(
                mx.split(conv_out, [self.key_dim, 2 * self.key_dim], -1),
                [self.num_k_heads, self.num_k_heads, self.num_v_heads],
                [self.head_k_dim, self.head_k_dim, self.head_v_dim],
            )
        ]

        state_in = cache[1] if cache else None
        # rms_norm adds eps to mean(x^2); FLA's l2norm adds 1e-6 to sum(x^2)
        eps = 1e-6 / self.head_k_dim
        inv_scale = k.shape[-1] ** -0.5
        q = (inv_scale**2) * mx.fast.rms_norm(q, None, eps)
        k = inv_scale * mx.fast.rms_norm(k, None, eps)

        def update(steps=None, extra=None):
            return gated_delta_update(
                q,
                k,
                v,
                a,
                b,
                self.A_log,
                self.dt_bias,
                state_in,
                mask,
                use_kernel=not self.training,
                steps=steps,
                chain=chain if steps is None else None,
                extra=extra,
            )

        out, state = update()

        if cache is not None:
            cache[1] = state
            cache.advance(S)
            if cache.keep_states and S > 1:
                self._keep_rollback(cache, update, conv_input, S)
        return out, z

    def _keep_rollback(self, cache, update, conv_input, S):
        """Let the cache rebuild the state after the first ``steps`` rows of this
        forward, plus the row ``extra`` (a sibling row, -1 for none) after them."""
        n_keep = self.conv_kernel_size - 1

        def rollback(steps, extra=None):
            kept = steps if extra is None else steps + (extra >= 0)
            # The conv window: the last rows of [conv state; the kept rows]
            pos = kept - n_keep + mx.arange(n_keep)
            src = n_keep + (pos if extra is None else mx.where(pos < steps, pos, extra))
            return update(steps=steps, extra=extra)[1], mx.take(conv_input, src, axis=1)

        cache.rollback, cache.steps, cache.conv_input = rollback, S, conv_input

    def _mixer_fused(self, proj, cache, chain=None):
        """The same with the small ops merged; returns (out, proj rows, column of z)."""
        B, S, _ = proj.shape
        conv_state = cache[0]
        if conv_state is None:
            conv_state = mx.zeros(
                (B, self.conv_kernel_size - 1, self.conv_dim), dtype=proj.dtype
            )
        q, k, v, g, beta, cache[0] = fused_ops.gdn_in(self, proj, conv_state, chain)
        state = cache[1]
        if state is None:
            state = mx.zeros(
                (B, self.num_v_heads, self.head_v_dim, self.head_k_dim), mx.float32
            )

        def update(steps=None, extra=None):
            return gated_delta_kernel(
                q,
                k,
                v,
                g,
                beta,
                state,
                steps=steps,
                chain=chain if steps is None else None,
                extra=extra,
            )

        out, cache[1] = update()
        cache.advance(S)
        if cache.keep_states and S > 1:
            # Only evaluated when a partial acceptance trims the conv window
            conv_input = mx.concatenate([conv_state, proj[..., : self.conv_dim]], axis=1)
            self._keep_rollback(cache, update, conv_input, S)
        # z is read in place from the projection rows
        return out, proj.reshape(B * S, -1), self.conv_dim


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

        if args.num_experts <= 0:
            self.mlp = MLP(args.hidden_size, args.intermediate_size)
        elif args.shared_expert_intermediate_size == args.moe_intermediate_size:
            self.mlp = SparseMoeBlock(args)
        else:
            self.mlp = Qwen3NextSparseMoeBlock(args)

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
        pending: Optional[mx.array] = None,
        split: bool = False,
        chain: Optional[int] = None,
    ):
        """``pending`` is an output not yet added to ``x`` (the add is merged into the
        norm); with ``split`` the result is returned as such a pair (h, mlp output)."""
        fused = not self.training
        if self.is_linear:
            # A sharded GDN applies sum_gradients to its input, which needs an array.
            proj = None if self.linear_attn.sharding_group else self.linear_attn.in_proj
            x, xn = prep_add_rms_norm(self.input_layernorm, x, pending, proj, fused)
            r = self.linear_attn(xn, mask, cache, chain)
        else:
            x, xn = prep_add_rms_norm(
                self.input_layernorm, x, pending, self.self_attn.qkv_proj, fused
            )
            r = self.self_attn(xn, mask, cache, chain)
        h, hn = prep_add_rms_norm(
            self.post_attention_layernorm,
            x,
            r,
            getattr(self.mlp, "gate_up_proj", None),
            fused,
        )
        if split and fused and isinstance(self.mlp, SparseMoeBlock):
            m = self.mlp(hn, slots=True)
        else:
            m = self.mlp(hn)
        return (h, m) if split else h + m


class Qwen3_5TextModel(PipelineMixin, nn.Module):
    # Layers per mx.async_eval while generating (0 disables)
    eval_every = 8

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
        chain: Optional[int] = None,
    ) -> mx.array:
        """``chain``: the rows past the first ``chain`` rows are siblings of chain
        rows 1.., each a branch of one token (speculative verification)."""
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
        if chain is not None:
            fa_mask = create_sibling_mask(
                chain, hidden_states.shape[1], cache[self.fa_idx].offset
            )
        elif self.fa_idx is not None:
            fa_mask = create_attention_mask(hidden_states, cache[self.fa_idx])
        if self.ssm_idx is not None:
            ssm_mask = create_ssm_mask(hidden_states, cache[self.ssm_idx])

        # Receive from the previous process in the pipeline
        if pipeline_rank < pipeline_size - 1:
            hidden_states = mx.distributed.recv_like(hidden_states, (pipeline_rank + 1))

        # Start the GPU on a chunk of layers while Python builds the rest
        chunk = self.eval_every
        if cache[0] is None or self.training or pipeline_size > 1:
            chunk = 0
        pending = None
        for i, (layer, c) in enumerate(zip(self.pipeline_layers, cache), 1):
            mask = ssm_mask if layer.is_linear else fa_mask
            hidden_states, pending = layer(
                hidden_states, mask=mask, cache=c, pending=pending, split=True, chain=chain
            )
            if chunk and i % chunk == 0 and i < len(cache):
                mx.async_eval(hidden_states, pending)
        if pipeline_size > 1:
            hidden_states, pending = hidden_states + fused_ops.slot_sum(pending), None

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

        return prep_add_rms_norm(
            self.norm, hidden_states, pending, None, not self.training
        )[1]


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
    return fuse_experts(weights)


def fuse_experts(weights):
    """Fuse the experts' gate and up and store the shared expert as the last expert.

    Only when the shared expert has the experts' shapes; the expert tensors are
    indexed by expert on axis 0 and by output row on axis 1.
    """
    marker = ".mlp.switch_mlp.gate_proj."
    for key in [k for k in weights if marker in k]:
        prefix, param = key.split(marker)
        shared = [
            f"{prefix}.mlp.shared_expert.{part}.{param}"
            for part in ("gate_proj", "up_proj", "down_proj")
        ]
        up = f"{prefix}.mlp.switch_mlp.up_proj.{param}"
        down = f"{prefix}.mlp.switch_mlp.down_proj.{param}"
        if (
            shared[0] not in weights
            or weights[shared[0]].shape != weights[key].shape[1:]
        ):
            continue
        gate_up = mx.concatenate([weights.pop(key), weights.pop(up)], axis=1)
        shared_gate_up = mx.concatenate(
            [weights.pop(shared[0]), weights.pop(shared[1])]
        )
        weights[f"{prefix}.mlp.switch_mlp.gate_up_proj.{param}"] = mx.concatenate(
            [gate_up, shared_gate_up[None]]
        )
        weights[down] = mx.concatenate(
            [weights.pop(down), weights.pop(shared[2])[None]]
        )
        gate = f"{prefix}.mlp.gate.{param}"
        weights[gate] = mx.concatenate(
            [weights[gate], weights.pop(f"{prefix}.mlp.shared_expert_gate.{param}")]
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
        chain: Optional[int] = None,
    ) -> mx.array:
        hidden = self.model(inputs, cache, input_embeddings=input_embeddings, chain=chain)
        if self.args.tie_word_embeddings:
            out = self.model.embed_tokens.as_linear(hidden)
        else:
            out = qlinear(self.lm_head, hidden)
        return (out, hidden) if return_hidden else out

    def warmup(self, rows=(1, 2, 3, 4, 5, 6, 7, 8, 16, 32), siblings=(11, 6)):
        """Compile the kernels of every verify row count on one layer of each kind
        and the head, so a generation does not pay the JIT (see qmv_small)."""
        if not mx.metal.is_available():
            return
        layers = list({l.is_linear: l for l in self.layers}.values())
        cache = [ArraysCache(size=2) if l.is_linear else KVCache() for l in layers]
        fa = next((c for c in cache if isinstance(c, KVCache)), None)
        for s, chain in [(r, None) for r in rows] + [siblings]:
            h = self.model.embed_tokens(mx.zeros((1, s), mx.uint32))
            if chain:
                fa_mask = create_sibling_mask(chain, s, fa.offset if fa else 0)
            else:
                fa_mask = create_attention_mask(h, fa)
            pending = None
            for layer, c in zip(layers, cache):
                mask = create_ssm_mask(h, c) if layer.is_linear else fa_mask
                h, pending = layer(h, mask, c, pending, split=True, chain=chain)
            h = prep_add_rms_norm(self.model.norm, h, pending, None, True)[1]
            if self.args.tie_word_embeddings:
                mx.eval(self.model.embed_tokens.as_linear(h))
            else:
                mx.eval(qlinear(self.lm_head, h))

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
        chain: Optional[int] = None,
    ):
        return self.language_model(
            inputs,
            cache=cache,
            input_embeddings=input_embeddings,
            return_hidden=return_hidden,
            chain=chain,
        )

    @property
    def model(self):
        return self.language_model.model

    def warmup(self):
        self.language_model.warmup()

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
            elif isinstance(layer.mlp, SparseMoeBlock):
                layer.mlp.sharding_group = group
                shard_inplace(
                    layer.mlp.switch_mlp.gate_up_proj,
                    "all-to-sharded",
                    segments=2,
                    group=group,
                )
                shard_inplace(
                    layer.mlp.switch_mlp.down_proj, "sharded-to-all", group=group
                )
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
