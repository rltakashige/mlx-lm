# Copyright © 2026 Apple Inc.

"""The multi-token prediction head of Qwen3.8-Flash-Next (qwen4_exp).

One full-attention decoder layer (sparse attention + MoE, no PLE) between the
hyper-connection streams: its input is the target's pre-mixer 4-stream state
projected per stream, plus the projected embedding of the token, and its
output is mixed for the shared output head. The pre-mixer output of the head
feeds the next draft step.
"""

import functools
from dataclasses import dataclass
from typing import Optional

import mlx.core as mx
import mlx.nn as nn

from .base import BaseModelArgs, create_attention_mask
from .qwen3_5_mtp import Candidates
from .qwen4_exp import DecoderLayer, GatedResidual, QSAKVCache, RMSNorm, TextArgs
from .qwen4_exp import fuse_hyper_connections, fuse_projections


@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str
    text_config: dict
    model_path: Optional[str] = None


class Model(nn.Module):
    """Multi-token prediction head drafting one token ahead of a qwen4_exp target."""

    needs_hidden = True

    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        text_args = TextArgs.from_dict(
            {
                **args.text_config,
                "layer_types": ["full_attention"],
                "num_hidden_layers": 1,
                "ple_layer_ids": [],
            }
        )
        self.hc, self.dims = text_args.hc_count, text_args.hidden_size
        self.moe_dims = text_args.moe_intermediate_size
        eps = text_args.rms_norm_eps
        self.pre_fc_norm_embedding = RMSNorm(self.dims, eps=eps)
        # One statistic over every stream, as the reference
        self.pre_fc_norm_hidden = RMSNorm(self.hc * self.dims, eps=eps)
        self.fc_embedding = nn.Linear(self.dims, self.dims, bias=False)
        self.fc_hidden = nn.Linear(self.dims, self.dims, bias=False)
        self.layers = [DecoderLayer(text_args, 0)]
        self.hyper_connection_mixer = GatedResidual(text_args, use_combine=False)

    def bind(self, target):
        """Share the target's embedding and output head (no copies)."""
        text_model = getattr(target, "language_model", target)
        embed_tokens = text_model.model.embed_tokens
        self.embed_tokens = embed_tokens.__call__
        if text_model.args.tie_word_embeddings:
            head, self.lm_head = embed_tokens, embed_tokens.as_linear
        else:
            head, self.lm_head = text_model.lm_head, text_model.lm_head.__call__
        # A partial keeps the head out of this module's parameters
        self.candidates = functools.partial(Candidates, head)

    def __call__(self, inputs: mx.array, hidden: mx.array, cache=None, head=None):
        """``hidden`` is the target's pre-mixer state (B, L, hc * dims); ``head``
        replaces the output head (see ``CandidateHead``)."""
        B, L, _ = hidden.shape
        emb = self.fc_embedding(self.pre_fc_norm_embedding(self.embed_tokens(inputs)))
        streams = self.pre_fc_norm_hidden(hidden).reshape(B, L, self.hc, self.dims)
        h = (self.fc_hidden(streams) + emb[..., None, :]).reshape(B, L, -1)
        if cache is None:
            cache = [None]
        mask = create_attention_mask(h, cache[0])
        h, pending = self.layers[0](h, mask, cache[0])
        out, _, h = self.hyper_connection_mixer.mix(h, pending)
        return (self.lm_head if head is None else head)(out), h

    def make_cache(self):
        return [QSAKVCache() for _ in self.layers]

    def sanitize(self, weights):
        out = {}
        for k, v in weights.items():
            for prefix in ("model.language_model.mtp.", "language_model.mtp.", "model.mtp.", "mtp."):
                if k.startswith(prefix):
                    k = k[len(prefix) :]
                    break
            else:
                continue
            if k.endswith("mlp.experts.gate_up_proj"):
                base = k[: -len("experts.gate_up_proj")]
                gate, up = mx.split(v, 2, axis=-2)
                out[base + "switch_mlp.gate_proj.weight"] = gate
                out[base + "switch_mlp.up_proj.weight"] = up
                continue
            if k.endswith("mlp.experts.down_proj"):
                out[k[: -len("experts.down_proj")] + "switch_mlp.down_proj.weight"] = v
                continue
            out[k] = v
        self._match_shared_expert(out)
        return fuse_hyper_connections(fuse_projections(out))

    def _match_shared_expert(self, weights):
        """Requantize a shared expert stored at another group size or bit width
        like the routed experts, so it can be fused as the last expert."""
        for i in range(len(self.layers)):
            for part, K in (("gate_proj", self.dims), ("up_proj", self.dims), ("down_proj", self.moe_dims)):
                shared = f"layers.{i}.mlp.shared_expert.{part}"
                routed = f"layers.{i}.mlp.switch_mlp.{part}"
                if f"{shared}.scales" not in weights or f"{routed}.scales" not in weights:
                    continue
                params = lambda name: (
                    K // weights[f"{name}.scales"].shape[-1],
                    32 * weights[f"{name}.weight"].shape[-1] // K,
                )
                if params(shared) == params(routed):
                    continue
                x = mx.dequantize(
                    weights[f"{shared}.weight"],
                    weights[f"{shared}.scales"],
                    weights.get(f"{shared}.biases"),
                    group_size=params(shared)[0],
                    bits=params(shared)[1],
                )
                gs, bits = params(routed)
                weights[f"{shared}.weight"], weights[f"{shared}.scales"], weights[f"{shared}.biases"] = mx.quantize(
                    x, group_size=gs, bits=bits
                )

    @property
    def quant_predicate(self):
        def predicate(path, _):
            if path.endswith("mlp.gate") or path.endswith("shared_expert_gate"):
                return {"group_size": 64, "bits": 8}
            return True

        return predicate
