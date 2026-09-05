# Copyright © 2026 Apple Inc.

import json
from dataclasses import dataclass
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn

from .base import BaseModelArgs, create_attention_mask
from .cache import KVCache
from .qwen3_5 import DecoderLayer, TextModelArgs, fuse_projections


@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str
    text_config: dict
    block_size: int = 3


class Model(nn.Module):
    """Multi-token prediction head drafting one token ahead of a Qwen3.5 target."""

    needs_hidden = True

    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        text_args = TextModelArgs.from_dict(
            {**args.text_config, "full_attention_interval": 1}
        )
        dim, eps = text_args.hidden_size, text_args.rms_norm_eps
        self.fc = nn.Linear(2 * dim, dim, bias=False)
        self.pre_fc_norm_embedding = nn.RMSNorm(dim, eps=eps)
        self.pre_fc_norm_hidden = nn.RMSNorm(dim, eps=eps)
        self.layers = [DecoderLayer(text_args, i) for i in range(args.block_size - 2)]
        self.norm = nn.RMSNorm(dim, eps=eps)

    def bind(self, target):
        """Share the target's embedding and output head (no copies)."""
        text_model = getattr(target, "language_model", target)
        embed_tokens = text_model.model.embed_tokens
        self.embed_tokens = embed_tokens.__call__
        if text_model.args.tie_word_embeddings:
            self.lm_head = embed_tokens.as_linear
        else:
            self.lm_head = text_model.lm_head.__call__

    def __call__(self, inputs: mx.array, hidden: mx.array, cache=None):
        h = mx.concatenate(
            [
                self.pre_fc_norm_embedding(self.embed_tokens(inputs)),
                self.pre_fc_norm_hidden(hidden),
            ],
            axis=-1,
        )
        h = self.fc(h)
        if cache is None:
            cache = [None] * len(self.layers)
        mask = create_attention_mask(h, cache[0])
        for layer, c in zip(self.layers, cache):
            h = layer(h, mask, c)
        h = self.norm(h)
        return self.lm_head(h), h

    def make_cache(self):
        return [KVCache() for _ in self.layers]

    def sanitize(self, weights):
        # HF layout: "mtp." prefix and norm weights stored as (actual - 1)
        if any(k.startswith("mtp.") for k in weights):
            weights = {k[4:]: v for k, v in weights.items() if k.startswith("mtp.")}
            weights = {
                k: v + 1.0 if v.ndim == 1 and "norm" in k else v
                for k, v in weights.items()
            }
        return fuse_projections(weights)


def load_bundled(model_path):
    """Build the head from the ``mtp.*`` weights bundled with a target checkpoint."""
    model_path = Path(model_path)
    with open(model_path / "config.json") as f:
        config = json.load(f)
    text_config = config.get("text_config", config)
    with open(model_path / "model.safetensors.index.json") as f:
        weight_map = json.load(f)["weight_map"]
    files = {v for k, v in weight_map.items() if k.startswith("mtp.")}
    if not files:
        raise FileNotFoundError(f"No mtp.* weights found in {model_path}")
    weights = {}
    for wf in sorted(files):
        weights.update(
            {
                k: v
                for k, v in mx.load(str(model_path / wf)).items()
                if k.startswith("mtp.")
            }
        )
    block_size = text_config.get("mtp_num_hidden_layers", 1) + 2
    model = Model(ModelArgs("qwen3_5_mtp", text_config, block_size))
    model.load_weights(list(model.sanitize(weights).items()))
    if quantization := config.get("quantization"):
        nn.quantize(
            model,
            quantization["group_size"],
            quantization["bits"],
            mode=quantization.get("mode", "affine"),
        )
    model.eval()
    mx.eval(model.parameters())
    return model
