# Copyright © 2026 Apple Inc.

import functools
import json
from dataclasses import dataclass
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn

from .base import BaseModelArgs, create_attention_mask
from .cache import KVCache
from .qwen3_5 import DecoderLayer, TextModelArgs, fuse_projections
from .qwen3_5_moe import split_experts


@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str
    text_config: dict
    block_size: int = 3


class CandidateHead:
    """The output head scored on a candidate set of vocabulary rows only.

    The set is the first ``fixed`` rows (a slice, no copy) and the gathered rows
    ``ids`` (repeats allowed). The logits come back over ``self.ids``; a repeat
    of a row scores -inf, so a softmax over them is exact.
    """

    def __init__(self, head, fixed, ids):
        quantized = hasattr(head, "scales")
        params = [head.weight]
        self.kwargs = None
        if quantized:
            # The e2m1 formats have no biases
            params += [p for p in (head.scales, head.get("biases")) if p is not None]
            self.kwargs = dict(group_size=head.group_size, bits=head.bits, mode=head.mode)
        ids = mx.sort(ids)
        self.first = mx.concatenate([mx.array([True]), ids[1:] != ids[:-1]])
        self.first &= ids >= fixed
        self.ids = mx.concatenate([mx.arange(fixed, dtype=ids.dtype), ids])
        self.fixed = [p[:fixed] for p in params]
        self.rows = [mx.take(p, ids, axis=0) for p in params]

    def _logits(self, x, params):
        if self.kwargs is None:
            return x @ params[0].T
        return mx.quantized_matmul(x, *params, transpose=True, **self.kwargs)

    def __call__(self, x):
        rows = mx.where(self.first, self._logits(x, self.rows), -mx.inf)
        if not self.fixed[0].shape[0]:
            return rows
        return mx.concatenate([self._logits(x, self.fixed), rows], axis=-1)


class Candidates:
    """Candidate rows for the drafts of one cycle: the first ``fixed`` rows of the
    vocabulary (the frequent tokens of a BPE order), the ``size`` rows with the
    largest running softmax mass in the target's outputs, the recent tokens and
    the last drafts."""

    def __init__(self, head, fixed, size=8192, window=2048, decay=0.98):
        self.head, self.window, self.decay = head, window, decay
        vocab = head.weight.shape[0]
        self.size, self.fixed = min(size, vocab), min(fixed, vocab)
        self.score = None
        self.context = None

    def observe(self, logits):
        """Add the softmax mass of the target's logits (.., V) to the score."""
        logits = logits.reshape(-1, logits.shape[-1]).astype(mx.float32)
        mass = mx.softmax(logits, axis=-1).sum(axis=0)
        self.score = mass if self.score is None else self.decay * self.score + mass

    def extend(self, tokens):
        if self.context is not None:
            tokens = mx.concatenate([self.context, tokens])
        self.context = tokens[-self.window :]

    def make_head(self, *recent):
        """The head for the next drafts, or None before the first observation."""
        if self.score is None:
            return None
        top = mx.argpartition(self.score, kth=-self.size)[-self.size :]
        ids = [top.astype(mx.uint32), self.context, *recent]
        ids = mx.concatenate([i.astype(mx.uint32) for i in ids])
        return CandidateHead(self.head, self.fixed, ids)


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
            head, self.lm_head = embed_tokens, embed_tokens.as_linear
        else:
            head, self.lm_head = text_model.lm_head, text_model.lm_head.__call__
        # A partial keeps the head out of this module's parameters
        self.candidates = functools.partial(Candidates, head)

    def __call__(self, inputs: mx.array, hidden: mx.array, cache=None, head=None):
        """``head`` replaces the output head (see ``CandidateHead``)."""
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
        return (self.lm_head if head is None else head)(h), h

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
        return fuse_projections(split_experts(weights))


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
