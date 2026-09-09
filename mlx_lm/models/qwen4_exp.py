# Copyright © 2026 Apple Inc.

"""Qwen3.8-Flash-Next (HF model_type qwen4_exp), text only.

Compared to qwen3_5: four residual streams mixed by hyper-connections, a hashed
n-gram embedding (PLE) at one layer whose 51B-parameter table stays on disk and
is gathered row by row on the host, Qwen Sparse Attention (an indexer selects
key blocks; applied as a boolean mask on a dense SDPA), and a GDN output gate
that is a sigmoid. Norm gains are stored zero-centered: y = norm(x) * (1 + w).
"""

import json
import math
import struct
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from . import fused_ops
from .base import BaseModelArgs, create_attention_mask, create_ssm_mask
from .base import scaled_dot_product_attention
from .cache import ArraysCache, KVCache
from .qmv_small import qlinear
from .qwen3_5 import GatedDeltaNet as Qwen3_5GatedDeltaNet
from .qwen3_5 import SparseMoeBlock, fuse_projections


@dataclass
class TextArgs(BaseModelArgs):
    model_type: str = "qwen4_exp_text"
    hidden_size: int = 2560
    num_hidden_layers: int = 48
    num_attention_heads: int = 24
    num_key_value_heads: int = 2
    head_dim: int = 256
    vocab_size: int = 248320
    rms_norm_eps: float = 1e-6
    layer_types: List[str] = field(default_factory=list)
    full_attention_interval: int = 4
    tie_word_embeddings: bool = False
    # MoE
    num_experts: int = 512
    num_experts_per_tok: int = 10
    moe_intermediate_size: int = 640
    shared_expert_intermediate_size: int = 640
    norm_topk_prob: bool = True
    # Gated DeltaNet
    linear_num_key_heads: int = 16
    linear_num_value_heads: int = 48
    linear_key_head_dim: int = 128
    linear_value_head_dim: int = 128
    linear_conv_kernel_dim: int = 4
    output_gate_type: str = "sigmoid"
    # Hyper-connections
    hc_count: int = 4
    hc_lowrank: int = 320
    # Qwen Sparse Attention
    indexer_n_heads: int = 4
    indexer_kv_heads: int = 1
    indexer_head_dim: int = 128
    indexer_budget: int = 2048
    indexer_compress_ratio: int = 4
    # n-gram embedding (PLE)
    ngram_size: int = 3
    heads_per_ngram: int = 8
    ngram_vocab_size_base: int = 20_000_000
    make_ngram_vocab_size_divisible_by: int = 128
    split_ngram_parts: int = 128
    ple_embed_dim: Optional[int] = None
    ple_layer_ids: List[int] = field(default_factory=lambda: [2])
    ple_conv_kernel_size: int = 4
    seed: int = 1234
    eos_token_id: Any = 248044
    # Rope
    rope_parameters: Optional[Dict[str, Any]] = None
    rope_theta: float = 10_000_000.0
    partial_rotary_factor: float = 0.25
    mtp_num_hidden_layers: int = 1

    def __post_init__(self):
        rp = self.rope_parameters or {}
        self.rope_theta = float(rp.get("rope_theta", self.rope_theta))
        self.partial_rotary_factor = float(
            rp.get("partial_rotary_factor", self.partial_rotary_factor)
        )
        if self.ple_embed_dim is None:
            self.ple_embed_dim = self.hidden_size
        if not self.layer_types:
            k = self.full_attention_interval
            self.layer_types = [
                "full_attention" if (i + 1) % k == 0 else "linear_attention"
                for i in range(self.num_hidden_layers)
            ]
        if isinstance(self.eos_token_id, list):
            self.eos_token_id = self.eos_token_id[0]


@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str
    text_config: dict
    # Directory of the checkpoint: the n-gram table is memory mapped from it
    model_path: Optional[str] = None

    @classmethod
    def from_dict(cls, params):
        if "text_config" not in params:
            params = {**params, "text_config": params}
        return super().from_dict(params)


# ------------------------------------------------------------------- norms


class RMSNorm(nn.Module):
    """RMSNorm with a zero-centered gain: y = norm(x) * (1 + weight).

    With ``group_size`` the statistic is taken over each group of the last axis
    (one per hyper-connection stream); the gain still spans the whole axis.
    """

    def __init__(self, dims: int, eps: float = 1e-6, group_size: Optional[int] = None):
        super().__init__()
        self.weight = mx.zeros(dims)
        self.eps = eps
        self.group_size = group_size

    def gain(self):
        # 1 + w is computed once per weight array, outside the parameters
        cached = getattr(self, "_gain_cache", None)
        if cached is None or cached[0] is not self.weight:
            cached = (self.weight, 1 + self.weight)
            object.__setattr__(self, "_gain_cache", cached)
        return cached[1]

    def __call__(self, x: mx.array) -> mx.array:
        if self.group_size is None:
            return mx.fast.rms_norm(x, self.gain(), self.eps)
        shape = x.shape
        x = x.reshape(*shape[:-1], -1, self.group_size)
        return mx.fast.rms_norm(x, None, self.eps).reshape(shape) * self.gain()


class RMSNormGated(nn.Module):
    """The GDN output norm: norm(x) * weight * sigmoid(gate) (a conventional gain)."""

    def __init__(self, dims: int, eps: float = 1e-6, activation: str = "sigmoid"):
        super().__init__()
        self.weight = mx.ones(dims)
        self.eps = eps
        self.activation = activation

    def __call__(self, x: mx.array, gate: mx.array) -> mx.array:
        y = mx.fast.rms_norm(x, self.weight, self.eps)
        act = mx.sigmoid if self.activation == "sigmoid" else nn.silu
        return y * act(gate)


# ------------------------------------------------------- hyper-connections


class GatedResidual(nn.Module):
    """Mix the ``hc_count`` residual streams into one block input and, with
    ``use_combine``, the weights that inject the block output back per stream."""

    def __init__(self, args: TextArgs, use_combine: bool = True):
        super().__init__()
        self.hc = args.hc_count
        self.dims = args.hidden_size
        hc_dims = self.hc * self.dims
        self.hc_norm = RMSNorm(hc_dims, eps=args.rms_norm_eps, group_size=self.dims)
        self.input_mix_weight_down = nn.Linear(hc_dims, args.hc_lowrank, bias=False)
        self.input_mix_weight_up = nn.Linear(args.hc_lowrank, hc_dims, bias=False)
        if use_combine:
            self.block_inject_weight = nn.Linear(hc_dims, self.hc, bias=False)

    def __call__(self, hyper: mx.array):
        normed = self.hc_norm(hyper)
        mix = nn.silu(self.input_mix_weight_down(normed) / self.hc)
        mix = mx.sigmoid(self.input_mix_weight_up(mix))
        streams = normed.reshape(*normed.shape[:-1], self.hc, self.dims)
        mixed = (mix.reshape(streams.shape) * streams).mean(axis=-2)
        if "block_inject_weight" not in self:
            return mixed
        inject = 2 * mx.sigmoid(self.block_inject_weight(normed) / self.hc)
        return mixed, inject

    def combine(self, hyper: mx.array, x: mx.array, inject: mx.array) -> mx.array:
        return hyper + (x[..., None, :] * inject[..., None]).reshape(hyper.shape)


# ------------------------------------------------------------ n-gram / PLE

_MASK64 = (1 << 64) - 1
_SPLITMIX_GAMMA = 0x9E3779B97F4A7C15
_SPLITMIX_M1 = 0xBF58476D1CE4E5B9
_SPLITMIX_M2 = 0x94D049BB133111EB
_PRIME_1 = 10007
_SHARD_MARKER = ".ple.ple_embedding.ngram_embedding.shards."
_BF16 = np.dtype("<u2")


def _splitmix64(value: int) -> int:
    value = (value + _SPLITMIX_GAMMA) & _MASK64
    value = ((value ^ (value >> 30)) * _SPLITMIX_M1) & _MASK64
    value = ((value ^ (value >> 27)) * _SPLITMIX_M2) & _MASK64
    return (value ^ (value >> 31)) & _MASK64


def layer_multipliers(vocab_size: int, ngram_size: int, ple_layer_index: int, seed: int):
    """The odd hash multipliers of one PLE layer (as the reference derives them)."""
    half_bound = max(1, (((1 << 63) - 1) // max(vocab_size, 1)) // 2)
    base_seed = seed + _PRIME_1 * ple_layer_index
    return [
        2 * (_splitmix64((base_seed + _SPLITMIX_GAMMA * (i + 1)) & _MASK64) % half_bound) + 1
        for i in range(ngram_size)
    ]


def _is_prime(value: int) -> bool:
    if value < 2:
        return False
    if value % 2 == 0:
        return value == 2
    return all(value % d for d in range(3, math.isqrt(value) + 1, 2))


def _nth_prime_after(start: int, count: int) -> int:
    prime = start
    for _ in range(count):
        prime += 1
        while not _is_prime(prime):
            prime += 1
    return prime


def _to_signed(values):
    """Python ints modulo 2**64 as numpy int64 (the reference multiplies in int64)."""
    return np.array([v - (1 << 64) if v >= (1 << 63) else v for v in values], np.int64)


class NGramHasher:
    """Row ids of the n-gram table for a token history, on the host in numpy."""

    def __init__(self, args: TextArgs, ple_layer_index: int):
        self.ngram_size = args.ngram_size
        self.context_len = args.ngram_size - 1
        self.heads_per_ngram = args.heads_per_ngram
        self.ngram_heads = self.context_len * args.heads_per_ngram
        self.eos = int(args.eos_token_id)
        sizes, offsets, total = [], [], 0
        for head in range(self.ngram_heads):
            g = ple_layer_index * self.ngram_heads + head
            size = _nth_prime_after(args.ngram_vocab_size_base - 1, g + 1)
            sizes.append(size)
            offsets.append(total)
            total += size
        div = args.make_ngram_vocab_size_divisible_by
        self.rows = -(-total // div) * div
        self.sizes = np.array(sizes, np.int64)
        self.offsets = np.array(offsets, np.int64)
        self.multipliers = _to_signed(
            layer_multipliers(args.vocab_size, args.ngram_size, ple_layer_index, args.seed)
        )

    def _shift(self, ids: np.ndarray, shift: int) -> np.ndarray:
        """Shift right by ``shift`` without crossing an EOS (the reference rule)."""
        if shift == 0:
            return ids
        B, T = ids.shape
        pos = np.arange(T)
        eos_pos = np.where(ids == self.eos, pos[None], -1)
        prev_eos = np.concatenate(
            [np.full((B, 1), -1), np.maximum.accumulate(eos_pos, axis=1)[:, :-1]], axis=1
        )
        in_segment = pos[None] - (prev_eos + 1)
        src = pos - shift
        shifted = np.take_along_axis(ids, np.broadcast_to(np.maximum(src, 0)[None], (B, T)), 1)
        ok = (in_segment >= shift) & (src[None] >= 0)
        return np.where(ok, shifted, self.eos)

    def __call__(self, history: np.ndarray) -> np.ndarray:
        """(B, T) token history -> (B, T, ngram_heads) row ids (int64)."""
        history = np.asarray(history, np.int64)
        shifted = [self._shift(history, s) for s in range(self.ngram_size)]
        blocks = []
        with np.errstate(over="ignore"):
            for ngram in range(2, self.ngram_size + 1):
                lo = (ngram - 2) * self.heads_per_ngram
                hi = lo + self.heads_per_ngram
                mixed = shifted[0] * self.multipliers[0]
                for p in range(1, ngram):
                    mixed = np.bitwise_xor(mixed, shifted[p] * self.multipliers[p])
                blocks.append(mixed[..., None] % self.sizes[lo:hi] + self.offsets[lo:hi])
        return np.concatenate(blocks, axis=-1)


def _bf16_to_f32(bits: np.ndarray) -> np.ndarray:
    return (bits.astype(np.uint32) << 16).view(np.float32)


def dequantize_rows(weight: np.ndarray, scales: np.ndarray, biases: np.ndarray, group_size=32):
    """4-bit affine rows (n, K/8) uint32 with bf16 scales/biases (n, K/group) -> (n, K) f32."""
    n = weight.shape[0]
    shifts = (4 * np.arange(8, dtype=np.uint32))[None, None]
    q = ((weight[..., None] >> shifts) & 0xF).reshape(n, -1, group_size).astype(np.float32)
    return (q * _bf16_to_f32(scales)[..., None] + _bf16_to_f32(biases)[..., None]).reshape(n, -1)


class NGramTable:
    """Rows of the n-gram table, read from the checkpoint shards on the host.

    The shards are memory mapped at their safetensors byte ranges; a row is
    dequantized when read and kept in an LRU of ``cache_rows`` rows. The
    counters (``lookups``, ``rows``, ``hits``, ``misses``, ``bytes``, ``seconds``)
    are for measurement.
    """

    def __init__(self, model_path: Union[str, Path], prefix: str, cache_rows: int = 1 << 16):
        model_path = Path(model_path)
        with open(model_path / "model.safetensors.index.json") as f:
            weight_map = json.load(f)["weight_map"]
        marker = prefix + _SHARD_MARKER
        shard_ids = sorted({int(k[len(marker) :].split(".")[0]) for k in weight_map if k.startswith(marker)})
        if not shard_ids:
            raise ValueError(f"No n-gram shards under {marker}* in {model_path}")
        headers = {}
        self.shards = []
        self.starts = [0]
        for i in shard_ids:
            arrays = {}
            for name in ("weight", "scales", "biases"):
                key = f"{marker}{i}.{name}"
                file = weight_map[key]
                if file not in headers:
                    with open(model_path / file, "rb") as f:
                        n = struct.unpack("<Q", f.read(8))[0]
                        headers[file] = (8 + n, json.loads(f.read(n)))
                base, header = headers[file]
                info = header[key]
                dtype = np.dtype("<u4") if name == "weight" else _BF16
                arrays[name] = np.memmap(
                    model_path / file,
                    dtype=dtype,
                    mode="r",
                    offset=base + info["data_offsets"][0],
                    shape=tuple(info["shape"]),
                )
            self.shards.append(arrays)
            self.starts.append(self.starts[-1] + arrays["weight"].shape[0])
        self.rows_total = self.starts[-1]
        self.width = self.shards[0]["weight"].shape[1] * 8
        self.group_size = self.width // self.shards[0]["scales"].shape[1]
        self.starts = np.array(self.starts, np.int64)
        self.cache_rows = cache_rows
        self.lru = OrderedDict()
        self.lookups = self.rows = self.hits = self.misses = self.bytes = 0
        self.seconds = 0.0
        self.prewarm_seconds = None

    def prewarm(self):
        """Read the shard files once so the rows come from the page cache: a cold
        row costs ~0.3 ms of disk latency, a warm one ~10 us (M5 Max)."""
        tic = time.perf_counter()
        buf = bytearray(64 << 20)
        for file in sorted({arr.filename for arrays in self.shards for arr in arrays.values()}):
            with open(file, "rb", buffering=0) as f:
                while f.readinto(buf):
                    pass
        self.prewarm_seconds = time.perf_counter() - tic

    def _read(self, ids: np.ndarray) -> np.ndarray:
        """Dequantized rows (n, width) of distinct ``ids``, read shard by shard."""
        out = np.empty((len(ids), self.width), np.float32)
        shard_of = np.searchsorted(self.starts, ids, side="right") - 1
        for s in np.unique(shard_of):
            sel = np.flatnonzero(shard_of == s)
            local = ids[sel] - self.starts[s]
            arrays = self.shards[s]
            w = np.asarray(arrays["weight"][local])
            sc = np.asarray(arrays["scales"][local])
            b = np.asarray(arrays["biases"][local])
            self.bytes += w.nbytes + sc.nbytes + b.nbytes
            out[sel] = dequantize_rows(w, sc, b, self.group_size)
        return out

    def __call__(self, ids: np.ndarray) -> np.ndarray:
        """Rows (n, width) float32 of the row ids ``ids`` (n,)."""
        tic = time.perf_counter()
        ids = np.asarray(ids, np.int64).reshape(-1)
        out = np.empty((len(ids), self.width), np.float32)
        missing = []
        for i, r in enumerate(ids.tolist()):
            row = self.lru.get(r)
            if row is None:
                missing.append(i)
            else:
                self.lru.move_to_end(r)
                out[i] = row
        self.hits += len(ids) - len(missing)
        if missing:
            uniq, inverse = np.unique(ids[missing], return_inverse=True)
            rows = self._read(uniq)
            out[missing] = rows[inverse]
            self.misses += len(uniq)
            self.hits += len(missing) - len(uniq)
            for r, row in zip(uniq.tolist(), rows):
                self.lru[r] = row
            while len(self.lru) > self.cache_rows:
                self.lru.popitem(last=False)
        self.lookups += 1
        self.rows += len(ids)
        self.seconds += time.perf_counter() - tic
        return out

    def stats(self):
        return dict(
            lookups=self.lookups, rows=self.rows, hits=self.hits, misses=self.misses,
            bytes=self.bytes, seconds=self.seconds,
        )


class NGramEmbedding(nn.Module):
    """The hashed n-gram embedding: row ids on the host, rows from ``NGramTable``
    (the checkpoint on disk) or from a small resident ``table`` (tests)."""

    def __init__(self, args: TextArgs, ple_layer_index: int):
        super().__init__()
        self.hasher = NGramHasher(args, ple_layer_index)
        self.width = args.ple_embed_dim // self.hasher.ngram_heads
        self.table = None
        # A resident table lives under a "_" key: outside the parameters
        self._resident = None

    def attach(self, model_path: str, prefix: str, prewarm: bool = True):
        self.table = NGramTable(model_path, prefix)
        if self.table.rows_total != self.hasher.rows or self.table.width != self.width:
            raise ValueError("The n-gram table on disk does not match the config")
        if prewarm:
            # Overlaps with the weight load
            threading.Thread(target=self.table.prewarm, daemon=True).start()

    def rows(self, ids: np.ndarray, dtype) -> mx.array:
        """Embeddings (.., ple_embed_dim) of row ids (.., ngram_heads)."""
        shape = (*ids.shape[:-1], ids.shape[-1] * self.width)
        if self.table is not None:
            return mx.array(self.table(ids.reshape(-1))).reshape(shape).astype(dtype)
        if self._resident is None:
            raise ValueError("The n-gram table is not loaded: pass model_path to the args")
        return mx.take(self._resident, mx.array(ids), axis=0).reshape(shape).astype(dtype)


class PLECache(ArraysCache):
    """The linear-attention cache plus the PLE conv state (2) and n-gram context (3)."""

    def __init__(self, left_padding=None):
        super().__init__(4, left_padding)
        self.ple_rollback = None
        self.ple_history = None

    def trim(self, n):
        steps = self.steps - min(n, max(self.steps - 1, 0))
        ple = self.ple_rollback
        n = super().trim(n)
        if n and ple is not None:
            self[2], self[3] = ple(steps)
        self.ple_rollback = None
        return n


class PLELayer(nn.Module):
    """Inject the hashed n-gram features of each token into the residual streams."""

    def __init__(self, args: TextArgs, ple_layer_index: int):
        super().__init__()
        self.dims = args.hidden_size
        self.hc = args.hc_count
        hc_dims = self.dims * self.hc
        self.ple_embedding = NGramEmbedding(args, ple_layer_index)
        self.dilation = args.ngram_size
        self.state_len = (args.ple_conv_kernel_size - 1) * self.dilation
        self.key_proj = nn.Linear(args.ple_embed_dim, hc_dims, bias=False)
        self.value_proj = nn.Linear(args.ple_embed_dim, self.dims, bias=False)
        self.norm_key = RMSNorm(hc_dims, eps=args.rms_norm_eps, group_size=self.dims)
        self.norm_query = RMSNorm(hc_dims, eps=args.rms_norm_eps, group_size=self.dims)
        self.norm_conv = RMSNorm(hc_dims, eps=args.rms_norm_eps, group_size=self.dims)
        self.conv1d = nn.Conv1d(
            hc_dims,
            hc_dims,
            kernel_size=args.ple_conv_kernel_size,
            dilation=self.dilation,
            groups=hc_dims,
            bias=False,
        )

    def _ngram_ids(self, ids: np.ndarray, cache) -> np.ndarray:
        """Row ids (B, L, heads) of the tokens ``ids`` (B, L) after the cached context."""
        hasher = self.ple_embedding.hasher
        B, L = ids.shape
        if cache is not None and cache[3] is not None:
            prev = np.array(cache[3], dtype=np.int64)
        else:
            prev = np.full((B, hasher.context_len), hasher.eos, np.int64)
        history = np.concatenate([prev, ids], axis=1)
        if cache is not None:
            cache[3] = mx.array(history[:, -hasher.context_len :])
            cache.ple_history = history
        return hasher(history)[:, -L:]

    def _short_conv(self, x: mx.array, cache) -> mx.array:
        B, L, _ = x.shape
        if cache is not None and cache[2] is not None:
            state = cache[2]
        else:
            state = mx.zeros((B, self.state_len, x.shape[-1]), dtype=x.dtype)
        conv_input = mx.concatenate([state, x], axis=1)
        if cache is not None:
            cache[2] = mx.contiguous(conv_input[:, -self.state_len :])
            if cache.keep_states and L > 1:
                # Let the cache rebuild the conv state and the context after ``steps`` rows
                n, ctx, history = self.state_len, self.ple_embedding.hasher.context_len, cache.ple_history
                cache.ple_rollback = lambda steps: (
                    conv_input[:, steps : steps + n],
                    mx.array(history[:, steps : steps + ctx]),
                )
        return nn.silu(self.conv1d(conv_input))

    def __call__(self, hyper: mx.array, ids: np.ndarray, cache) -> mx.array:
        B, L, _ = hyper.shape
        emb = self.ple_embedding.rows(self._ngram_ids(ids, cache), hyper.dtype)
        key = self.norm_key(self.key_proj(emb)).reshape(B, L, self.hc, self.dims)
        value = self.value_proj(emb)
        query = self.norm_query(hyper).reshape(B, L, self.hc, self.dims)
        gate = (key * query).sum(axis=-1, keepdims=True) / math.sqrt(self.dims)
        gate = mx.sign(gate) * mx.sqrt(mx.maximum(mx.abs(gate), 1e-6))
        gated = (mx.sigmoid(gate) * value[..., None, :]).reshape(hyper.shape)
        return gated + self._short_conv(self.norm_conv(gated), cache)


# ------------------------------------------------------------- attention


class QSAKVCache(KVCache):
    """KV cache with the indexer's raw keys (B, L, index_head_dim) alongside."""

    def __init__(self):
        super().__init__()
        self.index_keys = None

    def update_index(self, keys: mx.array) -> mx.array:
        """Append ``keys`` (B, S, D) at the current offset and return every key so far."""
        prev, S = self.offset, keys.shape[1]
        if self.index_keys is None or prev + S > self.index_keys.shape[1]:
            B, _, D = keys.shape
            n = ((self.step + S - 1) // self.step) * self.step
            new = mx.zeros((B, n, D), keys.dtype)
            if self.index_keys is not None:
                if prev % self.step != 0:
                    self.index_keys = self.index_keys[:, :prev]
                new = mx.concatenate([self.index_keys, new], axis=1)
            self.index_keys = new
        self.index_keys[:, prev : prev + S] = keys
        return self.index_keys[:, : prev + S]

    @property
    def state(self):
        keys = mx.array([]) if self.index_keys is None else self.index_keys[:, : self.offset]
        return (*super().state, keys)

    @state.setter
    def state(self, v):
        *kv, keys = v
        KVCache.state.fset(self, tuple(kv))
        self.index_keys = keys if keys.size else None

    def to_quantized(self, group_size: int = 64, bits: int = 4):
        raise ValueError("Qwen Sparse Attention does not support a quantized KV cache")

    @property
    def nbytes(self):
        return super().nbytes + (0 if self.index_keys is None else self.index_keys.nbytes)


class QSAIndexer(nn.Module):
    """Select, per query, a budget of compressed key blocks (a boolean mask).

    Returns None when every visible token fits in the budget: the dense causal
    attention is then exact.
    """

    def __init__(self, args: TextArgs, rotary_dims: int):
        super().__init__()
        self.n_heads = args.indexer_n_heads
        self.head_dim = args.indexer_head_dim
        self.ratio = args.indexer_compress_ratio
        self.block_topk = args.indexer_budget // self.ratio
        self.rotary_dims = rotary_dims
        self.rope_theta = args.rope_theta
        self.index_qk_proj = nn.Linear(
            args.hidden_size, (self.n_heads + args.indexer_kv_heads) * self.head_dim, bias=False
        )
        self.q_layernorm = RMSNorm(self.head_dim, eps=args.rms_norm_eps)
        self.k_layernorm = RMSNorm(self.head_dim, eps=args.rms_norm_eps)

    def _rope(self, x, offset=0, scale=1.0):
        return mx.fast.rope(
            x, self.rotary_dims, traditional=False, base=self.rope_theta, scale=scale, offset=offset
        )

    def __call__(self, x: mx.array, cache: Optional[QSAKVCache]) -> Optional[mx.array]:
        B, S, _ = x.shape
        qk = self.index_qk_proj(x)
        split = self.n_heads * self.head_dim
        q = qk[..., :split].reshape(B, S, self.n_heads, self.head_dim)
        raw_k = qk[..., split:].reshape(B, S, self.head_dim)
        offset = 0
        if cache is not None:
            offset = cache.offset
            raw_k = cache.update_index(raw_k)
        kv_len = raw_k.shape[1]
        n_blocks = kv_len // self.ratio
        # Every complete block fits in the budget: the causal mask is exact
        if n_blocks <= self.block_topk:
            return None

        pooled = raw_k[:, : n_blocks * self.ratio].reshape(B, n_blocks, self.ratio, -1)
        pooled = self.k_layernorm(pooled.astype(mx.float32).mean(axis=2).astype(raw_k.dtype))
        # The blocks are keyed at the position of their first token
        pooled = self._rope(pooled[:, None], scale=float(self.ratio))
        q = self._rope(self.q_layernorm(q).transpose(0, 2, 1, 3), offset=offset)
        scores = q.astype(mx.float32) @ pooled.astype(mx.float32).transpose(0, 1, 3, 2)
        scores = mx.maximum(scores, 0).sum(axis=1) / math.sqrt(self.head_dim)

        # A block is a candidate when it lies entirely before the query
        q_pos = offset + mx.arange(S)
        n_complete = (q_pos + 1) // self.ratio
        block_ids = mx.arange(n_blocks)
        visible = mx.broadcast_to(block_ids[None, None] < n_complete[None, :, None], scores.shape)
        scores = mx.where(visible, scores, -mx.inf)
        top = mx.argpartition(-scores, self.block_topk - 1, axis=-1)[..., : self.block_topk]
        picked = mx.take_along_axis(visible, top, axis=-1)
        keep = mx.put_along_axis(
            mx.zeros((B, S, n_blocks + 1), dtype=mx.bool_),
            mx.where(picked, top, n_blocks),
            mx.array(True),
            axis=-1,
        )[..., :n_blocks]
        keep = mx.repeat(keep, self.ratio, axis=-1)
        rest = kv_len - n_blocks * self.ratio
        if rest:
            keep = mx.concatenate([keep, mx.zeros((B, S, rest), dtype=mx.bool_)], axis=-1)
        # The query's own partial block is always visible
        tokens = mx.arange(kv_len)
        own = (tokens[None, :] >= (n_complete * self.ratio)[:, None]) & (tokens[None, :] <= q_pos[:, None])
        return (keep | own[None])[:, None]


class Attention(nn.Module):
    def __init__(self, args: TextArgs):
        super().__init__()
        self.n_heads = args.num_attention_heads
        self.n_kv_heads = args.num_key_value_heads
        self.head_dim = args.head_dim
        self.scale = self.head_dim**-0.5
        self.rotary_dims = int(self.head_dim * args.partial_rotary_factor)
        self.rope_theta = args.rope_theta
        dims = args.hidden_size
        # q (with its output gate) | k | v in one projection
        self.q_dim = 2 * self.n_heads * self.head_dim
        self.kv_dim = self.n_kv_heads * self.head_dim
        self.qkv_proj = nn.Linear(dims, self.q_dim + 2 * self.kv_dim, bias=False)
        self.o_proj = nn.Linear(self.n_heads * self.head_dim, dims, bias=False)
        self.q_norm = RMSNorm(self.head_dim, eps=args.rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=args.rms_norm_eps)
        self.indexer = QSAIndexer(args, self.rotary_dims)

    def __call__(self, x: mx.array, mask, cache: Optional[QSAKVCache]) -> mx.array:
        B, L, _ = x.shape
        sparse = self.indexer(x, cache)
        offset = cache.offset if cache is not None else 0

        qkv = self.qkv_proj(x)
        q, k, v = mx.split(qkv, [self.q_dim, self.q_dim + self.kv_dim], axis=-1)
        q, gate = mx.split(q.reshape(B, L, self.n_heads, -1), 2, axis=-1)
        gate = gate.reshape(B, L, -1)
        q = self.q_norm(q).transpose(0, 2, 1, 3)
        k = self.k_norm(k.reshape(B, L, self.n_kv_heads, -1)).transpose(0, 2, 1, 3)
        v = v.reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)
        q = mx.fast.rope(q, self.rotary_dims, traditional=False, base=self.rope_theta, scale=1.0, offset=offset)
        k = mx.fast.rope(k, self.rotary_dims, traditional=False, base=self.rope_theta, scale=1.0, offset=offset)
        if cache is not None:
            k, v = cache.update_and_fetch(k, v)

        if sparse is not None:
            # The selection is causal by construction, so it replaces the causal mask
            mask = sparse if mask is None or isinstance(mask, str) else (mask & sparse)
        out = scaled_dot_product_attention(q, k, v, cache=cache, scale=self.scale, mask=mask)
        out = out.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.o_proj(out * mx.sigmoid(gate))


# ------------------------------------------------------------------- GDN


class GatedDeltaNet(Qwen3_5GatedDeltaNet):
    """qwen3_5's GDN with a sigmoid output gate."""

    def __init__(self, args: TextArgs):
        super().__init__(args)
        self.norm = RMSNormGated(self.head_v_dim, eps=args.rms_norm_eps, activation=args.output_gate_type)

    def __call__(self, inputs, mask=None, cache=None, chain=None):
        B, S, _ = inputs.shape
        proj = qlinear(self.in_proj, inputs)
        if not self.training and fused_ops.gdn_in_ok(self, proj, mask, cache):
            out, _, z_off = self._mixer_fused(proj, cache, chain)
            z = proj[..., z_off : z_off + self.value_dim]
        else:
            out, z = self._mixer(proj, mask, cache, chain)
        z = z.reshape(B, S, self.num_v_heads, self.head_v_dim)
        return self.out_proj(self.norm(out, z).reshape(B, S, -1))


# ------------------------------------------------------------- decoder


class DecoderLayer(nn.Module):
    def __init__(self, args: TextArgs, layer_idx: int):
        super().__init__()
        self.is_linear = args.layer_types[layer_idx] == "linear_attention"
        if self.is_linear:
            self.linear_attn = GatedDeltaNet(args)
        else:
            self.self_attn = Attention(args)
        self.mlp = SparseMoeBlock(args)
        # ple_layer_ids are one-indexed
        if layer_idx + 1 in args.ple_layer_ids:
            self.ple = PLELayer(args, sorted(args.ple_layer_ids).index(layer_idx + 1))
        self.attn_hyper_connection = GatedResidual(args)
        self.mlp_hyper_connection = GatedResidual(args)

    def __call__(self, h: mx.array, mask, cache, ids: Optional[np.ndarray] = None):
        if "ple" in self:
            h = h + self.ple(h, ids, cache)
        x, inject = self.attn_hyper_connection(h)
        if self.is_linear:
            x = self.linear_attn(x, mask, cache)
        else:
            x = self.self_attn(x, mask, cache)
        h = self.attn_hyper_connection.combine(h, x, inject)
        x, inject = self.mlp_hyper_connection(h)
        return self.mlp_hyper_connection.combine(h, self.mlp(x), inject)


class Qwen4ExpModel(nn.Module):
    # Layers per mx.async_eval while generating (0 disables)
    eval_every = 8

    def __init__(self, args: TextArgs):
        super().__init__()
        self.args = args
        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [DecoderLayer(args, i) for i in range(args.num_hidden_layers)]
        # The final mixer stands in for a final norm
        self.hyper_connection_mixer = GatedResidual(args, use_combine=False)
        self.ssm_idx = next((i for i, l in enumerate(self.layers) if l.is_linear), None)
        self.fa_idx = next((i for i, l in enumerate(self.layers) if not l.is_linear), None)
        self.ple_idx = next((i for i, l in enumerate(self.layers) if "ple" in l), None)

    def __call__(self, inputs: mx.array, cache=None, input_embeddings=None, return_hidden=False):
        h = self.embed_tokens(inputs) if input_embeddings is None else input_embeddings
        if cache is None:
            cache = [None] * len(self.layers)
        fa_mask = create_attention_mask(h, cache[self.fa_idx]) if self.fa_idx is not None else None
        ssm_mask = create_ssm_mask(h, cache[self.ssm_idx]) if self.ssm_idx is not None else None
        # The n-gram hash needs the ids on the host: one sync per forward
        ids = np.array(inputs.tolist(), np.int64) if self.ple_idx is not None else None

        h = mx.tile(h, (1, 1, self.args.hc_count))
        chunk = 0 if (cache[0] is None or self.training) else self.eval_every
        first = self.ple_idx if self.ple_idx is not None else 0
        for i, (layer, c) in enumerate(zip(self.layers, cache)):
            h = layer(h, ssm_mask if layer.is_linear else fa_mask, c, ids)
            if chunk and i >= first and (i - first) % chunk == 0 and i < len(cache) - 1:
                mx.async_eval(h)
        out = self.hyper_connection_mixer(h)
        return (out, h) if return_hidden else out


class TextModel(nn.Module):
    def __init__(self, args: TextArgs):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.model = Qwen4ExpModel(args)
        if not args.tie_word_embeddings:
            self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)

    def __call__(self, inputs, cache=None, input_embeddings=None, return_hidden=False, chain=None):
        if chain is not None:
            raise ValueError("qwen4_exp does not verify sibling rows")
        out = self.model(inputs, cache, input_embeddings, return_hidden)
        hidden = out[1] if return_hidden else None
        out = out[0] if return_hidden else out
        if self.args.tie_word_embeddings:
            out = self.model.embed_tokens.as_linear(out)
        else:
            out = self.lm_head(out)
        return (out, hidden) if return_hidden else out

    @property
    def layers(self):
        return self.model.layers

    def make_cache(self):
        caches = []
        for layer in self.layers:
            if "ple" in layer:
                caches.append(PLECache())
            elif layer.is_linear:
                caches.append(ArraysCache(size=2))
            else:
                caches.append(QSAKVCache())
        return caches


class Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.language_model = TextModel(TextArgs.from_dict(args.text_config))
        self.ngram_prefix = "language_model.model.layers.{}"

    def __call__(self, inputs, cache=None, input_embeddings=None, return_hidden=False, chain=None):
        return self.language_model(inputs, cache, input_embeddings, return_hidden, chain)

    @property
    def model(self):
        return self.language_model.model

    @property
    def layers(self):
        return self.language_model.layers

    def make_cache(self):
        return self.language_model.make_cache()

    def sanitize(self, weights):
        out = {}
        for k, v in weights.items():
            if k.startswith(("vision_tower.", "model.visual.", "visual.")) or ".mtp." in k or k.startswith("mtp."):
                continue
            if k.startswith("model.language_model."):
                k = "language_model.model." + k[len("model.language_model.") :]
            elif k.startswith("lm_head."):
                k = "language_model." + k
            elif not k.startswith("language_model."):
                k = "language_model.model." + k
            # The n-gram table is read from disk (see NGramTable); its hash
            # buffers are derived from the config
            if ".ple.ple_embedding." in k:
                if _SHARD_MARKER in k:
                    self._resident_shard(k, v)
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
            if k.endswith("conv1d.weight") and v.ndim == 3 and v.shape[1] == 1:
                v = v.transpose(0, 2, 1)
            out[k] = v
        self._finish_resident()
        weights = fuse_projections(out)
        if self.args.model_path is not None:
            self.attach_table(self.args.model_path)
        return weights

    def _resident_shard(self, k, v):
        """Keep a small table (tests) in memory; a real one stays on disk."""
        layer, rest = k.split(_SHARD_MARKER)
        shard, name = rest.split(".")
        shards = self.__dict__.setdefault("_pending_shards", {}).setdefault(layer, {})
        shards.setdefault(int(shard), {})[name] = v

    def _finish_resident(self):
        pending = self.__dict__.pop("_pending_shards", {})
        for layer, shards in pending.items():
            parts = [shards[i] for i in sorted(shards)]
            rows = sum(p["weight"].shape[0] for p in parts)
            embedding = self.ngram_embedding(layer)
            if rows * self.width_bytes(parts[0]) > (1 << 30):
                continue
            if "scales" in parts[0]:
                table = mx.concatenate(
                    [
                        mx.dequantize(p["weight"], p["scales"], p.get("biases"), group_size=self._group(p), bits=4)
                        for p in parts
                    ]
                )
            else:
                table = mx.concatenate([p["weight"] for p in parts])
            embedding._resident = table

    @staticmethod
    def width_bytes(part):
        return part["weight"].shape[1] * part["weight"].dtype.size

    @staticmethod
    def _group(part):
        return part["weight"].shape[1] * 8 // part["scales"].shape[1]

    def ngram_embedding(self, prefix):
        layer = int(prefix.rsplit(".", 1)[1])
        return self.layers[layer].ple.ple_embedding

    def attach_table(self, model_path):
        """Memory map the n-gram table shards of the checkpoint at ``model_path``."""
        for i, layer in enumerate(self.layers):
            if "ple" in layer:
                layer.ple.ple_embedding.attach(model_path, self.ngram_prefix.format(i))

    @property
    def quant_predicate(self):
        def predicate(path, _):
            if path.endswith("mlp.gate") or path.endswith("shared_expert_gate"):
                return {"group_size": 64, "bits": 8}
            if _SHARD_MARKER in path:
                return {"group_size": 32, "bits": 4}
            return True

        return predicate

    @property
    def cast_predicate(self):
        return lambda path: not path.endswith("A_log")
