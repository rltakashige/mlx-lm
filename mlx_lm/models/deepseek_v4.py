import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import mlx.core as mx
import mlx.nn as nn

from .base import BaseModelArgs, scaled_dot_product_attention
from .cache import _BaseCache, RotatingKVCache
from .switch_layers import QuantizedSwitchLinear, SwiGLU, SwitchGLU, _gather_sort, _scatter_unsort


# Register a minimal HF config so AutoConfig / AutoTokenizer recognize
# ``deepseek_v4`` until huggingface/transformers#45616 merges. Once that PR
# ships, this registration becomes a no-op (``exist_ok=True``).
try:
    from transformers import AutoConfig, PretrainedConfig

    class _DeepseekV4HFConfig(PretrainedConfig):
        model_type = "deepseek_v4"

        def __init__(self, rope_scaling=None, **kwargs):
            self.rope_scaling = rope_scaling
            super().__init__(**kwargs)

    AutoConfig.register("deepseek_v4", _DeepseekV4HFConfig, exist_ok=True)
except ImportError:
    pass


@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str = "deepseek_v4"
    vocab_size: int = 129280
    hidden_size: int = 4096
    num_hidden_layers: int = 43
    num_attention_heads: int = 64
    num_key_value_heads: int = 1

    # MLA-style attention
    q_lora_rank: int = 1024
    o_lora_rank: int = 1024
    o_groups: int = 8
    head_dim: int = 512
    qk_rope_head_dim: int = 64
    attention_bias: bool = False
    sliding_window: int = 128
    compress_ratios: List[int] = field(default_factory=list)

    # Compressor / Indexer
    index_n_heads: int = 64
    index_head_dim: int = 128
    index_topk: int = 512
    compress_rope_theta: float = 160000.0

    # MoE
    moe_intermediate_size: int = 2048
    n_routed_experts: int = 256
    n_shared_experts: int = 1
    num_experts_per_tok: int = 6
    num_hash_layers: int = 3
    scoring_func: str = "sqrtsoftplus"
    topk_method: str = "noaux_tc"
    norm_topk_prob: bool = True
    routed_scaling_factor: float = 1.5
    swiglu_limit: float = 10.0

    # Hyper-Connections
    hc_mult: int = 4
    hc_sinkhorn_iters: int = 20
    hc_eps: float = 1e-6

    # MTP (dropped in sanitize)
    num_nextn_predict_layers: int = 1

    # RoPE / YaRN
    max_position_embeddings: int = 1048576
    rope_theta: float = 10000.0
    rope_scaling: Optional[Dict] = None
    rms_norm_eps: float = 1e-6

    quantization_config: Optional[Dict] = None


# --------------------------------------------------------------------------- #
# RoPE (traditional pair-wise, YaRN-aware, supports inverse rotation)         #
# --------------------------------------------------------------------------- #


class DeepseekV4RoPE(nn.Module):
    """Traditional (consecutive-pair) RoPE with optional YaRN frequency scaling
    and an ``inverse=True`` path that applies the conjugate rotation. V4 rotates
    K (which is also V since K==V for the shared head) and un-rotates the
    attention output on the same dims.

    Only the last ``dims`` of the input are rotated; preceding dims pass through.
    """

    def __init__(self, dims: int, base: float, scaling_config: Optional[Dict] = None):
        super().__init__()
        self.dims = dims
        inv_freq = 1.0 / (base ** (mx.arange(0, dims, 2, dtype=mx.float32) / dims))
        rope_type = None
        if scaling_config is not None:
            rope_type = scaling_config.get("type") or scaling_config.get("rope_type")

        if rope_type in ("yarn", "deepseek_yarn"):
            factor = scaling_config["factor"]
            orig = scaling_config["original_max_position_embeddings"]
            beta_fast = scaling_config.get("beta_fast", 32)
            beta_slow = scaling_config.get("beta_slow", 1)

            def correction_dim(num_rotations):
                return (
                    dims
                    * math.log(orig / (num_rotations * 2 * math.pi))
                    / (2 * math.log(base))
                )

            low = max(math.floor(correction_dim(beta_fast)), 0)
            high = min(math.ceil(correction_dim(beta_slow)), dims - 1)
            if low == high:
                high += 0.001

            ramp = (mx.arange(dims // 2, dtype=mx.float32) - low) / (high - low)
            smooth = 1 - mx.clip(ramp, 0, 1)
            inv_freq = inv_freq / factor * (1 - smooth) + inv_freq * smooth
        elif rope_type not in (None, "default", "linear"):
            raise ValueError(f"Unsupported DeepSeek-V4 RoPE type {rope_type!r}")

        # Tuple-wrap keeps inv_freq out of the module's parameter tree.
        self._inv_freq = (inv_freq,)

    @property
    def inv_freq(self) -> mx.array:
        return self._inv_freq[0]

    def __call__(self, x: mx.array, offset: int = 0, inverse: bool = False) -> mx.array:
        dtype = x.dtype
        T = x.shape[-2]
        pos = mx.arange(offset, offset + T, dtype=mx.float32)
        theta = pos[:, None] * self.inv_freq[None, :]
        if inverse:
            theta = -theta

        cos = mx.cos(theta)
        sin = mx.sin(theta)
        broadcast_shape = (1,) * (x.ndim - 2) + theta.shape
        cos = cos.reshape(broadcast_shape).astype(dtype)
        sin = sin.reshape(broadcast_shape).astype(dtype)

        rot = x[..., : self.dims].reshape(*x.shape[:-1], self.dims // 2, 2)
        x0, x1 = rot[..., 0], rot[..., 1]
        y = mx.stack((x0 * cos - x1 * sin, x0 * sin + x1 * cos), axis=-1)
        y = y.reshape(*x.shape[:-1], self.dims)
        if x.shape[-1] == self.dims:
            return y
        return mx.concatenate([y, x[..., self.dims :]], axis=-1)


# --------------------------------------------------------------------------- #
# Sinkhorn-based mHC (Manifold-constrained Hyper-Connections)                 #
# --------------------------------------------------------------------------- #


def hc_split_sinkhorn(
    mixes: mx.array,
    hc_scale: mx.array,
    hc_base: mx.array,
    hc_mult: int,
    iters: int,
    eps: float,
):
    """Reference algorithm (from DeepSeek V4 /inference/kernel.py):

        pre  = sigmoid(mixes[:, :hc]         * scale[0] + base[:hc]) + eps
        post = 2 * sigmoid(mixes[:, hc:2hc]  * scale[1] + base[hc:2hc])
        comb = mixes[:, 2hc:].reshape(hc,hc) * scale[2] + base[2hc:].reshape(hc,hc)
        comb = softmax(comb, axis=-1) + eps
        comb = comb / (comb.sum(-2) + eps)                 # initial col norm
        for _ in range(iters - 1):
            comb = comb / (comb.sum(-1) + eps)             # row norm
            comb = comb / (comb.sum(-2) + eps)             # col norm

    ``mixes`` has a leading batch shape ``[..., (2+hc)*hc]``; ``pre``/``post``
    end up ``[..., hc]`` and ``comb`` ``[..., hc, hc]``.
    """
    s0 = hc_scale[0]
    s1 = hc_scale[1]
    s2 = hc_scale[2]

    pre = mx.sigmoid(mixes[..., :hc_mult] * s0 + hc_base[:hc_mult]) + eps
    post = 2 * mx.sigmoid(
        mixes[..., hc_mult : 2 * hc_mult] * s1 + hc_base[hc_mult : 2 * hc_mult]
    )

    comb_logits = mixes[..., 2 * hc_mult :].reshape(
        *mixes.shape[:-1], hc_mult, hc_mult
    ) * s2 + hc_base[2 * hc_mult :].reshape(hc_mult, hc_mult)

    comb = mx.softmax(comb_logits, axis=-1, precise=True) + eps
    comb = comb / (comb.sum(axis=-2, keepdims=True) + eps)
    for _ in range(iters - 1):
        comb = comb / (comb.sum(axis=-1, keepdims=True) + eps)
        comb = comb / (comb.sum(axis=-2, keepdims=True) + eps)

    return pre, post, comb


class HyperConnection(nn.Module):
    """Per-block mHC: projects an ``[..., hc, D]`` state to ``pre``/``post``/``comb``.

    Parameter layout matches the reference checkpoint naming:
        fn    : ``[mix_hc, hc*D]``   (fp32)
        base  : ``[mix_hc]``         (fp32)
        scale : ``[3]``              (fp32)
    where ``mix_hc = (2 + hc_mult) * hc_mult``.
    """

    def __init__(
        self,
        dim: int,
        hc_mult: int,
        norm_eps: float,
        sinkhorn_iters: int,
        hc_eps: float,
    ):
        super().__init__()
        self.dim = dim
        self.hc_mult = hc_mult
        self.norm_eps = norm_eps
        self.sinkhorn_iters = sinkhorn_iters
        self.hc_eps = hc_eps
        mix_hc = (2 + hc_mult) * hc_mult
        hc_dim = hc_mult * dim
        self.fn = mx.zeros((mix_hc, hc_dim), dtype=mx.float32)
        self.base = mx.zeros((mix_hc,), dtype=mx.float32)
        self.scale = mx.zeros((3,), dtype=mx.float32)

    def hc_pre(self, x: mx.array):
        # x: [B, S, hc, D]  ->  (y [B, S, D], post [B, S, hc], comb [B, S, hc, hc])
        B, S, hc, D = x.shape
        dtype = x.dtype
        xf = x.reshape(B, S, hc * D).astype(mx.float32)
        inv = mx.rsqrt(xf.square().mean(axis=-1, keepdims=True) + self.norm_eps)
        mixes = (xf @ self.fn.T) * inv
        pre, post, comb = hc_split_sinkhorn(
            mixes, self.scale, self.base, hc, self.sinkhorn_iters, self.hc_eps
        )
        y = (pre[..., None] * x.astype(mx.float32)).sum(axis=2)
        return y.astype(dtype), post, comb

    def hc_post(
        self,
        f_out: mx.array,
        residual: mx.array,
        post: mx.array,
        comb: mx.array,
    ):
        # f_out    [B, S, D]
        # residual [B, S, hc, D]
        # post     [B, S, hc]      -> broadcast to [B, S, hc, D]
        # comb     [B, S, hc, hc]
        dtype = f_out.dtype
        term_new = post[..., None] * f_out[:, :, None, :].astype(mx.float32)
        term_res = mx.einsum(
            "bsij,bsjd->bsid",
            comb.astype(mx.float32),
            residual.astype(mx.float32),
        )
        return (term_new + term_res).astype(dtype)


class HyperHead(nn.Module):
    """Final head mHC: reduces ``[B, S, hc, D]`` -> ``[B, S, D]`` via sigmoid-
    weighted sum (no Sinkhorn normalization)."""

    def __init__(self, dim: int, hc_mult: int, norm_eps: float, hc_eps: float):
        super().__init__()
        self.dim = dim
        self.hc_mult = hc_mult
        self.norm_eps = norm_eps
        self.hc_eps = hc_eps
        self.fn = mx.zeros((hc_mult, hc_mult * dim), dtype=mx.float32)
        self.base = mx.zeros((hc_mult,), dtype=mx.float32)
        self.scale = mx.zeros((1,), dtype=mx.float32)

    def __call__(self, x: mx.array) -> mx.array:
        B, S, hc, D = x.shape
        dtype = x.dtype
        xf = x.reshape(B, S, hc * D).astype(mx.float32)
        inv = mx.rsqrt(xf.square().mean(axis=-1, keepdims=True) + self.norm_eps)
        mixes = (xf @ self.fn.T) * inv
        pre = mx.sigmoid(mixes * self.scale[0] + self.base) + self.hc_eps
        return (pre[..., None] * x.astype(mx.float32)).sum(axis=2).astype(dtype)


# --------------------------------------------------------------------------- #
# Cache                                                                        #
# --------------------------------------------------------------------------- #


class DeepseekV4Cache(_BaseCache):
    """Cache for layers with ``compress_ratio > 0``.

    Tracks four things:

    - ``window_kv``: rotating ring buffer of the last ``window`` raw KV rows,
      stored in temporal order via explicit temporal roll on fetch.
    - ``compressed_kv``: growing buffer of compressed KV rows, appended at
      prefill (bulk) and during decode once every ``ratio`` tokens.
    - ``comp_kv_state`` / ``comp_score_state``: fp32 accumulators the
      Compressor writes between decode steps. When ``overlap`` is on (ratio=4),
      the buffer holds ``2*ratio`` slots; the first ``ratio`` are the previous
      window and the next ``ratio`` the one currently filling.
    - Mirror buffers (``idx_*``) for the Indexer's own Compressor in ratio=4
      layers.

    ``offset`` is the total number of raw tokens ingested, matching the
    convention of the other mlx-lm caches.
    """

    def __init__(
        self,
        window: int,
        ratio: int,
        head_dim: int,
        has_indexer: bool,
        index_head_dim: int,
    ):
        self.window = window
        self.ratio = ratio
        self.head_dim = head_dim
        self.has_indexer = has_indexer
        self.index_head_dim = index_head_dim
        self.overlap = ratio == 4
        self.offset = 0

        self.window_kv: Optional[mx.array] = None
        self.compressed_kv: Optional[mx.array] = None
        self.compressed_len = 0

        coff = 2 if self.overlap else 1
        self._coff = coff
        self.comp_kv_state: Optional[mx.array] = None
        self.comp_score_state: Optional[mx.array] = None

        if has_indexer:
            self.idx_compressed_kv: Optional[mx.array] = None
            self.idx_compressed_len = 0
            self.idx_kv_state: Optional[mx.array] = None
            self.idx_score_state: Optional[mx.array] = None

    # ------------------------------------------------------------------ #
    # window KV                                                           #
    # ------------------------------------------------------------------ #
    def prefill_window(self, kv: mx.array) -> mx.array:
        """Prefill path: attend over the full raw KV. Persist the tail so the
        next decode step has the right sliding-window context."""
        self.window_kv = (
            kv[:, -self.window :, :] if kv.shape[1] > self.window else kv
        )
        return kv

    def decode_window(self, kv: mx.array) -> mx.array:
        """Decode path: extend the stored window with ``kv`` and trim to
        ``window``. Returns the updated window for attention."""
        if self.window_kv is None:
            merged = kv
        else:
            merged = mx.concatenate([self.window_kv, kv], axis=1)
        if merged.shape[1] > self.window:
            merged = merged[:, -self.window :, :]
        self.window_kv = merged
        return merged

    # ------------------------------------------------------------------ #
    # compressed KV                                                       #
    # ------------------------------------------------------------------ #
    def append_compressed(self, kv: mx.array, use_indexer_buffer: bool = False):
        if use_indexer_buffer:
            buf = self.idx_compressed_kv
            if buf is None:
                self.idx_compressed_kv = kv
            else:
                self.idx_compressed_kv = mx.concatenate([buf, kv], axis=1)
            self.idx_compressed_len = self.idx_compressed_kv.shape[1]
        else:
            buf = self.compressed_kv
            if buf is None:
                self.compressed_kv = kv
            else:
                self.compressed_kv = mx.concatenate([buf, kv], axis=1)
            self.compressed_len = self.compressed_kv.shape[1]

    # ------------------------------------------------------------------ #
    # mlx-lm _BaseCache interface                                         #
    # ------------------------------------------------------------------ #
    def make_mask(self, *args, **kwargs):
        # Masking is encoded via topk indices (-1 = masked), so no causal mask.
        return None

    def empty(self):
        return self.window_kv is None

    @property
    def nbytes(self):
        total = 0
        if self.window_kv is not None:
            total += self.window_kv.nbytes
        if self.compressed_kv is not None:
            total += self.compressed_kv.nbytes
        if self.has_indexer and self.idx_compressed_kv is not None:
            total += self.idx_compressed_kv.nbytes
        return total


# --------------------------------------------------------------------------- #
# Compressor                                                                  #
# --------------------------------------------------------------------------- #


class Compressor(nn.Module):
    """Learned gated pooling over ``ratio`` consecutive tokens.

    Prefill: chunk the input into windows of ``ratio`` tokens, softmax-gate
    across each window, sum to one compressed row per window; any tail shorter
    than ``ratio`` lives in the cache's ``comp_*_state`` buffer until enough
    tokens accumulate.

    Decode (S==1): append the new token's kv/score into the accumulator; if we
    just filled a window, emit one compressed row and rotate the buffer.
    """

    def __init__(
        self,
        dim: int,
        compress_ratio: int,
        head_dim: int,
        rope_head_dim: int,
        rms_norm_eps: float,
        rope: "DeepseekV4RoPE",
    ):
        super().__init__()
        self.dim = dim
        self.head_dim = head_dim
        self.rope_head_dim = rope_head_dim
        self.compress_ratio = compress_ratio
        self.overlap = compress_ratio == 4
        coff = 2 if self.overlap else 1
        self._coff = coff
        self.wkv = nn.Linear(dim, coff * head_dim, bias=False)
        self.wgate = nn.Linear(dim, coff * head_dim, bias=False)
        self.ape = mx.zeros((compress_ratio, coff * head_dim), dtype=mx.float32)
        self.norm = nn.RMSNorm(head_dim, eps=rms_norm_eps)
        self.rope = rope  # shared with attention, applies compress_rope_theta + YaRN

    def _overlap_transform_kv(self, kv: mx.array) -> mx.array:
        # kv: [B, n_windows, ratio, coff*d]; output [B, n_windows, 2*ratio, d]
        B, S, R, _ = kv.shape
        d = self.head_dim
        out = mx.zeros((B, S, 2 * R, d), dtype=kv.dtype)
        # second half: "current" window's last-d dims
        out[:, :, R:, :] = kv[:, :, :, d:]
        # first half: previous window's first-d dims
        out[:, 1:, :R, :] = kv[:, :-1, :, :d]
        return out

    def _overlap_transform_score(self, score: mx.array) -> mx.array:
        B, S, R, _ = score.shape
        d = self.head_dim
        out = mx.full((B, S, 2 * R, d), float("-inf"), dtype=score.dtype)
        out[:, :, R:, :] = score[:, :, :, d:]
        out[:, 1:, :R, :] = score[:, :-1, :, :d]
        return out

    def _apply_compressor_rope(
        self, compressed_kv: mx.array, offset_positions: mx.array
    ) -> mx.array:
        """Apply RoPE on the last ``rope_head_dim`` dims at the given positions.

        ``offset_positions`` is a 1-D int array of compressed positions (each
        compressed row's anchoring raw-token index).
        """
        rd = self.rope_head_dim
        out = compressed_kv
        # Treat as [B, 1, T, head_dim] so the rope's ``offset`` mechanism works
        # across a run of consecutive positions.
        # For prefill we have multiple rows with strictly increasing positions
        # at a fixed stride `ratio`, so we can apply rope in one shot with an
        # explicit theta table.
        B, T, D = out.shape
        dims = rd
        pos = offset_positions.astype(mx.float32)
        theta = pos[:, None] * self.rope.inv_freq[None, :]
        cos = mx.cos(theta).reshape(1, T, dims // 2).astype(out.dtype)
        sin = mx.sin(theta).reshape(1, T, dims // 2).astype(out.dtype)
        rope_part = out[..., -rd:].reshape(B, T, dims // 2, 2)
        x0, x1 = rope_part[..., 0], rope_part[..., 1]
        rotated = mx.stack((x0 * cos - x1 * sin, x0 * sin + x1 * cos), axis=-1)
        rotated = rotated.reshape(B, T, dims)
        return mx.concatenate([out[..., :-rd], rotated], axis=-1)

    def __call__(
        self,
        x: mx.array,
        cache: DeepseekV4Cache,
        offset: int,
        use_indexer_buffer: bool = False,
    ) -> Optional[mx.array]:
        """Ingest ``x`` and, if appropriate, emit compressed KV rows.

        Returns the freshly-produced compressed KV chunk (``[B, n_new, d]``)
        or ``None`` if nothing was produced this call. The caller is
        responsible for concatenating the result into the cache buffer (it is
        usually more convenient to let this routine do the append — we do both
        so the Attention module can still access the newly-emitted rows).
        """
        B, S, _ = x.shape
        xf = x.astype(mx.float32)
        kv = self.wkv(xf)
        score = self.wgate(xf)
        ratio = self.compress_ratio
        overlap = self.overlap
        d = self.head_dim
        coff_d = self._coff * d

        if use_indexer_buffer:
            state_kv = cache.idx_kv_state
            state_score = cache.idx_score_state
        else:
            state_kv = cache.comp_kv_state
            state_score = cache.comp_score_state

        # Lazy-init state buffers
        if state_kv is None:
            n_slots = self._coff * ratio
            state_kv = mx.zeros((B, n_slots, coff_d), dtype=mx.float32)
            state_score = mx.full((B, n_slots, coff_d), float("-inf"), dtype=mx.float32)

        if offset == 0:
            # Prefill: split into head_tokens (fully windowable) + tail (goes to state).
            remainder = S % ratio
            cutoff = S - remainder
            out_compressed = None

            if cutoff > 0:
                # carry: for overlap, we need the _last_ window of the head to
                # be available as the "previous" window for decode-phase overlap.
                kv_head = kv[:, :cutoff]  # [B, cutoff, coff_d]
                score_head = score[:, :cutoff]
                kv_head = kv_head.reshape(B, cutoff // ratio, ratio, coff_d)
                score_head = (
                    score_head.reshape(B, cutoff // ratio, ratio, coff_d) + self.ape
                )
                if overlap:
                    kv_trans = self._overlap_transform_kv(kv_head)
                    score_trans = self._overlap_transform_score(score_head)
                    weights = mx.softmax(score_trans, axis=2, precise=True)
                    compressed = (kv_trans * weights).sum(axis=2)  # [B, nw, d]
                else:
                    weights = mx.softmax(score_head, axis=2, precise=True)
                    compressed = (kv_head * weights).sum(axis=2)
                    compressed = compressed[..., :d]
                compressed = self.norm(compressed.astype(x.dtype))
                # Apply RoPE: compressed row k anchors at raw position
                # (k+1)*ratio - 1 for non-overlap, k*ratio + ratio - 1 for overlap
                # (matches ref: self.freqs_cis[:cutoff:ratio] = positions 0, r, 2r, ... ratio-1 actually...
                # Reference slices freqs_cis[:cutoff:ratio] which gives positions 0, r, 2r, ...
                # So compressed row k has position k*ratio.
                n_rows = compressed.shape[1]
                pos_anchor = mx.arange(0, n_rows, dtype=mx.int32) * ratio
                compressed = self._apply_compressor_rope(compressed, pos_anchor)
                out_compressed = compressed
                cache.append_compressed(compressed, use_indexer_buffer=use_indexer_buffer)

            # Stash tail into state
            if remainder > 0:
                tail_kv = kv[:, cutoff:, :]
                tail_score = score[:, cutoff:, :] + self.ape[:remainder]
                start_slot = ratio if overlap else 0
                state_kv[:, start_slot : start_slot + remainder, :] = tail_kv.astype(
                    mx.float32
                )
                state_score[
                    :, start_slot : start_slot + remainder, :
                ] = tail_score.astype(mx.float32)

            # On overlap, also stash the last full window's rows for future overlap.
            if overlap and cutoff > 0:
                prev_kv = kv[:, cutoff - ratio : cutoff, :]
                prev_score = score[:, cutoff - ratio : cutoff, :] + self.ape
                state_kv[:, :ratio, :] = prev_kv.astype(mx.float32)
                state_score[:, :ratio, :] = prev_score.astype(mx.float32)

            if use_indexer_buffer:
                cache.idx_kv_state = state_kv
                cache.idx_score_state = state_score
            else:
                cache.comp_kv_state = state_kv
                cache.comp_score_state = state_score
            return out_compressed

        # Decode path: S == 1
        assert S == 1, "Compressor decode expects single-token steps"
        pos_in_window = offset % ratio
        ape_row = self.ape[pos_in_window]
        new_kv = kv[:, 0, :].astype(mx.float32)
        new_score = (score[:, 0, :] + ape_row).astype(mx.float32)

        if overlap:
            slot = ratio + pos_in_window
        else:
            slot = pos_in_window
        state_kv[:, slot, :] = new_kv
        state_score[:, slot, :] = new_score

        should_compress = ((offset + 1) % ratio) == 0
        out_compressed = None

        if should_compress:
            if overlap:
                # Gather overlap-view: first half d from prev window, second half d from current
                first = state_kv[:, :ratio, :d]  # previous window, first d dims
                second = state_kv[:, ratio:, d:]  # current window, second d dims
                merged_kv = mx.concatenate([first, second], axis=1)  # [B, 2r, d]
                first_s = state_score[:, :ratio, :d]
                second_s = state_score[:, ratio:, d:]
                merged_score = mx.concatenate([first_s, second_s], axis=1)
                weights = mx.softmax(merged_score, axis=1, precise=True)
                compressed = (merged_kv * weights).sum(axis=1, keepdims=True)  # [B,1,d]
            else:
                weights = mx.softmax(state_score, axis=1, precise=True)
                compressed = (state_kv * weights).sum(axis=1, keepdims=True)
                compressed = compressed[..., :d]
            compressed = self.norm(compressed.astype(x.dtype))
            # Position of new compressed row anchor:
            # Reference uses freqs_cis[start_pos + 1 - ratio] (start of current window).
            anchor = mx.array([offset + 1 - ratio], dtype=mx.int32)
            compressed = self._apply_compressor_rope(compressed, anchor)
            out_compressed = compressed
            cache.append_compressed(compressed, use_indexer_buffer=use_indexer_buffer)

            if overlap:
                # Rotate: "current" becomes "previous"
                state_kv[:, :ratio, :] = state_kv[:, ratio:, :]
                state_score[:, :ratio, :] = state_score[:, ratio:, :]

        if use_indexer_buffer:
            cache.idx_kv_state = state_kv
            cache.idx_score_state = state_score
        else:
            cache.comp_kv_state = state_kv
            cache.comp_score_state = state_score

        return out_compressed


# --------------------------------------------------------------------------- #
# Indexer                                                                     #
# --------------------------------------------------------------------------- #


class Indexer(nn.Module):
    """Produces top-k compressed-row indices for the main attention."""

    def __init__(
        self,
        args: ModelArgs,
        compress_ratio: int,
        rope: "DeepseekV4RoPE",
    ):
        super().__init__()
        self.dim = args.hidden_size
        self.n_heads = args.index_n_heads
        self.head_dim = args.index_head_dim
        self.rope_head_dim = args.qk_rope_head_dim
        self.index_topk = args.index_topk
        self.q_lora_rank = args.q_lora_rank
        self.compress_ratio = compress_ratio
        self.softmax_scale = self.head_dim ** -0.5
        self.rope = rope

        self.wq_b = nn.Linear(
            args.q_lora_rank, self.n_heads * self.head_dim, bias=False
        )
        self.weights_proj = nn.Linear(args.hidden_size, self.n_heads, bias=False)
        self.compressor = Compressor(
            dim=args.hidden_size,
            compress_ratio=compress_ratio,
            head_dim=self.head_dim,
            rope_head_dim=args.qk_rope_head_dim,
            rms_norm_eps=args.rms_norm_eps,
            rope=rope,
        )

    def __call__(
        self,
        x: mx.array,
        qr: mx.array,
        cache: DeepseekV4Cache,
        offset: int,
    ) -> mx.array:
        """Return ``topk_idxs [B, S, index_topk]`` of int32 referencing the main
        attention's combined KV buffer (window followed by compressed rows).
        ``-1`` entries are masked out."""
        B, S, _ = x.shape
        ratio = self.compress_ratio
        rd = self.rope_head_dim
        # q via low-rank Q-LoRA shared with the attention
        q = self.wq_b(qr).reshape(B, S, self.n_heads, self.head_dim)
        q = mx.concatenate(
            [q[..., :-rd], self.rope(q[..., -rd:], offset=offset)], axis=-1
        )

        # Update the indexer's own compressed-KV cache
        self.compressor(x, cache, offset, use_indexer_buffer=True)
        idx_kv = cache.idx_compressed_kv  # [B, T, index_head_dim] (or None)

        per_head_weights = self.weights_proj(x) * (
            self.softmax_scale * (self.n_heads ** -0.5)
        )  # [B, S, n_heads]

        if idx_kv is None or idx_kv.shape[1] == 0:
            # No compressed rows yet — pass through -1s so the attention falls
            # back to the window-only indices.
            return mx.full((B, S, self.index_topk), -1, dtype=mx.int32)

        # index_score[b, s, h, t] = <q[b,s,h,:], idx_kv[b,t,:]>
        score = mx.einsum("bshd,btd->bsht", q.astype(idx_kv.dtype), idx_kv)
        score = mx.maximum(score, 0)  # ReLU
        score = (score * per_head_weights[..., None]).sum(axis=2)  # [B, S, T]

        # Mask future compressed rows during prefill.
        if offset == 0 and S > 1:
            s_range = mx.arange(1, S + 1, dtype=mx.int32)
            t_range = mx.arange(score.shape[-1], dtype=mx.int32)
            # row t is produced from raw positions [t*ratio, t*ratio + ratio - 1], so it
            # is visible to query at raw position s (1-indexed as s_range-1) iff
            # (t+1)*ratio <= s_range (mirrors the ref: `matrix >= arange(1, s+1)//ratio`)
            future = t_range[None, :] >= (s_range[:, None] // ratio)
            score = mx.where(future[None, :, :], mx.array(-mx.inf, mx.float32), score)

        topk = min(self.index_topk, idx_kv.shape[1])
        # mx.argpartition returns indices of the k smallest (along axis). We
        # want the k largest of ``score``, so partition by ``-score``.
        part = mx.argpartition(-score, kth=topk - 1, axis=-1)[..., :topk]
        # If the corresponding score is -inf (future-masked), mark -1.
        gathered_scores = mx.take_along_axis(score, part, axis=-1)
        top_idxs = mx.where(
            mx.isinf(gathered_scores) & (gathered_scores < 0),
            mx.array(-1, mx.int32),
            part.astype(mx.int32),
        )
        if topk < self.index_topk:
            pad = mx.full((B, S, self.index_topk - topk), -1, dtype=mx.int32)
            top_idxs = mx.concatenate([top_idxs, pad], axis=-1)
        return top_idxs.astype(mx.int32)


# --------------------------------------------------------------------------- #
# Attention                                                                   #
# --------------------------------------------------------------------------- #


def _get_window_topk_idxs(
    window: int, B: int, S: int, offset: int, window_cache_len: int
) -> mx.array:
    """Return ``[B, S, window]`` of indices into the window portion of ``kv_all``.

    Cache layout: entries ``[0..window_cache_len-1]`` are the last
    ``window_cache_len`` raw tokens in temporal order, ending at raw position
    ``offset + S - 1``. For query ``s`` (raw pos ``offset + s``), the valid
    sliding window is the `window` raw positions ``[offset+s-window+1, offset+s]``.
    In cache-index space that maps to the range ``[upper-window+1, upper]``
    where ``upper = s + window_cache_len - S``. Out-of-range slots are ``-1``.
    """
    upper = mx.arange(S, dtype=mx.int32) + (window_cache_len - S)  # [S]
    col_offsets = mx.arange(window, dtype=mx.int32) - (window - 1)  # [-W+1, ..., 0]
    idxs = upper[:, None] + col_offsets[None, :]  # [S, window]
    neg_one = mx.array(-1, mx.int32)
    idxs = mx.where(idxs < 0, neg_one, idxs)
    idxs = mx.where(idxs >= window_cache_len, neg_one, idxs)
    return mx.broadcast_to(idxs[None, ...], (B, S, window))


def _get_compress_topk_idxs(
    ratio: int,
    B: int,
    S: int,
    offset: int,
    compressed_cache_len: int,
    compress_idx_offset: int,
) -> mx.array:
    """For ratio=128 (no indexer), deterministic mapping: query at raw
    position ``offset+i`` can see compressed rows whose anchor raw position is
    at most ``offset+i - ratio + 1`` (row k anchors at raw position k*ratio
    during prefill, or at position k*ratio during decode — both give the same
    visibility rule ``(k+1)*ratio <= offset+i+1``).

    Returns ``[B, S, compressed_cache_len]`` of indices into the combined KV
    buffer (compressed_idx_offset + cache_row_k), -1 where not visible.
    """
    q_pos = offset + mx.arange(S, dtype=mx.int32)  # [S]
    k = mx.arange(compressed_cache_len, dtype=mx.int32)  # [T]
    # visible iff (k+1)*ratio <= q_pos + 1
    visible = (k + 1)[None, :] * ratio <= (q_pos + 1)[:, None]  # [S, T]
    idxs = mx.where(
        visible,
        k[None, :] + compress_idx_offset,
        mx.array(-1, mx.int32),
    )
    return mx.broadcast_to(idxs[None, ...], (B, S, compressed_cache_len))


def _sparse_attention(
    q: mx.array,
    kv_all: mx.array,
    topk_idxs: mx.array,
    attn_sink: mx.array,
    scale: float,
) -> mx.array:
    """Pure-MLX sparse attention.

    Args:
        q:          [B, n_heads, S, D]
        kv_all:     [B, N, D]               (single shared K=V head)
        topk_idxs:  [B, S, K]   int32       (-1 = masked)
        attn_sink:  [n_heads]               additive term in softmax denominator
        scale:      float

    Returns: [B, n_heads, S, D]
    """
    B, H, S, D = q.shape
    N = kv_all.shape[1]
    K = topk_idxs.shape[-1]

    # Gather per-query KV rows.
    clamped = mx.maximum(topk_idxs, 0)  # [B, S, K]
    flat_idxs = clamped + (mx.arange(B, dtype=clamped.dtype) * N)[:, None, None]
    gathered = kv_all.reshape(B * N, D)[flat_idxs.reshape(-1)]
    gathered = gathered.reshape(B, S, K, D)  # [B, S, K, D]

    # scores[b, h, s, k] = <q[b,h,s,:], gathered[b,s,k,:]>
    scores = mx.einsum("bhsd,bskd->bhsk", q, gathered) * scale

    # Mask invalid slots
    valid = (topk_idxs >= 0)[:, None, :, :]  # [B, 1, S, K]
    neg_inf = mx.array(-mx.inf, dtype=scores.dtype)
    scores = mx.where(valid, scores, neg_inf)

    sink = attn_sink.astype(scores.dtype).reshape(1, H, 1, 1)
    # Numerically stable softmax with an extra per-head sink slot in the denom.
    max_s = mx.maximum(scores.max(axis=-1, keepdims=True), sink)
    scores_exp = mx.exp(scores - max_s)
    sink_exp = mx.exp(sink - max_s)
    denom = scores_exp.sum(axis=-1, keepdims=True) + sink_exp
    attn = scores_exp / denom

    # Output = sum_k attn[...,k] * gathered[...,k,:]
    out = mx.einsum("bhsk,bskd->bhsd", attn, gathered)
    return out


class V4Attention(nn.Module):
    def __init__(self, args: ModelArgs, layer_id: int):
        super().__init__()
        self.args = args
        self.layer_id = layer_id
        self.dim = args.hidden_size
        self.n_heads = args.num_attention_heads
        self.head_dim = args.head_dim
        self.rope_head_dim = args.qk_rope_head_dim
        self.nope_head_dim = args.head_dim - args.qk_rope_head_dim
        self.n_groups = args.o_groups
        self.q_lora_rank = args.q_lora_rank
        self.o_lora_rank = args.o_lora_rank
        self.window = args.sliding_window
        self.eps = args.rms_norm_eps
        self.scale = self.head_dim ** -0.5

        ratios = args.compress_ratios or []
        self.compress_ratio = ratios[layer_id] if layer_id < len(ratios) else 0

        self.wq_a = nn.Linear(self.dim, self.q_lora_rank, bias=args.attention_bias)
        self.q_norm = nn.RMSNorm(self.q_lora_rank, eps=self.eps)
        self.wq_b = nn.Linear(self.q_lora_rank, self.n_heads * self.head_dim, bias=False)

        self.wkv = nn.Linear(self.dim, self.head_dim, bias=False)
        self.kv_norm = nn.RMSNorm(self.head_dim, eps=self.eps)

        self.attn_sink = mx.zeros((self.n_heads,), dtype=mx.float32)

        group_feat = (self.n_heads * self.head_dim) // self.n_groups
        self.wo_a = nn.Linear(group_feat, self.n_groups * self.o_lora_rank, bias=False)
        self.wo_b = nn.Linear(
            self.n_groups * self.o_lora_rank, self.dim, bias=args.attention_bias
        )

        if self.compress_ratio:
            base = args.compress_rope_theta
            scaling = args.rope_scaling
        else:
            base = args.rope_theta
            scaling = None
        self.rope = DeepseekV4RoPE(self.rope_head_dim, base, scaling)

        if self.compress_ratio:
            self.compressor = Compressor(
                dim=self.dim,
                compress_ratio=self.compress_ratio,
                head_dim=self.head_dim,
                rope_head_dim=self.rope_head_dim,
                rms_norm_eps=self.eps,
                rope=self.rope,
            )
            if self.compress_ratio == 4:
                self.indexer = Indexer(args, self.compress_ratio, self.rope)

    def _grouped_output_projection(self, out: mx.array) -> mx.array:
        """Grouped low-rank projection via wo_a. Handles both dense and quantized.

        wo_a has shape ``[n_groups * o_lora_rank, group_feat]`` (rows = output
        features). For each group, rows ``g * o_lora_rank : (g+1) * o_lora_rank``
        and the same input columns are used.
        """
        B, S = out.shape[:2]
        group_feat = (self.n_heads * self.head_dim) // self.n_groups
        out = out.reshape(B, S, self.n_groups, group_feat)

        if isinstance(self.wo_a, nn.QuantizedLinear):
            pieces = []
            for g in range(self.n_groups):
                rows = slice(g * self.o_lora_rank, (g + 1) * self.o_lora_rank)
                y = mx.quantized_matmul(
                    out[:, :, g, :],
                    self.wo_a.weight[rows],
                    scales=self.wo_a.scales[rows],
                    biases=self.wo_a.biases[rows] if self.wo_a.biases is not None else None,
                    transpose=True,
                    group_size=self.wo_a.group_size,
                    bits=self.wo_a.bits,
                )
                pieces.append(y)
            return mx.concatenate(pieces, axis=-1)

        wa = self.wo_a.weight.reshape(self.n_groups, self.o_lora_rank, group_feat)
        y = mx.einsum("bsgd,grd->bsgr", out, wa)
        return y.reshape(B, S, self.n_groups * self.o_lora_rank)

    def __call__(
        self,
        x: mx.array,
        cache: Optional[Any] = None,
    ) -> mx.array:
        B, S, _ = x.shape
        rd = self.rope_head_dim

        # Q
        qr = self.q_norm(self.wq_a(x))
        q = self.wq_b(qr).reshape(B, S, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)
        q = q * mx.rsqrt(q.square().mean(axis=-1, keepdims=True) + self.eps)

        # single-head KV (MQA with num_kv_heads=1)
        kv = self.kv_norm(self.wkv(x))  # [B, S, head_dim]

        offset = cache.offset if cache is not None else 0

        # RoPE (only the last rd dims)
        q_nope, q_pe = q[..., : -rd], q[..., -rd:]
        q_pe = self.rope(q_pe, offset=offset)
        q = mx.concatenate([q_nope, q_pe], axis=-1)

        kv_nope, kv_pe = kv[..., : -rd], kv[..., -rd:]
        # rope treats penultimate dim as sequence
        kv_pe = self.rope(kv_pe.reshape(B, 1, S, rd), offset=offset).reshape(B, S, rd)
        kv = mx.concatenate([kv_nope, kv_pe], axis=-1)  # [B, S, D]

        is_prefill = offset == 0

        if self.compress_ratio:
            if cache is None:
                cache = DeepseekV4Cache(
                    window=self.window,
                    ratio=self.compress_ratio,
                    head_dim=self.head_dim,
                    has_indexer=(self.compress_ratio == 4),
                    index_head_dim=self.args.index_head_dim,
                )
            if is_prefill:
                window_kv = cache.prefill_window(kv)
            else:
                window_kv = cache.decode_window(kv)
            _ = self.compressor(x, cache, offset)

            window_len = window_kv.shape[1]
            compressed = cache.compressed_kv
            compressed_len = 0 if compressed is None else compressed.shape[1]
            kv_all = (
                mx.concatenate([window_kv, compressed], axis=1)
                if compressed_len > 0
                else window_kv
            )

            win_idxs = _get_window_topk_idxs(
                self.window, B, S, offset, window_len
            )
            if self.compress_ratio == 4:
                compress_idxs = self.indexer(x, qr, cache, offset)
                compress_idxs = mx.where(
                    compress_idxs >= 0,
                    compress_idxs + window_len,
                    mx.array(-1, mx.int32),
                )
            else:
                compress_idxs = _get_compress_topk_idxs(
                    self.compress_ratio,
                    B,
                    S,
                    offset,
                    compressed_len,
                    window_len,
                )
            topk_idxs = mx.concatenate([win_idxs, compress_idxs], axis=-1)
            cache.offset = cache.offset + S
        else:
            # Pure sliding window. During prefill, attend over the full raw kv
            # and only persist the last ``window`` tokens for future decode.
            k = kv[:, None, :, :]
            if is_prefill:
                if cache is not None:
                    _ = cache.update_and_fetch(k, k)
                kv_all = kv
            else:
                if cache is not None:
                    k, _ = cache.update_and_fetch(k, k)
                    kv_all = k.squeeze(1)
                else:
                    kv_all = kv
            window_len = kv_all.shape[1]
            topk_idxs = _get_window_topk_idxs(
                self.window, B, S, offset, window_len
            )

        # Sparse attention
        o = _sparse_attention(q, kv_all, topk_idxs, self.attn_sink, self.scale)

        # Undo RoPE on V (K==V, V carries the rotation)
        o_nope, o_pe = o[..., : -rd], o[..., -rd:]
        o_pe = self.rope(o_pe, offset=offset, inverse=True)
        o = mx.concatenate([o_nope, o_pe], axis=-1)

        # [B, H, S, D] -> [B, S, H*D]
        o = o.transpose(0, 2, 1, 3).reshape(B, S, self.n_heads * self.head_dim)
        o = self._grouped_output_projection(o)
        return self.wo_b(o)


# --------------------------------------------------------------------------- #
# MoE                                                                          #
# --------------------------------------------------------------------------- #


def _score_func(scores: mx.array, func: str) -> mx.array:
    if func == "softmax":
        return mx.softmax(scores, axis=-1, precise=True)
    if func == "sigmoid":
        return mx.sigmoid(scores)
    # sqrtsoftplus = sqrt(softplus(x)) = sqrt(logaddexp(x, 0))
    return mx.sqrt(mx.logaddexp(scores, mx.zeros_like(scores)))


class MoEGate(nn.Module):
    def __init__(self, args: ModelArgs, layer_id: int):
        super().__init__()
        self.layer_id = layer_id
        self.n_routed = args.n_routed_experts
        self.top_k = args.num_experts_per_tok
        self.hash = layer_id < args.num_hash_layers
        self.score_func = args.scoring_func
        self.route_scale = args.routed_scaling_factor
        self.norm_topk_prob = args.norm_topk_prob

        self.weight = mx.zeros((self.n_routed, args.hidden_size))
        if self.hash:
            self.tid2eid = mx.zeros(
                (args.vocab_size, self.top_k), dtype=mx.int32
            )
        else:
            self.e_score_correction_bias = mx.zeros(
                (self.n_routed,), dtype=mx.float32
            )

    def __call__(self, x: mx.array, input_ids: Optional[mx.array] = None):
        scores = x.astype(mx.float32) @ self.weight.T.astype(mx.float32)
        scores = _score_func(scores, self.score_func)
        orig = scores
        if not self.hash:
            scores = scores + self.e_score_correction_bias
            inds = mx.stop_gradient(
                mx.argpartition(-scores, kth=self.top_k - 1, axis=-1)[..., : self.top_k]
            )
        else:
            ids = input_ids.reshape(-1)
            inds = self.tid2eid[ids]
            inds = inds.reshape(*x.shape[:-1], self.top_k)

        weights = mx.take_along_axis(orig, inds, axis=-1)
        if self.score_func != "softmax" and self.norm_topk_prob:
            weights = weights / (weights.sum(axis=-1, keepdims=True) + 1e-20)
        weights = (weights * self.route_scale).astype(x.dtype)
        return inds, weights


class DeepseekV4MLP(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int, swiglu_limit: float = 0.0):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.swiglu_limit = swiglu_limit

    def __call__(self, x: mx.array) -> mx.array:
        gate = self.gate_proj(x)
        up = self.up_proj(x)
        if self.swiglu_limit and self.swiglu_limit > 0:
            up = mx.clip(up, -self.swiglu_limit, self.swiglu_limit)
            gate = mx.minimum(gate, self.swiglu_limit)
        return self.down_proj(nn.silu(gate) * up)


_FP4_E2M1_TABLE = mx.array(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
     0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
    dtype=mx.float32,
)


def _dequant_fp4_to_bf16(
    weight: mx.array, scale: mx.array, scale_to_float
) -> mx.array:
    """Fallback FP4 → bf16 dequant used when not staying in native mxfp4."""
    bs = 32
    packed = weight.astype(mx.uint8)
    low = packed & 0x0F
    high = (packed >> 4) & 0x0F
    unpacked = mx.stack(
        [mx.take(_FP4_E2M1_TABLE, low), mx.take(_FP4_E2M1_TABLE, high)],
        axis=-1,
    ).reshape(weight.shape[0], weight.shape[1] * 2)
    s = mx.repeat(scale_to_float(scale), bs, axis=-1)
    return (unpacked * s).astype(mx.bfloat16)


def _dsv4_fp4_native(args: "ModelArgs") -> bool:
    """Return True iff the model config indicates DSV4's native FP4 routed-expert
    quantization — i.e. ``quantization_config = {"quant_method": "fp8",
    "fmt": "e4m3", "scale_fmt": "ue8m0", ...}``. In that layout, routed experts
    ship as FP4-packed uint8 (E2M1 codes) with per-32-col E8M0 scales, which maps
    losslessly onto MLX's ``mxfp4`` quantized format — so we can avoid dequanting
    them to bf16 at load time.
    """
    qc = args.quantization_config or {}
    return (
        qc.get("quant_method") == "fp8"
        and qc.get("fmt") == "e4m3"
        and qc.get("scale_fmt") == "ue8m0"
    )


class QuantizedSwitchGLU(nn.Module):
    """SwitchGLU whose projections are :class:`QuantizedSwitchLinear`. Mirrors
    ``switch_layers.SwitchGLU`` exactly in call semantics; different only in
    its parameter tree (weights/scales/biases instead of dense weights).
    """

    def __init__(
        self,
        input_dims: int,
        hidden_dims: int,
        num_experts: int,
        *,
        mode: str = "mxfp4",
        bits: int = 4,
        group_size: int = 32,
        activation=None,
    ):
        super().__init__()
        self.activation = activation or SwiGLU()
        kw = dict(
            bias=False, mode=mode, bits=bits, group_size=group_size
        )
        self.gate_proj = QuantizedSwitchLinear(input_dims, hidden_dims, num_experts, **kw)
        self.up_proj = QuantizedSwitchLinear(input_dims, hidden_dims, num_experts, **kw)
        self.down_proj = QuantizedSwitchLinear(hidden_dims, input_dims, num_experts, **kw)

    def __call__(self, x: mx.array, indices: mx.array) -> mx.array:
        x = mx.expand_dims(x, (-2, -3))
        do_sort = indices.size >= 64
        idx = indices
        inv_order = None
        if do_sort:
            x, idx, inv_order = _gather_sort(x, indices)
        x_up = self.up_proj(x, idx, sorted_indices=do_sort)
        x_gate = self.gate_proj(x, idx, sorted_indices=do_sort)
        x = self.down_proj(
            self.activation(x_up, x_gate), idx, sorted_indices=do_sort
        )
        if do_sort:
            x = _scatter_unsort(x, inv_order, indices.shape)
        return x.squeeze(-2)


class DeepseekV4MoE(nn.Module):
    def __init__(self, args: ModelArgs, layer_id: int):
        super().__init__()
        self.num_experts_per_tok = args.num_experts_per_tok
        if _dsv4_fp4_native(args):
            # Native DSV4 FP4 experts: build the MoE with pre-quantized
            # (mxfp4) SwitchLinear tensors so sanitize() can hand them
            # the raw packed bytes directly with no bf16 intermediate.
            self.switch_mlp = QuantizedSwitchGLU(
                args.hidden_size,
                args.moe_intermediate_size,
                args.n_routed_experts,
                mode="mxfp4",
                bits=4,
                group_size=32,
            )
        else:
            self.switch_mlp = SwitchGLU(
                args.hidden_size,
                args.moe_intermediate_size,
                args.n_routed_experts,
            )
        self.gate = MoEGate(args, layer_id)
        if args.n_shared_experts:
            self.shared_experts = DeepseekV4MLP(
                args.hidden_size,
                args.moe_intermediate_size * args.n_shared_experts,
                swiglu_limit=0.0,
            )

    def __call__(self, x: mx.array, input_ids: mx.array) -> mx.array:
        inds, weights = self.gate(x, input_ids)
        y = self.switch_mlp(x, inds)
        y = (y * weights[..., None]).sum(axis=-2).astype(y.dtype)
        if hasattr(self, "shared_experts"):
            y = y + self.shared_experts(x)
        return y


# --------------------------------------------------------------------------- #
# Block                                                                        #
# --------------------------------------------------------------------------- #


class DeepseekV4Block(nn.Module):
    def __init__(self, args: ModelArgs, layer_id: int):
        super().__init__()
        self.attn_norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.attn = V4Attention(args, layer_id)
        self.hc_attn = HyperConnection(
            args.hidden_size,
            args.hc_mult,
            args.rms_norm_eps,
            args.hc_sinkhorn_iters,
            args.hc_eps,
        )
        self.ffn_norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.ffn = DeepseekV4MoE(args, layer_id)
        self.hc_ffn = HyperConnection(
            args.hidden_size,
            args.hc_mult,
            args.rms_norm_eps,
            args.hc_sinkhorn_iters,
            args.hc_eps,
        )

    def __call__(
        self,
        h: mx.array,
        cache: Optional[Any],
        input_ids: mx.array,
    ) -> mx.array:
        # h: [B, S, hc, D]
        residual = h
        y, post, comb = self.hc_attn.hc_pre(h)
        y = self.attn_norm(y)
        y = self.attn(y, cache=cache)
        h = self.hc_attn.hc_post(y, residual, post, comb)

        residual = h
        y, post, comb = self.hc_ffn.hc_pre(h)
        y = self.ffn_norm(y)
        y = self.ffn(y, input_ids)
        h = self.hc_ffn.hc_post(y, residual, post, comb)
        return h


# --------------------------------------------------------------------------- #
# Top-level model                                                              #
# --------------------------------------------------------------------------- #


class DeepseekV4Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.vocab_size = args.vocab_size
        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [
            DeepseekV4Block(args, i) for i in range(args.num_hidden_layers)
        ]
        self.norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.hc_head = HyperHead(
            args.hidden_size, args.hc_mult, args.rms_norm_eps, args.hc_eps
        )

    def __call__(self, inputs: mx.array, cache: Optional[List[Any]] = None) -> mx.array:
        B, S = inputs.shape
        h = self.embed_tokens(inputs)  # [B, S, D]
        # Broadcast to hc_mult copies
        h = mx.broadcast_to(
            h[:, :, None, :],
            (B, S, self.args.hc_mult, h.shape[-1]),
        )
        h = mx.contiguous(h)

        if cache is None:
            cache = [None] * len(self.layers)

        for i, layer in enumerate(self.layers):
            h = layer(h, cache[i], inputs)

        h = self.hc_head(h)  # [B, S, D]
        return self.norm(h)


class Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.model = DeepseekV4Model(args)
        self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)

    def __call__(
        self, inputs: mx.array, cache: Optional[List[Any]] = None
    ) -> mx.array:
        h = self.model(inputs, cache)
        return self.lm_head(h)

    @property
    def layers(self):
        return self.model.layers

    @property
    def cast_predicate(self):
        def pred(k: str) -> bool:
            # Keep mHC parameters, attention sinks, and gate biases in fp32.
            keep_fp32 = (
                ".hc_attn." in k
                or ".hc_ffn." in k
                or ".hc_head." in k
                or "e_score_correction_bias" in k
                or "attn_sink" in k
            )
            return not keep_fp32
        return pred

    def make_cache(self):
        caches = []
        for layer in self.layers:
            r = layer.attn.compress_ratio
            if r == 0:
                caches.append(
                    RotatingKVCache(max_size=self.args.sliding_window)
                )
            else:
                caches.append(
                    DeepseekV4Cache(
                        window=self.args.sliding_window,
                        ratio=r,
                        head_dim=self.args.head_dim,
                        has_indexer=(r == 4),
                        index_head_dim=self.args.index_head_dim,
                    )
                )
        return caches

    # --------------------------------------------------------------------- #
    # Weight loading                                                        #
    # --------------------------------------------------------------------- #

    def sanitize(self, weights: Dict[str, mx.array]) -> Dict[str, mx.array]:
        """Follow the DeepSeek-V3.2 sanitize structure (MTP drop -> FP8 dequant
        -> expert stacking) with three DSV4-specific additions:

            1. ``F8_E8M0`` scale tensors arrive as ``uint8`` (see utils.py
               loader shim) and are decoded as ``2^(b-127)`` before use.
            2. Routed experts in Flash are FP4-packed ``uint8``; identify via
               ``scale.shape[-1] * 16 == weight.shape[-1]`` and dequant with a
               nibble table.
            3. DSV4 uses a different checkpoint key scheme (``embed`` /
               ``head`` / ``attn.*`` / ``ffn.*`` / ``hc_{attn,ffn}_{fn,base,scale}``
               / ``gate.bias``); remap into mlx-lm ``model.*`` space.
        """
        n_layers = self.args.num_hidden_layers

        # 1) Drop MTP blocks and layers past num_hidden_layers
        filtered = {}
        for k, v in weights.items():
            if k.startswith("mtp."):
                continue
            parts = k.split(".")
            if len(parts) >= 2 and parts[0] == "layers":
                try:
                    idx = int(parts[1])
                except ValueError:
                    filtered[k] = v
                    continue
                if idx >= n_layers:
                    continue
            filtered[k] = v
        weights = filtered

        def _scale_to_float(scale: mx.array) -> mx.array:
            if scale.dtype == mx.uint8:
                return mx.exp((scale.astype(mx.float32) - 127.0) * math.log(2.0))
            return scale.astype(mx.float32)

        def dequant_fp8_block(weight: mx.array, scale: mx.array) -> mx.array:
            # DSV3.2-style block dequant, plus F8_E8M0 scale decode.
            bs = 128
            w = mx.from_fp8(weight, dtype=mx.bfloat16)
            s = _scale_to_float(scale)
            m, n = w.shape
            pad_b = (-m) % bs
            pad_s = (-n) % bs
            w = mx.pad(w, ((0, pad_b), (0, pad_s)))
            w = w.reshape((m + pad_b) // bs, bs, (n + pad_s) // bs, bs)
            w = (w * s[:, None, :, None]).reshape(m + pad_b, n + pad_s)
            return w[:m, :n].astype(mx.bfloat16)

        # 2) Handle each `.scale` sibling entry:
        #    - Routed-expert FP4 packed weights (uint8 [out, in//2]) + E8M0 scale
        #      (uint8 [out, in//32]) → *reinterpret* as MLX's mxfp4 layout
        #      (uint32 [out, in//8] + uint8 scale). No dequant; no bf16
        #      intermediate. DSV4 Flash's ~138 GB of experts stays ~138 GB in
        #      memory.
        #    - FP8 weights (uint8 [M, N]) + E8M0 128x128 block scale → bf16
        #      dequant (same as DSV3.2). Only a few GB of non-expert weights
        #      hit this path.
        native_experts = _dsv4_fp4_native(self.args)
        dequanted = {}
        for k, v in weights.items():
            if not k.endswith(".scale"):
                if k not in dequanted:
                    dequanted[k] = v
                continue
            wk = k[: -len(".scale")] + ".weight"
            weight = weights.get(wk)
            if weight is None:
                dequanted[k] = v
                continue
            is_routed_expert = (
                ".ffn.experts." in wk
                and "shared_experts" not in wk
                and weight.dtype in (mx.int8, mx.uint8)
                and v.shape[-1] * 16 == weight.shape[-1]
            )
            if is_routed_expert and native_experts:
                # Reinterpret FP4 bytes as mxfp4 packed uint32. The bit layout
                # matches: DSV4 stores element 2i in low nibble, 2i+1 in high;
                # MLX mxfp4 packs 8 consecutive fp4 elements into one uint32
                # little-endian. So 4 consecutive bytes == 1 uint32 with the
                # same per-nibble ordering.
                packed = weight.astype(mx.uint8)
                if packed.shape[-1] % 4 != 0:
                    raise ValueError(
                        f"Expected FP4 packed last-dim divisible by 4, got shape "
                        f"{packed.shape} for {wk}"
                    )
                uint32_weight = packed.view(mx.uint32).reshape(
                    packed.shape[0], packed.shape[-1] // 4
                )
                # E8M0 scale is already uint8 with the exact shape MLX wants.
                dequanted[wk] = uint32_weight
                dequanted[k] = v.astype(mx.uint8)
            elif is_routed_expert:
                # Non-native path (e.g. someone re-quantizing): fall back to
                # dense bf16 dequant via the old table-lookup route.
                dequanted[wk] = _dequant_fp4_to_bf16(weight, v, _scale_to_float)
            elif weight.dtype in (mx.uint8,):
                dequanted[wk] = dequant_fp8_block(weight, v)
            else:
                dequanted[k] = v
                dequanted[wk] = weight
        weights = dequanted

        # 3) Remap keys to mlx-lm layout
        top_remap = {
            "embed.weight":   "model.embed_tokens.weight",
            "norm.weight":    "model.norm.weight",
            "head.weight":    "lm_head.weight",
            "hc_head_fn":     "model.hc_head.fn",
            "hc_head_base":   "model.hc_head.base",
            "hc_head_scale":  "model.hc_head.scale",
        }
        for src, dst in top_remap.items():
            if src in weights:
                weights[dst] = weights.pop(src)

        remapped = {}
        w_remap = {"w1": "gate_proj", "w2": "down_proj", "w3": "up_proj"}
        for k, v in weights.items():
            nk = k
            if nk.startswith("layers."):
                nk = "model." + nk
            nk = nk.replace(".ffn.gate.bias", ".ffn.gate.e_score_correction_bias")
            for sub in ("attn", "ffn"):
                for p in ("fn", "base", "scale"):
                    nk = nk.replace(f".hc_{sub}_{p}", f".hc_{sub}.{p}")
            for wo, wn in w_remap.items():
                nk = nk.replace(f".shared_experts.{wo}.", f".shared_experts.{wn}.")
            remapped[nk] = v
        weights = remapped

        # 4) Stack routed experts into SwitchGLU layout. When the MoE's
        #    switch_mlp is QuantizedSwitchLinear-backed (native FP4 path),
        #    stack both ``.weight`` (uint32) AND ``.scales`` (uint8). Otherwise
        #    stack only ``.weight`` (dense bf16).
        stack_scales = native_experts
        for l in range(n_layers):
            prefix = f"model.layers.{l}.ffn.experts"
            for src, dst in [("w1", "gate_proj"), ("w2", "down_proj"), ("w3", "up_proj")]:
                key0 = f"{prefix}.0.{src}.weight"
                if key0 in weights:
                    stack = [
                        weights.pop(f"{prefix}.{e}.{src}.weight")
                        for e in range(self.args.n_routed_experts)
                    ]
                    weights[f"model.layers.{l}.ffn.switch_mlp.{dst}.weight"] = (
                        mx.stack(stack)
                    )
                if stack_scales:
                    skey0 = f"{prefix}.0.{src}.scale"
                    if skey0 in weights:
                        sstack = [
                            weights.pop(f"{prefix}.{e}.{src}.scale")
                            for e in range(self.args.n_routed_experts)
                        ]
                        weights[
                            f"model.layers.{l}.ffn.switch_mlp.{dst}.scales"
                        ] = mx.stack(sstack)

        return weights
