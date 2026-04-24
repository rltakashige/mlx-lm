import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import mlx.core as mx
import mlx.nn as nn

from .base import BaseModelArgs, scaled_dot_product_attention
from .cache import ArraysCache, CacheList, RotatingKVCache
from .switch_layers import SwitchGLU

# Slot indices inside the ArraysCache that sits alongside the RotatingKVCache
# for compressed-attention layers (CacheList(RotatingKVCache, ArraysCache(6))).
# Trim is not supported on this composition (ArraysCache inherits is_trimmable
# = False); rollback is done by snapshotting ``cache.state`` and restoring via
# ``from_state``.
_C_COMPRESSED = 0
_C_COMP_KV_STATE = 1
_C_COMP_SCORE_STATE = 2
_C_IDX_COMPRESSED = 3
_C_IDX_KV_STATE = 4
_C_IDX_SCORE_STATE = 5
_N_COMPRESSED_SLOTS = 6


def _temporal_window_kv(cache) -> mx.array:
    """Return the sliding-window cache's keys in temporal order as ``[B, 1, L, D]``.

    ``RotatingKVCache._temporal_order(v)`` takes an array argument and returns
    a reordered copy; ``BatchRotatingKVCache._temporal_order()`` takes no
    argument and mutates ``self.keys`` in place. Dispatch via the presence of
    ``rotated`` (only on the batched variant).
    """
    if hasattr(cache, "rotated"):
        cache._temporal_order()
        keys = cache.keys
        idx = cache._idx
        return keys[..., :idx, :] if idx < keys.shape[2] else keys
    return cache._temporal_order(cache.keys)


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
        # mx.fast.rope takes the period (1/inv_freq), not the frequency.
        self._inv_freq = (inv_freq,)
        self._freqs = (1.0 / inv_freq,)

    @property
    def inv_freq(self) -> mx.array:
        return self._inv_freq[0]

    @property
    def freqs(self) -> mx.array:
        return self._freqs[0]

    def __call__(self, x: mx.array, offset=0, inverse: bool = False) -> mx.array:
        """``offset`` may be a Python int (all sequences at the same position)
        or an ``mx.array`` of shape ``[B]`` for per-batch-item positions.
        """
        scale = -1.0 if inverse else 1.0
        # mx.fast.rope is a single fused kernel; rotates only the last
        # ``self.dims`` dims and passes through the rest.
        return mx.fast.rope(
            x,
            self.dims,
            traditional=True,
            base=None,
            scale=scale,
            offset=offset,
            freqs=self.freqs,
        )


# --------------------------------------------------------------------------- #
# Sinkhorn-based mHC (Manifold-constrained Hyper-Connections)                 #
# --------------------------------------------------------------------------- #


@mx.compile
def _hc_split_sinkhorn_ops(
    mixes: mx.array,
    hc_scale: mx.array,
    hc_base: mx.array,
    hc_mult: int,
    iters: int,
    eps: float,
):
    """Compiled-op fallback used when the Metal kernel is unavailable (CPU, or
    odd hc_mult). ``mixes`` has shape ``[..., (2+hc)*hc]``; returns
    ``pre [..., hc]``, ``post [..., hc]``, ``comb [..., hc, hc]``.
    """
    mixes = mixes.astype(mx.float32)
    hc_scale = hc_scale.astype(mx.float32)
    hc_base = hc_base.astype(mx.float32)
    s0, s1, s2 = hc_scale[0], hc_scale[1], hc_scale[2]

    pre = mx.sigmoid(mixes[..., :hc_mult] * s0 + hc_base[:hc_mult]) + eps
    post = 2 * mx.sigmoid(
        mixes[..., hc_mult : 2 * hc_mult] * s1 + hc_base[hc_mult : 2 * hc_mult]
    )
    comb = mixes[..., 2 * hc_mult :].reshape(
        *mixes.shape[:-1], hc_mult, hc_mult
    ) * s2 + hc_base[2 * hc_mult :].reshape(hc_mult, hc_mult)
    comb = mx.softmax(comb, axis=-1, precise=True) + eps
    comb = comb / (comb.sum(axis=-2, keepdims=True) + eps)
    for _ in range(max(iters - 1, 0)):
        comb = comb / (comb.sum(axis=-1, keepdims=True) + eps)
        comb = comb / (comb.sum(axis=-2, keepdims=True) + eps)
    return pre, post, comb


def _make_hc_split_sinkhorn_kernel():
    # Single Metal kernel doing sigmoid+softmax+N Sinkhorn iters in registers.
    # Unrolls on HC and ITERS as template params, so each layer's HC call is
    # one dispatch instead of ~40 from the compiled-op path.
    if mx.default_device() != mx.gpu or not mx.metal.is_available():
        return None

    source = """
        uint idx = thread_position_in_grid.x;
        constexpr int MIX = (2 + HC) * HC;
        float epsv = static_cast<float>(eps[0]);

        auto mix = mixes + idx * MIX;
        auto pre_out = pre + idx * HC;
        auto post_out = post + idx * HC;
        auto comb_out = comb + idx * HC * HC;

        float pre_scale = static_cast<float>(scale[0]);
        float post_scale = static_cast<float>(scale[1]);
        float comb_scale = static_cast<float>(scale[2]);

        for (int i = 0; i < HC; ++i) {
            float z = static_cast<float>(mix[i]) * pre_scale
                + static_cast<float>(base[i]);
            pre_out[i] = 1.0f / (1.0f + metal::fast::exp(-z)) + epsv;
        }
        for (int i = 0; i < HC; ++i) {
            int off = HC + i;
            float z = static_cast<float>(mix[off]) * post_scale
                + static_cast<float>(base[off]);
            post_out[i] = 2.0f / (1.0f + metal::fast::exp(-z));
        }

        float c[HC * HC];
        for (int i = 0; i < HC; ++i) {
            float row_max = -INFINITY;
            for (int j = 0; j < HC; ++j) {
                int cidx = i * HC + j;
                int off = 2 * HC + cidx;
                float v = static_cast<float>(mix[off]) * comb_scale
                    + static_cast<float>(base[off]);
                c[cidx] = v;
                row_max = metal::max(row_max, v);
            }
            float row_sum = 0.0f;
            for (int j = 0; j < HC; ++j) {
                int cidx = i * HC + j;
                float v = metal::fast::exp(c[cidx] - row_max);
                c[cidx] = v;
                row_sum += v;
            }
            float inv_sum = 1.0f / row_sum;
            for (int j = 0; j < HC; ++j) {
                int cidx = i * HC + j;
                c[cidx] = c[cidx] * inv_sum + epsv;
            }
        }
        for (int j = 0; j < HC; ++j) {
            float col_sum = 0.0f;
            for (int i = 0; i < HC; ++i) {
                col_sum += c[i * HC + j];
            }
            float inv_denom = 1.0f / (col_sum + epsv);
            for (int i = 0; i < HC; ++i) {
                c[i * HC + j] *= inv_denom;
            }
        }
        for (int iter = 1; iter < ITERS; ++iter) {
            for (int i = 0; i < HC; ++i) {
                float row_sum = 0.0f;
                for (int j = 0; j < HC; ++j) {
                    row_sum += c[i * HC + j];
                }
                float inv_denom = 1.0f / (row_sum + epsv);
                for (int j = 0; j < HC; ++j) {
                    c[i * HC + j] *= inv_denom;
                }
            }
            for (int j = 0; j < HC; ++j) {
                float col_sum = 0.0f;
                for (int i = 0; i < HC; ++i) {
                    col_sum += c[i * HC + j];
                }
                float inv_denom = 1.0f / (col_sum + epsv);
                for (int i = 0; i < HC; ++i) {
                    c[i * HC + j] *= inv_denom;
                }
            }
        }
        for (int i = 0; i < HC * HC; ++i) {
            comb_out[i] = c[i];
        }
    """
    return mx.fast.metal_kernel(
        name="deepseek_v4_hc_split_sinkhorn",
        input_names=["mixes", "scale", "base", "eps"],
        output_names=["pre", "post", "comb"],
        source=source,
    )


_hc_split_sinkhorn_kernel = _make_hc_split_sinkhorn_kernel()


def hc_split_sinkhorn(
    mixes: mx.array,
    hc_scale: mx.array,
    hc_base: mx.array,
    hc_mult: int,
    iters: int,
    eps,
):
    """Dispatch to the Metal kernel if available, else the compiled-op fallback.

    ``mixes`` has shape ``[..., (2+hc)*hc]``; returns ``pre [..., hc]``,
    ``post [..., hc]``, ``comb [..., hc, hc]``.
    """
    if _hc_split_sinkhorn_kernel is None:
        eps_val = eps.item() if isinstance(eps, mx.array) else float(eps)
        return _hc_split_sinkhorn_ops(
            mixes, hc_scale, hc_base, hc_mult, iters, eps_val
        )
    eps_arr = eps if isinstance(eps, mx.array) else mx.array([eps], dtype=mx.float32)
    return _hc_split_sinkhorn_kernel(
        inputs=[mixes, hc_scale, hc_base, eps_arr],
        template=[("HC", hc_mult), ("ITERS", iters)],
        grid=(mixes.size // ((2 + hc_mult) * hc_mult), 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[
            (*mixes.shape[:-1], hc_mult),
            (*mixes.shape[:-1], hc_mult),
            (*mixes.shape[:-1], hc_mult, hc_mult),
        ],
        output_dtypes=[mx.float32, mx.float32, mx.float32],
    )


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
        # Cache so the kernel launch doesn't allocate a new mx.array each call.
        self._eps_arr = mx.array([hc_eps], dtype=mx.float32)

    def hc_pre(self, x: mx.array):
        # x: [B, S, hc, D]  ->  (y [B, S, D], post [B, S, hc], comb [B, S, hc, hc])
        B, S, hc, D = x.shape
        dtype = x.dtype
        # fast.rms_norm is a fused kernel; normalizing before the matmul is
        # equivalent to (xf @ fn.T) * rsqrt(...) since the scale is scalar/row.
        xf = mx.fast.rms_norm(
            x.reshape(B, S, hc * D).astype(mx.float32), None, self.norm_eps
        )
        mixes = xf @ self.fn.T
        pre, post, comb = hc_split_sinkhorn(
            mixes, self.scale, self.base, hc, self.sinkhorn_iters, self._eps_arr
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
        y = post[..., None] * f_out[:, :, None, :].astype(mx.float32)
        y = y + mx.matmul(comb.astype(mx.float32), residual.astype(mx.float32))
        return y.astype(dtype)


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
        xf = mx.fast.rms_norm(
            x.reshape(B, S, hc * D).astype(mx.float32), None, self.norm_eps
        )
        mixes = xf @ self.fn.T
        pre = mx.sigmoid(mixes * self.scale[0] + self.base) + self.hc_eps
        return (pre[..., None] * x.astype(mx.float32)).sum(axis=2).astype(dtype)


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
        self, compressed_kv: mx.array, first_pos: int
    ) -> mx.array:
        """Apply RoPE on the last ``rope_head_dim`` dims at strided positions
        ``first_pos, first_pos + ratio, first_pos + 2*ratio, ...``.

        ``first_pos`` must be divisible by ``compress_ratio`` (caller guarantees).
        """
        rd = self.rope_head_dim
        # mx.fast.rope generates positions as ``scale * (offset + arange(T))``.
        # We want ``first_pos + arange(T) * ratio`` = ``ratio * (first_pos/ratio + arange(T))``.
        rotated = mx.fast.rope(
            compressed_kv[..., -rd:],
            rd,
            traditional=True,
            base=None,
            scale=float(self.compress_ratio),
            offset=first_pos // self.compress_ratio,
            freqs=self.rope.freqs,
        )
        return mx.concatenate([compressed_kv[..., :-rd], rotated], axis=-1)

    def _call_non_overlap(
        self,
        x: mx.array,
        state: "ArraysCache",
        offset: int,
        slot_compressed: int,
        slot_kv_state: int,
        slot_score_state: int,
    ) -> Optional[mx.array]:
        """Concat-buffer pattern (PR #1192 style). Non-overlap only: each
        emit is independent, no cross-window tokens needed in buffer.

        State layout:
          slot_kv_state:    [B, buf_len, coff_d] — incomplete-window carry
          slot_score_state: [B, buf_len, coff_d] — same, for gate scores
          slot_compressed:  [B, n_emitted, d]    — growing pool of compressed rows
        """
        B, S, _ = x.shape
        # Run wkv/wgate in input dtype (bf16); promote only for softmax (below).
        # Casting x to fp32 before the projections doubles weight-read BW and
        # is the decode-bound cost — PR #1192 avoids it.
        kv = self.wkv(x)     # [B, S, coff_d] == [B, S, d] for non-overlap
        score = self.wgate(x)
        ratio = self.compress_ratio
        d = self.head_dim

        buf_kv = state[slot_kv_state]
        buf_score = state[slot_score_state]
        buf_len = 0 if buf_kv is None else buf_kv.shape[1]
        if buf_len:
            kv = mx.concatenate([buf_kv, kv], axis=1)
            score = mx.concatenate([buf_score, score], axis=1)

        total = kv.shape[1]
        usable = (total // ratio) * ratio
        # Raw-token position of the first token in the concatenated buffer.
        pool_base = offset - buf_len

        state[slot_kv_state] = kv[:, usable:] if usable < total else None
        state[slot_score_state] = score[:, usable:] if usable < total else None

        if usable == 0:
            return None

        W = usable // ratio
        kv_win = kv[:, :usable].reshape(B, W, ratio, -1)
        score_win = score[:, :usable].reshape(B, W, ratio, -1) + self.ape.astype(
            score.dtype
        )
        weights = mx.softmax(
            score_win.astype(mx.float32), axis=2, precise=True
        ).astype(kv_win.dtype)
        compressed = (kv_win * weights).sum(axis=2)[..., :d]
        compressed = self.norm(compressed.astype(x.dtype))

        compressed = self._apply_compressor_rope(compressed, pool_base)

        pool = state[slot_compressed]
        state[slot_compressed] = (
            compressed if pool is None else mx.concatenate([pool, compressed], axis=1)
        )
        return compressed

    def __call__(
        self,
        x: mx.array,
        state: "ArraysCache",
        offset,
        slot_compressed: int,
        slot_kv_state: int,
        slot_score_state: int,
    ) -> Optional[mx.array]:
        if isinstance(offset, mx.array):
            offset = int(offset.max().item())
        if not self.overlap:
            return self._call_non_overlap(
                x, state, offset, slot_compressed, slot_kv_state, slot_score_state
            )

        B, S, _ = x.shape
        # Run wkv/wgate in input dtype — fp32 here doubles weight BW (decode-bound).
        kv = self.wkv(x)
        score = self.wgate(x)
        ratio = self.compress_ratio
        overlap = self.overlap
        d = self.head_dim
        coff_d = self._coff * d

        state_kv = state[slot_kv_state]
        state_score = state[slot_score_state]

        if state_kv is None:
            n_slots = self._coff * ratio
            state_kv = mx.zeros((B, n_slots, coff_d), dtype=kv.dtype)
            state_score = mx.full((B, n_slots, coff_d), float("-inf"), dtype=score.dtype)

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
                    weights = mx.softmax(
                        score_trans.astype(mx.float32), axis=2, precise=True
                    ).astype(kv_trans.dtype)
                    compressed = (kv_trans * weights).sum(axis=2)  # [B, nw, d]
                else:
                    weights = mx.softmax(
                        score_head.astype(mx.float32), axis=2, precise=True
                    ).astype(kv_head.dtype)
                    compressed = (kv_head * weights).sum(axis=2)
                    compressed = compressed[..., :d]
                compressed = self.norm(compressed.astype(x.dtype))
                # Apply RoPE: compressed row k anchors at raw position
                # (k+1)*ratio - 1 for non-overlap, k*ratio + ratio - 1 for overlap
                # (matches ref: self.freqs_cis[:cutoff:ratio] = positions 0, r, 2r, ... ratio-1 actually...
                # Reference slices freqs_cis[:cutoff:ratio] which gives positions 0, r, 2r, ...
                # So compressed row k has position k*ratio.
                n_rows = compressed.shape[1]
                compressed = self._apply_compressor_rope(compressed, 0)
                out_compressed = compressed
                buf = state[slot_compressed]
                state[slot_compressed] = (
                    compressed if buf is None else mx.concatenate([buf, compressed], axis=1)
                )

            # Stash tail into state
            if remainder > 0:
                tail_kv = kv[:, cutoff:, :]
                tail_score = score[:, cutoff:, :] + self.ape[:remainder].astype(
                    score.dtype
                )
                start_slot = ratio if overlap else 0
                state_kv[:, start_slot : start_slot + remainder, :] = tail_kv
                state_score[:, start_slot : start_slot + remainder, :] = tail_score

            # On overlap, also stash the last full window's rows for future overlap.
            if overlap and cutoff > 0:
                prev_kv = kv[:, cutoff - ratio : cutoff, :]
                prev_score = score[:, cutoff - ratio : cutoff, :] + self.ape.astype(
                    score.dtype
                )
                state_kv[:, :ratio, :] = prev_kv
                state_score[:, :ratio, :] = prev_score

            state[slot_kv_state] = state_kv
            state[slot_score_state] = state_score
            return out_compressed

        # Decode path: offset > 0. Typically S=1 (single-step decode) but can be
        # S>1 for chunked prefill continuations (e.g. prefix cache + new prompt
        # tokens). Loop one token at a time so the accumulator/emit logic stays
        # correct.
        last_compressed = None
        for i in range(S):
            step_offset = offset + i
            pos_in_window = step_offset % ratio
            slot = ratio + pos_in_window  # overlap branch only; non-overlap short-circuited
            state_kv[:, slot, :] = kv[:, i, :]
            state_score[:, slot, :] = score[:, i, :] + self.ape[pos_in_window].astype(
                score.dtype
            )

            if ((step_offset + 1) % ratio) != 0:
                continue

            if overlap:
                first = state_kv[:, :ratio, :d]
                second = state_kv[:, ratio:, d:]
                merged_kv = mx.concatenate([first, second], axis=1)
                first_s = state_score[:, :ratio, :d]
                second_s = state_score[:, ratio:, d:]
                merged_score = mx.concatenate([first_s, second_s], axis=1)
                weights = mx.softmax(
                    merged_score.astype(mx.float32), axis=1, precise=True
                ).astype(merged_kv.dtype)
                compressed = (merged_kv * weights).sum(axis=1, keepdims=True)
            else:
                weights = mx.softmax(
                    state_score.astype(mx.float32), axis=1, precise=True
                ).astype(state_kv.dtype)
                compressed = (state_kv * weights).sum(axis=1, keepdims=True)
                compressed = compressed[..., :d]
            compressed = self.norm(compressed.astype(x.dtype))
            compressed = self._apply_compressor_rope(compressed, step_offset + 1 - ratio)
            last_compressed = compressed
            buf = state[slot_compressed]
            state[slot_compressed] = (
                compressed if buf is None else mx.concatenate([buf, compressed], axis=1)
            )
            if overlap:
                state_kv[:, :ratio, :] = state_kv[:, ratio:, :]
                state_score[:, :ratio, :] = state_score[:, ratio:, :]

        out_compressed = last_compressed

        state[slot_kv_state] = state_kv
        state[slot_score_state] = state_score
        return out_compressed


# --------------------------------------------------------------------------- #
# Indexer                                                                     #
# --------------------------------------------------------------------------- #


class Indexer(nn.Module):
    """Scores per-query visibility over the main compressed KV buffer and
    returns the top-k compressed-row indices per query. V4Attention turns
    those indices into a boolean mask on the compressed portion of SDPA's KV
    so each query only attends to its top-k far-context slots.
    """

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
        state: "ArraysCache",
        offset,
    ) -> Optional[mx.array]:
        """Update the Indexer's own compressed buffer and return
        ``top_idxs [B, S, k]`` (``k = min(index_topk, pooled_len)``) into that
        buffer. Returns ``None`` if no compressed rows exist yet. Indices are
        always in ``[0, pooled_len)``; no ``-1`` padding, no explicit
        future-masking at the Indexer level — causal visibility of the main
        attention's compressed portion is enforced by
        ``_compressed_visibility`` during prefill.
        """
        B, S, _ = x.shape
        rd = self.rope_head_dim

        self.compressor(
            x, state, offset,
            slot_compressed=_C_IDX_COMPRESSED,
            slot_kv_state=_C_IDX_KV_STATE,
            slot_score_state=_C_IDX_SCORE_STATE,
        )
        idx_kv = state[_C_IDX_COMPRESSED]
        if idx_kv is None or idx_kv.shape[1] == 0:
            return None

        q = self.wq_b(qr).reshape(B, S, self.n_heads, self.head_dim)
        q = mx.concatenate(
            [q[..., :-rd], self.rope(q[..., -rd:], offset=offset)], axis=-1
        )
        per_head_weights = self.weights_proj(x) * (
            self.softmax_scale * (self.n_heads ** -0.5)
        )
        score = mx.einsum("bshd,btd->bsht", q.astype(idx_kv.dtype), idx_kv)
        score = mx.maximum(score, 0)  # ReLU
        score = (score * per_head_weights[..., None]).sum(axis=2)  # [B, S, T]

        k = min(self.index_topk, idx_kv.shape[1])
        return mx.argpartition(-score, kth=k - 1, axis=-1)[..., :k].astype(
            mx.int32
        )


# --------------------------------------------------------------------------- #
# Attention                                                                   #
# --------------------------------------------------------------------------- #


def _build_window_mask(
    B: int,
    S: int,
    offset,
    window: int,
    window_len: int,
) -> mx.array:
    """Sliding-window causal mask of shape ``[B, 1, S, window_len]``."""
    if isinstance(offset, mx.array):
        off = offset.astype(mx.int32).reshape(-1)
        q_pos = off[:, None] + mx.arange(S, dtype=mx.int32)
        end = off + S
        cache_k = mx.arange(window_len, dtype=mx.int32)
        raw_pos_at_k = end[:, None] - window_len + cache_k[None, :]
        win_visible = (
            (raw_pos_at_k[:, None, :] <= q_pos[:, :, None])
            & (raw_pos_at_k[:, None, :] > q_pos[:, :, None] - window)
        )
    else:
        q_pos = mx.broadcast_to(
            offset + mx.arange(S, dtype=mx.int32)[None, :], (B, S)
        )
        cache_k = mx.arange(window_len, dtype=mx.int32)
        raw_pos_at_k = (offset + S) - window_len + cache_k
        win_visible = (
            (raw_pos_at_k[None, None, :] <= q_pos[:, :, None])
            & (raw_pos_at_k[None, None, :] > q_pos[:, :, None] - window)
        )
    return win_visible[:, None, :, :]


def _compressed_visibility(
    B: int,
    S: int,
    offset,
    compressed_len: int,
    ratio: int,
) -> mx.array:
    """Bool mask ``[B, 1, S, compressed_len]``: row ``k`` visible to query at
    raw pos ``p`` iff ``(k+1)*ratio <= p+1``."""
    if isinstance(offset, mx.array):
        off = offset.astype(mx.int32).reshape(-1)
        q_pos = off[:, None] + mx.arange(S, dtype=mx.int32)
    else:
        q_pos = mx.broadcast_to(
            offset + mx.arange(S, dtype=mx.int32)[None, :], (B, S)
        )
    k = mx.arange(compressed_len, dtype=mx.int32)
    comp_visible = (k + 1)[None, None, :] * ratio <= (q_pos + 1)[:, :, None]
    return comp_visible[:, None, :, :]


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
            # Batched over groups: one quantized_matmul for all groups instead
            # of a per-group Python loop (O(n_groups) → 1 kernel dispatch).
            out_g = out.transpose(2, 0, 1, 3)  # [G, B, S, group_feat]
            weight = self.wo_a.weight.reshape(self.n_groups, self.o_lora_rank, -1)[:, None]
            scales = self.wo_a.scales.reshape(self.n_groups, self.o_lora_rank, -1)[:, None]
            biases = (
                None
                if self.wo_a.biases is None
                else self.wo_a.biases.reshape(self.n_groups, self.o_lora_rank, -1)[:, None]
            )
            y = mx.quantized_matmul(
                out_g,
                weight,
                scales=scales,
                biases=biases,
                transpose=True,
                group_size=self.wo_a.group_size,
                bits=self.wo_a.bits,
                mode=self.wo_a.mode,
            )
            return y.transpose(1, 2, 0, 3).reshape(B, S, self.n_groups * self.o_lora_rank)

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
        q = mx.fast.rms_norm(q, None, self.eps)

        # single-head KV (MQA with num_kv_heads=1)
        kv = self.kv_norm(self.wkv(x))  # [B, S, head_dim]

        # Compressed-attention layers carry CacheList(RotatingKVCache, ArraysCache(6)).
        # Pure sliding-window layers carry a plain RotatingKVCache.
        if self.compress_ratio and cache is not None:
            win_cache = cache.caches[0]
            state_cache = cache.caches[1]
        else:
            win_cache = cache
            state_cache = None
        offset = win_cache.offset if win_cache is not None else 0
        # BatchRotatingKVCache stores offset as an mx.array and mutates it
        # in place via ``self.offset += S`` during update_and_fetch. Snapshot
        # here so our local ``offset`` reflects the pre-update position for
        # every downstream use (RoPE, Compressor, Indexer, topk helpers).
        if isinstance(offset, mx.array):
            offset = offset + 0

        # RoPE (only the last rd dims)
        q_nope, q_pe = q[..., : -rd], q[..., -rd:]
        q_pe = self.rope(q_pe, offset=offset)
        q = mx.concatenate([q_nope, q_pe], axis=-1)

        kv_nope, kv_pe = kv[..., : -rd], kv[..., -rd:]
        kv_pe = self.rope(kv_pe, offset=offset)  # RoPE works on 3D [B, S, rd]
        kv = mx.concatenate([kv_nope, kv_pe], axis=-1)  # [B, S, D]

        if self.compress_ratio:
            if state_cache is None:
                win_cache = RotatingKVCache(max_size=self.window)
                state_cache = ArraysCache(_N_COMPRESSED_SLOTS)
            k4 = kv[:, None, :, :]
            # Use the cache's own return value — ring-ordered post-wrap, but
            # that's fine for S=1 where all cached tokens are valid context and
            # for S>1 where ``_update_concat`` has already reordered to
            # temporal internally. No extra ``_temporal_order`` roll per layer.
            win_keys, _ = win_cache.update_and_fetch(k4, k4)
            window_kv = win_keys.squeeze(1)
            _ = self.compressor(
                x, state_cache, offset,
                slot_compressed=_C_COMPRESSED,
                slot_kv_state=_C_COMP_KV_STATE,
                slot_score_state=_C_COMP_SCORE_STATE,
            )
            indexer_topk = (
                self.indexer(x, qr, state_cache, offset)
                if self.compress_ratio == 4
                else None
            )
            compressed = state_cache[_C_COMPRESSED]
            compressed_len = 0 if compressed is None else compressed.shape[1]
        else:
            k = kv[:, None, :, :]
            if cache is not None:
                k_ret, _ = cache.update_and_fetch(k, k)
                window_kv = k_ret.squeeze(1)
            else:
                window_kv = kv
            compressed = None
            compressed_len = 0
            indexer_topk = None

        window_len = window_kv.shape[1]
        # Decode (S=1) fast path: gather the Indexer's top-k compressed rows
        # directly into SDPA's key tensor so the kernel only attends to those
        # K rows + the window. For prefill (S>1) fall back to attending to the
        # full compressed buffer with a scatter-based mask — the gather-flat
        # approach would over-include across query positions, changing output.
        use_gather = (
            S == 1 and compressed_len > 0 and indexer_topk is not None
        )
        if use_gather:
            d = compressed.shape[-1]
            # take_along_axis on a broadcast of the pool is ~15% faster at b=1
            # than reshape+flat-gather (specialized fused kernel vs fancy-index).
            expanded = mx.broadcast_to(
                compressed[:, None, None, :, :], (B, 1, S, compressed_len, d)
            )
            idx = mx.broadcast_to(
                indexer_topk[:, None, :, :, None],
                (B, 1, S, indexer_topk.shape[-1], d),
            )
            gathered = mx.take_along_axis(expanded, idx, axis=3).reshape(B, -1, d)
            kv_all = mx.concatenate([window_kv, gathered], axis=1)
        elif compressed_len > 0:
            kv_all = mx.concatenate([window_kv, compressed], axis=1)
        else:
            kv_all = window_kv

        # Decode (S=1): every token in the window cache is valid past context
        # and we're not doing causal masking (only one query), so skip the
        # window mask entirely. For the gather path the Indexer already
        # returns only valid indices (no -1 padding), so no compressed mask
        # either.
        if S == 1:
            # Pool rows are emitted only after a full window of raw tokens
            # has been processed, so at any decode step every pool row is in
            # the past — no compressed-visibility mask needed.
            mask = None
        else:
            win_mask = _build_window_mask(B, S, offset, self.window, window_len)
            if compressed_len > 0:
                comp_mask = _compressed_visibility(
                    B, S, offset, compressed_len, self.compress_ratio
                )
                if indexer_topk is not None:
                    k_range = mx.arange(compressed_len, dtype=mx.int32)
                    selected = (
                        indexer_topk[..., None] == k_range[None, None, None, :]
                    ).any(axis=-2)[:, None, :, :]
                    comp_mask = comp_mask & selected
                mask = mx.concatenate([win_mask, comp_mask], axis=-1)
            else:
                mask = win_mask

        kv_all_4d = kv_all[:, None, :, :]
        o = scaled_dot_product_attention(
            q, kv_all_4d, kv_all_4d,
            cache=None, scale=self.scale, mask=mask,
            sinks=self.attn_sink.astype(q.dtype),
        )

        o_nope, o_pe = o[..., : -rd], o[..., -rd:]
        o_pe = self.rope(o_pe, offset=offset, inverse=True)
        o = mx.concatenate([o_nope, o_pe], axis=-1)

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


@mx.compile
def _limited_swiglu(gate: mx.array, up: mx.array, limit: float) -> mx.array:
    if limit and limit > 0:
        gate = mx.minimum(gate, limit)
        up = mx.clip(up, -limit, limit)
    return nn.silu(gate) * up


class _DSV4SwiGLU(nn.Module):
    """SwiGLU with optional clipping of ``gate`` / ``up`` to ``limit``, wrapped
    in ``mx.compile`` so the silu+clip+min+mul stack runs as a single fused
    kernel. Used for routed experts (limit = args.swiglu_limit = 10.0) and
    shared experts (limit = 0 — no clip, still benefits from fusion)."""

    def __init__(self, limit: float):
        super().__init__()
        self.limit = limit

    def __call__(self, x: mx.array, gate: mx.array) -> mx.array:
        return _limited_swiglu(gate, x, self.limit)


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
        return self.down_proj(
            _limited_swiglu(self.gate_proj(x), self.up_proj(x), self.swiglu_limit)
        )


class DeepseekV4MoE(nn.Module):
    def __init__(self, args: ModelArgs, layer_id: int):
        super().__init__()
        self.num_experts_per_tok = args.num_experts_per_tok
        # Routed experts ship as FP4 (E2M1) with E8M0 per-32 scales — a 1:1
        # match for MLX's mxfp4. Build a dense SwitchGLU and immediately
        # quantize its three projections in-place; no bf16 intermediate.
        self.switch_mlp = SwitchGLU(
            args.hidden_size,
            args.moe_intermediate_size,
            args.n_routed_experts,
            activation=_DSV4SwiGLU(args.swiglu_limit),
        )
        for name in ("gate_proj", "up_proj", "down_proj"):
            sub = getattr(self.switch_mlp, name)
            setattr(
                self.switch_mlp,
                name,
                sub.to_quantized(group_size=32, bits=4, mode="mxfp4"),
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
        h = mx.broadcast_to(
            h[:, :, None, :],
            (B, S, self.args.hc_mult, h.shape[-1]),
        )
        h = mx.contiguous(h)

        if cache is None:
            cache = [None] * len(self.layers)

        for i, layer in enumerate(self.layers):
            h = layer(h, cache[i], inputs)

        h = self.hc_head(h)
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
                caches.append(RotatingKVCache(max_size=self.args.sliding_window))
            else:
                # Compose the compressed-attention layer's state from existing
                # cache primitives: a RotatingKVCache for the sliding window
                # (offset tracking, trim, ring-buffer memory) plus a 6-slot
                # ArraysCache for the Compressor/Indexer buffers.
                win = RotatingKVCache(max_size=self.args.sliding_window)
                state = ArraysCache(6)
                caches.append(CacheList(win, state))
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
        #    - Routed-expert FP4 (uint8 [out, in//2]) + E8M0 (uint8 [out, in//32])
        #      → reinterpret as MLX mxfp4 (uint32 [out, in//8] + uint8 scale).
        #      The bit layout matches: element 2i in low nibble, 2i+1 in high;
        #      MLX's mxfp4 packs 8 fp4 values per uint32 in little-endian, so a
        #      run of 4 uint8 bytes reads as the right uint32. No allocation.
        #    - FP8 (uint8 [M, N]) + E8M0 128x128 block scale → bf16 dequant.
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
            if is_routed_expert:
                packed = weight.astype(mx.uint8)
                dequanted[wk] = packed.view(mx.uint32).reshape(
                    packed.shape[0], packed.shape[-1] // 4
                )
                dequanted[k] = v.astype(mx.uint8)
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

        # 4) Stack per-expert {weight, scales} into the SwitchGLU layout.
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
