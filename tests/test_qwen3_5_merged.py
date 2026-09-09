# Copyright © 2026 Apple Inc.

import os
import copy
import unittest

os.environ.setdefault("MLX_ENABLE_TF32", "0")

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from mlx_lm.models import fused_ops, qmv_small, qwen3_5
from mlx_lm.models.cache import make_prompt_cache
from mlx_lm.models.gated_delta import gated_delta_update
from mlx_lm.models.qwen3_next import Qwen3NextRMSNormGated

CONFIG = {
    "model_type": "qwen3_5",
    "hidden_size": 64,
    "intermediate_size": 96,
    "num_hidden_layers": 4,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "head_dim": 16,
    "vocab_size": 128,
    "linear_num_value_heads": 4,
    "linear_num_key_heads": 2,
    "linear_key_head_dim": 128,
    "linear_value_head_dim": 128,
    "linear_conv_kernel_dim": 4,
    "rms_norm_eps": 1e-5,
    "full_attention_interval": 4,
    "tie_word_embeddings": False,
    "max_position_embeddings": 512,
}


def _bf16_values():
    """Every finite bf16 value below 1e30."""
    bits = np.arange(0, 65536, dtype=np.uint16)
    f = (bits.astype(np.uint32) << 16).view(np.float32)
    return mx.array(f[np.isfinite(f) & (np.abs(f) < 1e30)]).astype(mx.bfloat16)


def _model(seed=0, a_log_dtype=mx.bfloat16):
    mx.random.seed(seed)
    args = qwen3_5.ModelArgs.from_dict({"model_type": "qwen3_5", "text_config": CONFIG})
    model = qwen3_5.Model(args)
    model.eval()
    model.set_dtype(mx.bfloat16)
    # The Qwen3.8 checkpoints store A_log in bf16; a converted model may keep float32
    for layer in model.layers:
        if layer.is_linear:
            layer.linear_attn.A_log = layer.linear_attn.A_log.astype(a_log_dtype)
    mx.eval(model.parameters())
    return model


def _gdn_reference(net, proj, conv_state):
    """The ops chain of the GDN mixer inputs from the fused projection."""
    B, S, _ = proj.shape
    qkv, z, b, a = mx.split(
        proj,
        [net.conv_dim, net.conv_dim + net.value_dim, net.conv_dim + net.value_dim + net.num_v_heads],
        axis=-1,
    )
    conv_input = mx.concatenate([conv_state, qkv], axis=1)
    state_out = mx.contiguous(conv_input[:, -(net.conv_kernel_size - 1) :, :])
    conv_out = nn.silu(net.conv1d(conv_input))
    q, k, v = [
        t.reshape(B, S, h, d)
        for t, h, d in zip(
            mx.split(conv_out, [net.key_dim, 2 * net.key_dim], -1),
            [net.num_k_heads, net.num_k_heads, net.num_v_heads],
            [net.head_k_dim, net.head_k_dim, net.head_v_dim],
        )
    ]
    eps = 1e-6 / net.head_k_dim
    inv_scale = net.head_k_dim**-0.5
    q = (inv_scale**2) * mx.fast.rms_norm(q, None, eps)
    k = inv_scale * mx.fast.rms_norm(k, None, eps)
    from mlx_lm.models.gated_delta import compute_g

    return q, k, v, compute_g(net.A_log, a, net.dt_bias), mx.sigmoid(b), state_out


@unittest.skipUnless(mx.metal.is_available(), "Metal only")
class TestMergedKernels(unittest.TestCase):
    def test_gdn_in_matches_ops(self):
        for a_log_dtype in (mx.bfloat16, mx.float32):
            net = _model(a_log_dtype=a_log_dtype).layers[0].linear_attn
            self._check_gdn_in(net)

    def _check_gdn_in(self, net):
        for B, S in ((1, 1), (1, 4), (2, 3), (1, 8)):
            mx.random.seed(B * 10 + S)
            proj = (mx.random.normal((B, S, net.in_proj.weight.shape[0])) * 2).astype(mx.bfloat16)
            state = (mx.random.normal((B, 3, net.conv_dim)) * 2).astype(mx.bfloat16)
            outs = fused_ops.gdn_in(net, proj, state)
            refs = _gdn_reference(net, proj, state)
            for name, o, r in zip(("q", "k", "v", "g", "beta", "state"), outs, refs):
                self.assertEqual(o.dtype, r.dtype, name)
                self.assertTrue(mx.array_equal(o, r).item(), f"{name} B={B} S={S}")

    def test_gate_math_over_all_bf16_values(self):
        net = _model().layers[0].linear_attn
        x = _bf16_values()
        Hv = net.num_v_heads
        n = x.size // Hv * Hv
        # b and a rows made of every bf16 value; the conv part is random
        proj = (mx.random.normal((n // Hv, 1, net.in_proj.weight.shape[0])) * 2).astype(mx.bfloat16)
        ab = x[:n].reshape(n // Hv, 1, Hv)
        proj[..., net.conv_dim + net.value_dim :] = mx.concatenate([ab, ab], axis=-1)
        state = mx.zeros((n // Hv, 3, net.conv_dim), mx.bfloat16)
        _, _, _, g, beta, _ = fused_ops.gdn_in(net, proj, state)
        _, _, _, rg, rbeta, _ = _gdn_reference(net, proj, state)
        self.assertTrue(mx.array_equal(g, rg).item())
        self.assertTrue(mx.array_equal(beta, rbeta).item())

    def test_gated_norm_matches_ops(self):
        D, heads = 128, 4
        norm = Qwen3NextRMSNormGated(D, eps=1e-6)
        norm.weight = (mx.random.normal((D,)) * 0.2 + 1).astype(mx.bfloat16)
        x = (mx.random.normal((2, 3, heads, D)) * 2).astype(mx.bfloat16)
        wide = (mx.random.normal((6, 7 + heads * D)) * 2).astype(mx.bfloat16)
        z = wide[:, 7:].reshape(2, 3, heads, D)
        ref = norm(x, z).reshape(2, 3, heads * D)
        out = fused_ops.gated_norm(norm, x, wide, 7)
        self.assertTrue(mx.array_equal(out, ref).item())

    def test_add_rms_norm_matches_ops(self):
        for k in (5120, 2048, 64):
            norm = nn.RMSNorm(k, eps=1e-6)
            norm.weight = (mx.random.normal((k,)) * 0.2 + 1).astype(mx.bfloat16)
            x = (mx.random.normal((1, 3, k)) * 2).astype(mx.bfloat16)
            r = (mx.random.normal((1, 3, k)) * 2).astype(mx.bfloat16)
            h, out = fused_ops.add_rms_norm(norm, x, r)
            self.assertTrue(mx.array_equal(h, x + r).item(), k)
            self.assertTrue(mx.array_equal(out, norm(x + r)).item(), k)
            _, out = fused_ops.add_rms_norm(norm, x)
            self.assertTrue(mx.array_equal(out, norm(x)).item(), k)

    def test_prep_with_residual_and_gate_offset(self):
        k, m = 2048, 4
        w = (mx.random.normal((k,)) * 0.2 + 1).astype(mx.bfloat16)
        x = (mx.random.normal((m, k)) * 2).astype(mx.bfloat16)
        r = (mx.random.normal((m, k)) * 2).astype(mx.bfloat16)
        p = qmv_small.prep(x, "rms_norm", w, eps=1e-6, residual=r)
        ref = qmv_small.prep(x + r, "rms_norm", w, eps=1e-6)
        self.assertTrue(mx.array_equal(p.h, x + r).item())
        for a, b in ((p.x16, ref.x16), (p.xsum, ref.xsum), (p.rscale, ref.rscale)):
            self.assertTrue(mx.array_equal(a, b).item())
        # The gated norm prep reads the gate inside a wider row
        D, heads = 128, 4
        wn = (mx.random.normal((D,)) * 0.2 + 1).astype(mx.bfloat16)
        y = (mx.random.normal((m, heads * D)) * 2).astype(mx.bfloat16)
        wide = (mx.random.normal((m, 16 + heads * D + 8)) * 2).astype(mx.bfloat16)
        z = mx.contiguous(wide[:, 16 : 16 + heads * D])
        p = qmv_small.prep(y, "gated_norm", wide, wn, eps=1e-6, d=D, gs=wide.shape[1], go=16)
        ref = qmv_small.prep(y, "gated_norm", z, wn, eps=1e-6, d=D)
        for a, b in ((p.x16, ref.x16), (p.xsum, ref.xsum), (p.rscale, ref.rscale)):
            self.assertTrue(mx.array_equal(a, b).item())

    def test_attention_kernels_match_ops(self):
        args = qwen3_5.TextModelArgs.from_dict(
            {**CONFIG, "hidden_size": 128, "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 256}
        )
        attn = qwen3_5.Attention(args)
        attn.set_dtype(mx.bfloat16)
        mx.eval(attn.parameters())
        H, Hkv, Dh = 4, 2, 256
        for B, L, offset in ((1, 1, 0), (1, 4, 513), (2, 3, 100), (1, 6, 131000)):
            qkv = (mx.random.normal((B, L, 2 * (H + Hkv) * Dh)) * 2).astype(mx.bfloat16)
            q, k, v = fused_ops.attn_qkv(attn, qkv, offset)
            # the ops chain
            q_dim, kv_dim = 2 * H * Dh, Hkv * Dh
            qo, ko, vo = mx.split(qkv, [q_dim, q_dim + kv_dim], axis=-1)
            queries, gate = mx.split(qo.reshape(B, L, H, -1), 2, axis=-1)
            rq = attn.rope(attn.q_norm(queries).transpose(0, 2, 1, 3), offset=offset)
            rk = attn.rope(attn.k_norm(ko.reshape(B, L, Hkv, -1)).transpose(0, 2, 1, 3), offset=offset)
            rv = vo.reshape(B, L, Hkv, -1).transpose(0, 2, 1, 3)
            for name, a, b in (("q", q, rq), ("k", k, rk), ("v", v, rv)):
                self.assertTrue(mx.array_equal(a, b).item(), f"{name} B={B} L={L} offset={offset}")
            # the output gate
            x = (mx.random.normal((B, H, L, Dh)) * 2).astype(mx.bfloat16)
            ref = x.transpose(0, 2, 1, 3).reshape(B, L, -1) * mx.sigmoid(gate.reshape(B, L, -1))
            out = fused_ops.attn_gate(x, qkv.reshape(B * L, -1))
            self.assertTrue(mx.array_equal(out, ref).item())
            # the gate prep reading the attention layouts in place
            layout = (L, H, Dh, qkv.shape[-1])
            p = qmv_small.prep(x, "gate", qkv.reshape(B * L, -1), attn=layout)
            xr = x.transpose(0, 2, 1, 3).reshape(B * L, -1)
            pr = qmv_small.prep(xr, "gate", mx.contiguous(gate.reshape(B * L, -1)))
            # The padded rows of xsum and rscale are not written
            M = B * L
            for a, b in ((p.x16, pr.x16), (p.xsum[:, :M], pr.xsum[:, :M]), (p.rscale[:M], pr.rscale[:M])):
                self.assertTrue(mx.array_equal(a, b).item())

    def test_moe_slots_and_slot_sum(self):
        """The MoE slot rows sum to the block output; the consumers sum slot rows as mx.sum."""
        from mlx_lm.models import moe_small

        m = 4
        # The whole block against the ops, and its slot rows
        import sys

        sys.path.insert(0, os.path.dirname(__file__))
        from test_qwen3_5_fusion import MOE_CONFIG, _moe_block
        from mlx.utils import tree_map

        config = {**MOE_CONFIG, "hidden_size": 2048, "moe_intermediate_size": 512,
                  "shared_expert_intermediate_size": 512, "num_experts": 16, "num_experts_per_tok": 8}
        block = _moe_block(config, scale=0.05)
        nn.quantize(block, 64, 4)
        block.update(tree_map(lambda p: p.astype(mx.bfloat16) if p.dtype == mx.float32 else p, block.parameters()))
        for mm in (1, 3, 8):
            xx = mx.random.normal((1, mm, 2048)).astype(mx.bfloat16)
            lg = block.gate(xx)
            inds = mx.argpartition(lg[..., :16], kth=-8, axis=-1)[..., -8:]
            y = moe_small.experts(block, xx, lg, inds).astype(mx.float32)
            ys = moe_small.experts(block, xx, lg, inds, slots=True)
            self.assertTrue(mx.array_equal(ys.sum(axis=-2).astype(mx.float32), y).item())
            from unittest import mock

            with mock.patch.object(moe_small, "routes", return_value=False):
                expected = block(xx).astype(mx.float32)
            tol = 0.03 * mx.abs(expected).max().item()
            self.assertLess(mx.abs(y - expected).max().item(), tol, mm)
        # residual slot rows
        for R, Kd in ((9, 2048), (3, 5120), (1, 2048)):
            norm = nn.RMSNorm(Kd, eps=1e-6)
            norm.weight = (mx.random.normal((Kd,)) * 0.2 + 1).astype(mx.bfloat16)
            xx = (mx.random.normal((1, m, Kd)) * 2).astype(mx.bfloat16)
            rr = (mx.random.normal((1, m, R, Kd)) * 2).astype(mx.bfloat16)
            h, out = fused_ops.add_rms_norm(norm, xx, rr.reshape(-1, Kd), R)
            ref_h = xx + rr.sum(axis=-2)
            self.assertTrue(mx.array_equal(h, ref_h).item(), (R, Kd))
            self.assertTrue(mx.array_equal(out, norm(ref_h)).item(), (R, Kd))
            p = qmv_small.prep(xx.reshape(m, Kd), "rms_norm", norm.weight, eps=1e-6, residual=(rr.reshape(-1, Kd), R))
            pr = qmv_small.prep(ref_h.reshape(m, Kd), "rms_norm", norm.weight, eps=1e-6)
            self.assertTrue(mx.array_equal(p.h, ref_h.reshape(m, Kd)).item())
            for a, b in ((p.x16, pr.x16), (p.xsum[:, :m], pr.xsum[:, :m]), (p.rscale[:m], pr.rscale[:m])):
                self.assertTrue(mx.array_equal(a, b).item(), (R, Kd))

    def test_model_matches_unfused(self):
        """The decode and verify forwards of the merged path equal the ops path bitwise."""
        model = _model()
        mx.random.seed(3)
        prompt = mx.random.randint(0, CONFIG["vocab_size"], (1, 12))
        toks = mx.random.randint(0, CONFIG["vocab_size"], (1, 6))
        outs = []
        for enabled in (True, False):
            fused_ops._ENABLED = enabled
            cache = make_prompt_cache(model)
            mx.eval(model(prompt, cache=cache))
            res = []
            for s in (1, 4, 6):
                logits, hidden = model(toks[:, :s], cache=cache, return_hidden=True)
                mx.eval(logits, hidden)
                res += [logits, hidden]
            res += [c[0] for c in cache if not hasattr(c, "keys")]
            res += [c[1] for c in cache if not hasattr(c, "keys")]
            outs.append(res)
        fused_ops._ENABLED = True
        for a, b in zip(*outs):
            self.assertTrue(mx.array_equal(a, b).item())

    def test_sibling_rows_match_paths(self):
        """The merged tree forward equals the ops tree forward bitwise; its chain
        rows are those of a plain forward and each sibling row is the last row of
        its chain path, up to the bf16 rounding of the masked attention kernel."""
        model = _model()
        mx.random.seed(4)
        prompt = mx.random.randint(0, CONFIG["vocab_size"], (1, 12))
        k = 3
        rows = mx.random.randint(0, CONFIG["vocab_size"], (1, 2 * k + 1))
        trees = []
        for enabled in (True, False):
            fused_ops._ENABLED = enabled
            cache = make_prompt_cache(model)
            mx.eval(model(prompt, cache=cache))
            # The ArraysCache state is a list the forward mutates: restore copies
            base = [copy.deepcopy(c.state) for c in cache]

            def restore():
                for c, st in zip(cache, base):
                    c.state = copy.deepcopy(st)

            tree = model(rows, cache=cache, chain=k + 1)
            mx.eval(tree)
            trees.append(tree)
            restore()
            plain = model(rows, cache=cache)
            mx.eval(plain)
            self.assertTrue(
                mx.allclose(tree[:, : k + 1], plain[:, : k + 1], atol=0.1, rtol=0.05).item()
            )
            for i in range(1, k + 1):
                restore()
                ref = model(rows[:, list(range(i)) + [k + i]], cache=cache)
                mx.eval(ref)
                self.assertTrue(
                    mx.allclose(tree[:, k + i], ref[:, -1], atol=0.1, rtol=0.05).item(),
                    (enabled, i),
                )
        fused_ops._ENABLED = True
        self.assertTrue(mx.array_equal(*trees).item())


if __name__ == "__main__":
    unittest.main()


@unittest.skipUnless(mx.metal.is_available(), "Metal only")
class TestLatencyKernels(unittest.TestCase):
    """The router matvec and the two-pass sdpa are bitwise the MLX kernels they replace."""

    def test_router_matches_quantized_matmul(self):
        from mlx_lm.models import moe_small

        for N, K in ((257, 2048), (513, 2560), (129, 1024), (8, 512)):
            gate = nn.QuantizedLinear(K, N, bias=False, group_size=64, bits=8)
            for seed in range(20):
                mx.random.seed(seed)
                w = mx.random.randint(0, 2**32 - 1, gate.weight.shape, dtype=mx.uint32)
                gate.weight = w
                gate.scales = (mx.random.uniform(0.001, 0.05, gate.scales.shape) * (1 + seed % 3)).astype(mx.bfloat16)
                gate.biases = (mx.random.normal(gate.biases.shape) * 0.5).astype(mx.bfloat16)
                x = (mx.random.normal((1, 1, K)) * (0.5 + seed)).astype(mx.bfloat16)
                self.assertTrue(moe_small.router_ok(gate, x))
                got = moe_small.router(gate, x)
                ref = gate(x)
                self.assertEqual(got.shape, ref.shape)
                self.assertTrue(
                    mx.array_equal(got.view(mx.uint16), ref.view(mx.uint16)).item(),
                    f"N={N} K={K} seed={seed}",
                )

    def test_sdpa_two_pass_matches_sdpa_vector(self):
        for H, Hkv, D in ((16, 2, 256), (8, 2, 128), (4, 4, 64)):
            scale = D**-0.5
            for N in (1, 2, 31, 32, 33, 64, 100, 257, 500, 512, 777, 1000, 1023):
                mx.random.seed(N)
                cap = (N + 255) // 256 * 256
                q = (mx.random.normal((1, H, 1, D)) * 2).astype(mx.bfloat16)
                k = (mx.random.normal((1, Hkv, cap, D)) * 2).astype(mx.bfloat16)
                v = (mx.random.normal((1, Hkv, cap, D)) * 2).astype(mx.bfloat16)
                kv, vv = k[..., :N, :], v[..., :N, :]
                self.assertTrue(fused_ops.sdpa_ok(q, kv, None))
                got = fused_ops.sdpa_two_pass(q, kv, vv, scale)
                ref = mx.fast.scaled_dot_product_attention(q, kv, vv, scale=scale)
                self.assertEqual(got.shape, ref.shape)
                self.assertTrue(
                    mx.array_equal(got.view(mx.uint16), ref.view(mx.uint16)).item(),
                    f"H={H} Hkv={Hkv} D={D} N={N}",
                )

    def test_topk_prep_matches_argpartition_and_prep(self):
        from mlx_lm.models import moe_small

        E, top_k, K = 256, 8, 2048
        for seed in range(60):
            mx.random.seed(seed)
            lg = mx.random.normal((1, 1, E + 1)) * 3
            if seed % 3 == 1:
                lg = mx.round(lg * 2) / 2  # many exact ties
            if seed % 3 == 2:
                lg = mx.round(lg) / 4
            logits = lg.astype(mx.bfloat16)
            x = (mx.random.normal((1, 1, K)) * (1 + seed % 5)).astype(mx.bfloat16)
            ref_inds = mx.argpartition(logits[..., :E], kth=-top_k, axis=-1)[..., -top_k:].reshape(-1)
            x16, xsum, rscale, plan, inds = moe_small.topk_prep(x.reshape(1, K), logits, E, top_k)
            self.assertTrue(mx.array_equal(inds, ref_inds).item(), f"seed {seed}: {inds.tolist()} vs {ref_inds.tolist()}")
            r16, rsum, rrs, rplan = moe_small._prep(x.reshape(1, K), "copy", ref_inds, 1, top_k, E)
            # Only column 0 of xsum and rscale[0] are written for one row
            for name, a, b in (("x16", x16, r16), ("xsum", xsum[:, :1], rsum[:, :1]), ("rscale", rscale[:1], rrs[:1]), ("plan", plan, rplan)):
                self.assertTrue(mx.array_equal(a, b).item(), f"{name} seed {seed}")
