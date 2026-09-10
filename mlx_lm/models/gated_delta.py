import os
from functools import partial
from typing import Optional, Tuple

import mlx.core as mx
import mlx.nn as nn

# For the shapes it supports, the packed kernel is bitwise-identical by
# construction to an explicit-tree comparator kernel that the tests pin it
# against (see _make_gated_delta_packed_kernel). Every other shape (masks,
# vector gates, Dk != 128) and the MLX_GDN_PACKED=0 kill-switch use the
# original simd_sum kernels, unchanged.
_ENABLE_GDN_PACKED = os.environ.get("MLX_GDN_PACKED", "1") != "0"


@partial(mx.compile, shapeless=True)
def compute_g(A_log, a, dt_bias):
    return mx.exp(-mx.exp(A_log.astype(mx.float32)) * nn.softplus(a + dt_bias))


@partial(mx.compile, shapeless=True)
def compute_lower_bound_g(A_log, a, dt_bias, lower_bound):
    return mx.exp(
        lower_bound
        * mx.sigmoid(
            mx.exp(A_log.astype(mx.float32)) * (a.astype(mx.float32) + dt_bias)
        )
    )


def _generic_step(st, q, k, v, y, g, beta):
    """One recurrence step of the generic kernel on the state array ``st``."""
    return f"""
            float kv_mem = 0.0f;
            for (int i = 0; i < n_per_t; ++i) {{
              auto s_idx = n_per_t * dk_idx + i;
              {st}[i] = {st}[i] * {g};
              kv_mem += {st}[i] * {k}[s_idx];
            }}
            kv_mem = simd_sum(kv_mem);

            auto delta = ({v}[dv_idx] - kv_mem) * {beta};

            float out = 0.0f;
            for (int i = 0; i < n_per_t; ++i) {{
              auto s_idx = n_per_t * dk_idx + i;
              {st}[i] = {st}[i] + {k}[s_idx] * delta;
              out += {st}[i] * {q}[s_idx];
            }}
            out = simd_sum(out);
            if (thread_index_in_simdgroup == 0) {{
              {y}[dv_idx] = static_cast<InT>(out);
            }}"""


def _sibling_source(step, copy, g_stride):
    """The branch step of a sibling row: row C - 1 + t of a chain of C rows is
    computed from the state before chain step t, which is left unchanged."""
    return f"""
          if (t >= 1 && C - 1 + t < T) {{
            const int off = C - 1;
            auto qs = q_ + off * Hk * Dk;
            auto ks = k_ + off * Hk * Dk;
            auto vs = v_ + off * Hv * Dv;
            auto ys = y + off * Hv * Dv;
            auto gs = g_ + off * {g_stride};
            auto bs = beta_ + off * Hv;
            {copy}
            {step}
          }}"""


def _extra_source(step, q0, v0, g0, beta0):
    """One more step on row E (a sibling of the kept chain) after the chain steps.
    ``yb`` is the row-0 output pointer (``y`` has advanced with the chain)."""
    return f"""
        if (E >= 0) {{
          auto qs = {q0} + E * Hk * Dk;
          auto ks = {q0.replace("q", "k", 1)} + E * Hk * Dk;
          auto vs = {v0} + E * Hv * Dv;
          auto ys = yb + E * Hv * Dv;
          auto gs = {g0};
          auto bs = {beta0} + E * Hv;
          {step}
        }}"""


def _make_gated_delta_kernel(has_mask=False, vectorized=False, return_states=False):
    if not mx.metal.is_available():
        return None
    mask_source = "mask[b_idx * T + t]" if has_mask else "true"

    # Optionally record the state after every timestep for cache rollback
    if return_states:
        states_setup = (
            "auto states_ = states + ((b_idx * T * Hv + hv_idx) * Dv + dv_idx) * Dk;"
        )
        states_write = """for (int i = 0; i < n_per_t; ++i) {
            states_[n_per_t * dk_idx + i] = static_cast<StT>(state[i]);
          }
          states_ += Hv * Dv * Dk;"""
    else:
        states_setup = states_write = ""

    # Configure g indexing based on whether gating is vectorized
    if vectorized:
        g_comment = "// g: [B, T, Hv, Dk]"
        g_setup = "auto g_ = g + (b_idx * T * Hv + hv_idx) * Dk;"
        g_access, gs_access, g_stride = "g_[s_idx]", "gs[s_idx]", "Hv * Dk"
        g_advance = "g_ += Hv * Dk;"
    else:
        g_comment = "// g: [B, T, Hv]"
        g_setup = "auto g_ = g + b_idx * T * Hv;"
        g_access, gs_access, g_stride = "g_[hv_idx]", "gs[hv_idx]", "Hv"
        g_advance = "g_ += Hv;"
    chain_step = _generic_step("state", "q_", "k_", "v_", "y", g_access, "beta_[hv_idx]")
    sibling = extra = ""
    if not has_mask:
        sibling = _sibling_source(
            _generic_step("sst", "qs", "ks", "vs", "ys", gs_access, "bs[hv_idx]"),
            "float sst[n_per_t];\n            for (int i = 0; i < n_per_t; ++i) sst[i] = state[i];",
            g_stride,
        )
        g0 = (
            "g + (b_idx * T * Hv + hv_idx) * Dk + E * Hv * Dk"
            if vectorized
            else "g + b_idx * T * Hv + E * Hv"
        )
        extra = _extra_source(
            _generic_step("state", "qs", "ks", "vs", "ys", gs_access, "bs[hv_idx]"),
            "q + b_idx * T * Hk * Dk + hk_idx * Dk",
            "v + b_idx * T * Hv * Dv + hv_idx * Dv",
            g0,
            "beta + b_idx * T * Hv",
        )

    source = f"""
        auto n = thread_position_in_grid.z;
        auto b_idx = n / Hv;
        auto hv_idx = n % Hv;
        auto hk_idx = hv_idx / (Hv / Hk);
        constexpr int n_per_t = Dk / 32;

        // q, k: [B, T, Hk, Dk]
        auto q_ = q + b_idx * T * Hk * Dk + hk_idx * Dk;
        auto k_ = k + b_idx * T * Hk * Dk + hk_idx * Dk;

        // v, y: [B, T, Hv, Dv]
        auto v_ = v + b_idx * T * Hv * Dv + hv_idx * Dv;
        auto yb = y + b_idx * T * Hv * Dv + hv_idx * Dv;
        y = yb;

        auto dk_idx = thread_position_in_threadgroup.x;
        auto dv_idx = thread_position_in_grid.y;

        // state_in, state_out: [B, Hv, Dv, Dk]
        auto i_state = state_in + (n * Dv + dv_idx) * Dk;
        auto o_state = state_out + (n * Dv + dv_idx) * Dk;
        {states_setup}

        float state[n_per_t];
        for (int i = 0; i < n_per_t; ++i) {{
          auto s_idx = n_per_t * dk_idx + i;
          state[i] = static_cast<float>(i_state[s_idx]);
        }}

        {g_comment}
        {g_setup}
        auto beta_ = beta + b_idx * T * Hv;

        for (int t = 0; t < steps; ++t) {{
          {sibling}
          if ({mask_source}) {{
            {chain_step}
          }} else {{
            y[dv_idx] = static_cast<InT>(0);
          }}
          {states_write}
          // Increment data pointers to next time step
          q_ += Hk * Dk;
          k_ += Hk * Dk;
          v_ += Hv * Dv;
          y += Hv * Dv;
          {g_advance}
          beta_ += Hv;
        }}
        {extra}
        for (int i = 0; i < n_per_t; ++i) {{
          auto s_idx = n_per_t * dk_idx + i;
          o_state[s_idx] = static_cast<StT>(state[i]);
        }}
    """
    inputs = ["q", "k", "v", "g", "beta", "state_in", "T", "steps", "C", "E"]
    if has_mask:
        inputs.append("mask")

    suffix = ""
    if vectorized:
        suffix += "_vec"
    if has_mask:
        suffix += "_mask"
    outputs = ["y", "state_out"]
    if return_states:
        suffix += "_states"
        outputs.append("states")

    return mx.fast.metal_kernel(
        name=f"gated_delta_step{suffix}",
        input_names=inputs,
        output_names=outputs,
        source=source,
    )


def _make_gated_delta_kernel_xtree():
    """Scalar-gate, unmasked kernel with an explicitly-written reduction.

    This is the unpacked comparator for the packed kernel: it replaces the
    two simd_sum calls with the ascending butterfly (shuffle_xor 1,2,4,8,16)
    written out in source, so the reduction order is a contract of this file
    rather than of the simd_sum lowering. On current Apple GPUs this is the
    same tree simd_sum lowers to, so it is bit-identical to the generic
    kernel there. Only shapes eligible for the packed kernel ever run it;
    masked, vector-gate and Dk != 128 paths keep the original kernels.
    """
    if not mx.metal.is_available():
        return None

    source = """
        auto n = thread_position_in_grid.z;
        auto b_idx = n / Hv;
        auto hv_idx = n % Hv;
        auto hk_idx = hv_idx / (Hv / Hk);
        constexpr int n_per_t = Dk / 32;

        // q, k: [B, T, Hk, Dk]
        auto q_ = q + b_idx * T * Hk * Dk + hk_idx * Dk;
        auto k_ = k + b_idx * T * Hk * Dk + hk_idx * Dk;

        // v, y: [B, T, Hv, Dv]
        auto v_ = v + b_idx * T * Hv * Dv + hv_idx * Dv;
        y += b_idx * T * Hv * Dv + hv_idx * Dv;

        auto dk_idx = thread_position_in_threadgroup.x;
        auto dv_idx = thread_position_in_grid.y;

        // state_in, state_out: [B, Hv, Dv, Dk]
        auto i_state = state_in + (n * Dv + dv_idx) * Dk;
        auto o_state = state_out + (n * Dv + dv_idx) * Dk;

        float state[n_per_t];
        for (int i = 0; i < n_per_t; ++i) {
          auto s_idx = n_per_t * dk_idx + i;
          state[i] = static_cast<float>(i_state[s_idx]);
        }

        // g, beta: [B, T, Hv]
        auto g_ = g + b_idx * T * Hv;
        auto beta_ = beta + b_idx * T * Hv;

        for (int t = 0; t < T; ++t) {
          float kv_mem = 0.0f;
          for (int i = 0; i < n_per_t; ++i) {
            auto s_idx = n_per_t * dk_idx + i;
            state[i] = state[i] * g_[hv_idx];
            kv_mem += state[i] * k_[s_idx];
          }
          kv_mem += simd_shuffle_xor(kv_mem, 1);
          kv_mem += simd_shuffle_xor(kv_mem, 2);
          kv_mem += simd_shuffle_xor(kv_mem, 4);
          kv_mem += simd_shuffle_xor(kv_mem, 8);
          kv_mem += simd_shuffle_xor(kv_mem, 16);

          auto delta = (v_[dv_idx] - kv_mem) * beta_[hv_idx];

          float out = 0.0f;
          for (int i = 0; i < n_per_t; ++i) {
            auto s_idx = n_per_t * dk_idx + i;
            state[i] = state[i] + k_[s_idx] * delta;
            out += state[i] * q_[s_idx];
          }
          out += simd_shuffle_xor(out, 1);
          out += simd_shuffle_xor(out, 2);
          out += simd_shuffle_xor(out, 4);
          out += simd_shuffle_xor(out, 8);
          out += simd_shuffle_xor(out, 16);
          if (thread_index_in_simdgroup == 0) {
            y[dv_idx] = static_cast<InT>(out);
          }
          q_ += Hk * Dk;
          k_ += Hk * Dk;
          v_ += Hv * Dv;
          y += Hv * Dv;
          g_ += Hv;
          beta_ += Hv;
        }
        for (int i = 0; i < n_per_t; ++i) {
          auto s_idx = n_per_t * dk_idx + i;
          o_state[s_idx] = static_cast<StT>(state[i]);
        }
    """
    return mx.fast.metal_kernel(
        name="gated_delta_step_xtree",
        input_names=["q", "k", "v", "g", "beta", "state_in", "T"],
        output_names=["y", "state_out"],
        source=source,
    )


def _make_gated_delta_packed_kernel():
    """Make the scalar-gate Dk=128 prefill specialization.

    The generic kernel assigns one 32-lane SIMD-group to each value row. For
    Dk=128 that leaves every lane with only four state elements and performs
    two full-SIMD reductions per row and token. This kernel instead packs
    eight independent value rows into a SIMD-group: four lanes own each row
    and each lane keeps 32 contiguous state elements in registers.

    The reduction reproduces the unpacked comparator
    (_make_gated_delta_kernel_xtree) bitwise BY CONSTRUCTION: both use the
    same explicitly-written ascending butterfly (shuffle_xor 1,2,4,8,16)
    rather than relying on how simd_sum lowers. The butterfly's first three
    levels combine partials that live in a single packed lane (IEEE addition
    is commutative, so the local pairwise tree is bit-identical), and the
    last two levels map onto shuffle_xor(1) and shuffle_xor(2) within the
    four-lane row group. Each 4-element partial keeps the comparator's
    sequential order, so y and the state are bit-identical to it on any
    device. On current Apple GPUs the explicit tree is also bit-identical
    to the simd_sum-based generic kernel.
    """
    if not mx.metal.is_available():
        return None

    def step(st, q, k, v, y, g, beta):
        return f"""
          float gt = static_cast<float>({g});

          // Partials mirror the generic kernel: each 4-element chain is one
          // original lane's sequential accumulation.
          float part[partials_per_lane];
          for (int pb = 0; pb < partials_per_lane; ++pb) {{
            float acc = 0.0f;
            for (int i = 0; i < 4; ++i) {{
              int e = pb * 4 + i;
              {st}[e] = {st}[e] * gt;
              acc += {st}[e] * static_cast<float>({k}[e]);
            }}
            part[pb] = acc;
          }}
          // Butterfly levels xor 1,2,4 stay inside this lane (commutative
          // pairwise tree); levels xor 8,16 become the row-group shuffles.
          float kv_mem =
              ((part[0] + part[1]) + (part[2] + part[3])) +
              ((part[4] + part[5]) + (part[6] + part[7]));
          kv_mem += simd_shuffle_xor(kv_mem, 1);
          kv_mem += simd_shuffle_xor(kv_mem, 2);

          auto delta =
              (static_cast<float>({v}[dv_idx]) - kv_mem) *
              static_cast<float>({beta});

          for (int pb = 0; pb < partials_per_lane; ++pb) {{
            float acc = 0.0f;
            for (int i = 0; i < 4; ++i) {{
              int e = pb * 4 + i;
              {st}[e] = {st}[e] + static_cast<float>({k}[e]) * delta;
              acc += {st}[e] * static_cast<float>({q}[e]);
            }}
            part[pb] = acc;
          }}
          float out =
              ((part[0] + part[1]) + (part[2] + part[3])) +
              ((part[4] + part[5]) + (part[6] + part[7]));
          out += simd_shuffle_xor(out, 1);
          out += simd_shuffle_xor(out, 2);
          if (lane_in_row == 0) {{
            {y}[dv_idx] = static_cast<InT>(out);
          }}"""

    chain_step = step("state", "q_", "k_", "v_", "y", "g_[hv_idx]", "beta_[hv_idx]")
    sibling = _sibling_source(
        step("sst", "qs", "ks", "vs", "ys", "gs[hv_idx]", "bs[hv_idx]"),
        "float sst[values_per_lane];\n            for (int i = 0; i < values_per_lane; ++i) sst[i] = state[i];",
        "Hv",
    )
    extra = _extra_source(
        step("state", "qs", "ks", "vs", "ys", "gs[hv_idx]", "bs[hv_idx]"),
        "q + (b_idx * T * Hk + hk_idx) * Dk + lane_in_row * values_per_lane",
        "v + (b_idx * T * Hv + hv_idx) * Dv",
        "g + b_idx * T * Hv + E * Hv",
        "beta + b_idx * T * Hv",
    )
    source = f"""
        constexpr int lanes_per_row = 4;
        constexpr int rows_per_simdgroup = 32 / lanes_per_row;
        constexpr int values_per_lane = Dk / lanes_per_row;
        constexpr int partials_per_lane = values_per_lane / 4;

        auto n = thread_position_in_grid.z;
        auto b_idx = n / Hv;
        auto hv_idx = n % Hv;
        auto hk_idx = hv_idx / (Hv / Hk);

        auto lane = thread_index_in_simdgroup;
        auto row_in_simdgroup = lane / lanes_per_row;
        auto lane_in_row = lane & (lanes_per_row - 1);
        auto row_group = thread_position_in_grid.y;
        auto dv_idx = row_group * rows_per_simdgroup + row_in_simdgroup;

        // q, k: [B, T, Hk, Dk]
        auto q_ = q + (b_idx * T * Hk + hk_idx) * Dk + lane_in_row * values_per_lane;
        auto k_ = k + (b_idx * T * Hk + hk_idx) * Dk + lane_in_row * values_per_lane;

        // v, y: [B, T, Hv, Dv]
        auto v_ = v + (b_idx * T * Hv + hv_idx) * Dv;
        auto yb = y + (b_idx * T * Hv + hv_idx) * Dv;
        y = yb;

        // state_in, state_out: [B, Hv, Dv, Dk]
        auto i_state = state_in + (n * Dv + dv_idx) * Dk + lane_in_row * values_per_lane;
        auto o_state = state_out + (n * Dv + dv_idx) * Dk + lane_in_row * values_per_lane;

        float state[values_per_lane];
        for (int i = 0; i < values_per_lane; ++i) {{
          state[i] = static_cast<float>(i_state[i]);
        }}

        // g, beta: [B, T, Hv]
        auto g_ = g + b_idx * T * Hv;
        auto beta_ = beta + b_idx * T * Hv;

        for (int t = 0; t < steps; ++t) {{
          {sibling}
          {chain_step}

          q_ += Hk * Dk;
          k_ += Hk * Dk;
          v_ += Hv * Dv;
          y += Hv * Dv;
          g_ += Hv;
          beta_ += Hv;
        }}
        {extra}

        for (int i = 0; i < values_per_lane; ++i) {{
          o_state[i] = static_cast<StT>(state[i]);
        }}
    """
    return mx.fast.metal_kernel(
        name="gated_delta_step_packed_btree",
        input_names=["q", "k", "v", "g", "beta", "state_in", "T", "steps", "C", "E"],
        output_names=["y", "state_out"],
        source=source,
    )


# Keyed by (has_mask, vectorized, return_states)
_gated_delta_kernels = {
    (m, v, s): _make_gated_delta_kernel(m, v, s)
    for m in (False, True)
    for v in (False, True)
    for s in (False, True)
}
_gated_delta_kernel_xtree = _make_gated_delta_kernel_xtree()
_gated_delta_kernel_packed = _make_gated_delta_packed_kernel()


@mx.compile
def _gated_delta_step_ops(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    g: mx.array,
    beta: mx.array,
    state: mx.array,
    mask: Optional[mx.array] = None,
) -> Tuple[mx.array, mx.array]:
    """
    Ops-based reference implementation for a single recurrent step.

    Shapes:
      - q, k: [B, H, Dk]
      - v: [B, H, Dv]
      - g: [B, H] or [B, H, Dk]
      - beta: [B, H]
      - state: [B, H, Dv, Dk]
    Returns:
      - y: [B, H, Dv]
      - new_state: [B, H, Dv, Dk]
    """

    # Decay
    old_state = state
    if g.ndim == 2:
        decay = g[..., None, None]
    elif g.ndim == 3:
        decay = g[..., None, :]
    else:
        raise ValueError(f"Unsupported gating shape {g.shape}")
    state = state * decay
    kv_mem = (state * k[..., None, :]).sum(axis=-1)  # [B, H, Dv]
    delta = (v - kv_mem) * beta[..., None]  # [B, H, Dv]
    state = state + k[..., None, :] * delta[..., None]
    # Output projection along key dim with q
    y = (state * q[..., None, :]).sum(axis=-1)  # [B, H, Dv]

    if mask is not None:
        mask = mx.expand_dims(mask, axis=(1, 2, 3))
        state = mx.where(mask, state, old_state)
    return y.astype(q.dtype), state


def _gated_delta_kernel_impl(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    g: mx.array,
    beta: mx.array,
    state: mx.array,
    mask: Optional[mx.array] = None,
    *,
    allow_packed: bool,
    return_states: bool = False,
    steps=None,
    chain=None,
    extra=None,
) -> Tuple[mx.array, ...]:
    B, T, Hk, Dk = k.shape
    Hv, Dv = v.shape[2:]
    input_type = q.dtype
    state_type = state.dtype

    # The packed kernel gives each lane Dk/4 state elements and packs 32/4
    # value rows into a SIMD-group, so it needs Dk == 128 and Dv divisible by
    # 8. It is otherwise generic in B, Hk, Hv and the input element type.
    # Vector gating, padding masks and per-step states use the original kernels.
    packed_eligible = (
        mask is None
        and g.ndim == 3
        and Dk == 128
        and Dv % 8 == 0
        and g.dtype == mx.float32
        and state.dtype == mx.float32
    )

    # Rows past the first ``chain`` rows are siblings: row chain - 1 + t branches
    # from the state before chain step t.
    if chain is not None and (mask is not None or return_states):
        raise ValueError("Sibling rows need an unmasked recurrence")
    # Only the chain rows are steps of the recurrence; ``extra`` (a row index,
    # -1 for none) is one more step after them
    if steps is None:
        steps = chain or T
    if extra is not None and (mask is not None or return_states):
        raise ValueError("An extra row needs an unmasked recurrence")
    inputs = [q, k, v, g, beta, state, T, steps, chain or T, -1 if extra is None else extra]
    if mask is not None:
        inputs.append(mask)
    output_shapes = [(B, T, Hv, Dv), state.shape]
    output_dtypes = [input_type, state_type]
    if return_states:
        output_shapes.append((B, T, *state.shape[1:]))
        output_dtypes.append(state_type)

    if packed_eligible and allow_packed and _ENABLE_GDN_PACKED and not return_states:
        kernel = _gated_delta_kernel_packed
        grid = (32, Dv // 8, B * Hv)
        threadgroup = (32, 2, 1)
    else:
        kernel = _gated_delta_kernels[(mask is not None, g.ndim == 4, return_states)]
        grid = (32, Dv, B * Hv)
        threadgroup = (32, 4, 1)

    return kernel(
        inputs=inputs,
        template=[
            ("InT", input_type),
            ("StT", state_type),
            ("Dk", Dk),
            ("Dv", Dv),
            ("Hk", Hk),
            ("Hv", Hv),
        ],
        grid=grid,
        threadgroup=threadgroup,
        output_shapes=output_shapes,
        output_dtypes=output_dtypes,
    )


def gated_delta_kernel_xtree(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    g: mx.array,
    beta: mx.array,
    state: mx.array,
    mask: Optional[mx.array] = None,
) -> Tuple[mx.array, mx.array]:
    """Explicit-tree comparator for the packed kernel (test use).

    Runs the unpacked layout with the same explicitly-written reduction tree
    the packed kernel uses, defining its bitwise contract independently of
    the simd_sum lowering.
    """
    assert mask is None
    B, T, Hk, Dk = k.shape
    Hv, Dv = v.shape[2:]
    return _gated_delta_kernel_xtree(
        inputs=[q, k, v, g, beta, state, T],
        template=[
            ("InT", q.dtype),
            ("StT", state.dtype),
            ("Dk", Dk),
            ("Dv", Dv),
            ("Hk", Hk),
            ("Hv", Hv),
        ],
        grid=(32, Dv, B * Hv),
        threadgroup=(32, 4, 1),
        output_shapes=[(B, T, Hv, Dv), state.shape],
        output_dtypes=[q.dtype, state.dtype],
    )


def gated_delta_kernel_unpacked(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    g: mx.array,
    beta: mx.array,
    state: mx.array,
    mask: Optional[mx.array] = None,
    steps=None,
    chain=None,
    extra=None,
) -> Tuple[mx.array, mx.array]:
    """Run the original one-value-row-per-SIMD-group kernel."""
    return _gated_delta_kernel_impl(
        q, k, v, g, beta, state, mask, allow_packed=False, steps=steps, chain=chain, extra=extra
    )


def gated_delta_kernel(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    g: mx.array,
    beta: mx.array,
    state: mx.array,
    mask: Optional[mx.array] = None,
    return_states: bool = False,
    steps=None,
    chain=None,
    extra=None,
) -> Tuple[mx.array, ...]:
    return _gated_delta_kernel_impl(
        q,
        k,
        v,
        g,
        beta,
        state,
        mask,
        allow_packed=True,
        return_states=return_states,
        steps=steps,
        chain=chain,
        extra=extra,
    )


def gated_delta_ops(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    g: mx.array,
    beta: mx.array,
    state: Optional[mx.array] = None,
    mask: Optional[mx.array] = None,
    return_states: bool = False,
    steps=None,
    chain=None,
    extra=None,
) -> Tuple[mx.array, ...]:
    """
    Ops-based reference implementation for prompt prefill (sequential loop).
    Supports both scalar and vectorized gating. With ``chain`` the rows past the
    first ``chain`` rows are siblings: row chain - 1 + t is one step from the
    state before chain step t. ``extra`` (a row index, -1 for none) is one more
    step after the first ``steps`` rows.

    Shapes:
      - q, k: [B, T, Hk, Dk]
      - v: [B, T, Hv, Dv]
      - g: [B, T, Hv] (scalar) or [B, T, Hv, Dk] (vectorized)
      - beta: [B, T, Hv]
      - state: [B, Hv, Dv, Dk]
    Returns:
      - y: [B, T, Hv, Dv]
      - state: [B, Hv, Dv, Dk]
    """
    B, T, Hk, Dk = q.shape
    Hv, Dv = v.shape[-2:]
    n_steps = T if steps is None else int(steps)
    if chain is not None:
        n_steps = chain
    if state is None:
        state = mx.zeros((B, Hv, Dv, Dk), dtype=mx.float32)

    if (repeat_factor := Hv // Hk) > 1:
        q = mx.repeat(q, repeat_factor, -2)
        k = mx.repeat(k, repeat_factor, -2)

    def step(t, state):
        return _gated_delta_step_ops(
            q[:, t],
            k[:, t],
            v[:, t],
            g[:, t],
            beta[:, t],
            state,
            None if mask is None else mask[:, t],
        )

    ys, states = [None] * T, []
    for t in range(n_steps):
        if chain is not None and 1 <= t and chain - 1 + t < T:
            ys[chain - 1 + t] = step(chain - 1 + t, state)[0]
        ys[t], state = step(t, state)
        states.append(state)
    if extra is not None and int(extra) >= 0:
        ys[int(extra)], state = step(int(extra), state)
    # The rows of the chain steps, or all rows when siblings or an extra row exist
    if chain is None and extra is None:
        ys = ys[:n_steps]
    y = mx.stack([mx.zeros_like(ys[0]) if r is None else r for r in ys], axis=1)
    if return_states:
        return y, state, mx.stack(states, axis=1)
    return y, state


def gated_delta_update(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    a: mx.array,
    b: mx.array,
    A_log: mx.array,
    dt_bias: mx.array,
    state: Optional[mx.array] = None,
    mask: Optional[mx.array] = None,
    use_kernel: bool = True,
    lower_bound: float | None = None,
    return_states: bool = False,
    steps=None,
    chain=None,
    extra=None,
) -> Tuple[mx.array, ...]:
    """Gated delta rule recurrence.

    Contract: callers fold the ``Dk**-0.5`` readout scale into q before calling
    (e.g. ``inv_scale = Dk**-0.5; q = inv_scale**2 * rms_norm(q, eps);
    k = inv_scale * rms_norm(k, eps)``). The helper applies no scale of its own.
    With ``return_states`` the state after every step ``[B, T, Hv, Dv, Dk]`` is
    also returned. ``steps`` (an int or a device scalar) runs only the first
    steps of the sequence. With ``chain`` the rows past the first ``chain`` rows
    are siblings of chain rows 1.. (see ``gated_delta_ops``).
    """
    beta = mx.sigmoid(b)
    if lower_bound is None:
        g = compute_g(A_log, a, dt_bias)
    else:
        g = compute_lower_bound_g(A_log, a, dt_bias, lower_bound)
    if state is None:
        B, _, Hk, Dk = q.shape
        Hv, Dv = v.shape[-2:]
        state = mx.zeros((B, Hv, Dv, Dk), dtype=mx.float32)

    if (
        not use_kernel
        or mx.default_device() != mx.gpu
        or not mx.metal.is_available()
        or k.shape[-1] < 32
        or k.shape[-1] % 32 != 0
    ):
        return gated_delta_ops(
            q, k, v, g, beta, state, mask, return_states, steps, chain, extra
        )
    return gated_delta_kernel(
        q, k, v, g, beta, state, mask, return_states, steps, chain, extra
    )
