# Copyright © 2026 Apple Inc.
"""Tensor-parallel helpers shared by MoE block classes for EXO."""
from typing import Callable, cast

import mlx.core as mx
import mlx.nn as nn

from .switch_layers import (
    QuantizedSwitchLinear,
    SwitchGLU,
    SwitchLinear,
    SwitchMLP,
    _gather_sort,
    _scatter_unsort,
)


def all_gather_last(x: mx.array, group: mx.distributed.Group) -> mx.array:
    """Fast all_gather over the last axis."""
    leading = x.shape[:-1]
    last = x.shape[-1]
    x2 = x.reshape(-1, last)
    xt = mx.contiguous(x2.T)
    g = mx.distributed.all_gather(xt, group=group)
    return mx.contiguous(g.T).reshape(*leading, last * group.size())


def splitk_override_for_unsharded(m: int, n_full: int, k: int) -> int:
    n = mx.compute_splitk_partitions(m, n_full, k)
    return n if n > 0 else -1


def linear_column_sharded_forward(
    lin: nn.Module, x: mx.array, group: mx.distributed.Group
) -> mx.array:
    """Per-rank weight: (N_full/group.size(), K). bf16 uses splitk override;
    quantized goes through the module directly (no splitk participation)."""
    if isinstance(lin, nn.QuantizedLinear):
        return lin.__call__(x)
    w = cast(mx.array, lin["weight"])
    per_rank_n, k = w.shape
    n_full = per_rank_n * group.size()
    m = x.shape[-2] if x.ndim >= 2 else 1
    mx.set_splitk_partitions_override(splitk_override_for_unsharded(m, n_full, k))
    try:
        if "bias" in lin:
            y = mx.addmm(cast(mx.array, lin["bias"]), x, w.T)
        else:
            y = mx.matmul(x, w.T)
    finally:
        mx.set_splitk_partitions_override(0)
    return y


def switch_mlp_activation(
    switch_mlp: SwitchGLU | SwitchMLP, x_gate: mx.array, x_up: mx.array
) -> mx.array:
    activation = getattr(switch_mlp, "activation", None)
    if activation is not None:
        return cast(mx.array, activation(x_up, x_gate))
    return nn.silu(x_gate) * x_up


def switch_linear_forward(
    sl: SwitchLinear | QuantizedSwitchLinear,
    x: mx.array,
    indices: mx.array,
    sorted_indices: bool,
) -> mx.array:
    """Forward a SwitchLinear/QuantizedSwitchLinear with column-sharded
    (output-dim-sliced) weights. Returns the per-rank output slice."""
    if isinstance(sl, QuantizedSwitchLinear):
        y = mx.gather_qmm(
            x,
            cast(mx.array, sl["weight"]),
            cast(mx.array, sl["scales"]),
            cast(mx.array | None, sl.get("biases")),
            rhs_indices=indices,
            transpose=True,
            group_size=sl.group_size,
            bits=sl.bits,
            mode=sl.mode,
            sorted_indices=sorted_indices,
        )
    else:
        y = mx.gather_mm(
            x,
            cast(mx.array, sl["weight"]).swapaxes(-1, -2),
            rhs_indices=indices,
            sorted_indices=sorted_indices,
        )
    if "bias" in sl:
        y = y + mx.expand_dims(cast(mx.array, sl["bias"])[indices], -2)
    return y


def switch_mlp_n_sharded_sharded_out(
    switch_mlp: SwitchGLU,
    x: mx.array,
    indices: mx.array,
    scores: mx.array,
    group: mx.distributed.Group,
) -> mx.array:
    """SwitchGLU with column-sharded down_proj. Returns (..., H/N) — still
    sharded on hidden dim; caller issues the final all_gather."""
    x_exp = mx.expand_dims(x, (-2, -3))
    do_sort = indices.size >= 64
    idx = indices
    inv_order = None
    if do_sort:
        x_exp, idx, inv_order = _gather_sort(x_exp, indices)

    x_up = switch_linear_forward(switch_mlp.up_proj, x_exp, idx, do_sort)
    x_gate = switch_linear_forward(switch_mlp.gate_proj, x_exp, idx, do_sort)
    hidden_shard = switch_mlp_activation(switch_mlp, x_gate, x_up)

    hidden_full = all_gather_last(hidden_shard, group)

    out_shard = switch_linear_forward(switch_mlp.down_proj, hidden_full, idx, do_sort)

    if do_sort:
        assert inv_order is not None
        out_shard = _scatter_unsort(out_shard, inv_order, indices.shape)
    out_shard = out_shard.squeeze(-2)
    return (out_shard * scores[..., None]).sum(axis=-2).astype(out_shard.dtype)


def switch_mlp_n_sharded(
    switch_mlp: SwitchGLU,
    x: mx.array,
    indices: mx.array,
    scores: mx.array,
    group: mx.distributed.Group,
) -> mx.array:
    out_shard = switch_mlp_n_sharded_sharded_out(switch_mlp, x, indices, scores, group)
    return all_gather_last(out_shard, group)


def switch_fc_n_sharded(
    switch_mlp: SwitchMLP,
    x: mx.array,
    indices: mx.array,
    group: mx.distributed.Group,
    activation: Callable[[mx.array], mx.array],
) -> mx.array:
    """SwitchMLP variant (fc1 -> activation -> fc2, both column-sharded).
    fc1 output is all_gathered before fc2; fc2 output is all_gathered after."""
    x_exp = mx.expand_dims(x, (-2, -3))
    do_sort = indices.size >= 64
    idx = indices
    inv_order = None
    if do_sort:
        x_exp, idx, inv_order = _gather_sort(x_exp, indices)

    h_shard = switch_linear_forward(switch_mlp.fc1, x_exp, idx, do_sort)
    h_shard = activation(h_shard)
    h_full = all_gather_last(h_shard, group)

    out_shard = switch_linear_forward(switch_mlp.fc2, h_full, idx, do_sort)
    out_full = all_gather_last(out_shard, group)

    if do_sort:
        assert inv_order is not None
        out_full = _scatter_unsort(out_full, inv_order, indices.shape)
    return out_full.squeeze(-2)


def mlp_n_sharded_sharded_out(
    mlp: nn.Module, x: mx.array, group: mx.distributed.Group
) -> mx.array:
    """SwiGLU or relu2 MLP with column-sharded down_proj. Returns (..., H/N)."""
    up = cast(nn.Linear, mlp.up_proj)
    dp = cast(nn.Linear, mlp.down_proj)

    x_up = linear_column_sharded_forward(up, x, group)
    if hasattr(mlp, "gate_proj"):
        gp = cast(nn.Linear, mlp.gate_proj)
        x_gate = linear_column_sharded_forward(gp, x, group)
        hidden_shard = nn.silu(x_gate) * x_up
    else:
        hidden_shard = nn.relu2(x_up)

    hidden_full = all_gather_last(hidden_shard, group)
    return linear_column_sharded_forward(dp, hidden_full, group)


def mlp_n_sharded(
    mlp: nn.Module, x: mx.array, group: mx.distributed.Group
) -> mx.array:
    out_shard = mlp_n_sharded_sharded_out(mlp, x, group)
    return all_gather_last(out_shard, group)
