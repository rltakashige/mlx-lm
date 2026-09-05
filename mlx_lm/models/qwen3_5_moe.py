# Copyright © 2026 Apple Inc.

from dataclasses import dataclass

from .base import BaseModelArgs
from .qwen3_5 import Model as Qwen3_5Model


@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str
    text_config: dict

    @classmethod
    def from_dict(cls, params):
        if "text_config" not in params:
            return cls(model_type=params["model_type"], text_config=params)
        return super().from_dict(params)


def split_experts(weights):
    """Split the stacked HF ``experts`` tensors into the switch_mlp projections."""
    for key in [k for k in weights if k.endswith(".experts.gate_up_proj")]:
        prefix = key[: -len("experts.gate_up_proj")]
        gate_up = weights.pop(key)
        mid = gate_up.shape[-2] // 2
        weights[f"{prefix}switch_mlp.gate_proj.weight"] = gate_up[..., :mid, :]
        weights[f"{prefix}switch_mlp.up_proj.weight"] = gate_up[..., mid:, :]
        weights[f"{prefix}switch_mlp.down_proj.weight"] = weights.pop(
            f"{prefix}experts.down_proj"
        )
    return weights


class Model(Qwen3_5Model):

    def sanitize(self, weights):
        new_weights = {}
        for key, value in weights.items():
            if key.startswith("vision_tower") or key.startswith("model.visual"):
                continue
            if key.startswith("model.language_model"):
                key = key.replace("model.language_model", "language_model.model")
            elif key.startswith("language_model."):
                pass
            else:
                key = "language_model." + key
            new_weights[key] = value

        return self.language_model.sanitize(split_experts(new_weights))
