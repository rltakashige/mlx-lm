# Copyright © 2026 Apple Inc.

from dataclasses import dataclass
from typing import Any, Optional, Union

import mlx.core as mx
import mlx.nn as nn

from . import kimi_linear
from .base import BaseModelArgs


@dataclass
class ModelArgs(BaseModelArgs):
    text_config: Union[kimi_linear.ModelArgs, dict]
    model_type: str = "kimi_k3"

    def __post_init__(self):
        if isinstance(self.text_config, dict):
            self.text_config = kimi_linear.ModelArgs.from_dict(self.text_config)
        if not self.text_config.mla_use_nope:
            raise ValueError("Kimi-K3 requires mla_use_nope=True.")


class Model(nn.Module):
    """Text-generation view of Kimi K3.

    MLX-LM does not consume K3's vision tower here, but retains the checkpoint's
    ``language_model`` namespace so its text weights load without duplication.
    """

    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.language_model = kimi_linear.Model(args.text_config)

    def __call__(
        self,
        inputs: mx.array,
        cache: Optional[Any] = None,
    ) -> mx.array:
        return self.language_model(inputs, cache)

    def sanitize(self, weights):
        language_weights = {}
        for key, value in weights.items():
            if key.startswith("language_model."):
                language_weights[key.removeprefix("language_model.")] = value
            elif key.startswith(("model.", "lm_head.")):
                language_weights[key] = value

        language_weights = self.language_model.sanitize(language_weights)
        return {
            f"language_model.{key}": value for key, value in language_weights.items()
        }

    @property
    def model(self):
        return self.language_model.model

    @property
    def layers(self):
        return self.language_model.layers

    def make_cache(self):
        return self.language_model.make_cache()

    @property
    def quant_predicate(self):
        return self.language_model.quant_predicate

    @property
    def cast_predicate(self):
        return self.language_model.cast_predicate
