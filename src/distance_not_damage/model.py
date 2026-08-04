from __future__ import annotations

import math
from collections.abc import Iterator
from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from distance_not_damage.config import (
    FineTuneParameterization,
    LoRAConfig,
    ModelConfig,
)


@dataclass(frozen=True)
class ModelOutput:
    logits: Tensor
    hidden: Tensor


class LoRALinear(nn.Module):
    """A frozen linear layer plus a trainable low-rank update.

    The effective weight is ``W_0 + (alpha / rank) * B @ A``. ``B`` starts at
    zero, so replacing a pretrained ``nn.Linear`` with this module leaves the
    model's function exactly unchanged before the first optimizer step.
    """

    def __init__(
        self,
        base_layer: nn.Linear,
        *,
        rank: int,
        alpha: float,
        dropout: float,
    ) -> None:
        super().__init__()
        if rank <= 0:
            raise ValueError("rank must be positive")
        if rank > min(base_layer.in_features, base_layer.out_features):
            raise ValueError(
                "rank cannot exceed the smaller input/output dimension: "
                f"rank={rank}, shape=({base_layer.out_features}, {base_layer.in_features})"
            )
        if alpha <= 0.0:
            raise ValueError("alpha must be positive")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")

        self.base_layer = base_layer
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        self.dropout = nn.Dropout(dropout)
        parameter_options = {
            "device": base_layer.weight.device,
            "dtype": base_layer.weight.dtype,
        }
        self.lora_a = nn.Parameter(
            torch.empty(rank, base_layer.in_features, **parameter_options)
        )
        self.lora_b = nn.Parameter(
            torch.zeros(base_layer.out_features, rank, **parameter_options)
        )
        self.adapter_enabled = True

        nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))
        for parameter in self.base_layer.parameters():
            parameter.requires_grad_(False)

    def forward(self, inputs: Tensor) -> Tensor:
        output = self.base_layer(inputs)
        if not self.adapter_enabled:
            return output
        low_rank_hidden = F.linear(self.dropout(inputs), self.lora_a)
        return output + self.scaling * F.linear(low_rank_hidden, self.lora_b)

    def effective_weight(self) -> Tensor:
        """Return the deployed weight without mutating or merging the adapter."""

        if not self.adapter_enabled:
            return self.base_layer.weight
        return self.base_layer.weight + self.scaling * (self.lora_b @ self.lora_a)

    @property
    def effective_bias(self) -> Tensor | None:
        return self.base_layer.bias


class TaskConditionedMLP(nn.Module):
    """Three-layer classifier used by the ParityMNIST reproduction.

    The final input feature is a task indicator: +1 for ParityMNIST and -1 for
    FashionMNIST. Returning the penultimate activation makes representation
    comparisons explicit and avoids hooks with hidden mutable state.
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        first_hidden, second_hidden = config.hidden_sizes
        self.input_layer = nn.Linear(config.input_features, first_hidden)
        self.hidden_layer = nn.Linear(first_hidden, second_hidden)
        self.output_layer = nn.Linear(second_hidden, config.output_classes)
        self.activation = nn.ReLU()

    def forward(self, images: Tensor, task_indicator: Tensor) -> ModelOutput:
        flat_images = images.flatten(start_dim=1)
        indicator = task_indicator.reshape(-1, 1).to(dtype=flat_images.dtype)
        features = torch.cat((flat_images, indicator), dim=1)
        hidden = self.activation(self.input_layer(features))
        hidden = self.activation(self.hidden_layer(hidden))
        return ModelOutput(logits=self.output_layer(hidden), hidden=hidden)


def configure_for_fine_tuning(
    model: TaskConditionedMLP,
    *,
    parameterization: FineTuneParameterization,
    lora: LoRAConfig,
) -> None:
    """Configure a pretrained model in place for full fine-tuning or LoRA."""

    if parameterization == FineTuneParameterization.FULL:
        for parameter in model.parameters():
            parameter.requires_grad_(True)
        return
    if parameterization != FineTuneParameterization.LORA:
        raise ValueError(f"Unsupported parameterization: {parameterization}")

    available_layers = {
        name: module for name, module in model.named_modules() if isinstance(module, nn.Linear)
    }
    unknown_targets = set(lora.target_modules) - available_layers.keys()
    if unknown_targets:
        unknown = ", ".join(sorted(unknown_targets))
        raise ValueError(f"Unknown LoRA target modules: {unknown}")

    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for module_name in lora.target_modules:
        base_layer = available_layers[module_name]
        _replace_module(
            model,
            module_name,
            LoRALinear(
                base_layer,
                rank=lora.rank,
                alpha=lora.alpha,
                dropout=lora.dropout,
            ),
        )


def trainable_parameters(model: nn.Module) -> list[nn.Parameter]:
    """Return optimizer parameters and fail clearly if none are trainable."""

    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not parameters:
        raise ValueError("Model has no trainable parameters")
    return parameters


def parameter_counts(model: nn.Module) -> dict[str, int]:
    return {
        "trainable_parameter_count": sum(
            parameter.numel() for parameter in model.parameters() if parameter.requires_grad
        ),
        "total_parameter_count": sum(parameter.numel() for parameter in model.parameters()),
    }


def effective_named_tensors(model: TaskConditionedMLP) -> dict[str, Tensor]:
    """Expose logically equivalent weights for full and LoRA parameterizations."""

    tensors: dict[str, Tensor] = {}
    for layer_name, layer in _logical_linear_layers(model):
        if isinstance(layer, LoRALinear):
            tensors[f"{layer_name}.weight"] = layer.effective_weight()
            if layer.effective_bias is not None:
                tensors[f"{layer_name}.bias"] = layer.effective_bias
        else:
            tensors[f"{layer_name}.weight"] = layer.weight
            if layer.bias is not None:
                tensors[f"{layer_name}.bias"] = layer.bias
    return tensors


def set_lora_adapters_enabled(model: nn.Module, *, enabled: bool) -> None:
    for module in model.modules():
        if isinstance(module, LoRALinear):
            module.adapter_enabled = enabled


def _logical_linear_layers(
    model: TaskConditionedMLP,
) -> Iterator[tuple[str, nn.Linear | LoRALinear]]:
    for layer_name in ("input_layer", "hidden_layer", "output_layer"):
        layer = getattr(model, layer_name)
        if not isinstance(layer, nn.Linear | LoRALinear):
            raise TypeError(f"{layer_name} is not a supported linear layer")
        yield layer_name, layer


def _replace_module(root: nn.Module, qualified_name: str, replacement: nn.Module) -> None:
    components = qualified_name.split(".")
    parent = root
    for component in components[:-1]:
        parent = getattr(parent, component)
    setattr(parent, components[-1], replacement)
