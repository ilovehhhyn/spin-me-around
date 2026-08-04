from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor
from torch.nn import functional as F
from torch.utils.data import DataLoader

from distance_not_damage.model import TaskConditionedMLP, effective_named_tensors

Batch = tuple[Tensor, Tensor, Tensor]


@dataclass(frozen=True)
class LinearProbe:
    """Closed-form ridge probe fitted on frozen hidden representations."""

    weights: Tensor
    feature_mean: Tensor
    feature_scale: Tensor


@torch.inference_mode()
def classification_metrics(
    model: TaskConditionedMLP,
    loader: DataLoader[Batch],
    *,
    device: torch.device,
    parity_task: bool,
) -> dict[str, float]:
    model.eval()
    correct = 0
    total = 0
    loss_total = 0.0
    for images, targets, indicators in loader:
        images = images.to(device)
        targets = targets.to(device)
        indicators = indicators.to(device)
        logits = model(images, indicators).logits
        predictions = logits.argmax(dim=-1)
        if parity_task:
            batch_correct = predictions.remainder(2) == targets.remainder(2)
        else:
            batch_correct = predictions == targets
        correct += int(batch_correct.sum())
        total += targets.numel()
        loss_total += float(F.cross_entropy(logits, targets, reduction="sum"))
    cross_entropy = loss_total / total
    return {
        "accuracy": correct / total,
        "cross_entropy": cross_entropy,
        "code_length_bits_per_example": cross_entropy / math.log(2.0),
    }


@torch.inference_mode()
def policy_shift_metrics(
    *,
    base_model: TaskConditionedMLP,
    current_model: TaskConditionedMLP,
    loader: DataLoader[Batch],
    device: torch.device,
) -> dict[str, float]:
    base_model.eval()
    current_model.eval()
    forward_kl_total = 0.0
    reverse_kl_total = 0.0
    example_count = 0
    for images, _, indicators in loader:
        images = images.to(device)
        indicators = indicators.to(device)
        base_logits = base_model(images, indicators).logits
        current_logits = current_model(images, indicators).logits
        base_log_probabilities = base_logits.log_softmax(dim=-1)
        current_log_probabilities = current_logits.log_softmax(dim=-1)
        base_probabilities = base_log_probabilities.exp()
        current_probabilities = current_log_probabilities.exp()
        forward = (base_probabilities * (base_log_probabilities - current_log_probabilities)).sum(
            dim=-1
        )
        reverse = (
            current_probabilities * (current_log_probabilities - base_log_probabilities)
        ).sum(dim=-1)
        forward_kl_total += float(forward.sum())
        reverse_kl_total += float(reverse.sum())
        example_count += images.shape[0]
    return {
        "forward_kl": forward_kl_total / example_count,
        "reverse_kl": reverse_kl_total / example_count,
    }


@torch.inference_mode()
def representation_cka(
    *,
    base_model: TaskConditionedMLP,
    current_model: TaskConditionedMLP,
    loader: DataLoader[Batch],
    device: torch.device,
    max_examples: int,
) -> float:
    base_hidden = _collect_hidden(base_model, loader, device=device, max_examples=max_examples)
    current_hidden = _collect_hidden(
        current_model, loader, device=device, max_examples=max_examples
    )
    return float(linear_cka(base_hidden, current_hidden))


def linear_cka(left: Tensor, right: Tensor) -> Tensor:
    """Linear centered-kernel alignment without materializing n-by-n kernels."""

    if left.shape[0] != right.shape[0]:
        raise ValueError("Representations must contain the same examples")
    left = left - left.mean(dim=0, keepdim=True)
    right = right - right.mean(dim=0, keepdim=True)
    cross_covariance = left.T @ right
    left_covariance = left.T @ left
    right_covariance = right.T @ right
    numerator = cross_covariance.square().sum()
    denominator = torch.sqrt(
        left_covariance.square().sum() * right_covariance.square().sum()
    ).clamp_min(torch.finfo(left.dtype).eps)
    return numerator / denominator


def parameter_distance(
    base_model: TaskConditionedMLP, current_model: TaskConditionedMLP
) -> dict[str, float]:
    squared_l2 = 0.0
    l1 = 0.0
    base_tensors = effective_named_tensors(base_model)
    current_tensors = effective_named_tensors(current_model)
    if base_tensors.keys() != current_tensors.keys():
        raise ValueError("Models do not expose identical effective parameter names")
    for name, base_tensor in base_tensors.items():
        difference = current_tensors[name].detach() - base_tensor.detach()
        squared_l2 += float(difference.square().sum())
        l1 += float(difference.abs().sum())
    return {"parameter_l2": squared_l2**0.5, "parameter_l1": l1}


@torch.inference_mode()
def fit_linear_probe(
    model: TaskConditionedMLP,
    loader: DataLoader[Batch],
    *,
    device: torch.device,
    max_examples: int,
    class_count: int,
    ridge: float,
) -> LinearProbe:
    """Fit a deterministic multiclass ridge probe without updating the model."""

    if ridge <= 0.0:
        raise ValueError("ridge must be positive")
    hidden, targets = _collect_hidden_and_targets(
        model,
        loader,
        device=device,
        max_examples=max_examples,
    )
    hidden = hidden.to(dtype=torch.float64)
    feature_mean = hidden.mean(dim=0)
    feature_scale = hidden.std(dim=0, unbiased=False).clamp_min(1e-8)
    standardized = (hidden - feature_mean) / feature_scale
    design = torch.cat(
        (standardized, torch.ones(standardized.shape[0], 1, dtype=standardized.dtype)),
        dim=1,
    )
    target_matrix = F.one_hot(targets, num_classes=class_count).to(dtype=design.dtype)
    penalty = torch.eye(design.shape[1], dtype=design.dtype)
    penalty[-1, -1] = 0.0  # Do not regularize the intercept.
    weights = torch.linalg.solve(
        design.T @ design + ridge * penalty,
        design.T @ target_matrix,
    )
    return LinearProbe(
        weights=weights,
        feature_mean=feature_mean,
        feature_scale=feature_scale,
    )


@torch.inference_mode()
def linear_probe_accuracy(
    probe: LinearProbe,
    model: TaskConditionedMLP,
    loader: DataLoader[Batch],
    *,
    device: torch.device,
) -> float:
    hidden, targets = _collect_hidden_and_targets(
        model,
        loader,
        device=device,
        max_examples=len(loader.dataset),
    )
    hidden = hidden.to(dtype=probe.weights.dtype)
    standardized = (hidden - probe.feature_mean) / probe.feature_scale
    design = torch.cat(
        (standardized, torch.ones(standardized.shape[0], 1, dtype=standardized.dtype)),
        dim=1,
    )
    predictions = (design @ probe.weights).argmax(dim=-1)
    return float((predictions == targets).to(dtype=torch.float64).mean())


@torch.inference_mode()
def _collect_hidden(
    model: TaskConditionedMLP,
    loader: DataLoader[Batch],
    *,
    device: torch.device,
    max_examples: int,
) -> Tensor:
    hidden, _ = _collect_hidden_and_targets(
        model,
        loader,
        device=device,
        max_examples=max_examples,
    )
    return hidden


@torch.inference_mode()
def _collect_hidden_and_targets(
    model: TaskConditionedMLP,
    loader: DataLoader[Batch],
    *,
    device: torch.device,
    max_examples: int,
) -> tuple[Tensor, Tensor]:
    model.eval()
    chunks: list[Tensor] = []
    target_chunks: list[Tensor] = []
    collected = 0
    for images, targets, indicators in loader:
        if collected >= max_examples:
            break
        remaining = max_examples - collected
        images = images[:remaining].to(device)
        indicators = indicators[:remaining].to(device)
        hidden = model(images, indicators).hidden.detach().cpu()
        chunks.append(hidden)
        target_chunks.append(targets[:remaining].detach().cpu())
        collected += hidden.shape[0]
    return torch.cat(chunks, dim=0), torch.cat(target_chunks, dim=0)
