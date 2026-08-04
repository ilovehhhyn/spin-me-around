from __future__ import annotations

import math
from collections.abc import Callable

import torch
from torch import Tensor
from torch.nn import functional as F
from torch.optim import AdamW, Optimizer
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

from distance_not_damage.config import (
    PretrainingConfig,
    RunSpec,
    SchedulerName,
    SweepConfig,
)
from distance_not_damage.model import TaskConditionedMLP, trainable_parameters
from distance_not_damage.objectives import fine_tuning_loss

Batch = tuple[Tensor, Tensor, Tensor]
CheckpointCallback = Callable[[int, int, dict[str, float]], None]


def pretrain_model(
    *,
    model: TaskConditionedMLP,
    loader: DataLoader[Batch],
    config: PretrainingConfig,
    device: torch.device,
) -> None:
    model.train()
    optimizer = AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    total_steps = config.epochs * len(loader)
    scheduler = build_scheduler(
        optimizer,
        name=SchedulerName.COSINE_WITH_WARMUP,
        total_steps=total_steps,
        warmup_fraction=config.warmup_fraction,
    )
    for _ in range(config.epochs):
        for images, targets, indicators in loader:
            images = images.to(device)
            targets = targets.to(device)
            indicators = indicators.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(images, indicators).logits
            loss = F.cross_entropy(logits, targets)
            loss.backward()
            optimizer.step()
            scheduler.step()


def fine_tune_model(
    *,
    model: TaskConditionedMLP,
    base_model: TaskConditionedMLP,
    loader: DataLoader[Batch],
    spec: RunSpec,
    sweep_config: SweepConfig,
    device: torch.device,
    on_checkpoint: CheckpointCallback,
) -> None:
    model.train()
    base_model.eval()
    for parameter in base_model.parameters():
        parameter.requires_grad_(False)

    optimizer_parameters = trainable_parameters(model)
    optimizer = AdamW(
        optimizer_parameters,
        lr=spec.learning_rate,
        weight_decay=sweep_config.weight_decay,
    )
    total_steps = spec.epochs * len(loader)
    scheduler = build_scheduler(
        optimizer,
        name=spec.scheduler,
        total_steps=total_steps,
        warmup_fraction=sweep_config.warmup_fraction,
    )
    checkpoint_steps = build_checkpoint_steps(
        total_steps=total_steps,
        sweep_config=sweep_config,
    )

    step = 0
    diagnostics_accumulator: dict[str, float] = {}
    diagnostics_count = 0
    for epoch_index in range(spec.epochs):
        for images, digits, indicators in loader:
            step += 1
            images = images.to(device)
            digits = digits.to(device)
            indicators = indicators.to(device)

            optimizer.zero_grad(set_to_none=True)
            current_logits = model(images, indicators).logits
            with torch.no_grad():
                base_logits = base_model(images, indicators).logits
            loss, diagnostics = fine_tuning_loss(
                method=spec.method,
                current_logits=current_logits,
                base_logits=base_logits,
                digits=digits,
                group_size=sweep_config.group_size,
                grpo_kl_coefficient=sweep_config.grpo_kl_coefficient,
            )
            loss.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                optimizer_parameters, sweep_config.max_grad_norm
            )
            optimizer.step()
            scheduler.step()

            batch_diagnostics = {
                **diagnostics,
                "loss": float(loss.detach()),
                "gradient_norm": float(gradient_norm.detach()),
                "learning_rate": scheduler.get_last_lr()[0],
            }
            for name, value in batch_diagnostics.items():
                diagnostics_accumulator[name] = diagnostics_accumulator.get(name, 0.0) + value
            diagnostics_count += 1

            if step in checkpoint_steps:
                averaged = {
                    name: value / diagnostics_count
                    for name, value in diagnostics_accumulator.items()
                }
                on_checkpoint(step, epoch_index + 1, averaged)
                diagnostics_accumulator.clear()
                diagnostics_count = 0
                model.train()


def build_checkpoint_steps(*, total_steps: int, sweep_config: SweepConfig) -> set[int]:
    """Combine dense logarithmic early steps with run-relative checkpoints."""

    if total_steps <= 0:
        raise ValueError("total_steps must be positive")
    checkpoint_steps = {
        max(1, math.ceil(fraction * total_steps)) for fraction in sweep_config.checkpoint_fractions
    }
    checkpoint_steps.update(
        step for step in sweep_config.early_checkpoint_steps if step <= total_steps
    )
    checkpoint_steps.add(total_steps)
    return checkpoint_steps


def build_scheduler(
    optimizer: Optimizer,
    *,
    name: SchedulerName,
    total_steps: int,
    warmup_fraction: float,
) -> LambdaLR:
    if total_steps <= 0:
        raise ValueError("total_steps must be positive")
    warmup_steps = int(total_steps * warmup_fraction)

    def multiplier(step_index: int) -> float:
        completed_steps = step_index + 1
        if warmup_steps > 0 and completed_steps <= warmup_steps:
            return completed_steps / warmup_steps
        if name == SchedulerName.CONSTANT_WITH_WARMUP:
            return 1.0
        if name == SchedulerName.COSINE_WITH_WARMUP:
            decay_steps = max(1, total_steps - warmup_steps)
            progress = min(1.0, (completed_steps - warmup_steps) / decay_steps)
            return 0.5 * (1.0 + math.cos(math.pi * progress))
        raise ValueError(f"Unknown scheduler: {name}")

    return LambdaLR(optimizer, lr_lambda=multiplier)
