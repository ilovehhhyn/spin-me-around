from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any

import yaml


class FineTuneMethod(StrEnum):
    SFT_1 = "sft_1"
    SFT_2 = "sft_2"
    ORACLE_SFT = "oracle_sft"
    REINFORCE_10 = "reinforce_10"
    GRPO = "grpo"
    GRPO_KL = "grpo_kl"


class FineTuneParameterization(StrEnum):
    FULL = "full"
    LORA = "lora"


class SchedulerName(StrEnum):
    CONSTANT_WITH_WARMUP = "constant_with_warmup"
    COSINE_WITH_WARMUP = "cosine_with_warmup"


@dataclass(frozen=True)
class DataConfig:
    root: Path = Path("data")
    pretrain_examples_per_task: int = 500
    fine_tune_examples: int | None = None
    train_batch_size: int = 128
    eval_batch_size: int = 512
    num_workers: int = 0

    def __post_init__(self) -> None:
        if self.pretrain_examples_per_task <= 0:
            raise ValueError("pretrain_examples_per_task must be positive")
        if self.fine_tune_examples is not None and self.fine_tune_examples <= 0:
            raise ValueError("fine_tune_examples must be positive when set")
        if self.fine_tune_examples is not None and self.fine_tune_examples % 10 != 0:
            raise ValueError("fine_tune_examples must be divisible by 10")
        if self.train_batch_size <= 0 or self.eval_batch_size <= 0:
            raise ValueError("batch sizes must be positive")
        if self.num_workers < 0:
            raise ValueError("num_workers cannot be negative")


@dataclass(frozen=True)
class ModelConfig:
    image_features: int = 28 * 28
    task_features: int = 1
    hidden_sizes: tuple[int, int] = (512, 256)
    output_classes: int = 10

    @property
    def input_features(self) -> int:
        return self.image_features + self.task_features


@dataclass(frozen=True)
class LoRAConfig:
    rank: int = 8
    alpha: float = 8.0
    dropout: float = 0.0
    target_modules: tuple[str, ...] = (
        "input_layer",
        "hidden_layer",
        "output_layer",
    )

    def __post_init__(self) -> None:
        if self.rank <= 0:
            raise ValueError("LoRA rank must be positive")
        if self.alpha <= 0.0:
            raise ValueError("LoRA alpha must be positive")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("LoRA dropout must be in [0, 1)")
        if not self.target_modules:
            raise ValueError("LoRA target_modules cannot be empty")
        if len(self.target_modules) != len(set(self.target_modules)):
            raise ValueError("LoRA target_modules cannot contain duplicates")


@dataclass(frozen=True)
class PretrainingConfig:
    epochs: int = 30
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    warmup_fraction: float = 0.1

    def __post_init__(self) -> None:
        _validate_optimization_values(
            epochs=self.epochs,
            learning_rate=self.learning_rate,
            warmup_fraction=self.warmup_fraction,
        )


@dataclass(frozen=True)
class SweepConfig:
    methods: tuple[FineTuneMethod, ...] = (FineTuneMethod.SFT_1,)
    parameterizations: tuple[FineTuneParameterization, ...] = (
        FineTuneParameterization.FULL,
    )
    full_learning_rates: tuple[float, ...] = (1e-4,)
    lora_learning_rates: tuple[float, ...] = (1e-3,)
    schedulers: tuple[SchedulerName, ...] = (SchedulerName.CONSTANT_WITH_WARMUP,)
    epochs: tuple[int, ...] = (1,)
    seeds: tuple[int, ...] = (17,)
    warmup_fraction: float = 0.1
    weight_decay: float = 0.0
    max_grad_norm: float = 1.0
    group_size: int = 16
    grpo_kl_coefficient: float = 0.1
    checkpoint_fractions: tuple[float, ...] = (0.25, 0.5, 1.0)

    def __post_init__(self) -> None:
        if not self.methods or not self.parameterizations or not self.schedulers:
            raise ValueError("methods, parameterizations, and schedulers cannot be empty")
        if not self.epochs or not self.seeds:
            raise ValueError("epochs and seeds cannot be empty")
        for parameterization in self.parameterizations:
            learning_rates = self.learning_rates_for(parameterization)
            if not learning_rates:
                raise ValueError(
                    f"learning rates for {parameterization.value} cannot be empty"
                )
            if any(value <= 0 for value in learning_rates):
                raise ValueError("all learning rates must be positive")
        if any(value <= 0 for value in self.epochs):
            raise ValueError("all epoch counts must be positive")
        if not 0.0 <= self.warmup_fraction < 1.0:
            raise ValueError("warmup_fraction must be in [0, 1)")
        if self.max_grad_norm <= 0.0:
            raise ValueError("max_grad_norm must be positive")
        if self.group_size < 2:
            raise ValueError("group_size must be at least two")
        if self.grpo_kl_coefficient < 0.0:
            raise ValueError("grpo_kl_coefficient cannot be negative")
        if any(not 0.0 < value <= 1.0 for value in self.checkpoint_fractions):
            raise ValueError("checkpoint fractions must be in (0, 1]")

    def learning_rates_for(
        self, parameterization: FineTuneParameterization
    ) -> tuple[float, ...]:
        if parameterization == FineTuneParameterization.FULL:
            return self.full_learning_rates
        if parameterization == FineTuneParameterization.LORA:
            return self.lora_learning_rates
        raise ValueError(f"Unsupported parameterization: {parameterization}")


@dataclass(frozen=True)
class EvaluationConfig:
    cka_max_examples: int = 1_024
    probe_max_train_examples: int = 500
    probe_ridge: float = 1e-2

    def __post_init__(self) -> None:
        if self.cka_max_examples <= 1:
            raise ValueError("cka_max_examples must exceed one")
        if self.probe_max_train_examples <= 1:
            raise ValueError("probe_max_train_examples must exceed one")
        if self.probe_ridge <= 0.0:
            raise ValueError("probe_ridge must be positive")


@dataclass(frozen=True)
class ExperimentConfig:
    output_dir: Path = Path("runs/week1")
    device: str = "auto"
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    lora: LoRAConfig = field(default_factory=LoRAConfig)
    pretraining: PretrainingConfig = field(default_factory=PretrainingConfig)
    sweep: SweepConfig = field(default_factory=SweepConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)

    @classmethod
    def from_yaml(cls, path: Path) -> ExperimentConfig:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise ValueError(f"Expected a YAML mapping in {path}")

        data_raw = raw.get("data", {})
        model_raw = raw.get("model", {})
        lora_raw = raw.get("lora", {})
        pretraining_raw = raw.get("pretraining", {})
        sweep_raw = raw.get("sweep", {})
        evaluation_raw = raw.get("evaluation", {})

        return cls(
            output_dir=Path(raw.get("output_dir", "runs/week1")),
            device=str(raw.get("device", "auto")),
            data=DataConfig(
                root=Path(data_raw.get("root", "data")),
                pretrain_examples_per_task=int(
                    data_raw.get("pretrain_examples_per_task", 500)
                ),
                fine_tune_examples=(
                    int(data_raw["fine_tune_examples"])
                    if data_raw.get("fine_tune_examples") is not None
                    else None
                ),
                train_batch_size=int(data_raw.get("train_batch_size", 128)),
                eval_batch_size=int(data_raw.get("eval_batch_size", 512)),
                num_workers=int(data_raw.get("num_workers", 0)),
            ),
            model=ModelConfig(
                image_features=int(model_raw.get("image_features", 28 * 28)),
                task_features=int(model_raw.get("task_features", 1)),
                hidden_sizes=tuple(model_raw.get("hidden_sizes", [512, 256])),
                output_classes=int(model_raw.get("output_classes", 10)),
            ),
            lora=LoRAConfig(
                rank=int(lora_raw.get("rank", 8)),
                alpha=float(lora_raw.get("alpha", 8.0)),
                dropout=float(lora_raw.get("dropout", 0.0)),
                target_modules=tuple(
                    str(value)
                    for value in lora_raw.get(
                        "target_modules",
                        ["input_layer", "hidden_layer", "output_layer"],
                    )
                ),
            ),
            pretraining=PretrainingConfig(
                epochs=int(pretraining_raw.get("epochs", 30)),
                learning_rate=float(pretraining_raw.get("learning_rate", 1e-3)),
                weight_decay=float(pretraining_raw.get("weight_decay", 0.0)),
                warmup_fraction=float(pretraining_raw.get("warmup_fraction", 0.1)),
            ),
            sweep=SweepConfig(
                methods=tuple(
                    FineTuneMethod(value) for value in sweep_raw.get("methods", ["sft_1"])
                ),
                parameterizations=tuple(
                    FineTuneParameterization(value)
                    for value in sweep_raw.get("parameterizations", ["full"])
                ),
                full_learning_rates=tuple(
                    float(value)
                    for value in sweep_raw.get("learning_rates", {}).get("full", [1e-4])
                ),
                lora_learning_rates=tuple(
                    float(value)
                    for value in sweep_raw.get("learning_rates", {}).get("lora", [1e-3])
                ),
                schedulers=tuple(
                    SchedulerName(value)
                    for value in sweep_raw.get("schedulers", ["constant_with_warmup"])
                ),
                epochs=tuple(int(value) for value in sweep_raw.get("epochs", [1])),
                seeds=tuple(int(value) for value in sweep_raw.get("seeds", [17])),
                warmup_fraction=float(sweep_raw.get("warmup_fraction", 0.1)),
                weight_decay=float(sweep_raw.get("weight_decay", 0.0)),
                max_grad_norm=float(sweep_raw.get("max_grad_norm", 1.0)),
                group_size=int(sweep_raw.get("group_size", 16)),
                grpo_kl_coefficient=float(sweep_raw.get("grpo_kl_coefficient", 0.1)),
                checkpoint_fractions=tuple(
                    float(value)
                    for value in sweep_raw.get("checkpoint_fractions", [0.25, 0.5, 1.0])
                ),
            ),
            evaluation=EvaluationConfig(
                cka_max_examples=int(evaluation_raw.get("cka_max_examples", 1_024)),
                probe_max_train_examples=int(
                    evaluation_raw.get("probe_max_train_examples", 500)
                ),
                probe_ridge=float(evaluation_raw.get("probe_ridge", 1e-2)),
            ),
        )


@dataclass(frozen=True)
class RunSpec:
    method: FineTuneMethod
    parameterization: FineTuneParameterization
    learning_rate: float
    scheduler: SchedulerName
    epochs: int
    seed: int
    lora_rank: int | None = None

    def __post_init__(self) -> None:
        if self.parameterization == FineTuneParameterization.LORA:
            if self.lora_rank is None or self.lora_rank <= 0:
                raise ValueError("A positive lora_rank is required for LoRA runs")
        elif self.lora_rank is not None:
            raise ValueError("lora_rank must be omitted for full fine-tuning runs")

    @property
    def identifier(self) -> str:
        learning_rate = f"{self.learning_rate:.3e}".replace("+", "")
        parameterization = self.parameterization.value
        if self.lora_rank is not None:
            parameterization = f"{parameterization}_r{self.lora_rank}"
        return (
            f"method={self.method.value}__parameterization={parameterization}"
            f"__lr={learning_rate}__scheduler={self.scheduler.value}"
            f"__epochs={self.epochs}__seed={self.seed}"
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "method": self.method.value,
            "parameterization": self.parameterization.value,
            "lora_rank": self.lora_rank,
            "learning_rate": self.learning_rate,
            "scheduler": self.scheduler.value,
            "epochs": self.epochs,
            "seed": self.seed,
        }


def _validate_optimization_values(
    *, epochs: int, learning_rate: float, warmup_fraction: float
) -> None:
    if epochs <= 0:
        raise ValueError("epochs must be positive")
    if learning_rate <= 0.0:
        raise ValueError("learning_rate must be positive")
    if not 0.0 <= warmup_fraction < 1.0:
        raise ValueError("warmup_fraction must be in [0, 1)")
