from __future__ import annotations

import hashlib
import json
import os
import random
from dataclasses import asdict
from itertools import product
from pathlib import Path
from typing import Any

import numpy as np
import torch

from distance_not_damage.config import (
    ExperimentConfig,
    FineTuneParameterization,
    RunSpec,
)
from distance_not_damage.data import DataBundle, prepare_data
from distance_not_damage.metrics import (
    LinearProbe,
    classification_metrics,
    fit_linear_probe,
    linear_probe_accuracy,
    parameter_distance,
    policy_shift_metrics,
    representation_cka,
)
from distance_not_damage.model import (
    TaskConditionedMLP,
    configure_for_fine_tuning,
    parameter_counts,
)
from distance_not_damage.provenance import capture_run_provenance
from distance_not_damage.training import fine_tune_model, pretrain_model


class ExperimentRunner:
    """Own datasets, cached base checkpoints, and single-writer result creation."""

    def __init__(self, config: ExperimentConfig) -> None:
        self.config = config
        self.device = resolve_device(config.device)
        self._data_by_seed: dict[int, DataBundle] = {}
        self._base_state_by_seed: dict[int, dict[str, torch.Tensor]] = {}
        self._base_metrics_by_seed: dict[int, dict[str, float]] = {}
        self._fixed_probe_by_seed: dict[int, LinearProbe] = {}

    def run(self, spec: RunSpec) -> dict[str, Any]:
        if (
            spec.parameterization == FineTuneParameterization.LORA
            and spec.lora_rank != self.config.lora.rank
        ):
            raise ValueError("RunSpec lora_rank must match the experiment LoRA configuration")
        run_directory = self.config.output_dir / spec.identifier
        completion_path = run_directory / "summary.json"
        if completion_path.exists():
            return json.loads(completion_path.read_text(encoding="utf-8"))
        if run_directory.exists():
            raise FileExistsError(
                f"Incomplete run directory already exists: {run_directory}. "
                "Inspect it and move it aside before retrying."
            )
        run_directory.mkdir(parents=True)
        provenance = capture_run_provenance(Path.cwd())
        _atomic_write_json(self.config.as_dict(), run_directory / "resolved_config.json")
        _atomic_write_json(provenance.as_dict(), run_directory / "provenance.json")

        seed_everything(spec.seed)
        data = self._data(spec.seed)
        base_state, base_metrics = self._base(spec.seed, data)
        base_model = TaskConditionedMLP(self.config.model).to(self.device)
        base_model.load_state_dict(base_state)
        current_model = TaskConditionedMLP(self.config.model).to(self.device)
        current_model.load_state_dict(base_state)
        configure_for_fine_tuning(
            current_model,
            parameterization=spec.parameterization,
            lora=self.config.lora,
        )
        fixed_probe = self._fixed_probe(spec.seed, base_model, data)
        counts = parameter_counts(current_model)
        initial_function_error = self._maximum_logit_difference(
            base_model,
            current_model,
            data.parity_eval_loader,
        )

        records = [
            self._evaluate(
                spec=spec,
                step=0,
                epoch=0,
                base_model=base_model,
                current_model=current_model,
                data=data,
                base_metrics=base_metrics,
                fixed_probe=fixed_probe,
                parameter_count_metrics=counts,
                training_metrics={},
            )
        ]

        def on_checkpoint(step: int, epoch: int, training_metrics: dict[str, float]) -> None:
            records.append(
                self._evaluate(
                    spec=spec,
                    step=step,
                    epoch=epoch,
                    base_model=base_model,
                    current_model=current_model,
                    data=data,
                    base_metrics=base_metrics,
                    fixed_probe=fixed_probe,
                    parameter_count_metrics=counts,
                    training_metrics=training_metrics,
                )
            )

        fine_tune_model(
            model=current_model,
            base_model=base_model,
            loader=data.parity_train_loader,
            spec=spec,
            sweep_config=self.config.sweep,
            device=self.device,
            on_checkpoint=on_checkpoint,
        )

        _atomic_torch_save(current_model.state_dict(), run_directory / "final_model.pt")
        _atomic_write_text(
            "".join(json.dumps(record, sort_keys=True) + "\n" for record in records),
            run_directory / "metrics.jsonl",
        )
        summary: dict[str, Any] = {
            "schema_version": provenance.schema_version,
            "run": spec.as_dict(),
            "device": str(self.device),
            "provenance": provenance.as_dict(),
            "base_metrics": base_metrics,
            "lora": asdict(self.config.lora),
            "initial_function_max_abs_error": initial_function_error,
            **counts,
            "final_metrics": records[-1],
            "record_count": len(records),
        }
        _atomic_write_json(summary, completion_path)
        return summary

    def sweep_specs(self) -> list[RunSpec]:
        sweep = self.config.sweep
        run_settings = product(
            sweep.methods,
            sweep.parameterizations,
            sweep.schedulers,
            sweep.epochs,
            sweep.seeds,
        )
        specs: list[RunSpec] = []
        for method, parameterization, scheduler, epochs, seed in run_settings:
            for learning_rate in sweep.learning_rates_for(parameterization):
                specs.append(
                    RunSpec(
                        method=method,
                        parameterization=parameterization,
                        learning_rate=learning_rate,
                        scheduler=scheduler,
                        epochs=epochs,
                        seed=seed,
                        lora_rank=(
                            self.config.lora.rank
                            if parameterization == FineTuneParameterization.LORA
                            else None
                        ),
                    )
                )
        return specs

    def _data(self, seed: int) -> DataBundle:
        if seed not in self._data_by_seed:
            self._data_by_seed[seed] = prepare_data(self.config.data, seed)
        return self._data_by_seed[seed]

    def _base(
        self, seed: int, data: DataBundle
    ) -> tuple[dict[str, torch.Tensor], dict[str, float]]:
        if seed in self._base_state_by_seed:
            return self._base_state_by_seed[seed], self._base_metrics_by_seed[seed]

        cache_directory = self.config.output_dir / "base_checkpoints"
        cache_directory.mkdir(parents=True, exist_ok=True)
        fingerprint = self._pretraining_fingerprint(seed)
        checkpoint_path = cache_directory / f"base__seed={seed}__{fingerprint}.pt"
        metrics_path = checkpoint_path.with_suffix(".json")
        model = TaskConditionedMLP(self.config.model).to(self.device)
        if checkpoint_path.exists():
            state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
            model.load_state_dict(state)
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        else:
            seed_everything(seed)
            pretrain_model(
                model=model,
                loader=data.pretrain_loader,
                config=self.config.pretraining,
                device=self.device,
            )
            metrics = self._base_evaluation(model, data)
            state = {name: value.detach().cpu() for name, value in model.state_dict().items()}
            _atomic_torch_save(state, checkpoint_path)
            _atomic_write_json(metrics, metrics_path)

        self._base_state_by_seed[seed] = state
        self._base_metrics_by_seed[seed] = metrics
        return state, metrics

    def _fixed_probe(
        self,
        seed: int,
        base_model: TaskConditionedMLP,
        data: DataBundle,
    ) -> LinearProbe:
        if seed not in self._fixed_probe_by_seed:
            self._fixed_probe_by_seed[seed] = fit_linear_probe(
                base_model,
                data.fashion_probe_train_loader,
                device=self.device,
                max_examples=self.config.evaluation.probe_max_train_examples,
                class_count=self.config.model.output_classes,
                ridge=self.config.evaluation.probe_ridge,
            )
        return self._fixed_probe_by_seed[seed]

    def _base_evaluation(self, model: TaskConditionedMLP, data: DataBundle) -> dict[str, float]:
        parity = classification_metrics(
            model, data.parity_eval_loader, device=self.device, parity_task=True
        )
        fashion = classification_metrics(
            model, data.fashion_eval_loader, device=self.device, parity_task=False
        )
        return {
            "parity_accuracy": parity["accuracy"],
            "fashion_accuracy": fashion["accuracy"],
            "parity_cross_entropy": parity["cross_entropy"],
            "fashion_cross_entropy": fashion["cross_entropy"],
            "parity_code_length_bits_per_example": parity["code_length_bits_per_example"],
            "fashion_code_length_bits_per_example": fashion["code_length_bits_per_example"],
        }

    def _evaluate(
        self,
        *,
        spec: RunSpec,
        step: int,
        epoch: int,
        base_model: TaskConditionedMLP,
        current_model: TaskConditionedMLP,
        data: DataBundle,
        base_metrics: dict[str, float],
        fixed_probe: LinearProbe,
        parameter_count_metrics: dict[str, int],
        training_metrics: dict[str, float],
    ) -> dict[str, Any]:
        parity = classification_metrics(
            current_model,
            data.parity_eval_loader,
            device=self.device,
            parity_task=True,
        )
        fashion = classification_metrics(
            current_model,
            data.fashion_eval_loader,
            device=self.device,
            parity_task=False,
        )
        shift = policy_shift_metrics(
            base_model=base_model,
            current_model=current_model,
            loader=data.parity_eval_loader,
            device=self.device,
        )
        distances = parameter_distance(base_model, current_model)
        cka = representation_cka(
            base_model=base_model,
            current_model=current_model,
            loader=data.fashion_eval_loader,
            device=self.device,
            max_examples=self.config.evaluation.cka_max_examples,
        )
        fresh_probe = fit_linear_probe(
            current_model,
            data.fashion_probe_train_loader,
            device=self.device,
            max_examples=self.config.evaluation.probe_max_train_examples,
            class_count=self.config.model.output_classes,
            ridge=self.config.evaluation.probe_ridge,
        )
        fixed_probe_accuracy = linear_probe_accuracy(
            fixed_probe,
            current_model,
            data.fashion_eval_loader,
            device=self.device,
        )
        fresh_probe_accuracy = linear_probe_accuracy(
            fresh_probe,
            current_model,
            data.fashion_eval_loader,
            device=self.device,
        )
        parity_bits = parity["code_length_bits_per_example"]
        fashion_bits = fashion["code_length_bits_per_example"]
        return {
            **spec.as_dict(),
            "step": step,
            "epoch": epoch,
            "parity_accuracy": parity["accuracy"],
            "fashion_accuracy": fashion["accuracy"],
            "forgetting": base_metrics["fashion_accuracy"] - fashion["accuracy"],
            "parity_cross_entropy": parity["cross_entropy"],
            "fashion_cross_entropy": fashion["cross_entropy"],
            "parity_code_length_bits_per_example": parity_bits,
            "fashion_code_length_bits_per_example": fashion_bits,
            "parity_code_length_change_bits_per_example": (
                parity_bits - base_metrics["parity_code_length_bits_per_example"]
            ),
            "fashion_code_length_increase_bits_per_example": (
                fashion_bits - base_metrics["fashion_code_length_bits_per_example"]
            ),
            "representation_cka": cka,
            "fashion_fixed_probe_accuracy": fixed_probe_accuracy,
            "fashion_fresh_probe_accuracy": fresh_probe_accuracy,
            "fashion_probe_reorientation_gap": (fresh_probe_accuracy - fixed_probe_accuracy),
            **parameter_count_metrics,
            **shift,
            **distances,
            **{f"train_{name}": value for name, value in training_metrics.items()},
        }

    @staticmethod
    @torch.inference_mode()
    def _maximum_logit_difference(
        base_model: TaskConditionedMLP,
        current_model: TaskConditionedMLP,
        loader: torch.utils.data.DataLoader,
    ) -> float:
        base_model.eval()
        current_model.eval()
        images, _, indicators = next(iter(loader))
        device = next(base_model.parameters()).device
        images = images.to(device)
        indicators = indicators.to(device)
        base_logits = base_model(images, indicators).logits
        current_logits = current_model(images, indicators).logits
        return float((base_logits - current_logits).abs().max())

    def _pretraining_fingerprint(self, seed: int) -> str:
        relevant = {
            "metrics_schema": 2,
            "seed": seed,
            "data": {
                **asdict(self.config.data),
                "root": str(self.config.data.root),
            },
            "model": asdict(self.config.model),
            "pretraining": asdict(self.config.pretraining),
        }
        payload = json.dumps(relevant, sort_keys=True).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()[:12]


def resolve_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def _atomic_torch_save(value: Any, destination: Path) -> None:
    temporary = destination.with_name(destination.name + ".tmp")
    torch.save(value, temporary)
    os.replace(temporary, destination)


def _atomic_write_json(value: Any, destination: Path) -> None:
    _atomic_write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", destination)


def _atomic_write_text(value: str, destination: Path) -> None:
    temporary = destination.with_name(destination.name + ".tmp")
    temporary.write_text(value, encoding="utf-8")
    os.replace(temporary, destination)
