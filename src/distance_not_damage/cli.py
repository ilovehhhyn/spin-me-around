from __future__ import annotations

import argparse
import json
from pathlib import Path

from distance_not_damage.config import (
    ExperimentConfig,
    FineTuneMethod,
    FineTuneParameterization,
    RunSpec,
    SchedulerName,
)
from distance_not_damage.experiment import ExperimentRunner


def train_main() -> None:
    parser = argparse.ArgumentParser(description="Run one ParityMNIST fine-tuning experiment.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--method", type=FineTuneMethod, choices=list(FineTuneMethod))
    parser.add_argument(
        "--parameterization",
        type=FineTuneParameterization,
        choices=list(FineTuneParameterization),
    )
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--scheduler", type=SchedulerName, choices=list(SchedulerName))
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--seed", type=int)
    arguments = parser.parse_args()

    config = ExperimentConfig.from_yaml(arguments.config)
    parameterization = arguments.parameterization or config.sweep.parameterizations[0]
    spec = RunSpec(
        method=arguments.method or config.sweep.methods[0],
        parameterization=parameterization,
        learning_rate=(
            arguments.learning_rate
            if arguments.learning_rate is not None
            else config.sweep.learning_rates_for(parameterization)[0]
        ),
        scheduler=arguments.scheduler or config.sweep.schedulers[0],
        epochs=arguments.epochs or config.sweep.epochs[0],
        seed=arguments.seed if arguments.seed is not None else config.sweep.seeds[0],
        lora_rank=(config.lora.rank if parameterization == FineTuneParameterization.LORA else None),
    )
    summary = ExperimentRunner(config).run(spec)
    print(json.dumps(summary, indent=2, sort_keys=True))


def sweep_main() -> None:
    parser = argparse.ArgumentParser(description="Run a sequential ParityMNIST sweep.")
    parser.add_argument("--config", type=Path, required=True)
    arguments = parser.parse_args()

    config = ExperimentConfig.from_yaml(arguments.config)
    runner = ExperimentRunner(config)
    specs = runner.sweep_specs()
    for run_index, spec in enumerate(specs, start=1):
        print(f"[{run_index}/{len(specs)}] {spec.identifier}", flush=True)
        runner.run(spec)


if __name__ == "__main__":
    train_main()
