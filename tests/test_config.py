from pathlib import Path

from distance_not_damage.config import (
    ExperimentConfig,
    FineTuneMethod,
    FineTuneParameterization,
)


def test_smoke_config_loads() -> None:
    config = ExperimentConfig.from_yaml(Path("configs/week1_smoke.yaml"))

    assert config.data.fine_tune_examples == 2000
    assert config.sweep.methods == (
        FineTuneMethod.SFT_1,
        FineTuneMethod.REINFORCE_10,
    )
    assert config.sweep.parameterizations == (
        FineTuneParameterization.FULL,
        FineTuneParameterization.LORA,
    )
    assert config.sweep.learning_rates_for(FineTuneParameterization.FULL) == (1e-4,)
    assert config.sweep.learning_rates_for(FineTuneParameterization.LORA) == (1e-3,)
    assert config.lora.rank == 8
