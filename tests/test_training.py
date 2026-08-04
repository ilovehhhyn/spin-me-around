from distance_not_damage.config import SweepConfig
from distance_not_damage.training import build_checkpoint_steps


def test_checkpoint_schedule_combines_early_and_fractional_steps() -> None:
    config = SweepConfig(
        early_checkpoint_steps=(1, 2, 4, 8, 16),
        checkpoint_fractions=(0.5, 1.0),
    )

    steps = build_checkpoint_steps(total_steps=10, sweep_config=config)

    assert steps == {1, 2, 4, 5, 8, 10}
