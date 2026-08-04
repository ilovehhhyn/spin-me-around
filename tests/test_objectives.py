import torch

from distance_not_damage.config import FineTuneMethod
from distance_not_damage.objectives import (
    fine_tuning_loss,
    forward_policy_kl,
    oracle_distribution,
    reverse_policy_kl,
)


def test_oracle_distribution_has_only_parity_correct_support() -> None:
    base_logits = torch.tensor(
        [
            [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
            [9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0, 0.0],
        ]
    )
    digits = torch.tensor([2, 7])

    distribution = oracle_distribution(base_logits, digits)

    assert torch.allclose(distribution.sum(dim=1), torch.ones(2))
    assert torch.count_nonzero(distribution[0, 1::2]) == 0
    assert torch.count_nonzero(distribution[1, 0::2]) == 0


def test_policy_kl_is_zero_for_identical_logits() -> None:
    logits = torch.randn(8, 10)

    assert torch.allclose(forward_policy_kl(logits, logits), torch.zeros(()), atol=1e-7)
    assert torch.allclose(reverse_policy_kl(logits, logits), torch.zeros(()), atol=1e-7)


def test_every_objective_produces_finite_gradient() -> None:
    digits = torch.tensor([0, 1, 4, 7])
    base_logits = torch.randn(4, 10)

    for method in FineTuneMethod:
        current_logits = torch.randn(4, 10, requires_grad=True)
        loss, diagnostics = fine_tuning_loss(
            method=method,
            current_logits=current_logits,
            base_logits=base_logits,
            digits=digits,
            group_size=32,
            grpo_kl_coefficient=0.1,
        )
        loss.backward()

        assert torch.isfinite(loss)
        assert current_logits.grad is not None
        assert torch.isfinite(current_logits.grad).all()
        assert "policy_loss" in diagnostics
