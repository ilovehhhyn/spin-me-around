from __future__ import annotations

import torch
from torch import Tensor
from torch.nn import functional as F

from distance_not_damage.config import FineTuneMethod


def parity_mask(digits: Tensor, class_count: int) -> Tensor:
    """Return a Boolean mask of labels with the same parity as each digit."""

    labels = torch.arange(class_count, device=digits.device)
    return labels.unsqueeze(0).remainder(2) == digits.unsqueeze(1).remainder(2)


def oracle_distribution(base_logits: Tensor, digits: Tensor) -> Tensor:
    """Condition the base policy on the set of parity-correct labels.

    This is the minimum `KL(q || base)` distribution with support restricted to
    valid labels. It is also the rejection-sampling distribution induced by the
    base policy and binary parity reward.
    """

    base_probabilities = base_logits.softmax(dim=-1)
    valid = parity_mask(digits, base_logits.shape[-1])
    masked = base_probabilities * valid
    return masked / masked.sum(dim=-1, keepdim=True)


def forward_policy_kl(base_logits: Tensor, current_logits: Tensor) -> Tensor:
    """Compute mean `KL(base || current)` over a batch."""

    base_log_probabilities = base_logits.log_softmax(dim=-1)
    current_log_probabilities = current_logits.log_softmax(dim=-1)
    base_probabilities = base_log_probabilities.exp()
    return (
        (base_probabilities * (base_log_probabilities - current_log_probabilities))
        .sum(dim=-1)
        .mean()
    )


def reverse_policy_kl(base_logits: Tensor, current_logits: Tensor) -> Tensor:
    """Compute mean `KL(current || base)` over a batch."""

    base_log_probabilities = base_logits.log_softmax(dim=-1)
    current_log_probabilities = current_logits.log_softmax(dim=-1)
    current_probabilities = current_log_probabilities.exp()
    return (
        (current_probabilities * (current_log_probabilities - base_log_probabilities))
        .sum(dim=-1)
        .mean()
    )


def fine_tuning_loss(
    *,
    method: FineTuneMethod,
    current_logits: Tensor,
    base_logits: Tensor,
    digits: Tensor,
    group_size: int,
    grpo_kl_coefficient: float,
) -> tuple[Tensor, dict[str, float]]:
    """Compute one of the Week-1 SFT or on-policy objectives."""

    if method == FineTuneMethod.SFT_1:
        targets = digits.remainder(2)
        loss = F.cross_entropy(current_logits, targets)
        return loss, {"policy_loss": float(loss.detach())}

    if method == FineTuneMethod.SFT_2:
        choices = torch.randint(0, 2, digits.shape, device=digits.device)
        targets = digits.remainder(2) + 4 * choices
        loss = F.cross_entropy(current_logits, targets)
        return loss, {"policy_loss": float(loss.detach())}

    if method == FineTuneMethod.ORACLE_SFT:
        targets = oracle_distribution(base_logits, digits)
        log_probabilities = current_logits.log_softmax(dim=-1)
        loss = -(targets * log_probabilities).sum(dim=-1).mean()
        return loss, {"policy_loss": float(loss.detach())}

    policy_loss, reward_mean = _on_policy_loss(
        method=method,
        logits=current_logits,
        digits=digits,
        group_size=group_size,
    )
    if method == FineTuneMethod.GRPO_KL:
        kl_penalty = reverse_policy_kl(base_logits, current_logits)
        loss = policy_loss + grpo_kl_coefficient * kl_penalty
    else:
        kl_penalty = current_logits.new_zeros(())
        loss = policy_loss

    return loss, {
        "policy_loss": float(policy_loss.detach()),
        "reward_mean": float(reward_mean.detach()),
        "kl_penalty": float(kl_penalty.detach()),
    }


def _on_policy_loss(
    *, method: FineTuneMethod, logits: Tensor, digits: Tensor, group_size: int
) -> tuple[Tensor, Tensor]:
    log_probabilities = logits.log_softmax(dim=-1)
    probabilities = log_probabilities.exp()
    sampled_labels = torch.multinomial(probabilities, group_size, replacement=True)
    sampled_log_probabilities = log_probabilities.gather(dim=1, index=sampled_labels)
    rewards = (sampled_labels.remainder(2) == digits.unsqueeze(1).remainder(2)).to(logits.dtype)

    if method == FineTuneMethod.REINFORCE_10:
        weights = rewards
    elif method in (FineTuneMethod.GRPO, FineTuneMethod.GRPO_KL):
        centered = rewards - rewards.mean(dim=1, keepdim=True)
        scale = rewards.std(dim=1, keepdim=True, unbiased=False).clamp_min(1e-6)
        weights = centered / scale
    else:
        raise ValueError(f"Unsupported on-policy method: {method}")

    policy_loss = -(weights.detach() * sampled_log_probabilities).mean()
    return policy_loss, rewards.mean()
