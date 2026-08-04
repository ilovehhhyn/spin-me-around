import torch

from distance_not_damage.metrics import linear_cka


def test_linear_cka_is_one_for_identical_representations() -> None:
    representations = torch.randn(64, 16)

    value = linear_cka(representations, representations)

    assert torch.allclose(value, torch.ones(()), atol=1e-6)


def test_linear_cka_is_invariant_to_isotropic_scaling() -> None:
    representations = torch.randn(64, 16)

    value = linear_cka(representations, 3.0 * representations)

    assert torch.allclose(value, torch.ones(()), atol=1e-6)
