import torch
from torch.nn import functional as F

from distance_not_damage.config import (
    FineTuneParameterization,
    LoRAConfig,
    ModelConfig,
)
from distance_not_damage.metrics import parameter_distance
from distance_not_damage.model import (
    LoRALinear,
    TaskConditionedMLP,
    configure_for_fine_tuning,
    parameter_counts,
    set_lora_adapters_enabled,
    trainable_parameters,
)


def test_lora_starts_functionally_identical_and_updates_only_adapters() -> None:
    torch.manual_seed(17)
    config = ModelConfig()
    base_model = TaskConditionedMLP(config)
    current_model = TaskConditionedMLP(config)
    current_model.load_state_dict(base_model.state_dict())
    images = torch.randn(7, 1, 28, 28)
    indicators = torch.ones(7)
    base_logits = base_model(images, indicators).logits.detach()

    configure_for_fine_tuning(
        current_model,
        parameterization=FineTuneParameterization.LORA,
        lora=LoRAConfig(rank=8, alpha=8.0, dropout=0.0),
    )

    initial_logits = current_model(images, indicators).logits
    assert torch.equal(initial_logits, base_logits)
    assert parameter_distance(base_model, current_model)["parameter_l2"] == 0.0
    assert all(
        parameter.requires_grad
        for name, parameter in current_model.named_parameters()
        if name.endswith(("lora_a", "lora_b"))
    )
    assert all(
        not parameter.requires_grad
        for name, parameter in current_model.named_parameters()
        if "base_layer" in name
    )

    frozen_base = {
        name: parameter.detach().clone()
        for name, parameter in current_model.named_parameters()
        if "base_layer" in name
    }
    optimizer = torch.optim.SGD(trainable_parameters(current_model), lr=0.1)
    loss = F.cross_entropy(current_model(images, indicators).logits, torch.arange(7))
    loss.backward()
    optimizer.step()

    assert parameter_distance(base_model, current_model)["parameter_l2"] > 0.0
    for name, before in frozen_base.items():
        assert torch.equal(dict(current_model.named_parameters())[name], before)

    set_lora_adapters_enabled(current_model, enabled=False)
    assert torch.equal(current_model(images, indicators).logits, base_logits)


def test_rank_eight_configuration_has_expected_trainable_parameter_count() -> None:
    model = TaskConditionedMLP(ModelConfig()).to(dtype=torch.float64)
    configure_for_fine_tuning(
        model,
        parameterization=FineTuneParameterization.LORA,
        lora=LoRAConfig(rank=8, alpha=8.0, dropout=0.0),
    )

    counts = parameter_counts(model)

    expected = 8 * ((785 + 512) + (512 + 256) + (256 + 10))
    assert counts["trainable_parameter_count"] == expected
    assert isinstance(model.input_layer, LoRALinear)
    assert isinstance(model.hidden_layer, LoRALinear)
    assert isinstance(model.output_layer, LoRALinear)
    assert model.input_layer.lora_a.dtype == torch.float64
    assert model.input_layer.lora_a.device == model.input_layer.base_layer.weight.device
