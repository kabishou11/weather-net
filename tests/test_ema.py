import torch
from torch import nn


def test_model_ema_updates_floating_weights_with_decay() -> None:
    from src.weather_net.training import ModelEma

    model = nn.Linear(1, 1)
    with torch.no_grad():
        model.weight.fill_(1.0)
        model.bias.fill_(0.0)
    ema = ModelEma(model, decay=0.5)

    with torch.no_grad():
        model.weight.fill_(3.0)
        model.bias.fill_(2.0)
    ema.update(model)

    assert torch.allclose(ema.module.weight, torch.tensor([[2.0]]))
    assert torch.allclose(ema.module.bias, torch.tensor([1.0]))


def test_model_ema_rejects_invalid_decay() -> None:
    from src.weather_net.training import ModelEma

    model = nn.Linear(1, 1)

    try:
        ModelEma(model, decay=1.0)
    except ValueError as exc:
        assert "decay" in str(exc)
    else:
        raise AssertionError("ModelEma should reject decay >= 1")
