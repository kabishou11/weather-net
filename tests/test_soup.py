from pathlib import Path

import pytest
import torch


def _checkpoint(weight: float, bias: float = 0.0) -> dict[str, object]:
    return {
        "model_state": {
            "linear.weight": torch.tensor([[weight, weight + 1.0]]),
            "linear.bias": torch.tensor([bias]),
            "counter": torch.tensor(3, dtype=torch.long),
        },
        "class_to_idx": {"rain": 0},
        "model_name": "small_cnn",
        "image_size": 64,
        "macro_f1": weight,
    }


def test_average_state_dicts_weighted_float_tensors_and_keep_integer_tensors() -> None:
    from src.weather_net.soup import average_state_dicts

    averaged = average_state_dicts(
        [_checkpoint(1.0)["model_state"], _checkpoint(3.0)["model_state"]],
        weights=[0.25, 0.75],
    )

    assert torch.allclose(averaged["linear.weight"], torch.tensor([[2.5, 3.5]]))
    assert torch.equal(averaged["counter"], torch.tensor(3, dtype=torch.long))


def test_build_model_soup_rejects_mismatched_model_name(tmp_path: Path) -> None:
    from src.weather_net.soup import build_model_soup

    first = tmp_path / "first.pt"
    second = tmp_path / "second.pt"
    torch.save(_checkpoint(1.0), first)
    changed = _checkpoint(2.0)
    changed["model_name"] = "resnet18"
    torch.save(changed, second)

    with pytest.raises(ValueError, match="model_name"):
        build_model_soup([first, second])


def test_build_model_soup_preserves_metadata_and_averages_weights(tmp_path: Path) -> None:
    from src.weather_net.soup import build_model_soup

    first = tmp_path / "first.pt"
    second = tmp_path / "second.pt"
    torch.save(_checkpoint(1.0), first)
    torch.save(_checkpoint(3.0), second)

    soup = build_model_soup([first, second], weights=[1.0, 3.0])

    assert soup["model_name"] == "small_cnn"
    assert soup["class_to_idx"] == {"rain": 0}
    assert soup["soup"]["checkpoints"] == [str(first), str(second)]
    assert torch.allclose(soup["model_state"]["linear.weight"], torch.tensor([[2.5, 3.5]]))
