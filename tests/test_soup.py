from pathlib import Path

import pytest
import torch
import numpy as np


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


def _two_class_checkpoint(weight: float, bias: float = 0.0) -> dict[str, object]:
    checkpoint = _checkpoint(weight=weight, bias=bias)
    checkpoint["class_to_idx"] = {"rain": 0, "sunny": 1}
    checkpoint["model_state"] = {
        "linear.weight": torch.tensor([[weight, weight + 1.0], [weight + 2.0, weight + 3.0]]),
        "linear.bias": torch.tensor([bias, bias + 1.0]),
        "counter": torch.tensor(3, dtype=torch.long),
    }
    return checkpoint


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
    torch.save(_two_class_checkpoint(1.0), first)
    torch.save(_two_class_checkpoint(3.0), second)

    soup = build_model_soup([first, second], weights=[1.0, 3.0])

    assert soup["model_name"] == "small_cnn"
    assert soup["class_to_idx"] == {"rain": 0, "sunny": 1}
    assert soup["soup"]["checkpoints"] == [str(first), str(second)]
    assert torch.allclose(soup["model_state"]["linear.weight"], torch.tensor([[2.5, 3.5], [4.5, 5.5]]))


def test_search_oof_gated_soup_weights_keeps_or_improves_best_checkpoint() -> None:
    from src.weather_net.soup import search_oof_gated_soup_weights

    first_logits = np.array([[3.0, 0.0], [3.0, 0.0], [0.0, 3.0], [0.0, 3.0]], dtype=np.float64)
    second_logits = np.array([[3.0, 0.0], [0.0, 3.0], [0.0, 3.0], [3.0, 0.0]], dtype=np.float64)
    logits_by_checkpoint = np.stack([first_logits, second_logits], axis=0)
    y_true = np.array([0, 1, 1, 0], dtype=np.int64)

    result = search_oof_gated_soup_weights(
        logits_by_checkpoint=logits_by_checkpoint,
        y_true=y_true,
        class_names=["rain", "sunny"],
        max_steps=4,
    )

    assert result["score"] >= result["baseline_score"]
    assert pytest.approx(sum(result["weights"]), abs=1e-9) == 1.0
    assert result["selected_indices"]


def test_build_oof_gated_model_soup_uses_oof_weights_and_metadata(tmp_path: Path) -> None:
    from src.weather_net.soup import build_oof_gated_model_soup

    first = tmp_path / "first.pt"
    second = tmp_path / "second.pt"
    torch.save(_two_class_checkpoint(1.0), first)
    torch.save(_two_class_checkpoint(3.0), second)
    logits_by_checkpoint = np.stack(
        [
            np.array([[3.0, 0.0], [3.0, 0.0], [0.0, 3.0], [0.0, 3.0]], dtype=np.float64),
            np.array([[3.0, 0.0], [0.0, 3.0], [0.0, 3.0], [3.0, 0.0]], dtype=np.float64),
        ],
        axis=0,
    )
    y_true = np.array([0, 1, 1, 0], dtype=np.int64)

    soup = build_oof_gated_model_soup(
        checkpoints=[first, second],
        logits_by_checkpoint=logits_by_checkpoint,
        y_true=y_true,
        class_names=["rain", "sunny"],
        max_steps=4,
    )

    assert soup["soup"]["mode"] == "oof_gated_greedy"
    assert soup["soup"]["oof_score"] >= soup["soup"]["baseline_oof_score"]
    assert len(soup["soup"]["selected_indices"]) >= 1
    assert len(soup["soup"]["weights"]) == len(soup["soup"]["selected_indices"])
    assert len(soup["soup"]["candidate_weights"]) == 2
    assert soup["soup"]["gate_oof_ensemble_score"] == soup["soup"]["oof_score"]


def test_build_oof_gated_model_soup_rejects_checkpoint_oof_count_mismatch(tmp_path: Path) -> None:
    from src.weather_net.soup import build_oof_gated_model_soup

    checkpoint_path = tmp_path / "first.pt"
    torch.save(_checkpoint(1.0), checkpoint_path)

    with pytest.raises(ValueError, match="checkpoint count"):
        build_oof_gated_model_soup(
            checkpoints=[checkpoint_path],
            logits_by_checkpoint=np.zeros((2, 3, 2), dtype=np.float64),
            y_true=np.array([0, 1, 0], dtype=np.int64),
            class_names=["rain", "sunny"],
        )


def test_build_oof_gated_model_soup_rejects_class_mapping_mismatch(tmp_path: Path) -> None:
    from src.weather_net.soup import build_oof_gated_model_soup

    checkpoint_path = tmp_path / "first.pt"
    torch.save(_checkpoint(1.0), checkpoint_path)

    with pytest.raises(ValueError, match="class_names"):
        build_oof_gated_model_soup(
            checkpoints=[checkpoint_path],
            logits_by_checkpoint=np.zeros((1, 3, 2), dtype=np.float64),
            y_true=np.array([0, 1, 0], dtype=np.int64),
            class_names=["rain", "sunny"],
        )


def test_build_oof_gated_model_soup_ignores_unselected_checkpoint_state(tmp_path: Path) -> None:
    from src.weather_net.soup import build_oof_gated_model_soup

    first = tmp_path / "first.pt"
    second = tmp_path / "second.pt"
    torch.save(_two_class_checkpoint(1.0), first)
    unselected = _two_class_checkpoint(3.0)
    unselected["model_state"]["counter"] = torch.tensor(99, dtype=torch.long)
    torch.save(unselected, second)
    logits_by_checkpoint = np.stack(
        [
            np.array([[3.0, 0.0], [0.0, 3.0], [0.0, 3.0], [3.0, 0.0]], dtype=np.float64),
            np.array([[0.0, 3.0], [3.0, 0.0], [3.0, 0.0], [0.0, 3.0]], dtype=np.float64),
        ],
        axis=0,
    )

    soup = build_oof_gated_model_soup(
        checkpoints=[first, second],
        logits_by_checkpoint=logits_by_checkpoint,
        y_true=np.array([0, 1, 1, 0], dtype=np.int64),
        class_names=["rain", "sunny"],
        max_steps=4,
    )

    assert soup["soup"]["selected_indices"] == [0]
    assert soup["soup"]["checkpoints"] == [str(first)]
    assert torch.equal(soup["model_state"]["counter"], torch.tensor(3, dtype=torch.long))


def test_search_oof_gated_soup_weights_rejects_bad_oof_shapes() -> None:
    from src.weather_net.soup import search_oof_gated_soup_weights

    with pytest.raises(ValueError, match="same number of images"):
        search_oof_gated_soup_weights(
            logits_by_checkpoint=np.zeros((2, 3, 2), dtype=np.float64),
            y_true=np.array([0, 1], dtype=np.int64),
            class_names=["rain", "sunny"],
        )
    with pytest.raises(ValueError, match="class dimension"):
        search_oof_gated_soup_weights(
            logits_by_checkpoint=np.zeros((2, 3, 3), dtype=np.float64),
            y_true=np.array([0, 1, 0], dtype=np.int64),
            class_names=["rain", "sunny"],
        )
    with pytest.raises(ValueError, match="min_delta"):
        search_oof_gated_soup_weights(
            logits_by_checkpoint=np.zeros((2, 3, 2), dtype=np.float64),
            y_true=np.array([0, 1, 0], dtype=np.int64),
            class_names=["rain", "sunny"],
            min_delta=-0.1,
        )
