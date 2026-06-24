from pathlib import Path
import inspect

import pytest
from PIL import Image


def test_normalize_checkpoint_weights_defaults_to_equal_weights() -> None:
    from src.weather_net.inference import normalize_checkpoint_weights

    weights = normalize_checkpoint_weights([Path("a.pt"), Path("b.pt")], None)

    assert weights == [0.5, 0.5]


def test_normalize_checkpoint_weights_rejects_mismatched_count() -> None:
    from src.weather_net.inference import normalize_checkpoint_weights

    with pytest.raises(ValueError, match="same count"):
        normalize_checkpoint_weights([Path("a.pt"), Path("b.pt")], [1.0])


def test_normalize_checkpoint_weights_rejects_non_positive_sum() -> None:
    from src.weather_net.inference import normalize_checkpoint_weights

    with pytest.raises(ValueError, match="positive"):
        normalize_checkpoint_weights([Path("a.pt"), Path("b.pt")], [0.0, 0.0])


def test_resolve_checkpoint_weights_checks_decision_checkpoint_order() -> None:
    from src.weather_net.inference import resolve_checkpoint_weights
    from src.weather_net.postprocess import DecisionParams

    params = DecisionParams(
        class_names=["rain", "sunny"],
        temperature=1.0,
        bias=[0.0, 0.0],
        weights=[0.8, 0.2],
        scores={},
        checkpoints=["a.pt", "b.pt"],
    )

    weights = resolve_checkpoint_weights(
        checkpoints=[Path("a.pt"), Path("b.pt")],
        cli_weights=None,
        decision_params=params,
    )

    assert weights == [0.8, 0.2]
    with pytest.raises(ValueError, match="checkpoint order"):
        resolve_checkpoint_weights(
            checkpoints=[Path("b.pt"), Path("a.pt")],
            cli_weights=None,
            decision_params=params,
        )


def test_resolve_checkpoint_weights_allows_cli_weights_to_override_decision_weights() -> None:
    from src.weather_net.inference import resolve_checkpoint_weights
    from src.weather_net.postprocess import DecisionParams

    params = DecisionParams(
        class_names=["rain", "sunny"],
        temperature=1.0,
        bias=[0.0, 0.0],
        weights=[0.8, 0.2],
        scores={},
        checkpoints=["a.pt", "b.pt"],
    )

    weights = resolve_checkpoint_weights(
        checkpoints=[Path("b.pt"), Path("a.pt")],
        cli_weights=[0.25, 0.75],
        decision_params=params,
    )

    assert weights == [0.25, 0.75]


def test_predict_logits_accepts_transform_backend_argument() -> None:
    from src.weather_net.inference import predict_logits

    assert "transform_backend" in inspect.signature(predict_logits).parameters


def test_predict_probabilities_accepts_decision_params_path_argument() -> None:
    from src.weather_net.inference import predict_probabilities

    assert "decision_params_path" in inspect.signature(predict_probabilities).parameters


def _make_image(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (8, 8), (12, 34, 56)).save(path)


def test_load_unlabeled_rows_from_directory_uses_relative_image_ids(tmp_path: Path) -> None:
    from src.weather_net.inference import load_unlabeled_rows

    _make_image(tmp_path / "test" / "station_a" / "frame.jpg")
    _make_image(tmp_path / "test" / "station_b" / "frame.jpg")

    rows = load_unlabeled_rows(test_dir=tmp_path / "test")

    assert [row.image_id for row in rows] == [
        "station_a/frame.jpg",
        "station_b/frame.jpg",
    ]
