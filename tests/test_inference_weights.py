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


def test_predict_logits_accepts_transform_backend_argument() -> None:
    from src.weather_net.inference import predict_logits

    assert "transform_backend" in inspect.signature(predict_logits).parameters


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
