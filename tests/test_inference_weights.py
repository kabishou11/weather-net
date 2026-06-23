from pathlib import Path
import inspect

import pytest


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
