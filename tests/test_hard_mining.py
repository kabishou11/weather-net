import csv
from pathlib import Path

import pytest


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def test_compute_hard_sample_weights_prioritizes_errors_and_low_margin() -> None:
    from src.weather_net.hard_mining import compute_hard_sample_weights

    records = [
        {
            "image_id": "easy.jpg",
            "true_label": "rain",
            "pred_label": "rain",
            "correct": "True",
            "confidence": "0.950000",
            "margin": "0.800000",
            "loss": "0.050000",
        },
        {
            "image_id": "low_margin.jpg",
            "true_label": "rain",
            "pred_label": "rain",
            "correct": "True",
            "confidence": "0.550000",
            "margin": "0.050000",
            "loss": "0.600000",
        },
        {
            "image_id": "wrong.jpg",
            "true_label": "sunny",
            "pred_label": "rain",
            "correct": "False",
            "confidence": "0.900000",
            "margin": "0.700000",
            "loss": "2.000000",
        },
    ]

    weights = compute_hard_sample_weights(
        records,
        base_weight=1.0,
        error_boost=1.0,
        low_margin_boost=0.5,
        high_loss_boost=0.5,
        low_margin_threshold=0.1,
        high_loss_quantile=0.67,
        max_weight=2.5,
    )

    assert weights["easy.jpg"].sample_weight == pytest.approx(1.0)
    assert weights["low_margin.jpg"].sample_weight == pytest.approx(1.5)
    assert weights["wrong.jpg"].sample_weight == pytest.approx(2.5)
    assert weights["wrong.jpg"].hard_reason == "error+high_loss"


def test_apply_hard_mining_weights_updates_labeled_rows_only(tmp_path: Path) -> None:
    from src.weather_net.hard_mining import apply_hard_mining_weights

    train_csv = tmp_path / "train.csv"
    train_csv.write_text(
        "image,label,source,confidence,sample_weight\n"
        "easy.jpg,rain,labeled,1.000000,1.000000\n"
        "wrong.jpg,sunny,labeled,1.000000,1.000000\n"
        "pseudo.jpg,rain,pseudo,0.970000,0.979000\n",
        encoding="utf-8",
    )
    oof_csv = tmp_path / "oof_predictions.csv"
    oof_csv.write_text(
        "image_id,true_label,pred_label,correct,confidence,margin,loss\n"
        "easy.jpg,rain,rain,True,0.950000,0.800000,0.050000\n"
        "wrong.jpg,sunny,rain,False,0.900000,0.700000,2.000000\n"
        "pseudo.jpg,rain,sunny,False,0.990000,0.950000,3.000000\n",
        encoding="utf-8",
    )
    output_csv = tmp_path / "weighted.csv"
    hard_csv = tmp_path / "hard.csv"

    stats = apply_hard_mining_weights(
        train_csv=train_csv,
        oof_csv=oof_csv,
        output_csv=output_csv,
        hard_csv=hard_csv,
        error_boost=1.0,
        low_margin_boost=0.5,
        high_loss_boost=0.5,
        low_margin_threshold=0.1,
        high_loss_quantile=0.5,
        max_weight=2.5,
    )

    rows = _read_csv(output_csv)
    assert stats == {"input": 3, "updated": 1, "hard": 1, "skipped_pseudo": 1}
    assert rows[0]["sample_weight"] == "1.000000"
    assert rows[1]["sample_weight"] == "2.500000"
    assert rows[2]["sample_weight"] == "0.979000"
    hard_rows = _read_csv(hard_csv)
    assert hard_rows[0]["image"] == "wrong.jpg"
    assert hard_rows[0]["hard_reason"] == "error+high_loss"


def test_apply_hard_mining_matches_absolute_train_paths_by_name(tmp_path: Path) -> None:
    from src.weather_net.hard_mining import apply_hard_mining_weights

    train_image = tmp_path / "images" / "wrong.jpg"
    train_image.parent.mkdir(parents=True)
    train_image.write_bytes(b"fake")
    train_csv = tmp_path / "train.csv"
    train_csv.write_text(f"image,label\n{train_image},sunny\n", encoding="utf-8")
    oof_csv = tmp_path / "oof.csv"
    oof_csv.write_text(
        "image_id,true_label,pred_label,correct,confidence,margin,loss\n"
        "wrong.jpg,sunny,rain,False,0.900000,0.700000,2.000000\n",
        encoding="utf-8",
    )
    output_csv = tmp_path / "weighted.csv"

    stats = apply_hard_mining_weights(
        train_csv=train_csv,
        oof_csv=oof_csv,
        output_csv=output_csv,
        hard_csv=tmp_path / "hard.csv",
    )

    assert stats["updated"] == 1
    assert "2.500000" in output_csv.read_text(encoding="utf-8")


def test_apply_hard_mining_rejects_duplicate_oof_ids(tmp_path: Path) -> None:
    from src.weather_net.hard_mining import apply_hard_mining_weights

    train_csv = tmp_path / "train.csv"
    train_csv.write_text("image,label\nsame.jpg,rain\n", encoding="utf-8")
    oof_csv = tmp_path / "oof.csv"
    oof_csv.write_text(
        "image_id,true_label,pred_label,correct,confidence,margin,loss\n"
        "same.jpg,rain,rain,True,0.9,0.8,0.1\n"
        "same.jpg,rain,sunny,False,0.8,0.6,1.0\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Duplicate OOF"):
        apply_hard_mining_weights(
            train_csv=train_csv,
            oof_csv=oof_csv,
            output_csv=tmp_path / "weighted.csv",
            hard_csv=tmp_path / "hard.csv",
        )
