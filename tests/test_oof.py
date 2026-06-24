from pathlib import Path

import numpy as np


def test_oof_records_write_csv_npz_and_metrics(tmp_path: Path) -> None:
    from src.weather_net.oof import OofRecord, write_oof_artifacts

    records = [
        OofRecord(
            image_id="fold0/rain.jpg",
            image_path="/data/rain.jpg",
            fold=0,
            source="labeled",
            true_idx=0,
            true_label="rain",
            logits=[2.0, 0.0],
            checkpoint="fold0.pt",
            model_name="tiny",
        ),
        OofRecord(
            image_id="fold1/sunny.jpg",
            image_path="/data/sunny.jpg",
            fold=1,
            source="labeled",
            true_idx=1,
            true_label="sunny",
            logits=[0.0, 2.0],
            checkpoint="fold1.pt",
            model_name="tiny",
        ),
    ]

    outputs = write_oof_artifacts(
        output_dir=tmp_path / "oof",
        records=records,
        class_names=["rain", "sunny"],
        class_to_idx={"rain": 0, "sunny": 1},
        metadata={"folds": 2},
    )

    csv_text = outputs.csv_path.read_text(encoding="utf-8")
    assert "image_id,image_path,fold,source,true_idx,true_label,pred_idx" in csv_text
    assert "prob_rain" in csv_text
    assert "logit_sunny" in csv_text

    arrays = np.load(outputs.npz_path, allow_pickle=True)
    np.testing.assert_allclose(arrays["logits"], np.array([[2.0, 0.0], [0.0, 2.0]], dtype=np.float32))
    np.testing.assert_allclose(arrays["probs"].sum(axis=1), np.ones(2))
    assert arrays["y_true"].tolist() == [0, 1]
    assert arrays["fold"].tolist() == [0, 1]

    assert '"macro_f1": 1.0' in outputs.metrics_path.read_text(encoding="utf-8")


def test_oof_artifacts_reject_duplicate_image_ids(tmp_path: Path) -> None:
    import pytest
    from src.weather_net.oof import OofRecord, write_oof_artifacts

    records = [
        OofRecord(
            image_id="same.jpg",
            image_path="/data/a/same.jpg",
            fold=0,
            source="labeled",
            true_idx=0,
            true_label="rain",
            logits=[2.0, 0.0],
            checkpoint="fold0.pt",
            model_name="tiny",
        ),
        OofRecord(
            image_id="same.jpg",
            image_path="/data/b/same.jpg",
            fold=1,
            source="labeled",
            true_idx=1,
            true_label="sunny",
            logits=[0.0, 2.0],
            checkpoint="fold1.pt",
            model_name="tiny",
        ),
    ]

    with pytest.raises(ValueError, match="Duplicate OOF image ids"):
        write_oof_artifacts(
            output_dir=tmp_path / "oof",
            records=records,
            class_names=["rain", "sunny"],
            class_to_idx={"rain": 0, "sunny": 1},
            metadata={},
        )
