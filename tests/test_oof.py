from pathlib import Path

import numpy as np
import pytest


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
    assert "prob_0_rain" in csv_text
    assert "logit_1_sunny" in csv_text

    arrays = np.load(outputs.npz_path, allow_pickle=True)
    np.testing.assert_allclose(arrays["logits"], np.array([[2.0, 0.0], [0.0, 2.0]], dtype=np.float32))
    np.testing.assert_allclose(arrays["probs"].sum(axis=1), np.ones(2))
    assert arrays["y_true"].tolist() == [0, 1]
    assert arrays["fold"].tolist() == [0, 1]
    assert arrays["checkpoint"].tolist() == ["fold0.pt", "fold1.pt"]

    assert '"macro_f1": 1.0' in outputs.metrics_path.read_text(encoding="utf-8")


def test_oof_artifacts_reject_duplicate_image_ids(tmp_path: Path) -> None:
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


def test_oof_artifacts_reject_non_labeled_validation_sources(tmp_path: Path) -> None:
    from src.weather_net.oof import OofRecord, write_oof_artifacts

    records = [
        OofRecord(
            image_id="external.jpg",
            image_path="/data/external.jpg",
            fold=0,
            source="external_weapd",
            true_idx=0,
            true_label="rain",
            logits=[2.0, 0.0],
            checkpoint="fold0.pt",
            model_name="tiny",
        ),
    ]

    with pytest.raises(ValueError, match="OOF validation records must come from labeled rows"):
        write_oof_artifacts(
            output_dir=tmp_path / "oof",
            records=records,
            class_names=["rain", "sunny"],
            class_to_idx={"rain": 0, "sunny": 1},
            metadata={},
        )


def test_oof_csv_columns_include_class_indices_to_avoid_sanitized_name_collisions(tmp_path: Path) -> None:
    from src.weather_net.oof import OofRecord, write_oof_artifacts

    records = [
        OofRecord(
            image_id="sample.jpg",
            image_path="/data/sample.jpg",
            fold=0,
            source="labeled",
            true_idx=0,
            true_label="rain-snow",
            logits=[2.0, 0.0],
            checkpoint="fold0.pt",
            model_name="tiny",
        ),
    ]

    outputs = write_oof_artifacts(
        output_dir=tmp_path / "oof",
        records=records,
        class_names=["rain-snow", "rain_snow"],
        class_to_idx={"rain-snow": 0, "rain_snow": 1},
        metadata={},
    )

    header = outputs.csv_path.read_text(encoding="utf-8").splitlines()[0]

    assert "logit_0_rain_snow" in header
    assert "logit_1_rain_snow" in header
    assert len(header.split(",")) == len(set(header.split(",")))


def test_oof_artifacts_reject_class_mapping_order_mismatch(tmp_path: Path) -> None:
    from src.weather_net.oof import OofRecord, write_oof_artifacts

    records = [
        OofRecord(
            image_id="sample.jpg",
            image_path="/data/sample.jpg",
            fold=0,
            source="labeled",
            true_idx=0,
            true_label="rain",
            logits=[2.0, 0.0],
            checkpoint="fold0.pt",
            model_name="tiny",
        ),
    ]

    with pytest.raises(ValueError, match="class_to_idx"):
        write_oof_artifacts(
            output_dir=tmp_path / "oof",
            records=records,
            class_names=["rain", "sunny"],
            class_to_idx={"sunny": 0, "rain": 1},
            metadata={},
        )
