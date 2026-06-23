from pathlib import Path

import pytest
from PIL import Image


def _make_image(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (8, 8), (24, 48, 96)).save(path)


def test_csv_manifest_preserves_metadata_and_confidence_weight(tmp_path: Path) -> None:
    from src.weather_net.data import build_manifest_from_csv

    _make_image(tmp_path / "images" / "a.jpg")
    _make_image(tmp_path / "images" / "b.jpg")
    csv_path = tmp_path / "merged.csv"
    csv_path.write_text(
        "image,label,source,confidence\n"
        "a.jpg,rain,labeled,1.000000\n"
        "b.jpg,sunny,pseudo,0.800000\n",
        encoding="utf-8",
    )

    rows, _ = build_manifest_from_csv(csv_path, image_root=tmp_path / "images")

    assert rows[0].image_id == "a.jpg"
    assert rows[0].source == "labeled"
    assert rows[0].confidence == 1.0
    assert rows[0].sample_weight == 1.0
    assert rows[1].source == "pseudo"
    assert rows[1].confidence == 0.8
    assert rows[1].sample_weight == pytest.approx(0.86)


def test_unlabeled_csv_manifest_preserves_original_image_id(tmp_path: Path) -> None:
    from src.weather_net.data import build_unlabeled_manifest_from_csv

    _make_image(tmp_path / "images" / "nested" / "same.jpg")
    csv_path = tmp_path / "test.csv"
    csv_path.write_text("image\nnested/same.jpg\n", encoding="utf-8")

    rows = build_unlabeled_manifest_from_csv(csv_path, image_root=tmp_path / "images")

    assert rows[0].image_id == "nested/same.jpg"


def test_unlabeled_csv_manifest_accepts_id_column(tmp_path: Path) -> None:
    from src.weather_net.data import build_unlabeled_manifest_from_csv

    _make_image(tmp_path / "images" / "station_a" / "frame.jpg")
    csv_path = tmp_path / "test.csv"
    csv_path.write_text("id\nstation_a/frame.jpg\n", encoding="utf-8")

    rows = build_unlabeled_manifest_from_csv(csv_path, image_root=tmp_path / "images")

    assert rows[0].image_id == "station_a/frame.jpg"


def test_split_train_val_keeps_singleton_classes_in_train(tmp_path: Path) -> None:
    from src.weather_net.data import ManifestRow, split_train_val

    rows = [
        ManifestRow(path=tmp_path / "rain0.jpg", label=0, label_name="rain"),
        ManifestRow(path=tmp_path / "sunny0.jpg", label=1, label_name="sunny"),
        ManifestRow(path=tmp_path / "sunny1.jpg", label=1, label_name="sunny"),
    ]

    train_rows, val_rows = split_train_val(rows, val_fraction=0.5, seed=7)

    train_labels = {row.label_name for row in train_rows}
    assert "rain" in train_labels
    assert all(row.label_name != "rain" for row in val_rows)


def test_split_train_val_never_places_pseudo_rows_in_validation(tmp_path: Path) -> None:
    from src.weather_net.data import ManifestRow, split_train_val

    rows = [
        ManifestRow(path=tmp_path / "rain0.jpg", label=0, label_name="rain", source="labeled"),
        ManifestRow(path=tmp_path / "rain1.jpg", label=0, label_name="rain", source="labeled"),
        ManifestRow(path=tmp_path / "sunny0.jpg", label=1, label_name="sunny", source="labeled"),
        ManifestRow(path=tmp_path / "sunny1.jpg", label=1, label_name="sunny", source="labeled"),
        ManifestRow(path=tmp_path / "pseudo" / "rain2.jpg", label=0, label_name="rain", source="pseudo"),
        ManifestRow(path=tmp_path / "pseudo" / "sunny2.jpg", label=1, label_name="sunny", source="pseudo"),
    ]

    train_rows, val_rows = split_train_val(rows, val_fraction=0.5, seed=0)

    assert all(row.source != "pseudo" for row in val_rows)
    assert {row.path.name for row in train_rows} >= {"rain2.jpg", "sunny2.jpg"}


def test_iter_kfold_splits_reduces_folds_for_smallest_class(tmp_path: Path) -> None:
    from src.weather_net.data import ManifestRow, iter_kfold_splits

    rows = [
        ManifestRow(path=tmp_path / "rain0.jpg", label=0, label_name="rain"),
        ManifestRow(path=tmp_path / "rain1.jpg", label=0, label_name="rain"),
        ManifestRow(path=tmp_path / "sunny0.jpg", label=1, label_name="sunny"),
        ManifestRow(path=tmp_path / "sunny1.jpg", label=1, label_name="sunny"),
        ManifestRow(path=tmp_path / "sunny2.jpg", label=1, label_name="sunny"),
        ManifestRow(path=tmp_path / "sunny3.jpg", label=1, label_name="sunny"),
    ]

    splits = list(iter_kfold_splits(rows, folds=5, seed=7))

    assert len(splits) == 2
    for _, train_rows, val_rows in splits:
        assert {row.label_name for row in train_rows} == {"rain", "sunny"}
        assert {row.label_name for row in val_rows} == {"rain", "sunny"}


def test_iter_kfold_splits_never_places_pseudo_rows_in_validation(tmp_path: Path) -> None:
    from src.weather_net.data import ManifestRow, iter_kfold_splits

    rows = [
        ManifestRow(path=tmp_path / "rain0.jpg", label=0, label_name="rain", source="labeled"),
        ManifestRow(path=tmp_path / "rain1.jpg", label=0, label_name="rain", source="labeled"),
        ManifestRow(path=tmp_path / "sunny0.jpg", label=1, label_name="sunny", source="labeled"),
        ManifestRow(path=tmp_path / "sunny1.jpg", label=1, label_name="sunny", source="labeled"),
        ManifestRow(path=tmp_path / "pseudo" / "rain2.jpg", label=0, label_name="rain", source="pseudo"),
        ManifestRow(path=tmp_path / "pseudo" / "sunny2.jpg", label=1, label_name="sunny", source="pseudo"),
    ]

    splits = list(iter_kfold_splits(rows, folds=2, seed=3))

    assert len(splits) == 2
    for _, train_rows, val_rows in splits:
        assert all(row.source != "pseudo" for row in val_rows)
        assert {row.path.name for row in train_rows} >= {"rain2.jpg", "sunny2.jpg"}
