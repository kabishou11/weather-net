from pathlib import Path

from PIL import Image


def _make_image(path: Path, color: tuple[int, int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (8, 8), color).save(path)


def test_merge_external_training_rejects_external_duplicate_of_labeled_by_hash(tmp_path: Path) -> None:
    import csv
    import json

    from merge_external_training import merge_labeled_with_external_data

    labeled_root = tmp_path / "official"
    external_root = tmp_path / "external"
    _make_image(labeled_root / "rain.jpg", (10, 20, 30))
    _make_image(labeled_root / "sunny.jpg", (220, 210, 90))
    _make_image(external_root / "rain_copy.jpg", (10, 20, 30))
    _make_image(external_root / "sunny_extra.jpg", (250, 230, 120))

    labeled_csv = tmp_path / "official.csv"
    labeled_csv.write_text(
        "image,label\n"
        "rain.jpg,rain\n"
        "sunny.jpg,sunny\n",
        encoding="utf-8",
    )
    external_csv = tmp_path / "external.csv"
    external_csv.write_text(
        "image,label,source,sample_weight\n"
        "rain_copy.jpg,rain,external_road_weather_time,0.5\n"
        "sunny_extra.jpg,sunny,external_road_weather_time,0.5\n",
        encoding="utf-8",
    )

    summary = merge_labeled_with_external_data(
        train_csv=labeled_csv,
        train_dir=None,
        image_root=labeled_root,
        external_csv=external_csv,
        external_image_root=external_root,
        output_csv=tmp_path / "merged.csv",
        rejected_csv=tmp_path / "rejected.csv",
        audit_json=tmp_path / "audit.json",
        output_image_root=tmp_path,
        max_external_sample_weight=0.5,
    )

    rows = list(csv.DictReader((tmp_path / "merged.csv").open("r", encoding="utf-8")))
    rejected = list(csv.DictReader((tmp_path / "rejected.csv").open("r", encoding="utf-8")))
    audit = json.loads((tmp_path / "audit.json").read_text(encoding="utf-8"))

    assert summary["labeled_rows"] == 2
    assert summary["kept_external_rows"] == 1
    assert [row["source"] for row in rows] == ["labeled", "labeled", "external_road_weather_time"]
    assert rows[-1]["image"] == "external/sunny_extra.jpg"
    assert rejected[0]["image"] == "rain_copy.jpg"
    assert rejected[0]["reason"] == "duplicate_hash"
    assert rejected[0]["kept_source"] == "labeled"
    assert audit["rejected_by_reason"] == {"duplicate_hash": 1}


def test_merge_external_training_caps_weight_and_per_class_quota(tmp_path: Path) -> None:
    import csv

    from merge_external_training import merge_labeled_with_external_data

    labeled_root = tmp_path / "official"
    external_root = tmp_path / "external"
    _make_image(labeled_root / "rain.jpg", (10, 20, 30))
    _make_image(labeled_root / "sunny.jpg", (220, 210, 90))
    _make_image(external_root / "rain_a.jpg", (11, 21, 31))
    _make_image(external_root / "rain_b.jpg", (12, 22, 32))
    _make_image(external_root / "sunny_a.jpg", (221, 211, 91))

    labeled_csv = tmp_path / "official.csv"
    labeled_csv.write_text(
        "image,label\n"
        "rain.jpg,rain\n"
        "sunny.jpg,sunny\n",
        encoding="utf-8",
    )
    external_csv = tmp_path / "external.csv"
    external_csv.write_text(
        "image,label,source,sample_weight\n"
        "rain_a.jpg,rain,external_weapd,0.9\n"
        "rain_b.jpg,rain,external_weapd,0.4\n"
        "sunny_a.jpg,sunny,external_weapd,0.8\n",
        encoding="utf-8",
    )

    summary = merge_labeled_with_external_data(
        train_csv=labeled_csv,
        train_dir=None,
        image_root=labeled_root,
        external_csv=external_csv,
        external_image_root=external_root,
        output_csv=tmp_path / "merged.csv",
        rejected_csv=tmp_path / "rejected.csv",
        audit_json=tmp_path / "audit.json",
        output_image_root=tmp_path,
        max_external_per_labeled_class_ratio=1.0,
        max_external_sample_weight=0.5,
        clip_external_sample_weight=True,
    )

    rows = list(csv.DictReader((tmp_path / "merged.csv").open("r", encoding="utf-8")))
    external_rows = [row for row in rows if row["source"].startswith("external_")]
    rejected = list(csv.DictReader((tmp_path / "rejected.csv").open("r", encoding="utf-8")))

    assert summary["kept_external_rows"] == 2
    assert summary["clipped_external_sample_weight"] == 2
    assert [(row["label"], row["sample_weight"]) for row in external_rows] == [
        ("rain", "0.500000"),
        ("sunny", "0.500000"),
    ]
    assert rejected[0]["image"] == "rain_b.jpg"
    assert rejected[0]["reason"] == "class_quota_exceeded"


def test_merge_external_training_skips_labels_outside_labeled_classes(tmp_path: Path) -> None:
    import csv

    from merge_external_training import merge_labeled_with_external_data

    labeled_root = tmp_path / "official"
    external_root = tmp_path / "external"
    _make_image(labeled_root / "rain.jpg", (10, 20, 30))
    _make_image(external_root / "lightning.jpg", (240, 240, 20))

    labeled_csv = tmp_path / "official.csv"
    labeled_csv.write_text("image,label\nrain.jpg,rain\n", encoding="utf-8")
    external_csv = tmp_path / "external.csv"
    external_csv.write_text(
        "image,label,source,sample_weight\n"
        "lightning.jpg,lightning,external_weapd,0.2\n",
        encoding="utf-8",
    )

    summary = merge_labeled_with_external_data(
        train_csv=labeled_csv,
        train_dir=None,
        image_root=labeled_root,
        external_csv=external_csv,
        external_image_root=external_root,
        output_csv=tmp_path / "merged.csv",
        rejected_csv=tmp_path / "rejected.csv",
        audit_json=tmp_path / "audit.json",
        drop_unmapped_external=True,
    )

    rows = list(csv.DictReader((tmp_path / "merged.csv").open("r", encoding="utf-8")))
    rejected = list(csv.DictReader((tmp_path / "rejected.csv").open("r", encoding="utf-8")))
    assert summary["kept_external_rows"] == 0
    assert [row["source"] for row in rows] == ["labeled"]
    assert rejected[0]["reason"] == "label_not_in_labeled_classes"


def test_merge_external_training_cli_writes_outputs(tmp_path: Path) -> None:
    import json
    import subprocess
    import sys

    labeled_root = tmp_path / "official"
    external_root = tmp_path / "external"
    _make_image(labeled_root / "rain.jpg", (10, 20, 30))
    _make_image(external_root / "rain_extra.jpg", (11, 21, 31))
    labeled_csv = tmp_path / "official.csv"
    labeled_csv.write_text("image,label\nrain.jpg,rain\n", encoding="utf-8")
    external_csv = tmp_path / "external.csv"
    external_csv.write_text(
        "image,label,source,sample_weight\n"
        "rain_extra.jpg,rain,external_road_weather_time,0.7\n",
        encoding="utf-8",
    )

    output_csv = tmp_path / "merged.csv"
    rejected_csv = tmp_path / "rejected.csv"
    audit_json = tmp_path / "audit.json"
    completed = subprocess.run(
        [
            sys.executable,
            "merge_external_training.py",
            "--train-csv",
            str(labeled_csv),
            "--image-root",
            str(labeled_root),
            "--external-csv",
            str(external_csv),
            "--external-image-root",
            str(external_root),
            "--output-image-root",
            str(tmp_path),
            "--max-external-sample-weight",
            "0.5",
            "--clip-external-sample-weight",
            "--output-csv",
            str(output_csv),
            "--rejected-csv",
            str(rejected_csv),
            "--audit-json",
            str(audit_json),
        ],
        check=True,
        text=True,
        capture_output=True,
    )

    printed = json.loads(completed.stdout)
    audit = json.loads(audit_json.read_text(encoding="utf-8"))
    assert printed["total_rows"] == 2
    assert audit["clipped_external_sample_weight"] == 1
    assert output_csv.exists()
    assert rejected_csv.exists()


def test_merge_external_training_rejects_external_effective_weight_share(tmp_path: Path) -> None:
    import pytest

    from merge_external_training import merge_labeled_with_external_data

    labeled_root = tmp_path / "official"
    external_root = tmp_path / "external"
    _make_image(labeled_root / "rain.jpg", (10, 20, 30))
    _make_image(external_root / "rain_a.jpg", (11, 21, 31))
    _make_image(external_root / "rain_b.jpg", (12, 22, 32))
    labeled_csv = tmp_path / "official.csv"
    labeled_csv.write_text("image,label\nrain.jpg,rain\n", encoding="utf-8")
    external_csv = tmp_path / "external.csv"
    external_csv.write_text(
        "image,label,source,sample_weight\n"
        "rain_a.jpg,rain,external_weapd,0.5\n"
        "rain_b.jpg,rain,external_weapd,0.5\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="external effective weight share"):
        merge_labeled_with_external_data(
            train_csv=labeled_csv,
            train_dir=None,
            image_root=labeled_root,
            external_csv=external_csv,
            external_image_root=external_root,
            output_csv=tmp_path / "merged.csv",
            rejected_csv=tmp_path / "rejected.csv",
            audit_json=tmp_path / "audit.json",
            max_external_effective_weight_share=0.4,
        )


def test_merge_external_training_reports_duplicate_before_class_quota(tmp_path: Path) -> None:
    import csv

    from merge_external_training import merge_labeled_with_external_data

    labeled_root = tmp_path / "official"
    external_root = tmp_path / "external"
    _make_image(labeled_root / "rain.jpg", (10, 20, 30))
    _make_image(external_root / "rain_copy.jpg", (10, 20, 30))
    labeled_csv = tmp_path / "official.csv"
    labeled_csv.write_text("image,label\nrain.jpg,rain\n", encoding="utf-8")
    external_csv = tmp_path / "external.csv"
    external_csv.write_text(
        "image,label,source,sample_weight\n"
        "rain_copy.jpg,rain,external_weapd,0.5\n",
        encoding="utf-8",
    )

    merge_labeled_with_external_data(
        train_csv=labeled_csv,
        train_dir=None,
        image_root=labeled_root,
        external_csv=external_csv,
        external_image_root=external_root,
        output_csv=tmp_path / "merged.csv",
        rejected_csv=tmp_path / "rejected.csv",
        audit_json=tmp_path / "audit.json",
        max_external_per_labeled_class_ratio=0.0,
    )

    rejected = list(csv.DictReader((tmp_path / "rejected.csv").open("r", encoding="utf-8")))
    assert rejected[0]["reason"] == "duplicate_hash"
    assert rejected[0]["kept_source"] == "labeled"


def test_merge_external_training_rejects_non_labeled_base_rows(tmp_path: Path) -> None:
    import pytest

    from merge_external_training import merge_labeled_with_external_data

    labeled_root = tmp_path / "official"
    external_root = tmp_path / "external"
    _make_image(labeled_root / "pseudo.jpg", (10, 20, 30))
    _make_image(external_root / "rain.jpg", (11, 21, 31))
    labeled_csv = tmp_path / "base.csv"
    labeled_csv.write_text(
        "image,label,source,sample_weight\n"
        "pseudo.jpg,rain,pseudo,0.9\n",
        encoding="utf-8",
    )
    external_csv = tmp_path / "external.csv"
    external_csv.write_text(
        "image,label,source,sample_weight\n"
        "rain.jpg,rain,external_weapd,0.5\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="base training rows must be labeled"):
        merge_labeled_with_external_data(
            train_csv=labeled_csv,
            train_dir=None,
            image_root=labeled_root,
            external_csv=external_csv,
            external_image_root=external_root,
            output_csv=tmp_path / "merged.csv",
            rejected_csv=tmp_path / "rejected.csv",
            audit_json=tmp_path / "audit.json",
        )


def test_merge_external_training_rejects_external_source_typos(tmp_path: Path) -> None:
    import pytest

    from merge_external_training import merge_labeled_with_external_data

    labeled_root = tmp_path / "official"
    external_root = tmp_path / "external"
    _make_image(labeled_root / "rain.jpg", (10, 20, 30))
    _make_image(external_root / "rain_extra.jpg", (11, 21, 31))
    labeled_csv = tmp_path / "official.csv"
    labeled_csv.write_text("image,label\nrain.jpg,rain\n", encoding="utf-8")
    external_csv = tmp_path / "external.csv"
    external_csv.write_text(
        "image,label,source,sample_weight\n"
        "rain_extra.jpg,rain,externalbad,0.5\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="source=external"):
        merge_labeled_with_external_data(
            train_csv=labeled_csv,
            train_dir=None,
            image_root=labeled_root,
            external_csv=external_csv,
            external_image_root=external_root,
            output_csv=tmp_path / "merged.csv",
            rejected_csv=tmp_path / "rejected.csv",
            audit_json=tmp_path / "audit.json",
        )


def test_merge_external_training_rejects_non_finite_gate_values(tmp_path: Path) -> None:
    import math
    import pytest

    from merge_external_training import merge_labeled_with_external_data

    labeled_root = tmp_path / "official"
    external_root = tmp_path / "external"
    _make_image(labeled_root / "rain.jpg", (10, 20, 30))
    _make_image(external_root / "rain_extra.jpg", (11, 21, 31))
    labeled_csv = tmp_path / "official.csv"
    labeled_csv.write_text("image,label\nrain.jpg,rain\n", encoding="utf-8")
    external_csv = tmp_path / "external.csv"
    external_csv.write_text(
        "image,label,source,sample_weight\n"
        "rain_extra.jpg,rain,external_weapd,0.5\n",
        encoding="utf-8",
    )

    for kwargs in [
        {"max_external_per_labeled_class_ratio": math.nan},
        {"max_external_per_labeled_class_ratio": math.inf},
        {"max_external_sample_weight": math.nan},
        {"max_external_sample_weight": math.inf},
        {"max_external_effective_weight_share": math.nan},
        {"max_external_effective_weight_share": math.inf},
    ]:
        with pytest.raises(ValueError, match="finite"):
            merge_labeled_with_external_data(
                train_csv=labeled_csv,
                train_dir=None,
                image_root=labeled_root,
                external_csv=external_csv,
                external_image_root=external_root,
                output_csv=tmp_path / "merged.csv",
                rejected_csv=tmp_path / "rejected.csv",
                audit_json=tmp_path / "audit.json",
                **kwargs,
            )
