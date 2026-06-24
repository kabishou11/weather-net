from pathlib import Path

from PIL import Image


def _make_image(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (8, 8), (120, 120, 120)).save(path)


def test_preflight_rejects_external_rows_without_explicit_allowance(tmp_path: Path) -> None:
    import pytest

    from training_preflight import run_preflight_checks

    _make_image(tmp_path / "images" / "external.jpg")
    train_csv = tmp_path / "train.csv"
    train_csv.write_text(
        "image,label,source,sample_weight\n"
        "external.jpg,rain,external,0.2\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="external data"):
        run_preflight_checks(
            config_path=Path("configs/convnext_tiny.yaml"),
            train_csv=train_csv,
            image_root=tmp_path / "images",
            allow_external_data=False,
        )


def test_preflight_requires_teacher_columns_when_distillation_is_enabled(tmp_path: Path) -> None:
    import pytest

    from training_preflight import run_preflight_checks

    _make_image(tmp_path / "images" / "rain.jpg")
    train_csv = tmp_path / "train.csv"
    train_csv.write_text("image,label\nrain.jpg,rain\n", encoding="utf-8")
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "train:\n"
        "  distillation_alpha: 0.3\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="teacher"):
        run_preflight_checks(
            config_path=config_path,
            train_csv=train_csv,
            image_root=tmp_path / "images",
        )


def test_preflight_rejects_slow_inference_stats(tmp_path: Path) -> None:
    import json
    import pytest

    from training_preflight import run_preflight_checks

    _make_image(tmp_path / "images" / "rain.jpg")
    train_csv = tmp_path / "train.csv"
    train_csv.write_text("image,label\nrain.jpg,rain\n", encoding="utf-8")
    stats_path = tmp_path / "submission.stats.json"
    stats_path.write_text(
        json.dumps(
            {
                "images": 10,
                "seconds": 2.0,
                "checkpoints": ["a.pt", "b.pt"],
                "tta": True,
                "amp": False,
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="inference budget"):
        run_preflight_checks(
            config_path=Path("configs/convnext_tiny.yaml"),
            train_csv=train_csv,
            image_root=tmp_path / "images",
            inference_stats=stats_path,
            max_seconds_per_image=0.05,
            max_checkpoints=1,
            allow_tta=False,
        )


def test_preflight_passes_clean_training_manifest(tmp_path: Path) -> None:
    from training_preflight import run_preflight_checks

    _make_image(tmp_path / "images" / "rain.jpg")
    _make_image(tmp_path / "images" / "sunny.jpg")
    train_csv = tmp_path / "train.csv"
    train_csv.write_text(
        "image,label,teacher_rain,teacher_sunny\n"
        "rain.jpg,rain,0.9,0.1\n"
        "sunny.jpg,sunny,0.2,0.8\n",
        encoding="utf-8",
    )
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "train:\n"
        "  distillation_alpha: 0.2\n",
        encoding="utf-8",
    )

    result = run_preflight_checks(
        config_path=config_path,
        train_csv=train_csv,
        image_root=tmp_path / "images",
        min_images_per_class=1,
    )

    assert result["status"] == "pass"
    assert result["rows"] == 2
    assert result["classes"] == ["rain", "sunny"]
    assert result["teacher_distillation"] == {"enabled": True, "rows_with_teacher": 2}


def test_preflight_rejects_labels_outside_expected_classes(tmp_path: Path) -> None:
    import pytest

    from training_preflight import run_preflight_checks

    _make_image(tmp_path / "images" / "rain.jpg")
    _make_image(tmp_path / "images" / "rainbow.jpg")
    train_csv = tmp_path / "train.csv"
    train_csv.write_text(
        "image,label\n"
        "rain.jpg,rain\n"
        "rainbow.jpg,rainbow\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="unexpected classes"):
        run_preflight_checks(
            config_path=Path("configs/convnext_tiny.yaml"),
            train_csv=train_csv,
            image_root=tmp_path / "images",
            expected_classes=["rain", "snow"],
        )


def test_preflight_rejects_missing_images_and_duplicate_ids(tmp_path: Path) -> None:
    import pytest

    from training_preflight import run_preflight_checks

    _make_image(tmp_path / "images" / "rain.jpg")
    train_csv = tmp_path / "train.csv"
    train_csv.write_text(
        "image,label\n"
        "rain.jpg,rain\n"
        "rain.jpg,rain\n"
        "missing.jpg,rain\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="duplicate image_id"):
        run_preflight_checks(
            config_path=Path("configs/convnext_tiny.yaml"),
            train_csv=train_csv,
            image_root=tmp_path / "images",
            check_unique_image_id=True,
        )

    with pytest.raises(ValueError, match="missing image files"):
        run_preflight_checks(
            config_path=Path("configs/convnext_tiny.yaml"),
            train_csv=train_csv,
            image_root=tmp_path / "images",
            check_image_exists=True,
        )


def test_preflight_rejects_external_ratio_or_weight_above_gate(tmp_path: Path) -> None:
    import pytest

    from training_preflight import run_preflight_checks

    _make_image(tmp_path / "images" / "labeled.jpg")
    _make_image(tmp_path / "images" / "external.jpg")
    train_csv = tmp_path / "train.csv"
    train_csv.write_text(
        "image,label,source,sample_weight\n"
        "labeled.jpg,rain,labeled,1.0\n"
        "external.jpg,rain,external_weapd,0.5\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="external ratio"):
        run_preflight_checks(
            config_path=Path("configs/convnext_tiny.yaml"),
            train_csv=train_csv,
            image_root=tmp_path / "images",
            allow_external_data=True,
            external_max_ratio=0.25,
        )

    with pytest.raises(ValueError, match="external sample_weight"):
        run_preflight_checks(
            config_path=Path("configs/convnext_tiny.yaml"),
            train_csv=train_csv,
            image_root=tmp_path / "images",
            allow_external_data=True,
            external_max_sample_weight=0.3,
        )


def test_preflight_rejects_ambiguous_csv_and_imagefolder_inputs(tmp_path: Path) -> None:
    import pytest

    from training_preflight import run_preflight_checks

    _make_image(tmp_path / "folder" / "rain" / "rain.jpg")
    _make_image(tmp_path / "images" / "rain.jpg")
    train_csv = tmp_path / "train.csv"
    train_csv.write_text("image,label\nrain.jpg,rain\n", encoding="utf-8")

    with pytest.raises(ValueError, match="either --train-csv or --train-dir"):
        run_preflight_checks(
            config_path=Path("configs/convnext_tiny.yaml"),
            train_csv=train_csv,
            train_dir=tmp_path / "folder",
            image_root=tmp_path / "images",
        )


def test_preflight_rejects_pseudo_rows_outside_confidence_or_ratio_gate(tmp_path: Path) -> None:
    import pytest

    from training_preflight import run_preflight_checks

    _make_image(tmp_path / "images" / "labeled.jpg")
    _make_image(tmp_path / "images" / "pseudo_low.jpg")
    _make_image(tmp_path / "images" / "pseudo_high.jpg")
    train_csv = tmp_path / "train.csv"
    train_csv.write_text(
        "image,label,source,confidence,sample_weight\n"
        "labeled.jpg,rain,labeled,1.0,1.0\n"
        "pseudo_low.jpg,rain,pseudo,0.82,0.874\n"
        "pseudo_high.jpg,rain,pseudo,0.97,0.979\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="pseudo confidence"):
        run_preflight_checks(
            config_path=Path("configs/convnext_tiny.yaml"),
            train_csv=train_csv,
            image_root=tmp_path / "images",
            pseudo_min_confidence=0.9,
        )

    with pytest.raises(ValueError, match="pseudo ratio"):
        run_preflight_checks(
            config_path=Path("configs/convnext_tiny.yaml"),
            train_csv=train_csv,
            image_root=tmp_path / "images",
            pseudo_max_ratio=0.5,
        )


def test_preflight_rejects_teacher_distillation_with_pseudo_or_external_rows(tmp_path: Path) -> None:
    import pytest

    from training_preflight import run_preflight_checks

    _make_image(tmp_path / "images" / "labeled.jpg")
    _make_image(tmp_path / "images" / "pseudo.jpg")
    train_csv = tmp_path / "train.csv"
    train_csv.write_text(
        "image,label,source,confidence,teacher_rain\n"
        "labeled.jpg,rain,labeled,1.0,1.0\n"
        "pseudo.jpg,rain,pseudo,0.97,1.0\n",
        encoding="utf-8",
    )
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "train:\n"
        "  distillation_alpha: 0.3\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="teacher distillation preflight expects labeled rows only"):
        run_preflight_checks(
            config_path=config_path,
            train_csv=train_csv,
            image_root=tmp_path / "images",
        )
