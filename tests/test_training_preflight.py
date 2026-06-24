from pathlib import Path

import pytest
from PIL import Image


def _make_image(path: Path, color: tuple[int, int, int] = (120, 120, 120)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (8, 8), color).save(path)


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


def test_preflight_requires_explicit_sample_weight_for_external_rows(tmp_path: Path) -> None:
    from training_preflight import run_preflight_checks

    _make_image(tmp_path / "images" / "labeled.jpg")
    _make_image(tmp_path / "images" / "external.jpg")
    train_csv = tmp_path / "train.csv"
    train_csv.write_text(
        "image,label,source\n"
        "labeled.jpg,rain,labeled\n"
        "external.jpg,rain,external_road_weather_time\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="explicit sample_weight"):
        run_preflight_checks(
            config_path=Path("configs/convnext_tiny.yaml"),
            train_csv=train_csv,
            image_root=tmp_path / "images",
            allow_external_data=True,
        )


def test_server_strict_requires_external_merge_audit(tmp_path: Path) -> None:
    import json

    from training_preflight import run_preflight_checks

    _make_image(tmp_path / "images" / "labeled1.jpg", color=(1, 2, 3))
    _make_image(tmp_path / "images" / "labeled2.jpg", color=(2, 3, 4))
    _make_image(tmp_path / "images" / "external.jpg", color=(3, 4, 5))
    train_csv = tmp_path / "train.csv"
    train_csv.write_text(
        "image,label,source,sample_weight\n"
        "labeled1.jpg,rain,labeled,1.0\n"
        "labeled2.jpg,rain,labeled,1.0\n"
        "external.jpg,rain,external_road_weather_time,0.3\n",
        encoding="utf-8",
    )
    class_map = tmp_path / "class_to_idx.json"
    class_map.write_text(json.dumps({"rain": 0}), encoding="utf-8")

    with pytest.raises(ValueError, match="external merge audit"):
        run_preflight_checks(
            config_path=Path("configs/convnext_tiny.yaml"),
            profile="server-strict",
            train_csv=train_csv,
            image_root=tmp_path / "images",
            class_map=class_map,
            allow_external_data=True,
        )


def test_custom_preflight_allows_external_manifest_audit_without_merge_audit(tmp_path: Path) -> None:
    from training_preflight import run_preflight_checks

    _make_image(tmp_path / "images" / "labeled.jpg", color=(1, 2, 3))
    _make_image(tmp_path / "images" / "external.jpg", color=(3, 4, 5))
    train_csv = tmp_path / "train.csv"
    train_csv.write_text(
        "image,label,source,sample_weight\n"
        "labeled.jpg,rain,labeled,1.0\n"
        "external.jpg,rain,external_road_weather_time,0.3\n",
        encoding="utf-8",
    )

    result = run_preflight_checks(
        config_path=Path("configs/convnext_tiny.yaml"),
        profile="custom",
        train_csv=train_csv,
        image_root=tmp_path / "images",
        allow_external_data=True,
        external_max_ratio=0.5,
        external_max_sample_weight=0.5,
    )

    assert result["external_audit"] is None
    assert result["source_counts"] == {"external_road_weather_time": 1, "labeled": 1}


def test_preflight_rejects_external_test_split_paths(tmp_path: Path) -> None:
    from training_preflight import run_preflight_checks

    _make_image(tmp_path / "images" / "train.jpg")
    _make_image(tmp_path / "images" / "test_dataset" / "test_images" / "leak.jpg")
    train_csv = tmp_path / "train.csv"
    train_csv.write_text(
        "image,label,source,sample_weight\n"
        "train.jpg,rain,labeled,1.0\n"
        "test_dataset/test_images/leak.jpg,rain,external_road_weather_time,0.3\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="reserved test split"):
        run_preflight_checks(
            config_path=Path("configs/convnext_tiny.yaml"),
            train_csv=train_csv,
            image_root=tmp_path / "images",
            allow_external_data=True,
            external_max_ratio=0.5,
            external_max_sample_weight=0.5,
        )


def test_preflight_rejects_external_alien_test_paths(tmp_path: Path) -> None:
    from training_preflight import run_preflight_checks

    _make_image(tmp_path / "images" / "train.jpg")
    _make_image(tmp_path / "images" / "alien_test" / "alien.jpg")
    train_csv = tmp_path / "train.csv"
    train_csv.write_text(
        "image,label,source,sample_weight\n"
        "train.jpg,rain,labeled,1.0\n"
        "alien_test/alien.jpg,rain,external_vijay_multiclass_weather,0.2\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="reserved test split"):
        run_preflight_checks(
            config_path=Path("configs/convnext_tiny.yaml"),
            train_csv=train_csv,
            image_root=tmp_path / "images",
            allow_external_data=True,
            external_max_ratio=0.5,
            external_max_sample_weight=0.5,
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
    assert result["profile"] == "custom"
    assert result["rows"] == 2
    assert result["classes"] == ["rain", "sunny"]
    assert result["teacher_distillation"] == {"enabled": True, "rows_with_teacher": 2}


def test_preflight_uses_class_map_when_loading_csv(tmp_path: Path) -> None:
    import json

    from training_preflight import run_preflight_checks

    _make_image(tmp_path / "images" / "rain.jpg", color=(10, 20, 30))
    _make_image(tmp_path / "images" / "sunny.jpg", color=(30, 20, 10))
    train_csv = tmp_path / "train.csv"
    train_csv.write_text(
        "image,label\n"
        "rain.jpg,rain\n"
        "sunny.jpg,sunny\n",
        encoding="utf-8",
    )
    class_map = tmp_path / "class_to_idx.json"
    class_map.write_text(json.dumps({"sunny": 0, "rain": 1}), encoding="utf-8")

    result = run_preflight_checks(
        config_path=Path("configs/convnext_tiny.yaml"),
        train_csv=train_csv,
        image_root=tmp_path / "images",
        class_map=class_map,
    )

    assert result["classes"] == ["sunny", "rain"]


def test_server_strict_profile_requires_class_map_for_csv(tmp_path: Path) -> None:
    from training_preflight import run_preflight_checks

    _make_image(tmp_path / "images" / "rain.jpg")
    train_csv = tmp_path / "train.csv"
    train_csv.write_text("image,label\nrain.jpg,rain\n", encoding="utf-8")

    with pytest.raises(ValueError, match="requires data.class_map or --class-map"):
        run_preflight_checks(
            config_path=Path("configs/convnext_tiny.yaml"),
            profile="server-strict",
            train_csv=train_csv,
            image_root=tmp_path / "images",
        )


def test_server_strict_profile_enables_high_value_gates(tmp_path: Path) -> None:
    import json

    from training_preflight import run_preflight_checks

    samples = [
        ("rain1.jpg", "rain", (10, 20, 30)),
        ("rain2.jpg", "rain", (11, 20, 30)),
        ("sunny1.jpg", "sunny", (30, 20, 10)),
        ("sunny2.jpg", "sunny", (30, 21, 10)),
    ]
    for image_name, _label, color in samples:
        _make_image(tmp_path / "images" / image_name, color=color)
    train_csv = tmp_path / "train.csv"
    train_csv.write_text(
        "image,label\n" + "".join(f"{image_name},{label}\n" for image_name, label, _color in samples),
        encoding="utf-8",
    )
    class_map = tmp_path / "class_to_idx.json"
    class_map.write_text(json.dumps({"rain": 0, "sunny": 1}), encoding="utf-8")

    result = run_preflight_checks(
        config_path=Path("configs/convnext_tiny.yaml"),
        profile="server-strict",
        train_csv=train_csv,
        image_root=tmp_path / "images",
        class_map=class_map,
    )

    assert result["profile"] == "server-strict"
    assert result["image_integrity"] == {
        "checked_exists": True,
        "checked_readable_images": True,
        "checked_unique_image_id": True,
        "checked_unique_realpath": True,
        "checked_unique_image_hash": True,
    }
    assert result["labeled_label_counts"] == {"rain": 2, "sunny": 2}


def test_server_strict_rejects_labeled_support_below_requested_folds(tmp_path: Path) -> None:
    import json

    from training_preflight import run_preflight_checks

    samples = [
        ("rain1.jpg", "rain", (10, 20, 30)),
        ("rain2.jpg", "rain", (11, 20, 30)),
        ("sunny1.jpg", "sunny", (30, 20, 10)),
        ("sunny2.jpg", "sunny", (30, 21, 10)),
    ]
    for image_name, _label, color in samples:
        _make_image(tmp_path / "images" / image_name, color=color)
    train_csv = tmp_path / "train.csv"
    train_csv.write_text(
        "image,label,source\n" + "".join(f"{image_name},{label},labeled\n" for image_name, label, _color in samples),
        encoding="utf-8",
    )
    class_map = tmp_path / "class_to_idx.json"
    class_map.write_text(json.dumps({"rain": 0, "sunny": 1}), encoding="utf-8")
    config_path = tmp_path / "config.yaml"
    config_path.write_text("data:\n  folds: 3\n", encoding="utf-8")

    with pytest.raises(ValueError, match="labeled classes below requested folds"):
        run_preflight_checks(
            config_path=config_path,
            profile="server-strict",
            train_csv=train_csv,
            image_root=tmp_path / "images",
            class_map=class_map,
        )


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


def test_preflight_rejects_unreadable_images_when_requested(tmp_path: Path) -> None:
    from training_preflight import run_preflight_checks

    _make_image(tmp_path / "images" / "rain.jpg")
    bad_image = tmp_path / "images" / "bad.jpg"
    bad_image.parent.mkdir(parents=True, exist_ok=True)
    bad_image.write_bytes(b"not an image")
    train_csv = tmp_path / "train.csv"
    train_csv.write_text(
        "image,label\n"
        "rain.jpg,rain\n"
        "bad.jpg,rain\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="unreadable image files"):
        run_preflight_checks(
            config_path=Path("configs/convnext_tiny.yaml"),
            train_csv=train_csv,
            image_root=tmp_path / "images",
            check_readable_images=True,
        )


def test_preflight_rejects_duplicate_content_hash_across_sources(tmp_path: Path) -> None:
    from training_preflight import run_preflight_checks

    _make_image(tmp_path / "images" / "labeled.jpg")
    _make_image(tmp_path / "images" / "pseudo_copy.jpg")
    (tmp_path / "images" / "pseudo_copy.jpg").write_bytes((tmp_path / "images" / "labeled.jpg").read_bytes())
    train_csv = tmp_path / "train.csv"
    train_csv.write_text(
        "image,label,source,confidence\n"
        "labeled.jpg,rain,labeled,1.0\n"
        "pseudo_copy.jpg,rain,pseudo,0.97\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="duplicate image content hashes"):
        run_preflight_checks(
            config_path=Path("configs/convnext_tiny.yaml"),
            train_csv=train_csv,
            image_root=tmp_path / "images",
            check_unique_image_hash=True,
            pseudo_min_confidence=0.95,
            pseudo_max_ratio=1.0,
        )


def test_preflight_hash_check_reports_missing_images_cleanly(tmp_path: Path) -> None:
    from training_preflight import run_preflight_checks

    train_csv = tmp_path / "train.csv"
    train_csv.write_text(
        "image,label\n"
        "missing.jpg,rain\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="missing image files"):
        run_preflight_checks(
            config_path=Path("configs/convnext_tiny.yaml"),
            train_csv=train_csv,
            image_root=tmp_path / "images",
            check_unique_image_hash=True,
        )


def test_preflight_rejects_low_labeled_class_support_even_when_pseudo_exists(tmp_path: Path) -> None:
    from training_preflight import run_preflight_checks

    _make_image(tmp_path / "images" / "rain_labeled.jpg")
    _make_image(tmp_path / "images" / "rain_pseudo.jpg")
    _make_image(tmp_path / "images" / "sunny_pseudo.jpg")
    train_csv = tmp_path / "train.csv"
    train_csv.write_text(
        "image,label,source,confidence\n"
        "rain_labeled.jpg,rain,labeled,1.0\n"
        "rain_pseudo.jpg,rain,pseudo,0.98\n"
        "sunny_pseudo.jpg,sunny,pseudo,0.98\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="labeled classes below min_labeled_images_per_class"):
        run_preflight_checks(
            config_path=Path("configs/convnext_tiny.yaml"),
            train_csv=train_csv,
            image_root=tmp_path / "images",
            min_labeled_images_per_class=1,
            pseudo_min_confidence=0.95,
            pseudo_max_ratio=2.0,
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

    with pytest.raises(ValueError, match="teacher distillation preflight found disallowed sources"):
        run_preflight_checks(
            config_path=config_path,
            train_csv=train_csv,
            image_root=tmp_path / "images",
        )


def test_preflight_allows_pseudo_teacher_distillation_with_explicit_gates(tmp_path: Path) -> None:
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

    result = run_preflight_checks(
        config_path=config_path,
        train_csv=train_csv,
        image_root=tmp_path / "images",
        allow_pseudo_teacher_distillation=True,
        pseudo_min_confidence=0.95,
        pseudo_max_ratio=1.0,
    )

    assert result["teacher_distillation"]["enabled"] is True
    assert result["source_counts"] == {"labeled": 1, "pseudo": 1}


@pytest.mark.parametrize("confidence", ["nan", "inf", "1.2", "-0.1"])
def test_preflight_rejects_invalid_pseudo_confidence(tmp_path: Path, confidence: str) -> None:
    from training_preflight import run_preflight_checks

    _make_image(tmp_path / "images" / "labeled.jpg")
    _make_image(tmp_path / "images" / "pseudo.jpg")
    train_csv = tmp_path / "train.csv"
    train_csv.write_text(
        "image,label,source,confidence\n"
        "labeled.jpg,rain,labeled,1.0\n"
        f"pseudo.jpg,rain,pseudo,{confidence}\n",
        encoding="utf-8",
    )
    config_path = tmp_path / "config.yaml"
    config_path.write_text("train:\n  distillation_alpha: 0.0\n", encoding="utf-8")

    with pytest.raises(ValueError, match=r"confidence must be finite and in \[0, 1\]"):
        run_preflight_checks(
            config_path=config_path,
            train_csv=train_csv,
            image_root=tmp_path / "images",
        )


def test_preflight_rejects_pseudo_teacher_top1_disagreement(tmp_path: Path) -> None:
    from training_preflight import run_preflight_checks

    _make_image(tmp_path / "images" / "labeled.jpg")
    _make_image(tmp_path / "images" / "pseudo.jpg")
    _make_image(tmp_path / "images" / "sunny.jpg")
    train_csv = tmp_path / "train.csv"
    train_csv.write_text(
        "image,label,source,confidence,teacher_rain,teacher_sunny\n"
        "labeled.jpg,rain,labeled,1.0,0.9,0.1\n"
        "sunny.jpg,sunny,labeled,1.0,0.1,0.9\n"
        "pseudo.jpg,rain,pseudo,0.97,0.1,0.9\n",
        encoding="utf-8",
    )
    config_path = tmp_path / "config.yaml"
    config_path.write_text("train:\n  distillation_alpha: 0.3\n", encoding="utf-8")

    with pytest.raises(ValueError, match="pseudo teacher top1 differs from pseudo label"):
        run_preflight_checks(
            config_path=config_path,
            train_csv=train_csv,
            image_root=tmp_path / "images",
            allow_pseudo_teacher_distillation=True,
            pseudo_min_confidence=0.95,
            pseudo_max_ratio=1.0,
        )
