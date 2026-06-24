from pathlib import Path

import json
import pytest
from PIL import Image


def _make_image(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (8, 8), (32, 64, 96)).save(path)


def test_merge_image_folder_and_pseudo_labels_writes_training_csv(tmp_path: Path) -> None:
    from src.weather_net.pseudo_merge import merge_training_with_pseudo_labels

    _make_image(tmp_path / "train" / "rain" / "r0.jpg")
    _make_image(tmp_path / "train" / "sunny" / "s0.jpg")
    _make_image(tmp_path / "unlabeled" / "u0.jpg")
    _make_image(tmp_path / "unlabeled" / "u1.jpg")
    pseudo_csv = tmp_path / "pseudo.csv"
    pseudo_csv.write_text(
        "image,label,confidence\n"
        "u0.jpg,rain,0.970000\n"
        "u1.jpg,sunny,0.910000\n",
        encoding="utf-8",
    )

    output_csv = tmp_path / "merged.csv"
    stats = merge_training_with_pseudo_labels(
        train_dir=tmp_path / "train",
        train_csv=None,
        image_root=None,
        pseudo_csv=pseudo_csv,
        pseudo_image_root=tmp_path / "unlabeled",
        output_csv=output_csv,
        min_confidence=0.95,
    )

    assert stats == {
        "labeled": 2,
        "pseudo": 1,
        "total": 3,
        "skipped_duplicates": 0,
        "skipped_labeled_duplicates": 0,
        "skipped_duplicate_pseudo": 0,
    }
    assert output_csv.read_text(encoding="utf-8") == (
        "image,label,source,confidence,sample_weight\n"
        f"{tmp_path / 'train' / 'rain' / 'r0.jpg'},rain,labeled,1.000000,1.000000\n"
        f"{tmp_path / 'train' / 'sunny' / 's0.jpg'},sunny,labeled,1.000000,1.000000\n"
        f"{tmp_path / 'unlabeled' / 'u0.jpg'},rain,pseudo,0.970000,0.979000\n"
    )


def test_merge_skips_pseudo_rows_that_duplicate_labeled_paths(tmp_path: Path) -> None:
    from src.weather_net.pseudo_merge import merge_training_with_pseudo_labels

    duplicate = tmp_path / "train" / "rain" / "same.jpg"
    _make_image(duplicate)
    pseudo_csv = tmp_path / "pseudo.csv"
    pseudo_csv.write_text(
        "image,label,confidence\n"
        "rain/same.jpg,rain,0.990000\n",
        encoding="utf-8",
    )

    output_csv = tmp_path / "merged.csv"
    stats = merge_training_with_pseudo_labels(
        train_dir=tmp_path / "train",
        train_csv=None,
        image_root=None,
        pseudo_csv=pseudo_csv,
        pseudo_image_root=tmp_path / "train",
        output_csv=output_csv,
        min_confidence=0.95,
    )

    assert stats == {
        "labeled": 1,
        "pseudo": 0,
        "total": 1,
        "skipped_duplicates": 1,
        "skipped_labeled_duplicates": 1,
        "skipped_duplicate_pseudo": 0,
    }
    assert output_csv.read_text(encoding="utf-8").count("same.jpg") == 1


def test_merge_deduplicates_pseudo_rows_by_highest_confidence(tmp_path: Path) -> None:
    from src.weather_net.pseudo_merge import merge_training_with_pseudo_labels

    _make_image(tmp_path / "train" / "rain" / "r0.jpg")
    _make_image(tmp_path / "unlabeled" / "same.jpg")
    pseudo_csv = tmp_path / "pseudo.csv"
    pseudo_csv.write_text(
        "image,label,confidence\n"
        "same.jpg,rain,0.960000\n"
        "same.jpg,rain,0.990000\n",
        encoding="utf-8",
    )

    output_csv = tmp_path / "merged.csv"
    stats = merge_training_with_pseudo_labels(
        train_dir=tmp_path / "train",
        train_csv=None,
        image_root=None,
        pseudo_csv=pseudo_csv,
        pseudo_image_root=tmp_path / "unlabeled",
        output_csv=output_csv,
        min_confidence=0.95,
    )

    assert stats == {
        "labeled": 1,
        "pseudo": 1,
        "total": 2,
        "skipped_duplicates": 1,
        "skipped_labeled_duplicates": 0,
        "skipped_duplicate_pseudo": 1,
    }
    assert output_csv.read_text(encoding="utf-8").count("same.jpg") == 1
    assert f"{tmp_path / 'unlabeled' / 'same.jpg'},rain,pseudo,0.990000,0.993000" in output_csv.read_text(
        encoding="utf-8"
    )


def test_merge_rejects_conflicting_pseudo_duplicates(tmp_path: Path) -> None:
    from src.weather_net.pseudo_merge import merge_training_with_pseudo_labels

    _make_image(tmp_path / "train" / "rain" / "r0.jpg")
    _make_image(tmp_path / "train" / "sunny" / "s0.jpg")
    _make_image(tmp_path / "unlabeled" / "same.jpg")
    pseudo_csv = tmp_path / "pseudo.csv"
    pseudo_csv.write_text(
        "image,label,confidence\n"
        "same.jpg,rain,0.990000\n"
        "same.jpg,sunny,0.980000\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Conflicting pseudo labels"):
        merge_training_with_pseudo_labels(
            train_dir=tmp_path / "train",
            train_csv=None,
            image_root=None,
            pseudo_csv=pseudo_csv,
            pseudo_image_root=tmp_path / "unlabeled",
            output_csv=tmp_path / "merged.csv",
            min_confidence=0.95,
        )


def test_merge_rejects_pseudo_labels_outside_labeled_classes(tmp_path: Path) -> None:
    from src.weather_net.pseudo_merge import merge_training_with_pseudo_labels

    _make_image(tmp_path / "train" / "rain" / "r0.jpg")
    _make_image(tmp_path / "unlabeled" / "u0.jpg")
    pseudo_csv = tmp_path / "pseudo.csv"
    pseudo_csv.write_text(
        "image,label,confidence\n"
        "u0.jpg,snow_typo,0.990000\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="unknown labels"):
        merge_training_with_pseudo_labels(
            train_dir=tmp_path / "train",
            train_csv=None,
            image_root=None,
            pseudo_csv=pseudo_csv,
            pseudo_image_root=tmp_path / "unlabeled",
            output_csv=tmp_path / "merged.csv",
            min_confidence=0.95,
        )


@pytest.mark.parametrize("confidence", ["nan", "inf", "1.200000", "-0.100000"])
def test_merge_rejects_invalid_pseudo_confidence(tmp_path: Path, confidence: str) -> None:
    from src.weather_net.pseudo_merge import merge_training_with_pseudo_labels

    _make_image(tmp_path / "train" / "rain" / "r0.jpg")
    _make_image(tmp_path / "unlabeled" / "u0.jpg")
    pseudo_csv = tmp_path / "pseudo.csv"
    pseudo_csv.write_text(
        "image,label,confidence\n"
        f"u0.jpg,rain,{confidence}\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=r"confidence must be finite and in \[0, 1\]"):
        merge_training_with_pseudo_labels(
            train_dir=tmp_path / "train",
            train_csv=None,
            image_root=None,
            pseudo_csv=pseudo_csv,
            pseudo_image_root=tmp_path / "unlabeled",
            output_csv=tmp_path / "merged.csv",
            min_confidence=0.95,
        )


def test_merge_preserves_labeled_sample_weight_from_csv(tmp_path: Path) -> None:
    from src.weather_net.pseudo_merge import merge_training_with_pseudo_labels

    _make_image(tmp_path / "images" / "rain.jpg")
    _make_image(tmp_path / "images" / "pseudo.jpg")
    train_csv = tmp_path / "train.csv"
    train_csv.write_text(
        "image,label,sample_weight\n"
        "rain.jpg,rain,0.750000\n",
        encoding="utf-8",
    )
    pseudo_csv = tmp_path / "pseudo.csv"
    pseudo_csv.write_text(
        "image,label,confidence\n"
        "pseudo.jpg,rain,0.960000\n",
        encoding="utf-8",
    )

    output_csv = tmp_path / "merged.csv"
    merge_training_with_pseudo_labels(
        train_dir=None,
        train_csv=train_csv,
        image_root=tmp_path / "images",
        pseudo_csv=pseudo_csv,
        pseudo_image_root=tmp_path / "images",
        output_csv=output_csv,
        min_confidence=0.95,
    )

    assert output_csv.read_text(encoding="utf-8") == (
        "image,label,source,confidence,sample_weight\n"
        f"{tmp_path / 'images' / 'rain.jpg'},rain,labeled,1.000000,0.750000\n"
        f"{tmp_path / 'images' / 'pseudo.jpg'},rain,pseudo,0.960000,0.972000\n"
    )


def test_merge_pseudo_teacher_columns_requires_explicit_allowance(tmp_path: Path) -> None:
    from src.weather_net.pseudo_merge import merge_training_with_pseudo_labels

    _make_image(tmp_path / "images" / "rain.jpg")
    _make_image(tmp_path / "images" / "pseudo.jpg")
    train_csv = tmp_path / "train.csv"
    train_csv.write_text("image,label\nrain.jpg,rain\n", encoding="utf-8")
    pseudo_csv = tmp_path / "pseudo.csv"
    pseudo_csv.write_text(
        "image,label,confidence,teacher_rain\n"
        "pseudo.jpg,rain,0.970000,1.0\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="pseudo teacher columns require explicit allowance"):
        merge_training_with_pseudo_labels(
            train_dir=None,
            train_csv=train_csv,
            image_root=tmp_path / "images",
            pseudo_csv=pseudo_csv,
            pseudo_image_root=tmp_path / "images",
            output_csv=tmp_path / "merged.csv",
            min_confidence=0.95,
        )


def test_merge_pseudo_teacher_columns_when_allowed(tmp_path: Path) -> None:
    from src.weather_net.pseudo_merge import merge_training_with_pseudo_labels

    _make_image(tmp_path / "images" / "rain.jpg")
    _make_image(tmp_path / "images" / "sunny.jpg")
    _make_image(tmp_path / "images" / "pseudo.jpg")
    train_csv = tmp_path / "train.csv"
    train_csv.write_text("image,label\nrain.jpg,rain\nsunny.jpg,sunny\n", encoding="utf-8")
    pseudo_csv = tmp_path / "pseudo.csv"
    pseudo_csv.write_text(
        "image,label,confidence,teacher_rain,teacher_sunny\n"
        "pseudo.jpg,rain,0.970000,0.91,0.09\n",
        encoding="utf-8",
    )

    output_csv = tmp_path / "merged.csv"
    stats = merge_training_with_pseudo_labels(
        train_dir=None,
        train_csv=train_csv,
        image_root=tmp_path / "images",
        pseudo_csv=pseudo_csv,
        pseudo_image_root=tmp_path / "images",
        output_csv=output_csv,
        min_confidence=0.95,
        allow_pseudo_teacher=True,
    )

    assert stats["pseudo"] == 1
    assert output_csv.read_text(encoding="utf-8") == (
        "image,label,source,confidence,sample_weight,teacher_rain,teacher_sunny\n"
        f"{tmp_path / 'images' / 'rain.jpg'},rain,labeled,1.000000,1.000000,1.00000000,0.00000000\n"
        f"{tmp_path / 'images' / 'sunny.jpg'},sunny,labeled,1.000000,1.000000,0.00000000,1.00000000\n"
        f"{tmp_path / 'images' / 'pseudo.jpg'},rain,pseudo,0.970000,0.979000,0.91000000,0.09000000\n"
    )


def test_merge_preserves_labeled_soft_teacher_columns_when_allowed(tmp_path: Path) -> None:
    from src.weather_net.pseudo_merge import merge_training_with_pseudo_labels

    _make_image(tmp_path / "images" / "rain.jpg")
    _make_image(tmp_path / "images" / "sunny.jpg")
    _make_image(tmp_path / "images" / "pseudo.jpg")
    train_csv = tmp_path / "train_teacher.csv"
    train_csv.write_text(
        "image,label,teacher_rain,teacher_sunny\n"
        "rain.jpg,rain,0.82,0.18\n"
        "sunny.jpg,sunny,0.07,0.93\n",
        encoding="utf-8",
    )
    pseudo_csv = tmp_path / "pseudo.csv"
    pseudo_csv.write_text(
        "image,label,confidence,teacher_rain,teacher_sunny\n"
        "pseudo.jpg,rain,0.970000,0.91,0.09\n",
        encoding="utf-8",
    )

    output_csv = tmp_path / "merged.csv"
    merge_training_with_pseudo_labels(
        train_dir=None,
        train_csv=train_csv,
        image_root=tmp_path / "images",
        pseudo_csv=pseudo_csv,
        pseudo_image_root=tmp_path / "images",
        output_csv=output_csv,
        min_confidence=0.95,
        allow_pseudo_teacher=True,
    )

    assert output_csv.read_text(encoding="utf-8") == (
        "image,label,source,confidence,sample_weight,teacher_rain,teacher_sunny\n"
        f"{tmp_path / 'images' / 'rain.jpg'},rain,labeled,1.000000,1.000000,0.82000000,0.18000000\n"
        f"{tmp_path / 'images' / 'sunny.jpg'},sunny,labeled,1.000000,1.000000,0.07000000,0.93000000\n"
        f"{tmp_path / 'images' / 'pseudo.jpg'},rain,pseudo,0.970000,0.979000,0.91000000,0.09000000\n"
    )


def test_merge_writes_pseudo_teacher_audit_json(tmp_path: Path) -> None:
    from src.weather_net.pseudo_merge import merge_training_with_pseudo_labels

    _make_image(tmp_path / "images" / "rain.jpg")
    _make_image(tmp_path / "images" / "sunny.jpg")
    _make_image(tmp_path / "images" / "pseudo_rain.jpg")
    _make_image(tmp_path / "images" / "pseudo_sunny.jpg")
    train_csv = tmp_path / "train.csv"
    train_csv.write_text("image,label\nrain.jpg,rain\nsunny.jpg,sunny\n", encoding="utf-8")
    pseudo_csv = tmp_path / "pseudo.csv"
    pseudo_csv.write_text(
        "image,label,confidence,teacher_rain,teacher_sunny\n"
        "pseudo_rain.jpg,rain,0.970000,0.91,0.09\n"
        "pseudo_sunny.jpg,sunny,0.980000,0.12,0.88\n",
        encoding="utf-8",
    )

    audit_json = tmp_path / "merge_audit.json"
    stats = merge_training_with_pseudo_labels(
        train_dir=None,
        train_csv=train_csv,
        image_root=tmp_path / "images",
        pseudo_csv=pseudo_csv,
        pseudo_image_root=tmp_path / "images",
        output_csv=tmp_path / "merged.csv",
        min_confidence=0.95,
        allow_pseudo_teacher=True,
        audit_json=audit_json,
    )

    audit = json.loads(audit_json.read_text(encoding="utf-8"))
    assert audit["stats"] == stats
    assert audit["class_names"] == ["rain", "sunny"]
    assert audit["pseudo_per_class"] == {"rain": 1, "sunny": 1}
    assert audit["pseudo_teacher"]["rows"] == 2
    assert audit["pseudo_teacher"]["mean_top1_probability"] == pytest.approx(0.895)
    assert audit["pseudo_teacher"]["mean_margin"] == pytest.approx(0.79)


def test_merge_rejects_extra_pseudo_teacher_class(tmp_path: Path) -> None:
    from src.weather_net.pseudo_merge import merge_training_with_pseudo_labels

    _make_image(tmp_path / "images" / "rain.jpg")
    _make_image(tmp_path / "images" / "sunny.jpg")
    _make_image(tmp_path / "images" / "pseudo.jpg")
    train_csv = tmp_path / "train.csv"
    train_csv.write_text("image,label\nrain.jpg,rain\nsunny.jpg,sunny\n", encoding="utf-8")
    pseudo_csv = tmp_path / "pseudo.csv"
    pseudo_csv.write_text(
        "image,label,confidence,teacher_rain,teacher_sunny,teacher_dew\n"
        "pseudo.jpg,rain,0.970000,0.91,0.09,0.00\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="unknown classes"):
        merge_training_with_pseudo_labels(
            train_dir=None,
            train_csv=train_csv,
            image_root=tmp_path / "images",
            pseudo_csv=pseudo_csv,
            pseudo_image_root=tmp_path / "images",
            output_csv=tmp_path / "merged.csv",
            min_confidence=0.95,
            allow_pseudo_teacher=True,
        )


def test_merge_rejects_pseudo_teacher_top1_disagreement(tmp_path: Path) -> None:
    from src.weather_net.pseudo_merge import merge_training_with_pseudo_labels

    _make_image(tmp_path / "images" / "rain.jpg")
    _make_image(tmp_path / "images" / "sunny.jpg")
    _make_image(tmp_path / "images" / "pseudo.jpg")
    train_csv = tmp_path / "train.csv"
    train_csv.write_text("image,label\nrain.jpg,rain\nsunny.jpg,sunny\n", encoding="utf-8")
    pseudo_csv = tmp_path / "pseudo.csv"
    pseudo_csv.write_text(
        "image,label,confidence,teacher_rain,teacher_sunny\n"
        "pseudo.jpg,rain,0.970000,0.10,0.90\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="teacher top1 must agree with pseudo label"):
        merge_training_with_pseudo_labels(
            train_dir=None,
            train_csv=train_csv,
            image_root=tmp_path / "images",
            pseudo_csv=pseudo_csv,
            pseudo_image_root=tmp_path / "images",
            output_csv=tmp_path / "merged.csv",
            min_confidence=0.95,
            allow_pseudo_teacher=True,
        )
