from pathlib import Path

import numpy as np
import pytest
from PIL import Image


def _make_image(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (8, 8), (120, 120, 120)).save(path)


def _write_oof(path: Path, probs: np.ndarray, image_ids: list[str], class_names: list[str]) -> None:
    np.savez_compressed(
        path,
        probs=probs.astype(np.float32),
        y_true=np.asarray([0, 1], dtype=np.int64),
        image_id=np.asarray(image_ids, dtype=object),
        class_names=np.asarray(class_names, dtype=object),
        source=np.asarray(["labeled" for _ in image_ids], dtype=object),
    )


def test_write_teacher_csv_from_oof_adds_class_probabilities(tmp_path: Path) -> None:
    from src.weather_net.teacher import write_teacher_csv_from_oof

    _make_image(tmp_path / "images" / "rain.jpg")
    _make_image(tmp_path / "images" / "sunny.jpg")
    train_csv = tmp_path / "train.csv"
    train_csv.write_text("image,label\nrain.jpg,rain\nsunny.jpg,sunny\n", encoding="utf-8")
    oof_path = tmp_path / "oof_probabilities.npz"
    _write_oof(
        oof_path,
        probs=np.asarray([[0.8, 0.2], [0.1, 0.9]], dtype=np.float32),
        image_ids=["rain.jpg", "sunny.jpg"],
        class_names=["rain", "sunny"],
    )

    stats = write_teacher_csv_from_oof(
        train_csv=train_csv,
        image_root=tmp_path / "images",
        oof_npz=oof_path,
        output_csv=tmp_path / "train_teacher.csv",
    )

    output = (tmp_path / "train_teacher.csv").read_text(encoding="utf-8")
    assert output.splitlines()[0] == "image,label,teacher_rain,teacher_sunny"
    assert "rain.jpg,rain,0.80000000,0.20000000" in output
    assert "sunny.jpg,sunny,0.10000000,0.90000000" in output
    assert stats["rows"] == 2
    assert stats["classes"] == ["rain", "sunny"]
    assert stats["output"] == str(tmp_path / "train_teacher.csv")
    assert stats["teacher_quality"]["top1_mismatch_rate"] == 0.0


def test_write_teacher_csv_from_oof_preserves_existing_columns(tmp_path: Path) -> None:
    from src.weather_net.teacher import write_teacher_csv_from_oof

    _make_image(tmp_path / "images" / "rain.jpg")
    _make_image(tmp_path / "images" / "sunny.jpg")
    train_csv = tmp_path / "train.csv"
    train_csv.write_text(
        "image,label,fold,source\n"
        "rain.jpg,rain,0,labeled\n"
        "sunny.jpg,sunny,1,labeled\n",
        encoding="utf-8",
    )
    oof_path = tmp_path / "oof_probabilities.npz"
    _write_oof(
        oof_path,
        probs=np.asarray([[0.7, 0.3], [0.2, 0.8]], dtype=np.float32),
        image_ids=["rain.jpg", "sunny.jpg"],
        class_names=["rain", "sunny"],
    )

    write_teacher_csv_from_oof(
        train_csv=train_csv,
        image_root=tmp_path / "images",
        oof_npz=oof_path,
        output_csv=tmp_path / "train_teacher.csv",
    )

    header = (tmp_path / "train_teacher.csv").read_text(encoding="utf-8").splitlines()[0]
    assert header == "image,label,fold,source,teacher_rain,teacher_sunny"


def test_write_teacher_csv_from_oof_rejects_missing_or_duplicate_oof_ids(tmp_path: Path) -> None:
    from src.weather_net.teacher import write_teacher_csv_from_oof

    _make_image(tmp_path / "images" / "rain.jpg")
    _make_image(tmp_path / "images" / "sunny.jpg")
    train_csv = tmp_path / "train.csv"
    train_csv.write_text("image,label\nrain.jpg,rain\nsunny.jpg,sunny\n", encoding="utf-8")
    missing_path = tmp_path / "missing_oof.npz"
    _write_oof(
        missing_path,
        probs=np.asarray([[1.0, 0.0]], dtype=np.float32),
        image_ids=["rain.jpg"],
        class_names=["rain", "sunny"],
    )

    with pytest.raises(ValueError, match="missing OOF teacher probabilities"):
        write_teacher_csv_from_oof(
            train_csv=train_csv,
            image_root=tmp_path / "images",
            oof_npz=missing_path,
            output_csv=tmp_path / "out.csv",
        )

    duplicate_path = tmp_path / "duplicate_oof.npz"
    _write_oof(
        duplicate_path,
        probs=np.asarray([[0.8, 0.2], [0.1, 0.9]], dtype=np.float32),
        image_ids=["rain.jpg", "rain.jpg"],
        class_names=["rain", "sunny"],
    )

    with pytest.raises(ValueError, match="duplicate OOF image_id"):
        write_teacher_csv_from_oof(
            train_csv=train_csv,
            image_root=tmp_path / "images",
            oof_npz=duplicate_path,
            output_csv=tmp_path / "out.csv",
        )


def test_write_teacher_csv_from_oof_rejects_class_mismatch_or_bad_probabilities(tmp_path: Path) -> None:
    from src.weather_net.teacher import write_teacher_csv_from_oof

    _make_image(tmp_path / "images" / "rain.jpg")
    _make_image(tmp_path / "images" / "sunny.jpg")
    train_csv = tmp_path / "train.csv"
    train_csv.write_text("image,label\nrain.jpg,rain\nsunny.jpg,sunny\n", encoding="utf-8")
    mismatch_path = tmp_path / "mismatch_oof.npz"
    _write_oof(
        mismatch_path,
        probs=np.asarray([[0.8, 0.2], [0.1, 0.9]], dtype=np.float32),
        image_ids=["rain.jpg", "sunny.jpg"],
        class_names=["sunny", "rain"],
    )

    with pytest.raises(ValueError, match="class_names"):
        write_teacher_csv_from_oof(
            train_csv=train_csv,
            image_root=tmp_path / "images",
            oof_npz=mismatch_path,
            output_csv=tmp_path / "out.csv",
        )

    bad_probs_path = tmp_path / "bad_probs_oof.npz"
    _write_oof(
        bad_probs_path,
        probs=np.asarray([[0.8, 0.8], [0.1, 0.9]], dtype=np.float32),
        image_ids=["rain.jpg", "sunny.jpg"],
        class_names=["rain", "sunny"],
    )

    with pytest.raises(ValueError, match="sum to 1"):
        write_teacher_csv_from_oof(
            train_csv=train_csv,
            image_root=tmp_path / "images",
            oof_npz=bad_probs_path,
            output_csv=tmp_path / "out.csv",
        )


def test_write_teacher_csv_from_oof_rejects_pseudo_or_external_training_rows(tmp_path: Path) -> None:
    from src.weather_net.teacher import write_teacher_csv_from_oof

    _make_image(tmp_path / "images" / "rain.jpg")
    _make_image(tmp_path / "images" / "sunny.jpg")
    train_csv = tmp_path / "train.csv"
    train_csv.write_text(
        "image,label,source,confidence\n"
        "rain.jpg,rain,labeled,1.0\n"
        "sunny.jpg,sunny,pseudo,0.97\n",
        encoding="utf-8",
    )
    oof_path = tmp_path / "oof_probabilities.npz"
    _write_oof(
        oof_path,
        probs=np.asarray([[0.8, 0.2], [0.1, 0.9]], dtype=np.float32),
        image_ids=["rain.jpg", "sunny.jpg"],
        class_names=["rain", "sunny"],
    )

    with pytest.raises(ValueError, match="teacher CSV must be built from labeled rows only"):
        write_teacher_csv_from_oof(
            train_csv=train_csv,
            image_root=tmp_path / "images",
            oof_npz=oof_path,
            output_csv=tmp_path / "out.csv",
        )


def test_write_teacher_csv_from_oof_rejects_low_quality_teacher(tmp_path: Path) -> None:
    from src.weather_net.teacher import write_teacher_csv_from_oof

    _make_image(tmp_path / "images" / "rain.jpg")
    _make_image(tmp_path / "images" / "sunny.jpg")
    train_csv = tmp_path / "train.csv"
    train_csv.write_text("image,label\nrain.jpg,rain\nsunny.jpg,sunny\n", encoding="utf-8")
    oof_path = tmp_path / "oof_probabilities.npz"
    _write_oof(
        oof_path,
        probs=np.asarray([[0.1, 0.9], [0.8, 0.2]], dtype=np.float32),
        image_ids=["rain.jpg", "sunny.jpg"],
        class_names=["rain", "sunny"],
    )

    with pytest.raises(ValueError, match="teacher top1 mismatch rate"):
        write_teacher_csv_from_oof(
            train_csv=train_csv,
            image_root=tmp_path / "images",
            oof_npz=oof_path,
            output_csv=tmp_path / "out.csv",
            max_top1_mismatch_rate=0.25,
        )

    with pytest.raises(ValueError, match="mean true-class teacher probability"):
        write_teacher_csv_from_oof(
            train_csv=train_csv,
            image_root=tmp_path / "images",
            oof_npz=oof_path,
            output_csv=tmp_path / "out.csv",
            min_mean_true_probability=0.5,
        )


def test_write_teacher_csv_from_oof_reports_quality_stats(tmp_path: Path) -> None:
    from src.weather_net.teacher import write_teacher_csv_from_oof

    _make_image(tmp_path / "images" / "rain.jpg")
    _make_image(tmp_path / "images" / "sunny.jpg")
    train_csv = tmp_path / "train.csv"
    train_csv.write_text("image,label\nrain.jpg,rain\nsunny.jpg,sunny\n", encoding="utf-8")
    oof_path = tmp_path / "oof_probabilities.npz"
    _write_oof(
        oof_path,
        probs=np.asarray([[0.8, 0.2], [0.1, 0.9]], dtype=np.float32),
        image_ids=["rain.jpg", "sunny.jpg"],
        class_names=["rain", "sunny"],
    )

    stats = write_teacher_csv_from_oof(
        train_csv=train_csv,
        image_root=tmp_path / "images",
        oof_npz=oof_path,
        output_csv=tmp_path / "train_teacher.csv",
        min_samples_per_class=1,
        max_top1_mismatch_rate=0.0,
        min_mean_true_probability=0.8,
    )

    assert stats["teacher_quality"]["top1_mismatch_rate"] == 0.0
    assert stats["teacher_quality"]["mean_true_probability"] == pytest.approx(0.85)
    assert stats["teacher_quality"]["per_class_counts"] == {"rain": 1, "sunny": 1}


def test_teacher_from_oof_cli_writes_output(monkeypatch, tmp_path: Path, capsys) -> None:
    from teacher_from_oof import main

    _make_image(tmp_path / "images" / "rain.jpg")
    _make_image(tmp_path / "images" / "sunny.jpg")
    train_csv = tmp_path / "train.csv"
    train_csv.write_text("image,label\nrain.jpg,rain\nsunny.jpg,sunny\n", encoding="utf-8")
    oof_path = tmp_path / "oof_probabilities.npz"
    _write_oof(
        oof_path,
        probs=np.asarray([[0.8, 0.2], [0.1, 0.9]], dtype=np.float32),
        image_ids=["rain.jpg", "sunny.jpg"],
        class_names=["rain", "sunny"],
    )
    monkeypatch.setattr(
        "sys.argv",
        [
            "teacher_from_oof.py",
            "--train-csv",
            str(train_csv),
            "--image-root",
            str(tmp_path / "images"),
            "--oof",
            str(oof_path),
            "--output",
            str(tmp_path / "train_teacher.csv"),
            "--max-top1-mismatch-rate",
            "0.0",
            "--min-mean-true-probability",
            "0.8",
        ],
    )

    main()

    assert (tmp_path / "train_teacher.csv").exists()
    assert '"rows": 2' in capsys.readouterr().out
