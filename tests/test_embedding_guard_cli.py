import sys
from pathlib import Path

import numpy as np
from PIL import Image


def _make_image(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (8, 8), (120, 140, 160)).save(path)


def test_embedding_guard_parse_args_accepts_required_files(monkeypatch) -> None:
    from embedding_guard import parse_args

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "embedding_guard.py",
            "--train-csv",
            "train.csv",
            "--pseudo-csv",
            "pseudo.csv",
            "--embeddings",
            "embeddings.npz",
            "--output",
            "filtered.csv",
            "--rejected-output",
            "rejected.csv",
        ],
    )

    args = parse_args()

    assert args.train_csv == Path("train.csv")
    assert args.pseudo_csv == Path("pseudo.csv")
    assert args.embeddings == Path("embeddings.npz")
    assert args.output == Path("filtered.csv")
    assert args.rejected_output == Path("rejected.csv")


def test_embedding_guard_main_filters_pseudo_labels(monkeypatch, tmp_path: Path, capsys) -> None:
    from embedding_guard import main

    _make_image(tmp_path / "images" / "rain_ref.jpg")
    _make_image(tmp_path / "images" / "sunny_ref.jpg")
    train_csv = tmp_path / "train.csv"
    train_csv.write_text(
        "image,label\n"
        "rain_ref.jpg,rain\n"
        "sunny_ref.jpg,sunny\n",
        encoding="utf-8",
    )
    pseudo_csv = tmp_path / "pseudo.csv"
    pseudo_csv.write_text(
        "image,label,confidence\n"
        "u0.jpg,rain,0.970000\n"
        "u1.jpg,rain,0.980000\n",
        encoding="utf-8",
    )
    embeddings = tmp_path / "embeddings.npz"
    np.savez_compressed(
        embeddings,
        image_id=np.array(["rain_ref.jpg", "sunny_ref.jpg", "u0.jpg", "u1.jpg"], dtype=object),
        embedding=np.array(
            [
                [1.0, 0.0],
                [0.0, 1.0],
                [0.95, 0.05],
                [0.05, 0.95],
            ],
            dtype=np.float32,
        ),
    )
    output = tmp_path / "filtered.csv"
    rejected = tmp_path / "rejected.csv"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "embedding_guard.py",
            "--train-csv",
            str(train_csv),
            "--image-root",
            str(tmp_path / "images"),
            "--pseudo-csv",
            str(pseudo_csv),
            "--embeddings",
            str(embeddings),
            "--output",
            str(output),
            "--rejected-output",
            str(rejected),
            "--min-similarity",
            "0.8",
            "--min-margin",
            "0.1",
        ],
    )

    main()

    assert '"kept": 1' in capsys.readouterr().out
    assert "u0.jpg,rain" in output.read_text(encoding="utf-8")
    assert "u1.jpg,rain" in rejected.read_text(encoding="utf-8")


def test_embedding_guard_output_can_feed_pseudo_merge(monkeypatch, tmp_path: Path) -> None:
    from embedding_guard import main
    from src.weather_net.pseudo_merge import merge_training_with_pseudo_labels

    _make_image(tmp_path / "images" / "rain_ref.jpg")
    _make_image(tmp_path / "images" / "sunny_ref.jpg")
    _make_image(tmp_path / "unlabeled" / "u0.jpg")
    _make_image(tmp_path / "unlabeled" / "u1.jpg")
    train_csv = tmp_path / "train.csv"
    train_csv.write_text(
        "image,label\n"
        "rain_ref.jpg,rain\n"
        "sunny_ref.jpg,sunny\n",
        encoding="utf-8",
    )
    pseudo_csv = tmp_path / "pseudo.csv"
    pseudo_csv.write_text(
        "image,label,confidence\n"
        "u0.jpg,rain,0.970000\n"
        "u1.jpg,rain,0.980000\n",
        encoding="utf-8",
    )
    embeddings = tmp_path / "embeddings.npz"
    np.savez_compressed(
        embeddings,
        image_id=np.array(["rain_ref.jpg", "sunny_ref.jpg", "u0.jpg", "u1.jpg"], dtype=object),
        embedding=np.array(
            [
                [1.0, 0.0],
                [0.0, 1.0],
                [0.95, 0.05],
                [0.05, 0.95],
            ],
            dtype=np.float32,
        ),
    )
    filtered = tmp_path / "filtered.csv"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "embedding_guard.py",
            "--train-csv",
            str(train_csv),
            "--image-root",
            str(tmp_path / "images"),
            "--pseudo-csv",
            str(pseudo_csv),
            "--embeddings",
            str(embeddings),
            "--output",
            str(filtered),
            "--rejected-output",
            str(tmp_path / "rejected.csv"),
        ],
    )
    main()

    stats = merge_training_with_pseudo_labels(
        train_dir=None,
        train_csv=train_csv,
        image_root=tmp_path / "images",
        pseudo_csv=filtered,
        pseudo_image_root=tmp_path / "unlabeled",
        output_csv=tmp_path / "merged.csv",
        min_confidence=0.95,
    )

    assert stats["pseudo"] == 1
