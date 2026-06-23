from pathlib import Path

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

    assert stats == {"labeled": 2, "pseudo": 1, "total": 3}
    assert output_csv.read_text(encoding="utf-8") == (
        "image,label,source,confidence\n"
        f"{tmp_path / 'train' / 'rain' / 'r0.jpg'},rain,labeled,1.000000\n"
        f"{tmp_path / 'train' / 'sunny' / 's0.jpg'},sunny,labeled,1.000000\n"
        f"{tmp_path / 'unlabeled' / 'u0.jpg'},rain,pseudo,0.970000\n"
    )
