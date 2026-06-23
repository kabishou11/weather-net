from pathlib import Path

from PIL import Image


def _make_image(path: Path, color: tuple[int, int, int] = (32, 64, 96)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (8, 8), color).save(path)


def test_image_folder_manifest_uses_sorted_classes(tmp_path: Path) -> None:
    from src.weather_net.data import build_manifest_from_image_folder

    _make_image(tmp_path / "train" / "rain" / "b.jpg")
    _make_image(tmp_path / "train" / "sunny" / "a.jpg")
    _make_image(tmp_path / "train" / "rain" / "a.jpg")

    manifest, class_to_idx = build_manifest_from_image_folder(tmp_path / "train")

    assert class_to_idx == {"rain": 0, "sunny": 1}
    assert [row.label_name for row in manifest] == ["rain", "rain", "sunny"]
    assert [row.label for row in manifest] == [0, 0, 1]
    assert [row.path.name for row in manifest] == ["a.jpg", "b.jpg", "a.jpg"]


def test_csv_manifest_respects_existing_class_mapping(tmp_path: Path) -> None:
    from src.weather_net.data import build_manifest_from_csv

    _make_image(tmp_path / "images" / "snow.jpg")
    _make_image(tmp_path / "images" / "fog.jpg")
    csv_path = tmp_path / "train.csv"
    csv_path.write_text("image,label\nsnow.jpg,snow\nfog.jpg,fog\n", encoding="utf-8")

    manifest, class_to_idx = build_manifest_from_csv(
        csv_path,
        image_root=tmp_path / "images",
        class_to_idx={"fog": 0, "snow": 1},
    )

    assert class_to_idx == {"fog": 0, "snow": 1}
    assert [row.label for row in manifest] == [1, 0]
    assert all(row.path.is_absolute() for row in manifest)


def test_image_folder_manifest_respects_existing_class_mapping(tmp_path: Path) -> None:
    from src.weather_net.data import build_manifest_from_image_folder

    _make_image(tmp_path / "val" / "snow" / "snow.jpg")
    _make_image(tmp_path / "val" / "fog" / "fog.jpg")

    manifest, class_to_idx = build_manifest_from_image_folder(
        tmp_path / "val",
        class_to_idx={"fog": 0, "snow": 1},
    )

    assert class_to_idx == {"fog": 0, "snow": 1}
    assert [row.label for row in manifest] == [0, 1]


def test_class_mapping_round_trips_json(tmp_path: Path) -> None:
    from src.weather_net.data import load_class_mapping, save_class_mapping

    mapping_path = tmp_path / "class_to_idx.json"
    save_class_mapping({"fog": 0, "rain": 1}, mapping_path)

    assert load_class_mapping(mapping_path) == {"fog": 0, "rain": 1}
