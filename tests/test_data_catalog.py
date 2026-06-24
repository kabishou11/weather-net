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


def test_csv_manifest_reads_teacher_probability_columns(tmp_path: Path) -> None:
    from src.weather_net.data import build_manifest_from_csv

    _make_image(tmp_path / "images" / "rain.jpg")
    _make_image(tmp_path / "images" / "sunny.jpg")
    csv_path = tmp_path / "train.csv"
    csv_path.write_text(
        "image,label,teacher_rain,teacher_sunny\n"
        "rain.jpg,rain,0.8,0.2\n"
        "sunny.jpg,sunny,0.1,0.9\n",
        encoding="utf-8",
    )

    manifest, class_to_idx = build_manifest_from_csv(
        csv_path,
        image_root=tmp_path / "images",
        class_to_idx={"rain": 0, "sunny": 1},
    )

    assert class_to_idx == {"rain": 0, "sunny": 1}
    assert manifest[0].teacher_probs == (0.8, 0.2)
    assert manifest[1].teacher_probs == (0.1, 0.9)


def test_csv_manifest_rejects_invalid_teacher_probability_columns(tmp_path: Path) -> None:
    import pytest

    from src.weather_net.data import build_manifest_from_csv

    _make_image(tmp_path / "images" / "rain.jpg")
    csv_path = tmp_path / "train.csv"
    csv_path.write_text(
        "image,label,teacher_rain,teacher_sunny\n"
        "rain.jpg,rain,0.9,0.9\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="teacher"):
        build_manifest_from_csv(
            csv_path,
            image_root=tmp_path / "images",
            class_to_idx={"rain": 0, "sunny": 1},
        )


def test_csv_manifest_requires_all_teacher_probability_columns(tmp_path: Path) -> None:
    import pytest

    from src.weather_net.data import build_manifest_from_csv

    _make_image(tmp_path / "images" / "rain.jpg")
    csv_path = tmp_path / "train.csv"
    csv_path.write_text(
        "image,label,teacher_rain\n"
        "rain.jpg,rain,1.0\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="missing classes"):
        build_manifest_from_csv(
            csv_path,
            image_root=tmp_path / "images",
            class_to_idx={"rain": 0, "sunny": 1},
        )


def test_training_csv_prefers_filename_over_id_primary_key(tmp_path: Path) -> None:
    from src.weather_net.data import build_manifest_from_csv

    _make_image(tmp_path / "images" / "rain.jpg")
    _make_image(tmp_path / "images" / "sunny.jpg")
    csv_path = tmp_path / "train.csv"
    csv_path.write_text(
        "id,filename,label\n"
        "100,rain.jpg,rain\n"
        "101,sunny.jpg,sunny\n",
        encoding="utf-8",
    )

    manifest, _ = build_manifest_from_csv(csv_path, image_root=tmp_path / "images")

    assert [row.image_id for row in manifest] == ["rain.jpg", "sunny.jpg"]
    assert [row.path.name for row in manifest] == ["rain.jpg", "sunny.jpg"]


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
