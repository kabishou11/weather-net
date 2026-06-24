from pathlib import Path

from PIL import Image


def _make_image(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (8, 8), (255, 255, 255)).save(path)


def test_external_manifest_maps_mwd_labels(tmp_path: Path) -> None:
    from external_dataset_manifest import DEFAULT_LABEL_MAPS, build_external_manifest

    root = tmp_path / "mwd"
    _make_image(root / "Shine" / "shine1.jpg")
    _make_image(root / "Sunrise" / "sunrise1.jpg")
    _make_image(root / "Rain" / "rain1.jpg")

    summary = build_external_manifest(
        image_root=root,
        output_csv=tmp_path / "mwd.csv",
        summary_json=tmp_path / "mwd.json",
        dataset_name="mwd",
        label_map=DEFAULT_LABEL_MAPS["mwd"],
    )

    assert summary["label_counts"] == {"rain": 1, "sunny": 2}
    assert (tmp_path / "mwd.csv").read_text(encoding="utf-8").splitlines()[0] == (
        "image,label,source,original_label"
    )


def test_external_manifest_maps_weapd_weather_variants(tmp_path: Path) -> None:
    from external_dataset_manifest import DEFAULT_LABEL_MAPS, build_external_manifest

    root = tmp_path / "weapd"
    _make_image(root / "fog/smog" / "fog1.jpg")
    _make_image(root / "hail" / "hail1.jpg")
    _make_image(root / "rime" / "rime1.jpg")
    _make_image(root / "rain" / "rain1.jpg")

    summary = build_external_manifest(
        image_root=root,
        output_csv=tmp_path / "weapd.csv",
        summary_json=tmp_path / "weapd.json",
        dataset_name="weapd",
        label_map=DEFAULT_LABEL_MAPS["weapd"],
    )

    assert summary["label_counts"] == {"fog": 1, "rain": 1, "snow": 2}


def test_external_manifest_rejects_empty_image_root(tmp_path: Path) -> None:
    import pytest

    from external_dataset_manifest import build_external_manifest

    with pytest.raises(ValueError, match="No image files"):
        build_external_manifest(
            image_root=tmp_path,
            output_csv=tmp_path / "empty.csv",
            summary_json=tmp_path / "empty.json",
            dataset_name="empty",
            label_map={},
        )


def test_external_manifest_drops_unmapped_labels_when_requested(tmp_path: Path) -> None:
    from external_dataset_manifest import build_external_manifest

    root = tmp_path / "external"
    _make_image(root / "rain" / "rain1.jpg")
    _make_image(root / "rainbow" / "rainbow1.jpg")

    summary = build_external_manifest(
        image_root=root,
        output_csv=tmp_path / "filtered.csv",
        summary_json=tmp_path / "filtered.json",
        dataset_name="external",
        label_map={"rain": "rain"},
        drop_unmapped=True,
    )

    assert summary["label_counts"] == {"rain": 1}
    assert summary["skipped_unmapped"] == {"rainbow": 1}


def test_external_manifest_rejects_empty_mapped_label(tmp_path: Path) -> None:
    import pytest

    from external_dataset_manifest import build_external_manifest

    root = tmp_path / "external"
    _make_image(root / "rain" / "rain1.jpg")

    with pytest.raises(ValueError, match="empty mapped label"):
        build_external_manifest(
            image_root=root,
            output_csv=tmp_path / "bad.csv",
            summary_json=tmp_path / "bad.json",
            dataset_name="external",
            label_map={"rain": ""},
        )
