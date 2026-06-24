from pathlib import Path
import math

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
        dataset_name="external_mwd",
        label_map=DEFAULT_LABEL_MAPS["mwd"],
        dataset_url="https://data.mendeley.com/datasets/4drtyfjtfy/1",
        license_name="CC BY 4.0",
        doi="10.17632/4drtyfjtfy.1",
        sample_weight=0.4,
    )

    assert summary["label_counts"] == {"rain": 1, "sunny": 2}
    assert summary["dataset_url"] == "https://data.mendeley.com/datasets/4drtyfjtfy/1"
    assert summary["license"] == "CC BY 4.0"
    assert summary["doi"] == "10.17632/4drtyfjtfy.1"
    assert isinstance(summary["label_map_sha256"], str)
    lines = (tmp_path / "mwd.csv").read_text(encoding="utf-8").splitlines()
    assert lines[0] == "image,label,source,original_label,dataset_url,license,doi,sample_weight"
    assert lines[1].endswith(",0.400000")


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
        dataset_name="external_weapd",
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
            dataset_name="external_empty",
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
        dataset_name="external_test",
        label_map={"rain": "rain"},
        drop_unmapped=True,
        allowed_labels=["rain"],
    )

    assert summary["label_counts"] == {"rain": 1}
    assert summary["skipped_unmapped"] == {"rainbow": 1}


def test_external_manifest_main_requires_label_boundary(monkeypatch, tmp_path: Path) -> None:
    import pytest
    import sys

    from external_dataset_manifest import main

    root = tmp_path / "external"
    _make_image(root / "rain" / "rain1.jpg")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "external_dataset_manifest.py",
            "--image-root",
            str(root),
            "--output-csv",
            str(tmp_path / "external.csv"),
            "--summary-json",
            str(tmp_path / "summary.json"),
            "--dataset-name",
            "external_test",
            "--label-map-preset",
            "mwd",
        ],
    )

    with pytest.raises(ValueError, match="requires --class-map or --allowed-labels"):
        main()


def test_external_manifest_main_allows_explicit_unbounded_label_exploration(monkeypatch, tmp_path: Path) -> None:
    import sys

    from external_dataset_manifest import main

    root = tmp_path / "external"
    _make_image(root / "rain" / "rain1.jpg")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "external_dataset_manifest.py",
            "--image-root",
            str(root),
            "--output-csv",
            str(tmp_path / "external.csv"),
            "--summary-json",
            str(tmp_path / "summary.json"),
            "--dataset-name",
            "external_test",
            "--label-map-preset",
            "mwd",
            "--allow-unbounded-labels",
            "--sample-weight",
            "0.3",
        ],
    )

    main()

    assert (tmp_path / "external.csv").exists()


def test_external_manifest_main_requires_explicit_sample_weight(monkeypatch, tmp_path: Path) -> None:
    import pytest
    import sys

    from external_dataset_manifest import main

    root = tmp_path / "external"
    _make_image(root / "rain" / "rain1.jpg")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "external_dataset_manifest.py",
            "--image-root",
            str(root),
            "--output-csv",
            str(tmp_path / "external.csv"),
            "--summary-json",
            str(tmp_path / "summary.json"),
            "--dataset-name",
            "external_test",
            "--label-map-preset",
            "mwd",
            "--allow-unbounded-labels",
        ],
    )

    with pytest.raises(ValueError, match="--sample-weight"):
        main()


def test_external_manifest_rejects_labels_outside_allowed_set(tmp_path: Path) -> None:
    import pytest

    from external_dataset_manifest import build_external_manifest

    root = tmp_path / "external"
    _make_image(root / "rain" / "rain1.jpg")
    _make_image(root / "rainbow" / "rainbow1.jpg")

    with pytest.raises(ValueError, match="outside allowed labels"):
        build_external_manifest(
            image_root=root,
            output_csv=tmp_path / "blocked.csv",
            summary_json=tmp_path / "blocked.json",
            dataset_name="external_test",
            label_map={"rain": "rain", "rainbow": "rainbow"},
            allowed_labels=["rain", "snow"],
        )


def test_external_manifest_uses_class_map_as_allowed_labels(tmp_path: Path) -> None:
    import json
    import pytest

    from external_dataset_manifest import build_external_manifest, load_allowed_labels

    class_map = tmp_path / "class_to_idx.json"
    class_map.write_text(json.dumps({"rain": 0, "snow": 1}) + "\n", encoding="utf-8")
    assert load_allowed_labels(allowed_labels=None, class_map=class_map) == ["rain", "snow"]

    root = tmp_path / "external"
    _make_image(root / "rain" / "rain1.jpg")
    _make_image(root / "dew" / "dew1.jpg")

    with pytest.raises(ValueError, match="outside allowed labels"):
        build_external_manifest(
            image_root=root,
            output_csv=tmp_path / "blocked.csv",
            summary_json=tmp_path / "blocked.json",
            dataset_name="external_test",
            label_map={"rain": "rain", "dew": "dew"},
            allowed_labels=load_allowed_labels(allowed_labels=None, class_map=class_map),
        )


def test_external_manifest_class_map_is_authoritative_when_allowed_labels_are_set(tmp_path: Path) -> None:
    import json
    import pytest

    from external_dataset_manifest import load_allowed_labels

    class_map = tmp_path / "class_to_idx.json"
    class_map.write_text(json.dumps({"rain": 0, "snow": 1}) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="outside class map"):
        load_allowed_labels(allowed_labels=["rainbow"], class_map=class_map)


def test_external_manifest_parse_args_accepts_class_map_and_allowed_labels(monkeypatch) -> None:
    import sys

    from external_dataset_manifest import parse_args

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "external_dataset_manifest.py",
            "--image-root",
            "images",
            "--output-csv",
            "external.csv",
            "--summary-json",
            "summary.json",
            "--dataset-name",
            "external_mwd",
            "--class-map",
            "class_to_idx.json",
            "--allowed-labels",
            "rain",
            "snow",
            "--allow-unbounded-labels",
            "--sample-weight",
            "0.25",
            "--skip-reserved-splits",
        ],
    )

    args = parse_args()

    assert args.class_map == Path("class_to_idx.json")
    assert args.allowed_labels == ["rain", "snow"]
    assert args.allow_unbounded_labels is True
    assert args.sample_weight == 0.25
    assert args.skip_reserved_splits is True


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
            dataset_name="external_test",
            label_map={"rain": ""},
        )


def test_external_manifest_rejects_reserved_dataset_names(tmp_path: Path) -> None:
    import pytest

    from external_dataset_manifest import build_external_manifest

    root = tmp_path / "external"
    _make_image(root / "rain" / "rain1.jpg")

    with pytest.raises(ValueError, match="dataset_name must start with external_"):
        build_external_manifest(
            image_root=root,
            output_csv=tmp_path / "bad.csv",
            summary_json=tmp_path / "bad.json",
            dataset_name="labeled",
            label_map={"rain": "rain"},
            allowed_labels=["rain"],
        )


def test_external_manifest_maps_vijay_weather_variants(tmp_path: Path) -> None:
    from external_dataset_manifest import DEFAULT_LABEL_MAPS, build_external_manifest

    root = tmp_path / "vijay"
    _make_image(root / "cloudy" / "cloudy1.jpg")
    _make_image(root / "foggy" / "foggy1.jpg")
    _make_image(root / "rainy" / "rainy1.jpg")
    _make_image(root / "shine" / "shine1.jpg")
    _make_image(root / "sunrise" / "sunrise1.jpg")

    summary = build_external_manifest(
        image_root=root,
        output_csv=tmp_path / "vijay.csv",
        summary_json=tmp_path / "vijay.json",
        dataset_name="external_vijay_multiclass_weather",
        label_map=DEFAULT_LABEL_MAPS["vijay_mwd"],
        allowed_labels=["cloudy", "fog", "rain", "sunny"],
    )

    assert summary["label_counts"] == {"cloudy": 1, "fog": 1, "rain": 1, "sunny": 2}


def test_external_manifest_rejects_reserved_imagefolder_test_splits(tmp_path: Path) -> None:
    import pytest

    from external_dataset_manifest import DEFAULT_LABEL_MAPS, build_external_manifest

    root = tmp_path / "vijay"
    _make_image(root / "cloudy" / "cloudy1.jpg")
    _make_image(root / "alien_test" / "alien1.jpg")

    with pytest.raises(ValueError, match="reserved test split"):
        build_external_manifest(
            image_root=root,
            output_csv=tmp_path / "vijay.csv",
            summary_json=tmp_path / "vijay.json",
            dataset_name="external_vijay_multiclass_weather",
            label_map=DEFAULT_LABEL_MAPS["vijay_mwd"],
            allowed_labels=["cloudy", "fog", "rain", "sunny"],
            sample_weight=0.2,
        )


def test_external_manifest_can_explicitly_skip_reserved_imagefolder_splits(tmp_path: Path) -> None:
    from external_dataset_manifest import DEFAULT_LABEL_MAPS, build_external_manifest

    root = tmp_path / "vijay"
    _make_image(root / "cloudy" / "cloudy1.jpg")
    _make_image(root / "alien_test" / "alien1.jpg")

    summary = build_external_manifest(
        image_root=root,
        output_csv=tmp_path / "vijay.csv",
        summary_json=tmp_path / "vijay.json",
        dataset_name="external_vijay_multiclass_weather",
        label_map=DEFAULT_LABEL_MAPS["vijay_mwd"],
        allowed_labels=["cloudy", "fog", "rain", "sunny"],
        sample_weight=0.2,
        skip_reserved_splits=True,
    )

    assert summary["label_counts"] == {"cloudy": 1}
    assert summary["skipped_reserved_splits"] == {"alien_test": 1}


def test_external_manifest_rejects_non_finite_sample_weight(tmp_path: Path) -> None:
    import pytest

    from external_dataset_manifest import build_external_manifest

    root = tmp_path / "external"
    _make_image(root / "rain" / "rain1.jpg")

    with pytest.raises(ValueError, match="sample_weight"):
        build_external_manifest(
            image_root=root,
            output_csv=tmp_path / "bad.csv",
            summary_json=tmp_path / "bad.json",
            dataset_name="external_bad",
            label_map={"rain": "rain"},
            allowed_labels=["rain"],
            sample_weight=math.inf,
        )


def test_road_weather_time_manifest_converts_train_json_and_keeps_period(tmp_path: Path) -> None:
    import csv
    import json

    from external_dataset_manifest import DEFAULT_LABEL_MAPS, build_road_weather_time_manifest

    root = tmp_path / "road"
    _make_image(root / "train_dataset" / "train_images" / "00001.jpg")
    _make_image(root / "train_dataset" / "train_images" / "00002.jpg")
    _make_image(root / "test_dataset" / "test_images" / "00003.jpg")
    annotations = {
        "annotations": [
            {"filename": "train_images\\00001.jpg", "period": "Morning", "weather": "Cloudy"},
            {"filename": "train_images\\00002.jpg", "period": "Night", "weather": "Rainy"},
        ]
    }
    annotation_json = root / "train_dataset" / "train.json"
    annotation_json.write_text(json.dumps(annotations), encoding="utf-8")

    summary = build_road_weather_time_manifest(
        annotation_json=annotation_json,
        image_root=root / "train_dataset",
        output_csv=tmp_path / "road.csv",
        summary_json=tmp_path / "road.json",
        dataset_name="external_road_weather_time",
        label_map=DEFAULT_LABEL_MAPS["road_weather_time"],
        allowed_labels=["cloudy", "rain"],
        dataset_url="https://www.kaggle.com/datasets/wjybuqi/weathertime-classification-with-road-images",
        license_name="Kaggle dataset; verify competition terms before redistribution",
        sample_weight=0.85,
    )

    assert summary["label_counts"] == {"cloudy": 1, "rain": 1}
    assert summary["period_counts"] == {"morning": 1, "night": 1}
    assert summary["total"] == 2
    rows = list(csv.DictReader((tmp_path / "road.csv").open("r", encoding="utf-8")))
    assert rows == [
        {
            "image": "train_images/00001.jpg",
            "label": "cloudy",
            "source": "external_road_weather_time",
            "original_label": "cloudy",
            "period": "morning",
            "dataset_url": "https://www.kaggle.com/datasets/wjybuqi/weathertime-classification-with-road-images",
            "license": "Kaggle dataset; verify competition terms before redistribution",
            "doi": "",
            "sample_weight": "0.850000",
        },
        {
            "image": "train_images/00002.jpg",
            "label": "rain",
            "source": "external_road_weather_time",
            "original_label": "rainy",
            "period": "night",
            "dataset_url": "https://www.kaggle.com/datasets/wjybuqi/weathertime-classification-with-road-images",
            "license": "Kaggle dataset; verify competition terms before redistribution",
            "doi": "",
            "sample_weight": "0.850000",
        },
    ]


def test_road_weather_time_manifest_rejects_test_paths_and_unknown_weather(tmp_path: Path) -> None:
    import json
    import pytest

    from external_dataset_manifest import DEFAULT_LABEL_MAPS, build_road_weather_time_manifest

    root = tmp_path / "road"
    _make_image(root / "train_dataset" / "train_images" / "00001.jpg")
    annotation_json = root / "train_dataset" / "train.json"
    annotation_json.write_text(
        json.dumps(
            {
                "annotations": [
                    {"filename": "test_dataset/test_images/00003.jpg", "period": "Morning", "weather": "Cloudy"},
                    {"filename": "train_images/00001.jpg", "period": "Morning", "weather": "Rainbow"},
                ]
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="test_dataset"):
        build_road_weather_time_manifest(
            annotation_json=annotation_json,
            image_root=root / "train_dataset",
            output_csv=tmp_path / "road.csv",
            summary_json=tmp_path / "road.json",
            dataset_name="external_road_weather_time",
            label_map=DEFAULT_LABEL_MAPS["road_weather_time"],
            allowed_labels=["cloudy", "rain"],
        )


def test_external_manifest_main_builds_road_weather_time_manifest(monkeypatch, tmp_path: Path) -> None:
    import json
    import sys

    from external_dataset_manifest import main

    root = tmp_path / "road"
    _make_image(root / "train_dataset" / "train_images" / "00001.jpg")
    annotation_json = root / "train_dataset" / "train.json"
    annotation_json.write_text(
        json.dumps({"annotations": [{"filename": "train_images\\00001.jpg", "period": "Dusk", "weather": "Foggy"}]}),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "external_dataset_manifest.py",
            "--format",
            "road-weather-time",
            "--annotation-json",
            str(annotation_json),
            "--image-root",
            str(root / "train_dataset"),
            "--output-csv",
            str(tmp_path / "road.csv"),
            "--summary-json",
            str(tmp_path / "road.json"),
            "--dataset-name",
            "external_road_weather_time",
            "--label-map-preset",
            "road_weather_time",
            "--allowed-labels",
            "fog",
            "--sample-weight",
            "0.8",
        ],
    )

    main()

    assert (tmp_path / "road.csv").read_text(encoding="utf-8").splitlines()[1].endswith(",0.800000")


def test_external_manifest_main_skips_reserved_imagefolder_splits(monkeypatch, tmp_path: Path) -> None:
    import json
    import sys

    from external_dataset_manifest import main

    root = tmp_path / "vijay"
    _make_image(root / "dataset" / "cloudy" / "cloudy1.jpg")
    _make_image(root / "dataset" / "alien_test" / "alien1.jpg")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "external_dataset_manifest.py",
            "--image-root",
            str(root / "dataset"),
            "--output-csv",
            str(tmp_path / "vijay.csv"),
            "--summary-json",
            str(tmp_path / "vijay.json"),
            "--dataset-name",
            "external_vijay_multiclass_weather",
            "--label-map-preset",
            "vijay_mwd",
            "--allow-unbounded-labels",
            "--skip-reserved-splits",
            "--sample-weight",
            "0.2",
        ],
    )

    main()

    summary = json.loads((tmp_path / "vijay.json").read_text(encoding="utf-8"))
    assert summary["label_counts"] == {"cloudy": 1}
    assert summary["skipped_reserved_splits"] == {"alien_test": 1}
