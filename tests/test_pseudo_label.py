from pathlib import Path


def test_select_pseudo_labels_filters_by_threshold() -> None:
    from src.weather_net.pseudo_label import select_pseudo_labels

    rows = select_pseudo_labels(
        image_paths=[Path("a.jpg"), Path("b.jpg"), Path("c.jpg")],
        probabilities=[
            [0.97, 0.03],
            [0.40, 0.60],
            [0.02, 0.98],
        ],
        class_names=["rain", "sunny"],
        threshold=0.95,
    )

    assert [(row.image.name, row.label, row.confidence) for row in rows] == [
        ("a.jpg", "rain", 0.97),
        ("c.jpg", "sunny", 0.98),
    ]


def test_select_pseudo_labels_filters_by_min_margin() -> None:
    from src.weather_net.pseudo_label import select_pseudo_labels

    rows = select_pseudo_labels(
        image_paths=[Path("ambiguous.jpg"), Path("clear.jpg")],
        probabilities=[
            [0.53, 0.47],
            [0.85, 0.15],
        ],
        class_names=["rain", "sunny"],
        threshold=0.5,
        min_margin=0.1,
    )

    assert [(row.image.name, row.label) for row in rows] == [("clear.jpg", "rain")]


def test_select_pseudo_labels_requires_tta_agreement() -> None:
    from src.weather_net.pseudo_label import select_pseudo_labels

    rows = select_pseudo_labels(
        image_paths=[Path("unstable.jpg"), Path("stable.jpg")],
        probabilities=[
            [0.96, 0.04],
            [0.94, 0.06],
        ],
        class_names=["rain", "sunny"],
        threshold=0.9,
        require_tta_agreement=True,
        agreement_probabilities=[
            [0.40, 0.60],
            [0.91, 0.09],
        ],
    )

    assert [(row.image.name, row.label) for row in rows] == [("stable.jpg", "rain")]


def test_select_pseudo_labels_supports_per_class_thresholds_and_caps() -> None:
    from src.weather_net.pseudo_label import select_pseudo_labels

    rows = select_pseudo_labels(
        image_paths=[
            Path("rain_high.jpg"),
            Path("rain_low.jpg"),
            Path("sunny_high.jpg"),
            Path("sunny_mid.jpg"),
            Path("sunny_low.jpg"),
        ],
        probabilities=[
            [0.97, 0.03],
            [0.91, 0.09],
            [0.04, 0.96],
            [0.08, 0.92],
            [0.11, 0.89],
        ],
        class_names=["rain", "sunny"],
        threshold=0.95,
        per_class_thresholds={"rain": 0.90, "sunny": 0.90},
        per_class_max_count={"rain": 2, "sunny": 1},
    )

    assert [(row.image.name, row.label, row.confidence) for row in rows] == [
        ("rain_high.jpg", "rain", 0.97),
        ("rain_low.jpg", "rain", 0.91),
        ("sunny_high.jpg", "sunny", 0.96),
    ]


def test_select_pseudo_labels_rejects_unknown_per_class_thresholds() -> None:
    import pytest
    from src.weather_net.pseudo_label import select_pseudo_labels

    with pytest.raises(ValueError, match="unknown classes"):
        select_pseudo_labels(
            image_paths=[Path("a.jpg")],
            probabilities=[[0.9, 0.1]],
            class_names=["rain", "sunny"],
            threshold=0.8,
            per_class_thresholds={"snow": 0.7},
        )


def test_select_pseudo_labels_rejects_negative_per_class_cap() -> None:
    import pytest
    from src.weather_net.pseudo_label import select_pseudo_labels

    with pytest.raises(ValueError, match="per-class max"):
        select_pseudo_labels(
            image_paths=[Path("a.jpg")],
            probabilities=[[0.9, 0.1]],
            class_names=["rain", "sunny"],
            threshold=0.8,
            per_class_max_count={"rain": -1},
        )


def test_write_pseudo_labels_preserves_image_id_when_available(tmp_path: Path) -> None:
    from src.weather_net.pseudo_label import PseudoLabelRow, write_pseudo_labels

    output_path = tmp_path / "pseudo.csv"
    write_pseudo_labels(
        output_path,
        [
            PseudoLabelRow(
                image=Path("/data/nested/a.jpg"),
                label="rain",
                confidence=0.99,
                image_id="nested/a.jpg",
            ),
        ],
    )

    assert output_path.read_text(encoding="utf-8") == (
        "image,label,confidence\n"
        "nested/a.jpg,rain,0.990000\n"
    )


def test_write_pseudo_labels_outputs_confidence_csv(tmp_path: Path) -> None:
    from src.weather_net.pseudo_label import PseudoLabelRow, write_pseudo_labels

    output_path = tmp_path / "pseudo.csv"
    write_pseudo_labels(
        output_path,
        [
            PseudoLabelRow(image=Path("/data/a.jpg"), label="rain", confidence=0.975432),
            PseudoLabelRow(image=Path("/data/b.jpg"), label="sunny", confidence=0.951),
        ],
    )

    assert output_path.read_text(encoding="utf-8") == (
        "image,label,confidence\n"
        "a.jpg,rain,0.975432\n"
        "b.jpg,sunny,0.951000\n"
    )


def test_count_pseudo_labels_by_class() -> None:
    from src.weather_net.pseudo_label import PseudoLabelRow, count_pseudo_labels_by_class

    counts = count_pseudo_labels_by_class(
        [
            PseudoLabelRow(image=Path("a.jpg"), label="rain", confidence=0.9),
            PseudoLabelRow(image=Path("b.jpg"), label="rain", confidence=0.91),
            PseudoLabelRow(image=Path("c.jpg"), label="sunny", confidence=0.92),
        ],
        class_names=["rain", "sunny", "fog"],
    )

    assert counts == {"rain": 2, "sunny": 1, "fog": 0}


def test_estimate_per_class_thresholds_from_oof_precision() -> None:
    from src.weather_net.pseudo_label import estimate_per_class_thresholds

    thresholds = estimate_per_class_thresholds(
        y_true=[0, 0, 1, 1, 1],
        probabilities=[
            [0.95, 0.05],
            [0.80, 0.20],
            [0.40, 0.60],
            [0.10, 0.90],
            [0.70, 0.30],
        ],
        class_names=["rain", "sunny"],
        target_precision=0.75,
        min_threshold=0.5,
        fallback_threshold=0.97,
    )

    assert thresholds == {"rain": 0.8, "sunny": 0.6}


def test_estimate_per_class_thresholds_uses_fallback_when_precision_unmet() -> None:
    from src.weather_net.pseudo_label import estimate_per_class_thresholds

    thresholds = estimate_per_class_thresholds(
        y_true=[0, 0],
        probabilities=[
            [0.20, 0.80],
            [0.30, 0.70],
        ],
        class_names=["rain", "sunny"],
        target_precision=0.9,
        min_threshold=0.5,
        fallback_threshold=0.99,
    )

    assert thresholds["sunny"] == 0.99


def test_estimate_per_class_thresholds_groups_tied_confidences() -> None:
    from src.weather_net.pseudo_label import estimate_per_class_thresholds

    thresholds = estimate_per_class_thresholds(
        y_true=[0, 1],
        probabilities=[
            [0.90, 0.10],
            [0.90, 0.10],
        ],
        class_names=["rain", "sunny"],
        target_precision=1.0,
        min_threshold=0.5,
        fallback_threshold=0.99,
    )

    assert thresholds["rain"] == 0.99


def test_estimate_per_class_thresholds_rejects_out_of_range_labels() -> None:
    import pytest
    from src.weather_net.pseudo_label import estimate_per_class_thresholds

    with pytest.raises(ValueError, match="valid class index"):
        estimate_per_class_thresholds(
            y_true=[0, 2],
            probabilities=[
                [0.80, 0.20],
                [0.10, 0.90],
            ],
            class_names=["rain", "sunny"],
        )
