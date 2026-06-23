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
