from pathlib import Path


def test_build_prediction_records_marks_errors_and_confidence() -> None:
    from src.weather_net.error_analysis import build_prediction_records

    records = build_prediction_records(
        image_paths=[Path("a.jpg"), Path("b.jpg")],
        y_true=[0, 1],
        probabilities=[
            [0.8, 0.2],
            [0.7, 0.3],
        ],
        class_names=["rain", "sunny"],
    )

    assert records[0].prediction == "rain"
    assert records[0].confidence == 0.8
    assert records[0].correct is True
    assert records[1].label == "sunny"
    assert records[1].prediction == "rain"
    assert records[1].correct is False


def test_write_error_analysis_csv_outputs_hard_examples_first(tmp_path: Path) -> None:
    from src.weather_net.error_analysis import build_prediction_records, write_error_analysis_csv

    records = build_prediction_records(
        image_paths=[Path("correct.jpg"), Path("wrong.jpg"), Path("low_conf.jpg")],
        y_true=[0, 1, 0],
        probabilities=[
            [0.9, 0.1],
            [0.8, 0.2],
            [0.55, 0.45],
        ],
        class_names=["rain", "sunny"],
    )

    output_path = tmp_path / "errors.csv"
    write_error_analysis_csv(output_path, records, low_confidence_threshold=0.6)

    assert output_path.read_text(encoding="utf-8") == (
        "image,label,prediction,confidence,correct,hard_reason\n"
        "wrong.jpg,sunny,rain,0.800000,0,error\n"
        "low_conf.jpg,rain,rain,0.550000,1,low_confidence\n"
    )


def test_confusion_pairs_counts_misclassified_pairs() -> None:
    from src.weather_net.error_analysis import confusion_pairs

    pairs = confusion_pairs(
        y_true=[0, 0, 1, 1, 2],
        y_pred=[1, 1, 0, 1, 0],
        class_names=["rain", "sunny", "fog"],
    )

    assert pairs == [
        {"label": "rain", "prediction": "sunny", "count": 2},
        {"label": "fog", "prediction": "rain", "count": 1},
        {"label": "sunny", "prediction": "rain", "count": 1},
    ]
