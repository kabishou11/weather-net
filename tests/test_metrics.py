from src.weather_net.metrics import classification_report


def test_classification_report_computes_macro_f1_and_confusion_matrix() -> None:
    report = classification_report(
        y_true=[0, 0, 1, 1],
        y_pred=[0, 1, 1, 1],
        class_names=["rain", "sunny"],
    )

    assert report.confusion_matrix == [[1, 1], [0, 2]]
    assert round(report.per_class_f1["rain"], 6) == round(2 / 3, 6)
    assert round(report.per_class_f1["sunny"], 6) == 0.8
    assert round(report.macro_f1, 6) == round(((2 / 3) + 0.8) / 2, 6)
    assert report.accuracy == 0.75
