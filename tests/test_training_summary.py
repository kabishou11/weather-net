from src.weather_net.metrics import ClassificationReport
from src.weather_net.training import build_fold_summary


def test_build_fold_summary_includes_report_details() -> None:
    report = ClassificationReport(
        macro_f1=0.75,
        accuracy=0.8,
        per_class_f1={"rain": 0.7, "sunny": 0.8},
        confusion_matrix=[[3, 1], [0, 4]],
    )

    summary = build_fold_summary(
        fold=1,
        best_epoch=3,
        best_val_loss=0.42,
        checkpoint="outputs/model_fold1.pt",
        report=report,
    )

    assert summary == {
        "fold": 1,
        "best_epoch": 3,
        "best_macro_f1": 0.75,
        "best_accuracy": 0.8,
        "best_val_loss": 0.42,
        "checkpoint": "outputs/model_fold1.pt",
        "per_class_f1": {"rain": 0.7, "sunny": 0.8},
        "confusion_matrix": [[3, 1], [0, 4]],
    }
