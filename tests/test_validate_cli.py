import sys
from pathlib import Path


def test_validate_parse_args_accepts_errors_csv(monkeypatch) -> None:
    from validate import parse_args

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "validate.py",
            "--checkpoint",
            "model.pt",
            "--val-dir",
            "val",
            "--errors-csv",
            "errors.csv",
        ],
    )

    args = parse_args()

    assert args.errors_csv == Path("errors.csv")


def test_infer_parse_args_accepts_submission_schema_controls(monkeypatch) -> None:
    from infer import parse_args

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "infer.py",
            "--checkpoint",
            "model.pt",
            "--test-dir",
            "test",
            "--sample-submission",
            "sample_submission.csv",
            "--image-column",
            "id",
            "--label-column",
            "weather",
            "--decision-params",
            "decision_params.json",
        ],
    )

    args = parse_args()

    assert args.sample_submission == Path("sample_submission.csv")
    assert args.image_column == "id"
    assert args.label_column == "weather"
    assert args.decision_params == Path("decision_params.json")
