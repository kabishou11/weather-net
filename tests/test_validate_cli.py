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


def test_validate_checkpoint_rejects_non_labeled_validation_csv(monkeypatch, tmp_path: Path) -> None:
    import pytest
    import torch
    from torch import nn

    import validate

    val_csv = tmp_path / "val.csv"
    val_csv.write_text(
        "image,label,source,confidence\n"
        "pseudo.jpg,rain,pseudo,0.97\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(
        validate,
        "load_checkpoint",
        lambda *_args, **_kwargs: (nn.Linear(1, 1), {"rain": 0}, 8),
    )
    monkeypatch.setattr(validate, "resolve_device", lambda _device: "cpu")

    with pytest.raises(ValueError, match="validation data must contain only labeled rows"):
        validate.validate_checkpoint(
            checkpoint=tmp_path / "model.pt",
            val_csv=val_csv,
            image_root=tmp_path,
            device="cpu",
        )
