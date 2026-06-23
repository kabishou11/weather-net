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
