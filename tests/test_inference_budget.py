import json
import sys
from pathlib import Path


def test_check_inference_budget_accepts_fast_single_model(tmp_path: Path) -> None:
    from inference_budget import check_inference_budget

    stats_path = tmp_path / "submission.stats.json"
    stats_path.write_text(
        json.dumps(
            {
                "images": 100,
                "seconds": 2.0,
                "images_per_second": 50.0,
                "checkpoints": ["model.pt"],
                "tta": False,
                "amp": False,
            }
        ),
        encoding="utf-8",
    )

    result = check_inference_budget(
        stats_path=stats_path,
        max_seconds_per_image=0.05,
        max_checkpoints=1,
        allow_tta=False,
    )

    assert result["status"] == "pass"
    assert result["seconds_per_image"] == 0.02
    assert result["recommendation"] == "submission_ready"


def test_check_inference_budget_rejects_slow_or_heavy_inference(tmp_path: Path) -> None:
    from inference_budget import check_inference_budget

    stats_path = tmp_path / "ensemble.stats.json"
    stats_path.write_text(
        json.dumps(
            {
                "images": 10,
                "seconds": 2.0,
                "images_per_second": 5.0,
                "checkpoints": ["a.pt", "b.pt"],
                "tta": True,
                "amp": False,
            }
        ),
        encoding="utf-8",
    )

    result = check_inference_budget(
        stats_path=stats_path,
        max_seconds_per_image=0.1,
        max_checkpoints=1,
        allow_tta=False,
    )

    assert result["status"] == "fail"
    assert "seconds_per_image" in result["violations"]
    assert "checkpoints" in result["violations"]
    assert "tta" in result["violations"]
    assert result["recommendation"] == "prefer_soup_or_fast_single_model"


def test_inference_budget_parse_args(monkeypatch) -> None:
    from inference_budget import parse_args

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "inference_budget.py",
            "--stats",
            "submission.stats.json",
            "--max-seconds-per-image",
            "0.05",
            "--max-checkpoints",
            "1",
            "--output",
            "budget.json",
        ],
    )

    args = parse_args()

    assert args.stats == Path("submission.stats.json")
    assert args.max_seconds_per_image == 0.05
    assert args.max_checkpoints == 1
    assert args.output == Path("budget.json")


def test_inference_budget_main_writes_output_and_exits_nonzero_on_failure(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    import pytest

    from inference_budget import main

    stats_path = tmp_path / "slow.stats.json"
    output_path = tmp_path / "budget.json"
    stats_path.write_text(
        json.dumps(
            {
                "images": 10,
                "seconds": 2.0,
                "images_per_second": 5.0,
                "checkpoints": ["a.pt", "b.pt"],
                "tta": False,
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "inference_budget.py",
            "--stats",
            str(stats_path),
            "--max-seconds-per-image",
            "0.1",
            "--max-checkpoints",
            "1",
            "--output",
            str(output_path),
        ],
    )

    with pytest.raises(SystemExit) as error:
        main()

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert error.value.code == 1
    assert payload["status"] == "fail"
    assert "checkpoints" in payload["violations"]
    assert '"status": "fail"' in capsys.readouterr().out
