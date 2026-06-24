import json
import sys
from pathlib import Path

import yaml


def test_build_ablation_configs_sets_usage_and_output_dirs(tmp_path: Path) -> None:
    from hard_finetune_ablation import build_ablation_configs
    from src.weather_net.config import load_config

    base_config = load_config(Path("configs/convnextv2_384_hard_finetune.yaml"))
    outputs = build_ablation_configs(
        base_config=base_config,
        output_root=tmp_path / "ablation",
        usages=["loss", "sampler", "both"],
    )

    assert [item.usage for item in outputs] == ["loss", "sampler", "both"]
    for item in outputs:
        payload = yaml.safe_load(item.config_path.read_text(encoding="utf-8"))
        assert payload["train"]["sample_weight_usage"] == item.usage
        assert "loss_warmup_epochs" in payload["train"]
        assert payload["train"]["output_dir"] == str(tmp_path / "ablation" / item.usage)
        expected_sampler = "auto" if item.usage == "loss" else "sample_weighted"
        assert payload["train"]["sampler_mode"] == expected_sampler


def test_summarize_ablation_results_selects_best_macro_f1(tmp_path: Path) -> None:
    from hard_finetune_ablation import summarize_ablation_results

    for usage, macro_f1, fog_f1 in [
        ("loss", 0.71, 0.62),
        ("sampler", 0.75, 0.69),
        ("both", 0.70, 0.58),
    ]:
        run_dir = tmp_path / usage
        run_dir.mkdir(parents=True)
        (run_dir / "training_summary.json").write_text(
            json.dumps(
                [
                    {
                        "fold": 0,
                        "best_macro_f1": macro_f1,
                        "per_class_f1": {"fog": fog_f1, "sunny": 0.8},
                    }
                ],
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

    summary = summarize_ablation_results(
        output_root=tmp_path,
        usages=["loss", "sampler", "both"],
    )

    assert summary["best_usage"] == "sampler"
    assert summary["results"][1]["usage"] == "sampler"
    assert summary["results"][1]["macro_f1_mean"] == 0.75
    assert summary["results"][1]["per_class_f1_mean"]["fog"] == 0.69


def test_summarize_ablation_results_rejects_unstable_tail_class_winner(tmp_path: Path) -> None:
    from hard_finetune_ablation import summarize_ablation_results

    for usage, macro_f1, fog_f1 in [
        ("loss", 0.72, 0.66),
        ("sampler", 0.74, 0.67),
        ("both", 0.78, 0.40),
    ]:
        run_dir = tmp_path / usage
        run_dir.mkdir(parents=True)
        (run_dir / "training_summary.json").write_text(
            json.dumps(
                [
                    {
                        "fold": 0,
                        "best_macro_f1": macro_f1,
                        "per_class_f1": {"fog": fog_f1, "sunny": 0.9},
                    }
                ],
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

    summary = summarize_ablation_results(
        output_root=tmp_path,
        usages=["loss", "sampler", "both"],
        baseline_usage="loss",
        min_per_class_f1=0.6,
        min_delta_macro_f1=0.0,
    )

    assert summary["best_usage"] == "sampler"
    rejected = {item["usage"]: item["selection_status"] for item in summary["results"]}
    assert rejected["both"] == "rejected_min_per_class_f1"


def test_summarize_ablation_results_tie_breaks_toward_lower_risk_usage(tmp_path: Path) -> None:
    from hard_finetune_ablation import summarize_ablation_results

    for usage, macro_f1 in [
        ("loss", 0.740),
        ("sampler", 0.741),
        ("both", 0.741),
    ]:
        run_dir = tmp_path / usage
        run_dir.mkdir(parents=True)
        (run_dir / "training_summary.json").write_text(
            json.dumps(
                [
                    {
                        "fold": 0,
                        "best_macro_f1": macro_f1,
                        "per_class_f1": {"fog": 0.7, "sunny": 0.8},
                    }
                ],
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

    summary = summarize_ablation_results(
        output_root=tmp_path,
        usages=["loss", "sampler", "both"],
        tie_epsilon=0.002,
    )

    assert summary["best_usage"] == "loss"


def test_hard_finetune_ablation_parse_args_accepts_usage_list(monkeypatch) -> None:
    from hard_finetune_ablation import parse_args

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "hard_finetune_ablation.py",
            "--base-config",
            "base.yaml",
            "--output-root",
            "outputs/ab",
            "--usage",
            "loss",
            "sampler",
            "--baseline-usage",
            "loss",
            "--min-per-class-f1",
            "0.6",
            "--min-delta-macro-f1",
            "0.005",
            "--tie-epsilon",
            "0.002",
            "--summarize",
        ],
    )

    args = parse_args()

    assert args.base_config == Path("base.yaml")
    assert args.output_root == Path("outputs/ab")
    assert args.usage == ["loss", "sampler"]
    assert args.baseline_usage == "loss"
    assert args.min_per_class_f1 == 0.6
    assert args.min_delta_macro_f1 == 0.005
    assert args.tie_epsilon == 0.002
    assert args.summarize is True


def test_write_summary_outputs_json_and_csv(tmp_path: Path) -> None:
    from hard_finetune_ablation import write_summary

    summary = {
        "best_usage": "sampler",
        "best_macro_f1_mean": 0.75,
        "results": [
            {
                "usage": "sampler",
                "macro_f1_mean": 0.75,
                "folds": 1,
                "output_dir": str(tmp_path / "sampler"),
            }
        ],
    }

    json_path, csv_path = write_summary(tmp_path, summary)

    assert json.loads(json_path.read_text(encoding="utf-8"))["best_usage"] == "sampler"
    assert "sampler,0.750000,1" in csv_path.read_text(encoding="utf-8")


def test_summarize_ablation_results_rejects_missing_macro_f1(tmp_path: Path) -> None:
    import pytest

    from hard_finetune_ablation import summarize_ablation_results

    run_dir = tmp_path / "loss"
    run_dir.mkdir(parents=True)
    (run_dir / "training_summary.json").write_text(
        json.dumps([{"fold": 0, "per_class_f1": {"fog": 0.5}}]),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="best_macro_f1"):
        summarize_ablation_results(tmp_path, usages=["loss"])
