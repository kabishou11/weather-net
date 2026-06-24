import json
from pathlib import Path


def test_build_rebalance_grid_generates_tau_and_crt_candidates(tmp_path: Path) -> None:
    from classifier_rebalance_grid import build_rebalance_grid

    candidates = build_rebalance_grid(
        checkpoint=Path("fold0.pt"),
        output_dir=tmp_path,
        tau_values=[0.5, 1.0],
        crt_sampler_modes=["sqrt", "class_balanced"],
        lws_sampler_modes=["sqrt"],
    )

    assert [candidate.name for candidate in candidates] == [
        "tau0p50",
        "tau1p00",
        "crt_sqrt",
        "crt_class_balanced",
        "lws_sqrt",
    ]
    assert candidates[0].method == "tau_norm"
    assert candidates[0].output == tmp_path / "fold0_tau0p50.pt"
    assert candidates[2].method == "crt"
    assert candidates[2].sampler_mode == "sqrt"
    assert candidates[4].method == "lws"
    assert candidates[4].output == tmp_path / "fold0_lws_sqrt.pt"


def test_build_rebalance_grid_rejects_duplicate_candidate_names(tmp_path: Path) -> None:
    import pytest

    from classifier_rebalance_grid import build_rebalance_grid

    with pytest.raises(ValueError, match="duplicate"):
        build_rebalance_grid(
            checkpoint=Path("fold0.pt"),
            output_dir=tmp_path,
            tau_values=[0.124, 0.12],
            crt_sampler_modes=[],
        )
    with pytest.raises(ValueError, match="duplicate"):
        build_rebalance_grid(
            checkpoint=Path("fold0.pt"),
            output_dir=tmp_path,
            tau_values=[],
            crt_sampler_modes=["sqrt", "sqrt"],
        )


def test_summarize_rebalance_grid_selects_stable_macro_f1_winner(tmp_path: Path) -> None:
    from classifier_rebalance_grid import RebalanceCandidate, summarize_rebalance_grid

    candidates = [
        RebalanceCandidate(name="baseline", method="baseline", checkpoint=Path("base.pt"), output=Path("base.pt")),
        RebalanceCandidate(name="tau0p50", method="tau_norm", checkpoint=Path("base.pt"), output=Path("tau.pt"), tau=0.5),
        RebalanceCandidate(
            name="crt_sqrt",
            method="crt",
            checkpoint=Path("base.pt"),
            output=Path("crt.pt"),
            sampler_mode="sqrt",
        ),
    ]
    for name, macro_f1, fog_f1 in [
        ("baseline", 0.740, 0.650),
        ("tau0p50", 0.748, 0.670),
        ("crt_sqrt", 0.755, 0.510),
    ]:
        path = tmp_path / f"{name}.json"
        path.write_text(
            json.dumps(
                {
                    "macro_f1": macro_f1,
                    "per_class_f1": {"fog": fog_f1, "sunny": 0.9},
                }
            ),
            encoding="utf-8",
        )
        for candidate in candidates:
            if candidate.name == name:
                candidate.metrics_json = path

    summary = summarize_rebalance_grid(
        candidates,
        baseline_name="baseline",
        min_delta_macro_f1=0.003,
        min_per_class_f1=0.6,
        tie_epsilon=0.001,
    )

    assert summary["best_name"] == "tau0p50"
    rejected = {item["name"]: item["selection_status"] for item in summary["results"]}
    assert rejected["crt_sqrt"] == "rejected_min_per_class_f1"
    assert summary["results"][1]["delta_macro_f1_vs_baseline"] == 0.008000000000000007


def test_run_rebalance_grid_dispatches_tau_and_crt(monkeypatch, tmp_path: Path) -> None:
    import classifier_rebalance_grid
    from classifier_rebalance_grid import run_rebalance_grid

    calls: list[tuple[str, Path, object, object]] = []

    def fake_tau(checkpoint, output, tau, head_key="auto"):
        calls.append(("tau", output, tau, head_key))
        return {"output": str(output)}

    def fake_crt(**kwargs):
        calls.append(
            (
                kwargs.get("rebalance_method", "crt"),
                kwargs["output_path"],
                kwargs["sampler_mode"],
                kwargs["head_prefix"],
            )
        )
        return {"output": str(kwargs["output_path"])}

    monkeypatch.setattr(classifier_rebalance_grid, "rebalance_checkpoint", fake_tau)
    monkeypatch.setattr(classifier_rebalance_grid, "retrain_classifier_head", fake_crt)

    candidates = run_rebalance_grid(
        checkpoint=Path("fold0.pt"),
        output_dir=tmp_path,
        tau_values=[0.5],
        crt_sampler_modes=["sqrt"],
        lws_sampler_modes=["sqrt"],
        train_csv=Path("train.csv"),
        run=True,
        epochs=2,
        head_key="head.fc.weight",
        head_prefix="head.fc",
    )

    assert [candidate.name for candidate in candidates] == ["tau0p50", "crt_sqrt", "lws_sqrt"]
    assert calls == [
        ("tau", tmp_path / "fold0_tau0p50.pt", 0.5, "head.fc.weight"),
        ("crt", tmp_path / "fold0_crt_sqrt.pt", "sqrt", "head.fc"),
        ("lws", tmp_path / "fold0_lws_sqrt.pt", "sqrt", "head.fc"),
    ]


def test_validate_rebalance_grid_writes_baseline_and_candidate_metrics(monkeypatch, tmp_path: Path) -> None:
    import classifier_rebalance_grid
    from classifier_rebalance_grid import build_rebalance_grid, validate_rebalance_grid

    calls: list[Path] = []

    def fake_validate_checkpoint(**kwargs):
        checkpoint = Path(kwargs["checkpoint"])
        calls.append(checkpoint)
        macro_f1 = 0.7
        if "tau" in checkpoint.stem:
            macro_f1 = 0.73
        if "crt" in checkpoint.stem:
            macro_f1 = 0.72
        return {"macro_f1": macro_f1, "per_class_f1": {"rain": 0.7, "sunny": 0.8}}

    monkeypatch.setattr(classifier_rebalance_grid, "validate_checkpoint", fake_validate_checkpoint)
    candidates = build_rebalance_grid(
        checkpoint=Path("fold0.pt"),
        output_dir=tmp_path,
        tau_values=[0.5],
        crt_sampler_modes=["sqrt"],
        lws_sampler_modes=[],
    )

    baseline = validate_rebalance_grid(
        checkpoint=Path("fold0.pt"),
        candidates=candidates,
        output_dir=tmp_path,
        val_csv=Path("val.csv"),
        val_image_root=Path("images"),
        batch_size=16,
        device="cpu",
    )

    assert baseline.metrics_json == tmp_path / "baseline_metrics.json"
    assert [candidate.metrics_json for candidate in candidates] == [
        tmp_path / "tau0p50_metrics.json",
        tmp_path / "crt_sqrt_metrics.json",
    ]
    assert json.loads((tmp_path / "baseline_metrics.json").read_text(encoding="utf-8"))["macro_f1"] == 0.7
    assert json.loads((tmp_path / "tau0p50_metrics.json").read_text(encoding="utf-8"))["macro_f1"] == 0.73
    assert calls == [Path("fold0.pt"), tmp_path / "fold0_tau0p50.pt", tmp_path / "fold0_crt_sqrt.pt"]


def test_parse_args_accepts_rebalance_grid_controls(monkeypatch, tmp_path: Path) -> None:
    import sys

    from classifier_rebalance_grid import parse_args

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "classifier_rebalance_grid.py",
            "--checkpoint",
            "fold0.pt",
            "--output-dir",
            str(tmp_path),
            "--tau",
            "0.5",
            "1.0",
            "--crt-sampler-mode",
            "sqrt",
            "class_balanced",
            "--lws-sampler-mode",
            "sqrt",
            "--head-key",
            "head.fc.weight",
            "--head-prefix",
            "head.fc",
            "--validate",
            "--val-csv",
            "val.csv",
            "--val-image-root",
            "images",
            "--val-batch-size",
            "16",
            "--min-delta-macro-f1",
            "0.003",
            "--min-per-class-f1",
            "0.6",
        ],
    )

    args = parse_args()

    assert args.checkpoint == Path("fold0.pt")
    assert args.output_dir == tmp_path
    assert args.tau == [0.5, 1.0]
    assert args.crt_sampler_mode == ["sqrt", "class_balanced"]
    assert args.lws_sampler_mode == ["sqrt"]
    assert args.head_key == "head.fc.weight"
    assert args.head_prefix == "head.fc"
    assert args.validate is True
    assert args.val_csv == Path("val.csv")
    assert args.val_image_root == Path("images")
    assert args.val_batch_size == 16
    assert args.min_delta_macro_f1 == 0.003
    assert args.min_per_class_f1 == 0.6


def test_validate_cli_args_rejects_mixed_manual_and_auto_metrics() -> None:
    import argparse

    import pytest

    from classifier_rebalance_grid import validate_cli_args

    args = argparse.Namespace(
        validate=True,
        val_dir=None,
        val_csv=Path("val.csv"),
        baseline_metrics=Path("baseline.json"),
        candidate_metrics=None,
    )

    with pytest.raises(ValueError, match="Do not combine"):
        validate_cli_args(args)


def test_validate_cli_args_requires_validation_manifest() -> None:
    import argparse

    import pytest

    from classifier_rebalance_grid import validate_cli_args

    args = argparse.Namespace(
        validate=True,
        val_dir=None,
        val_csv=None,
        baseline_metrics=None,
        candidate_metrics=None,
    )

    with pytest.raises(ValueError, match="requires --val-dir or --val-csv"):
        validate_cli_args(args)
