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


def test_build_fold_safe_rebalance_manifests_keeps_validation_labeled_only(tmp_path: Path) -> None:
    import csv
    import json

    from classifier_rebalance_grid import build_fold_safe_rebalance_manifests

    images = tmp_path / "images"
    images.mkdir()
    for name in ["rain1.jpg", "rain2.jpg", "sun1.jpg", "sun2.jpg", "pseudo.jpg", "external.jpg"]:
        (images / name).write_bytes(b"not-read-here")
    train_csv = tmp_path / "train.csv"
    train_csv.write_text(
        "image,label,source,confidence,sample_weight,teacher_rain,teacher_sunny\n"
        "rain1.jpg,rain,labeled,1.0,1.0,0.9,0.1\n"
        "rain2.jpg,rain,labeled,1.0,1.0,0.8,0.2\n"
        "sun1.jpg,sunny,labeled,1.0,1.0,0.1,0.9\n"
        "sun2.jpg,sunny,labeled,1.0,1.0,0.2,0.8\n"
        "pseudo.jpg,rain,pseudo,0.97,0.979,0.95,0.05\n"
        "external.jpg,sunny,external_weather,1.0,0.3,0.05,0.95\n",
        encoding="utf-8",
    )
    class_map = tmp_path / "class_to_idx.json"
    class_map.write_text(json.dumps({"rain": 0, "sunny": 1}), encoding="utf-8")

    summary = build_fold_safe_rebalance_manifests(
        train_csv=train_csv,
        train_dir=None,
        image_root=images,
        class_map=class_map,
        output_dir=tmp_path / "folds",
        folds=2,
        seed=7,
    )

    assert summary["fold_count"] == 2
    all_val_ids: set[str] = set()
    for item in summary["folds"]:
        train_rows = list(csv.DictReader(open(item["train_csv"], encoding="utf-8")))
        val_rows = list(csv.DictReader(open(item["val_csv"], encoding="utf-8")))
        assert item["val_source_counts"] == {"labeled": 2}
        assert item["train_source_counts"] == {"labeled": 2}
        assert all(row["source"] == "labeled" for row in val_rows)
        assert all(row["source"] == "labeled" for row in train_rows)
        assert "pseudo.jpg" not in {row["image"] for row in train_rows}
        assert "external.jpg" not in {row["image"] for row in train_rows}
        train_labeled_ids = {row["image"] for row in train_rows if row["source"] == "labeled"}
        val_ids = {row["image"] for row in val_rows}
        assert not (train_labeled_ids & val_ids)
        all_val_ids.update(val_ids)
        assert train_rows[0]["teacher_rain"] != ""
        assert train_rows[0]["sample_weight"] != ""
    assert all_val_ids == {"rain1.jpg", "rain2.jpg", "sun1.jpg", "sun2.jpg"}
    assert summary["excluded_source_counts"] == {"external_weather": 1, "pseudo": 1}
    assert summary["excluded_rows"] == 2
    assert summary["labeled_rows"] == 4
    summary_json = tmp_path / "folds" / "fold_safe_rebalance_manifests.json"
    assert json.loads(summary_json.read_text(encoding="utf-8"))["fold_count"] == 2


def test_build_fold_safe_rebalance_manifests_requires_class_map_for_csv(tmp_path: Path) -> None:
    import pytest

    from classifier_rebalance_grid import build_fold_safe_rebalance_manifests

    train_csv = tmp_path / "train.csv"
    train_csv.write_text("image,label\nrain.jpg,rain\n", encoding="utf-8")

    with pytest.raises(ValueError, match="requires --class-map"):
        build_fold_safe_rebalance_manifests(
            train_csv=train_csv,
            train_dir=None,
            image_root=tmp_path,
            class_map=None,
            output_dir=tmp_path / "folds",
            folds=2,
            seed=42,
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

    calls: list[tuple[str, Path, object, object, object]] = []

    def fake_tau(checkpoint, output, tau, head_key="auto"):
        calls.append(("tau", output, tau, head_key, None))
        return {"output": str(output)}

    def fake_crt(**kwargs):
        calls.append(
            (
                kwargs.get("rebalance_method", "crt"),
                kwargs["output_path"],
                kwargs["sampler_mode"],
                kwargs["head_prefix"],
                kwargs["rebalance_manifest_summary"],
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
        rebalance_manifest_summary=Path("fold_safe_rebalance_manifests.json"),
    )

    assert [candidate.name for candidate in candidates] == ["tau0p50", "crt_sqrt", "lws_sqrt"]
    assert calls == [
        ("tau", tmp_path / "fold0_tau0p50.pt", 0.5, "head.fc.weight", None),
        ("crt", tmp_path / "fold0_crt_sqrt.pt", "sqrt", "head.fc", Path("fold_safe_rebalance_manifests.json")),
        ("lws", tmp_path / "fold0_lws_sqrt.pt", "sqrt", "head.fc", Path("fold_safe_rebalance_manifests.json")),
    ]


def test_run_rebalance_grid_requires_summary_for_crt_lws_run(tmp_path: Path) -> None:
    import pytest

    from classifier_rebalance_grid import run_rebalance_grid

    with pytest.raises(ValueError, match="rebalance-manifest-summary"):
        run_rebalance_grid(
            checkpoint=Path("fold0.pt"),
            output_dir=tmp_path,
            tau_values=[],
            crt_sampler_modes=["sqrt"],
            lws_sampler_modes=[],
            train_csv=Path("fold0_rebalance_train.csv"),
            run=True,
        )


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


def test_summarize_fold_rebalance_grid_selects_cross_fold_winner(tmp_path: Path) -> None:
    from classifier_rebalance_grid import RebalanceCandidate, summarize_fold_rebalance_grid

    def candidate(fold: int, name: str, method: str, macro_f1: float, fog_f1: float) -> RebalanceCandidate:
        path = tmp_path / f"fold{fold}_{name}.json"
        path.write_text(
            json.dumps(
                {
                    "macro_f1": macro_f1,
                    "per_class_f1": {"fog": fog_f1, "sunny": 0.9},
                }
            ),
            encoding="utf-8",
        )
        return RebalanceCandidate(
            name=name,
            method=method,
            checkpoint=Path(f"fold{fold}.pt"),
            output=Path(f"fold{fold}_{name}.pt"),
            metrics_json=path,
        )

    summary = summarize_fold_rebalance_grid(
        {
            0: [
                candidate(0, "baseline", "baseline", 0.70, 0.62),
                candidate(0, "tau0p50", "tau_norm", 0.73, 0.64),
                candidate(0, "crt_sqrt", "crt", 0.76, 0.50),
            ],
            1: [
                candidate(1, "baseline", "baseline", 0.72, 0.63),
                candidate(1, "tau0p50", "tau_norm", 0.75, 0.66),
                candidate(1, "crt_sqrt", "crt", 0.77, 0.52),
            ],
        },
        min_delta_macro_f1=0.01,
        min_per_class_f1=0.6,
        tie_epsilon=0.001,
    )

    assert summary["fold_count"] == 2
    assert summary["best_name"] == "tau0p50"
    assert summary["baseline_mean_macro_f1"] == 0.71
    results = {item["name"]: item for item in summary["results"]}
    assert results["tau0p50"]["mean_macro_f1"] == 0.74
    assert results["tau0p50"]["delta_mean_macro_f1_vs_baseline"] == 0.030000000000000027
    assert results["crt_sqrt"]["selection_status"] == "rejected_min_per_class_f1"


def test_run_all_fold_rebalance_grid_uses_fold_safe_train_and_val(monkeypatch, tmp_path: Path) -> None:
    import classifier_rebalance_grid
    from classifier_rebalance_grid import RebalanceCandidate, run_all_fold_rebalance_grid

    summary_path = tmp_path / "fold_safe_rebalance_manifests.json"
    fold_entries = []
    for fold in [0, 1]:
        train_csv = tmp_path / f"fold{fold}_rebalance_train.csv"
        val_csv = tmp_path / f"fold{fold}_rebalance_val.csv"
        train_csv.write_text("image,label,source\n", encoding="utf-8")
        val_csv.write_text("image,label,source\n", encoding="utf-8")
        fold_entries.append(
            {
                "fold": fold,
                "train_csv": str(train_csv),
                "val_csv": str(val_csv),
                "train_rows": 4,
                "val_rows": 2,
                "train_source_counts": {"labeled": 4},
                "val_source_counts": {"labeled": 2},
                "train_row_digest": f"train-{fold}",
                "val_row_digest": f"val-{fold}",
            }
        )
    summary_path.write_text(json.dumps({"folds": fold_entries, "fold_count": 2}), encoding="utf-8")
    calls: list[tuple[str, int, Path, Path]] = []

    monkeypatch.setattr(
        classifier_rebalance_grid,
        "_checkpoint_fold_from_path",
        lambda path: 0 if Path(path).name == "fold0.pt" else 1,
    )

    def fake_run_rebalance_grid(**kwargs):
        fold = 0 if Path(kwargs["checkpoint"]).name == "fold0.pt" else 1
        calls.append(("run", fold, Path(kwargs["train_csv"]), Path(kwargs["output_dir"])))
        return [
            RebalanceCandidate(
                name="tau0p50",
                method="tau_norm",
                checkpoint=Path(kwargs["checkpoint"]),
                output=Path(kwargs["output_dir"]) / f"fold{fold}_tau.pt",
                tau=0.5,
            )
        ]

    def fake_validate_rebalance_grid(**kwargs):
        checkpoint = Path(kwargs["checkpoint"])
        fold = 0 if checkpoint.name == "fold0.pt" else 1
        output_dir = Path(kwargs["output_dir"])
        calls.append(("validate", fold, Path(kwargs["val_csv"]), output_dir))
        baseline = RebalanceCandidate(
            name="baseline",
            method="baseline",
            checkpoint=checkpoint,
            output=checkpoint,
            metrics_json=output_dir / "baseline_metrics.json",
        )
        baseline.metrics_json.write_text(
            json.dumps({"macro_f1": 0.70 + fold * 0.02, "per_class_f1": {"fog": 0.62, "sunny": 0.8}}),
            encoding="utf-8",
        )
        for candidate_item in kwargs["candidates"]:
            candidate_item.metrics_json = output_dir / f"{candidate_item.name}_metrics.json"
            candidate_item.metrics_json.write_text(
                json.dumps({"macro_f1": 0.74 + fold * 0.02, "per_class_f1": {"fog": 0.65, "sunny": 0.82}}),
                encoding="utf-8",
            )
        return baseline

    monkeypatch.setattr(classifier_rebalance_grid, "run_rebalance_grid", fake_run_rebalance_grid)
    monkeypatch.setattr(classifier_rebalance_grid, "validate_rebalance_grid", fake_validate_rebalance_grid)

    result = run_all_fold_rebalance_grid(
        checkpoints=[Path("fold1.pt"), Path("fold0.pt")],
        rebalance_manifest_summary=summary_path,
        output_dir=tmp_path / "grid",
        tau_values=[0.5],
        crt_sampler_modes=[],
        lws_sampler_modes=[],
        run=True,
        validate=True,
        image_root=Path("images"),
        min_delta_macro_f1=0.01,
        min_per_class_f1=0.6,
    )

    assert calls == [
        ("run", 0, tmp_path / "fold0_rebalance_train.csv", tmp_path / "grid" / "fold0"),
        ("validate", 0, tmp_path / "fold0_rebalance_val.csv", tmp_path / "grid" / "fold0"),
        ("run", 1, tmp_path / "fold1_rebalance_train.csv", tmp_path / "grid" / "fold1"),
        ("validate", 1, tmp_path / "fold1_rebalance_val.csv", tmp_path / "grid" / "fold1"),
    ]
    assert result["aggregate"]["best_name"] == "tau0p50"
    assert result["aggregate"]["fold_count"] == 2


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
            "--class-map",
            "class_to_idx.json",
            "--prepare-folds",
            "--folds",
            "5",
            "--seed",
            "123",
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

    assert args.checkpoint == [Path("fold0.pt")]
    assert args.output_dir == tmp_path
    assert args.tau == [0.5, 1.0]
    assert args.crt_sampler_mode == ["sqrt", "class_balanced"]
    assert args.lws_sampler_mode == ["sqrt"]
    assert args.class_map == Path("class_to_idx.json")
    assert args.prepare_folds is True
    assert args.folds == 5
    assert args.seed == 123
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
        checkpoint=Path("fold0.pt"),
        run=False,
        validate=True,
        val_dir=None,
        val_csv=Path("val.csv"),
        baseline_metrics=Path("baseline.json"),
        candidate_metrics=None,
        crt_sampler_mode=[],
        lws_sampler_mode=[],
        rebalance_manifest_summary=None,
    )

    with pytest.raises(ValueError, match="Do not combine"):
        validate_cli_args(args)


def test_validate_cli_args_requires_validation_manifest() -> None:
    import argparse

    import pytest

    from classifier_rebalance_grid import validate_cli_args

    args = argparse.Namespace(
        checkpoint=Path("fold0.pt"),
        run=False,
        validate=True,
        val_dir=None,
        val_csv=None,
        baseline_metrics=None,
        candidate_metrics=None,
        crt_sampler_mode=[],
        lws_sampler_mode=[],
        rebalance_manifest_summary=None,
    )

    with pytest.raises(ValueError, match="requires --val-dir or --val-csv"):
        validate_cli_args(args)


def test_validate_cli_args_requires_class_map_for_fold_safe_csv() -> None:
    import argparse

    import pytest

    from classifier_rebalance_grid import validate_cli_args

    args = argparse.Namespace(
        validate=False,
        run=False,
        val_dir=None,
        val_csv=None,
        baseline_metrics=None,
        candidate_metrics=None,
        prepare_folds=True,
        train_csv=Path("train.csv"),
        train_dir=None,
        class_map=None,
        crt_sampler_mode=[],
        lws_sampler_mode=[],
        rebalance_manifest_summary=None,
    )

    with pytest.raises(ValueError, match="requires --class-map"):
        validate_cli_args(args)


def test_validate_cli_args_allows_prepare_folds_without_checkpoint() -> None:
    import argparse

    from classifier_rebalance_grid import validate_cli_args

    args = argparse.Namespace(
        checkpoint=None,
        run=False,
        validate=False,
        val_dir=None,
        val_csv=None,
        baseline_metrics=None,
        candidate_metrics=None,
        prepare_folds=True,
        train_csv=None,
        train_dir=Path("train_dir"),
        class_map=None,
        crt_sampler_mode=[],
        lws_sampler_mode=[],
        rebalance_manifest_summary=None,
    )

    validate_cli_args(args)


def test_validate_cli_args_requires_checkpoint_for_grid_run() -> None:
    import argparse

    import pytest

    from classifier_rebalance_grid import validate_cli_args

    args = argparse.Namespace(
        checkpoint=None,
        run=False,
        validate=False,
        val_dir=None,
        val_csv=None,
        baseline_metrics=None,
        candidate_metrics=None,
        prepare_folds=False,
        train_csv=None,
        train_dir=None,
        class_map=None,
        crt_sampler_mode=[],
        lws_sampler_mode=[],
        rebalance_manifest_summary=None,
    )

    with pytest.raises(ValueError, match="--checkpoint"):
        validate_cli_args(args)


def test_validate_cli_args_requires_all_fold_summary_and_validation() -> None:
    import argparse

    import pytest

    from classifier_rebalance_grid import validate_cli_args

    args = argparse.Namespace(
        checkpoint=[Path("fold0.pt"), Path("fold1.pt")],
        run=True,
        validate=False,
        val_dir=None,
        val_csv=None,
        baseline_metrics=None,
        candidate_metrics=None,
        prepare_folds=False,
        train_csv=None,
        train_dir=None,
        class_map=None,
        crt_sampler_mode=["sqrt"],
        lws_sampler_mode=[],
        rebalance_manifest_summary=Path("fold_safe_rebalance_manifests.json"),
    )

    with pytest.raises(ValueError, match="requires --validate"):
        validate_cli_args(args)

    args.validate = True
    args.rebalance_manifest_summary = None
    with pytest.raises(ValueError, match="requires --rebalance-manifest-summary"):
        validate_cli_args(args)

    args.rebalance_manifest_summary = Path("fold_safe_rebalance_manifests.json")
    args.train_csv = Path("fold0_rebalance_train.csv")
    with pytest.raises(ValueError, match="per-fold train_csv"):
        validate_cli_args(args)


def test_validate_cli_args_allows_all_fold_validation_without_val_csv() -> None:
    import argparse

    from classifier_rebalance_grid import validate_cli_args

    args = argparse.Namespace(
        checkpoint=[Path("fold0.pt"), Path("fold1.pt")],
        run=True,
        validate=True,
        val_dir=None,
        val_csv=None,
        baseline_metrics=None,
        candidate_metrics=None,
        prepare_folds=False,
        train_csv=None,
        train_dir=None,
        class_map=None,
        crt_sampler_mode=["sqrt"],
        lws_sampler_mode=[],
        rebalance_manifest_summary=Path("fold_safe_rebalance_manifests.json"),
    )

    validate_cli_args(args)


def test_validate_cli_args_rejects_prepare_folds_with_run_or_validate() -> None:
    import argparse

    import pytest

    from classifier_rebalance_grid import validate_cli_args

    args = argparse.Namespace(
        checkpoint=None,
        run=True,
        validate=False,
        val_dir=None,
        val_csv=None,
        baseline_metrics=None,
        candidate_metrics=None,
        prepare_folds=True,
        train_csv=None,
        train_dir=Path("train_dir"),
        class_map=None,
        crt_sampler_mode=[],
        lws_sampler_mode=[],
        rebalance_manifest_summary=None,
    )

    with pytest.raises(ValueError, match="cannot be combined"):
        validate_cli_args(args)

    args.run = False
    args.validate = True
    with pytest.raises(ValueError, match="cannot be combined"):
        validate_cli_args(args)


def test_validate_cli_args_rejects_prepare_folds_with_ignored_runtime_inputs() -> None:
    import argparse

    import pytest

    from classifier_rebalance_grid import validate_cli_args

    args = argparse.Namespace(
        checkpoint=Path("fold0.pt"),
        run=False,
        validate=False,
        val_dir=None,
        val_csv=None,
        baseline_metrics=None,
        candidate_metrics=None,
        prepare_folds=True,
        train_csv=None,
        train_dir=Path("train_dir"),
        class_map=None,
        crt_sampler_mode=[],
        lws_sampler_mode=[],
        rebalance_manifest_summary=None,
    )

    with pytest.raises(ValueError, match="cannot be combined"):
        validate_cli_args(args)

    args.checkpoint = None
    args.val_csv = Path("val.csv")
    with pytest.raises(ValueError, match="cannot be combined"):
        validate_cli_args(args)

    args.val_csv = None
    args.rebalance_manifest_summary = Path("fold_safe_rebalance_manifests.json")
    with pytest.raises(ValueError, match="cannot be combined"):
        validate_cli_args(args)


def test_validate_cli_args_requires_rebalance_summary_for_crt_lws_run() -> None:
    import argparse

    import pytest

    from classifier_rebalance_grid import validate_cli_args

    args = argparse.Namespace(
        checkpoint=Path("fold0.pt"),
        run=True,
        validate=False,
        val_dir=None,
        val_csv=None,
        baseline_metrics=None,
        candidate_metrics=None,
        prepare_folds=False,
        train_csv=Path("fold0_rebalance_train.csv"),
        train_dir=None,
        class_map=None,
        crt_sampler_mode=["sqrt"],
        lws_sampler_mode=[],
        rebalance_manifest_summary=None,
    )

    with pytest.raises(ValueError, match="rebalance-manifest-summary"):
        validate_cli_args(args)
