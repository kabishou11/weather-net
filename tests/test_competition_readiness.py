from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
from PIL import Image


def _make_image(path: Path, color: tuple[int, int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (8, 8), color).save(path)


def _write_class_map(path: Path) -> None:
    path.write_text(json.dumps({"rain": 0, "sunny": 1}), encoding="utf-8")


def _write_config(path: Path, train_csv: Path, image_root: Path, class_map: Path, audit_json=None) -> None:
    audit_line = f"  external_audit_json: {audit_json}\n" if audit_json is not None else ""
    path.write_text(
        "data:\n"
        f"  train_csv: {train_csv}\n"
        f"  image_root: {image_root}\n"
        f"  class_map: {class_map}\n"
        f"{audit_line}"
        "  folds: 2\n"
        "train:\n"
        "  epochs: 1\n"
        "  batch_size: 2\n",
        encoding="utf-8",
    )


def _write_stats(path: Path, *, seconds: float = 0.1, checkpoints: list[str] | None = None, tta: bool = False) -> None:
    path.write_text(
        json.dumps(
            {
                "images": 10,
                "seconds": seconds,
                "checkpoints": checkpoints or ["soup.pt"],
                "tta": tta,
                "amp": True,
            }
        ),
        encoding="utf-8",
    )


def _write_decision_dir(path: Path, *, nested_accepted: bool = True) -> None:
    path.mkdir(parents=True, exist_ok=True)
    (path / "decision_params.json").write_text(
        json.dumps(
            {
                "version": 1,
                "class_names": ["rain", "sunny"],
                "weights": [1.0],
                "checkpoints": ["soup.pt"],
                "temperature": 1.0,
                "bias": [0.0, 0.0],
                "scores": {
                    "bias_accepted": 1.0 if nested_accepted else 0.0,
                    "nested_bias_accepted": 1.0 if nested_accepted else 0.0,
                },
            }
        ),
        encoding="utf-8",
    )
    (path / "nested_oof_scorecard.json").write_text(
        json.dumps(
            {
                "folds": [
                    {"fold": 0, "eval_rows": 2},
                    {"fold": 1, "eval_rows": 2},
                ],
                "pooled_scores": {
                    "ensemble_macro_f1": 0.8,
                    "temperature_macro_f1": 0.8,
                    "tuned_macro_f1": 0.81 if nested_accepted else 0.79,
                    "delta_tuned_vs_temperature_macro_f1": 0.01 if nested_accepted else -0.01,
                },
                "gate_threshold_delta_macro_f1": 0.0,
                "bias_accepted": 1 if nested_accepted else 0,
            }
        ),
        encoding="utf-8",
    )


def _write_training_output_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    checkpoint = path / "soup.pt"
    checkpoint.write_bytes(b"checkpoint")
    (path / "training_history.csv").write_text(
        "fold,epoch,train_loss,val_loss,macro_f1,accuracy\n0,1,0.7,0.6,0.8,0.85\n",
        encoding="utf-8",
    )
    (path / "training_history.json").write_text(
        json.dumps([{"fold": 0, "epoch": 1, "macro_f1": 0.8}]),
        encoding="utf-8",
    )
    (path / "training_curves.png").write_bytes(b"\x89PNG\r\n\x1a\n")
    oof_dir = path / "oof"
    oof_dir.mkdir()
    (oof_dir / "oof_metrics.json").write_text(
        json.dumps({"macro_f1": 0.8, "per_class_f1": {"rain": 0.8, "sunny": 0.8}}),
        encoding="utf-8",
    )
    (path / "training_summary.json").write_text(
        json.dumps(
            [
                {
                    "fold": 0,
                    "best_epoch": 1,
                    "best_macro_f1": 0.8,
                    "checkpoint": str(checkpoint),
                    "training_history_csv": str(path / "training_history.csv"),
                    "training_history_json": str(path / "training_history.json"),
                    "training_curves_png": str(path / "training_curves.png"),
                    "oof_metrics_json": str(oof_dir / "oof_metrics.json"),
                }
            ]
        ),
        encoding="utf-8",
    )


def _merge_fixture(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    from merge_external_training import merge_labeled_with_external_data

    image_root = tmp_path / "official"
    external_root = tmp_path / "external"
    _make_image(image_root / "rain1.jpg", (10, 20, 30))
    _make_image(image_root / "rain2.jpg", (11, 20, 30))
    _make_image(image_root / "sunny1.jpg", (30, 20, 10))
    _make_image(image_root / "sunny2.jpg", (30, 21, 10))
    _make_image(external_root / "rain_extra.jpg", (90, 91, 92))
    _make_image(external_root / "sunny_extra.jpg", (93, 94, 95))
    labeled_csv = tmp_path / "official.csv"
    external_csv = tmp_path / "external.csv"
    train_csv = tmp_path / "merged.csv"
    audit_json = tmp_path / "merged_audit.json"
    class_map = tmp_path / "class_to_idx.json"
    labeled_csv.write_text(
        "image,label\n"
        "rain1.jpg,rain\n"
        "rain2.jpg,rain\n"
        "sunny1.jpg,sunny\n"
        "sunny2.jpg,sunny\n",
        encoding="utf-8",
    )
    external_csv.write_text(
        "image,label,source,sample_weight\n"
        "rain_extra.jpg,rain,external_weapd,0.3\n"
        "sunny_extra.jpg,sunny,external_weapd,0.3\n",
        encoding="utf-8",
    )
    merge_labeled_with_external_data(
        train_csv=labeled_csv,
        train_dir=None,
        image_root=image_root,
        external_csv=external_csv,
        external_image_root=external_root,
        output_csv=train_csv,
        rejected_csv=tmp_path / "rejected.csv",
        audit_json=audit_json,
        output_image_root=tmp_path,
        max_external_sample_weight=0.5,
    )
    _write_class_map(class_map)
    return train_csv, tmp_path, class_map, audit_json


def test_readiness_fails_closed_when_server_strict_preflight_fails(tmp_path: Path) -> None:
    from competition_readiness import run_readiness_checks

    image_root = tmp_path / "images"
    _make_image(image_root / "rain.jpg", (10, 20, 30))
    train_csv = tmp_path / "train.csv"
    train_csv.write_text("image,label\nrain.jpg,rain\n", encoding="utf-8")
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "data:\n"
        f"  train_csv: {train_csv}\n"
        f"  image_root: {image_root}\n"
        "  folds: 2\n",
        encoding="utf-8",
    )

    report = run_readiness_checks(config_path=config_path)

    assert report["status"] == "fail"
    assert any(item["gate"] == "server_strict_preflight" for item in report["violations"])
    assert report["recommendation"] == "fix_readiness_violations_before_server_training"


def test_readiness_requires_nested_oof_scorecard_when_decision_required(tmp_path: Path) -> None:
    from competition_readiness import run_readiness_checks

    train_csv, image_root, class_map, audit_json = _merge_fixture(tmp_path)
    config_path = tmp_path / "config.yaml"
    _write_config(config_path, train_csv, image_root, class_map, audit_json)
    stats_path = tmp_path / "submission.stats.json"
    _write_stats(stats_path)
    decision_dir = tmp_path / "decision"
    decision_dir.mkdir()
    (decision_dir / "decision_params.json").write_text(
        json.dumps(
            {
                "version": 1,
                "class_names": ["rain", "sunny"],
                "temperature": 1.0,
                "bias": [0.0, 0.0],
                "weights": [1.0],
                "checkpoints": ["soup.pt"],
                "scores": {"bias_accepted": 1.0, "nested_bias_accepted": 1.0},
            }
        ),
        encoding="utf-8",
    )

    report = run_readiness_checks(
        config_path=config_path,
        allow_external_data=True,
        inference_stats=stats_path,
        decision_dir=decision_dir,
        require_nested_oof=True,
    )

    assert report["status"] == "fail"
    assert any(item["gate"] == "nested_oof_decision" for item in report["violations"])


def test_readiness_fails_on_inference_budget_violation(tmp_path: Path) -> None:
    from competition_readiness import run_readiness_checks

    train_csv, image_root, class_map, audit_json = _merge_fixture(tmp_path)
    config_path = tmp_path / "config.yaml"
    _write_config(config_path, train_csv, image_root, class_map, audit_json)
    stats_path = tmp_path / "slow.stats.json"
    _write_stats(stats_path, seconds=2.0, checkpoints=["a.pt", "b.pt"], tta=True)
    decision_dir = tmp_path / "decision"
    _write_decision_dir(decision_dir)

    report = run_readiness_checks(
        config_path=config_path,
        allow_external_data=True,
        inference_stats=stats_path,
        decision_dir=decision_dir,
        require_nested_oof=True,
        max_seconds_per_image=0.05,
        max_checkpoints=1,
        allow_tta=False,
    )

    assert report["status"] == "fail"
    assert any(item["gate"] == "inference_budget" for item in report["violations"])


def test_readiness_passes_clean_external_nested_and_budget_evidence(tmp_path: Path) -> None:
    from competition_readiness import run_readiness_checks

    train_csv, image_root, class_map, audit_json = _merge_fixture(tmp_path)
    config_path = tmp_path / "config.yaml"
    _write_config(config_path, train_csv, image_root, class_map, audit_json)
    stats_path = tmp_path / "submission.stats.json"
    _write_stats(stats_path)
    decision_dir = tmp_path / "decision"
    _write_decision_dir(decision_dir)

    report = run_readiness_checks(
        config_path=config_path,
        allow_external_data=True,
        inference_stats=stats_path,
        decision_dir=decision_dir,
        require_nested_oof=True,
    )

    assert report["status"] == "pass"
    assert report["recommendation"] == "ready_for_server_training"
    assert report["gates"]["server_strict_preflight"]["status"] == "pass"
    assert report["gates"]["nested_oof_decision"]["status"] == "pass"
    assert report["gates"]["inference_budget"]["status"] == "pass"


def test_readiness_checks_training_output_artifacts(tmp_path: Path) -> None:
    from competition_readiness import run_readiness_checks

    train_csv, image_root, class_map, audit_json = _merge_fixture(tmp_path)
    config_path = tmp_path / "config.yaml"
    _write_config(config_path, train_csv, image_root, class_map, audit_json)
    training_output_dir = tmp_path / "outputs"
    _write_training_output_dir(training_output_dir)

    report = run_readiness_checks(
        config_path=config_path,
        allow_external_data=True,
        training_output_dir=training_output_dir,
    )

    assert report["status"] == "pass"
    assert report["gates"]["training_artifacts"]["status"] == "pass"
    assert report["gates"]["training_artifacts"]["summary"] == str(training_output_dir / "training_summary.json")


def test_readiness_fails_when_training_curve_is_missing(tmp_path: Path) -> None:
    from competition_readiness import run_readiness_checks

    train_csv, image_root, class_map, audit_json = _merge_fixture(tmp_path)
    config_path = tmp_path / "config.yaml"
    _write_config(config_path, train_csv, image_root, class_map, audit_json)
    training_output_dir = tmp_path / "outputs"
    _write_training_output_dir(training_output_dir)
    (training_output_dir / "training_curves.png").unlink()

    report = run_readiness_checks(
        config_path=config_path,
        allow_external_data=True,
        training_output_dir=training_output_dir,
    )

    assert report["status"] == "fail"
    assert any(item["gate"] == "training_artifacts" for item in report["violations"])


def test_competition_readiness_cli_writes_report_and_exits_nonzero_on_failure(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from competition_readiness import main

    config_path = tmp_path / "config.yaml"
    config_path.write_text("data:\n  folds: 2\n", encoding="utf-8")
    output_path = tmp_path / "readiness.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "competition_readiness.py",
            "--config",
            str(config_path),
            "--output",
            str(output_path),
        ],
    )

    with pytest.raises(SystemExit) as error:
        main()

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert error.value.code == 1
    assert payload["status"] == "fail"
