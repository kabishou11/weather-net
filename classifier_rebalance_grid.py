from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from classifier_rebalance import rebalance_checkpoint, retrain_classifier_head
from validate import validate_checkpoint


@dataclass
class RebalanceCandidate:
    name: str
    method: str
    checkpoint: Path
    output: Path
    tau: float | None = None
    sampler_mode: str | None = None
    metrics_json: Path | None = None


def _tau_name(value: float) -> str:
    return f"tau{value:.2f}".replace(".", "p")


def _validate_unique_candidates(candidates: Sequence[RebalanceCandidate]) -> None:
    names = [candidate.name for candidate in candidates]
    outputs = [candidate.output for candidate in candidates]
    if len(set(names)) != len(names):
        raise ValueError("duplicate rebalance candidate names generated")
    if len(set(outputs)) != len(outputs):
        raise ValueError("duplicate rebalance candidate outputs generated")


def build_rebalance_grid(
    checkpoint: Path,
    output_dir: Path,
    tau_values: Sequence[float],
    crt_sampler_modes: Sequence[str],
    lws_sampler_modes: Sequence[str] = (),
) -> list[RebalanceCandidate]:
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = checkpoint.stem
    candidates: list[RebalanceCandidate] = []
    for tau in tau_values:
        value = float(tau)
        if value < 0:
            raise ValueError("tau values must be non-negative")
        name = _tau_name(value)
        candidates.append(
            RebalanceCandidate(
                name=name,
                method="tau_norm",
                checkpoint=checkpoint,
                output=output_dir / f"{stem}_{name}.pt",
                tau=value,
            )
        )
    valid_sampler_modes = {"none", "sqrt", "class_balanced"}
    for sampler_mode in crt_sampler_modes:
        if sampler_mode not in valid_sampler_modes:
            raise ValueError("crt sampler modes must be one of: none, sqrt, class_balanced")
        name = f"crt_{sampler_mode}"
        candidates.append(
            RebalanceCandidate(
                name=name,
                method="crt",
                checkpoint=checkpoint,
                output=output_dir / f"{stem}_{name}.pt",
                sampler_mode=sampler_mode,
            )
        )
    for sampler_mode in lws_sampler_modes:
        if sampler_mode not in valid_sampler_modes:
            raise ValueError("lws sampler modes must be one of: none, sqrt, class_balanced")
        name = f"lws_{sampler_mode}"
        candidates.append(
            RebalanceCandidate(
                name=name,
                method="lws",
                checkpoint=checkpoint,
                output=output_dir / f"{stem}_{name}.pt",
                sampler_mode=sampler_mode,
            )
        )
    _validate_unique_candidates(candidates)
    return candidates


def _read_metrics(path: Path) -> dict[str, object]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if "macro_f1" not in data:
        raise ValueError(f"metrics JSON missing macro_f1: {path}")
    per_class = data.get("per_class_f1", {})
    if not isinstance(per_class, dict):
        raise ValueError(f"metrics JSON per_class_f1 must be a mapping: {path}")
    return data


def summarize_rebalance_grid(
    candidates: Sequence[RebalanceCandidate],
    baseline_name: str = "baseline",
    min_delta_macro_f1: float = 0.0,
    min_per_class_f1: float = 0.0,
    tie_epsilon: float = 0.0,
) -> dict[str, object]:
    if not 0 <= min_per_class_f1 <= 1:
        raise ValueError("min_per_class_f1 must be in [0, 1]")
    if tie_epsilon < 0:
        raise ValueError("tie_epsilon must be non-negative")
    results: list[dict[str, object]] = []
    baseline_macro_f1: float | None = None
    for candidate in candidates:
        if candidate.metrics_json is None:
            raise ValueError(f"candidate metrics_json is required: {candidate.name}")
        metrics = _read_metrics(candidate.metrics_json)
        per_class = {str(label): float(score) for label, score in metrics.get("per_class_f1", {}).items()}
        macro_f1 = float(metrics["macro_f1"])
        min_class_f1 = min(per_class.values()) if per_class else 0.0
        item = {
            "name": candidate.name,
            "method": candidate.method,
            "checkpoint": str(candidate.output),
            "metrics_json": str(candidate.metrics_json),
            "macro_f1": macro_f1,
            "per_class_f1": per_class,
            "min_per_class_f1": float(min_class_f1),
            "tau": candidate.tau,
            "sampler_mode": candidate.sampler_mode,
        }
        results.append(item)
        if candidate.name == baseline_name:
            baseline_macro_f1 = macro_f1
    if baseline_macro_f1 is None:
        raise ValueError(f"Missing baseline candidate: {baseline_name}")
    for item in results:
        delta = float(item["macro_f1"]) - baseline_macro_f1
        item["delta_macro_f1_vs_baseline"] = float(delta)
        if float(item["min_per_class_f1"]) < min_per_class_f1:
            item["selection_status"] = "rejected_min_per_class_f1"
        elif item["name"] != baseline_name and delta < min_delta_macro_f1:
            item["selection_status"] = "rejected_min_delta_macro_f1"
        else:
            item["selection_status"] = "candidate"
    candidates_kept = [item for item in results if item["selection_status"] == "candidate"]
    if not candidates_kept:
        raise ValueError("No candidates passed rebalance grid gates")
    top_score = max(float(item["macro_f1"]) for item in candidates_kept)
    near_top = [item for item in candidates_kept if top_score - float(item["macro_f1"]) <= tie_epsilon]
    best = sorted(
        near_top,
        key=lambda item: (
            0 if item["name"] == baseline_name else 1,
            -float(item["min_per_class_f1"]),
            str(item["name"]),
        ),
    )[0]
    return {
        "best_name": best["name"],
        "best_checkpoint": best["checkpoint"],
        "best_macro_f1": best["macro_f1"],
        "baseline_macro_f1": baseline_macro_f1,
        "gates": {
            "min_delta_macro_f1": min_delta_macro_f1,
            "min_per_class_f1": min_per_class_f1,
            "tie_epsilon": tie_epsilon,
        },
        "results": results,
    }


def write_summary(output_dir: Path, summary: dict[str, object]) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "classifier_rebalance_grid_summary.json"
    csv_path = output_dir / "classifier_rebalance_grid_summary.csv"
    json_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["name", "method", "macro_f1", "delta_macro_f1_vs_baseline", "min_per_class_f1", "status", "checkpoint"])
        for item in summary["results"]:
            writer.writerow(
                [
                    item["name"],
                    item["method"],
                    f"{float(item['macro_f1']):.6f}",
                    f"{float(item['delta_macro_f1_vs_baseline']):.6f}",
                    f"{float(item['min_per_class_f1']):.6f}",
                    item["selection_status"],
                    item["checkpoint"],
                ]
            )
    return json_path, csv_path


def run_rebalance_grid(
    checkpoint: Path,
    output_dir: Path,
    tau_values: Sequence[float],
    crt_sampler_modes: Sequence[str],
    lws_sampler_modes: Sequence[str] = (),
    train_csv: Path | None = None,
    train_dir: Path | None = None,
    image_root: Path | None = None,
    run: bool = False,
    epochs: int = 4,
    batch_size: int = 32,
    lr: float = 1e-3,
    device: str = "auto",
    head_key: str = "auto",
    head_prefix: str = "auto",
) -> list[RebalanceCandidate]:
    candidates = build_rebalance_grid(checkpoint, output_dir, tau_values, crt_sampler_modes, lws_sampler_modes)
    if not run:
        return candidates
    for candidate in candidates:
        if candidate.method == "tau_norm":
            assert candidate.tau is not None
            rebalance_checkpoint(checkpoint, candidate.output, tau=candidate.tau, head_key=head_key)
        elif candidate.method == "crt":
            retrain_classifier_head(
                checkpoint_path=checkpoint,
                output_path=candidate.output,
                train_csv=train_csv,
                train_dir=train_dir,
                image_root=image_root,
                epochs=epochs,
                batch_size=batch_size,
                lr=lr,
                sampler_mode=str(candidate.sampler_mode),
                device=device,
                head_prefix=head_prefix,
            )
        elif candidate.method == "lws":
            retrain_classifier_head(
                checkpoint_path=checkpoint,
                output_path=candidate.output,
                train_csv=train_csv,
                train_dir=train_dir,
                image_root=image_root,
                epochs=epochs,
                batch_size=batch_size,
                lr=lr,
                sampler_mode=str(candidate.sampler_mode),
                device=device,
                head_prefix=head_prefix,
                rebalance_method="lws",
            )
    return candidates


def _write_metrics(path: Path, metrics: dict[str, object]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def validate_rebalance_grid(
    checkpoint: Path,
    candidates: Sequence[RebalanceCandidate],
    output_dir: Path,
    val_dir: Path | None = None,
    val_csv: Path | None = None,
    val_image_root: Path | None = None,
    batch_size: int = 64,
    device: str = "auto",
) -> RebalanceCandidate:
    if val_dir is None and val_csv is None:
        raise ValueError("Set val_dir or val_csv when validating rebalance grid")
    output_dir.mkdir(parents=True, exist_ok=True)
    baseline = RebalanceCandidate(
        name="baseline",
        method="baseline",
        checkpoint=checkpoint,
        output=checkpoint,
        metrics_json=output_dir / "baseline_metrics.json",
    )
    baseline_metrics = validate_checkpoint(
        checkpoint=checkpoint,
        val_dir=val_dir,
        val_csv=val_csv,
        image_root=val_image_root,
        batch_size=batch_size,
        device=device,
    )
    _write_metrics(baseline.metrics_json, baseline_metrics)
    for candidate in candidates:
        candidate.metrics_json = output_dir / f"{candidate.name}_metrics.json"
        metrics = validate_checkpoint(
            checkpoint=candidate.output,
            val_dir=val_dir,
            val_csv=val_csv,
            image_root=val_image_root,
            batch_size=batch_size,
            device=device,
        )
        _write_metrics(candidate.metrics_json, metrics)
    return baseline


def validate_cli_args(args: argparse.Namespace) -> None:
    if args.validate:
        if args.baseline_metrics is not None or args.candidate_metrics:
            raise ValueError("Do not combine --validate with --baseline-metrics or --candidate-metrics")
        if args.val_dir is None and args.val_csv is None:
            raise ValueError("--validate requires --val-dir or --val-csv")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build and summarize tau/crt classifier rebalance grids.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tau", type=float, nargs="*", default=[0.25, 0.5, 0.75, 1.0])
    parser.add_argument("--crt-sampler-mode", nargs="*", default=["sqrt", "class_balanced"])
    parser.add_argument("--lws-sampler-mode", nargs="*", default=[])
    parser.add_argument("--train-csv", type=Path, default=None)
    parser.add_argument("--train-dir", type=Path, default=None)
    parser.add_argument("--image-root", type=Path, default=None)
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--head-key", type=str, default="auto")
    parser.add_argument("--head-prefix", type=str, default="auto")
    parser.add_argument("--validate", action="store_true")
    parser.add_argument("--val-dir", type=Path, default=None)
    parser.add_argument("--val-csv", type=Path, default=None)
    parser.add_argument("--val-image-root", type=Path, default=None)
    parser.add_argument("--val-batch-size", type=int, default=64)
    parser.add_argument("--baseline-metrics", type=Path, default=None)
    parser.add_argument("--candidate-metrics", type=Path, nargs="*", default=None)
    parser.add_argument("--min-delta-macro-f1", type=float, default=0.0)
    parser.add_argument("--min-per-class-f1", type=float, default=0.0)
    parser.add_argument("--tie-epsilon", type=float, default=0.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    validate_cli_args(args)
    candidates = run_rebalance_grid(
        checkpoint=args.checkpoint,
        output_dir=args.output_dir,
        tau_values=args.tau,
        crt_sampler_modes=args.crt_sampler_mode,
        lws_sampler_modes=args.lws_sampler_mode,
        train_csv=args.train_csv,
        train_dir=args.train_dir,
        image_root=args.image_root,
        run=args.run,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=args.device,
        head_key=args.head_key,
        head_prefix=args.head_prefix,
    )
    payload: dict[str, object] = {
        "candidates": [candidate.__dict__ | {"checkpoint": str(candidate.checkpoint), "output": str(candidate.output)} for candidate in candidates]
    }
    baseline_metrics = args.baseline_metrics
    if args.validate:
        baseline = validate_rebalance_grid(
            checkpoint=args.checkpoint,
            candidates=candidates,
            output_dir=args.output_dir,
            val_dir=args.val_dir,
            val_csv=args.val_csv,
            val_image_root=args.val_image_root,
            batch_size=args.val_batch_size,
            device=args.device,
        )
        baseline_metrics = baseline.metrics_json
    if baseline_metrics is not None or args.candidate_metrics or args.validate:
        baseline = RebalanceCandidate(
            name="baseline",
            method="baseline",
            checkpoint=args.checkpoint,
            output=args.checkpoint,
            metrics_json=baseline_metrics,
        )
        if baseline.metrics_json is None:
            raise ValueError("--baseline-metrics is required when summarizing candidate metrics")
        if args.candidate_metrics is not None:
            if len(args.candidate_metrics) != len(candidates):
                raise ValueError("--candidate-metrics count must match generated candidate count")
            for candidate, metrics_path in zip(candidates, args.candidate_metrics):
                candidate.metrics_json = metrics_path
        summary = summarize_rebalance_grid(
            [baseline, *candidates],
            min_delta_macro_f1=args.min_delta_macro_f1,
            min_per_class_f1=args.min_per_class_f1,
            tie_epsilon=args.tie_epsilon,
        )
        json_path, csv_path = write_summary(args.output_dir, summary)
        payload = {**summary, "json": str(json_path), "csv": str(csv_path)}
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
