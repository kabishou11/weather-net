from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import yaml

from src.weather_net.config import load_config
from src.weather_net.hard_mining import apply_hard_mining_weights
from src.weather_net.training import train_config


@dataclass(frozen=True)
class ConfusionHardAblationConfig:
    name: str
    pair_confusion_boost: float
    config_path: Path
    weighted_csv: Path
    hard_csv: Path
    output_dir: Path


def _boost_name(value: float) -> str:
    return f"pair{value:.2f}".replace(".", "p")


def _validate_boosts(values: Sequence[float]) -> list[float]:
    if not values:
        raise ValueError("At least one pair_confusion_boost value is required")
    boosts = [float(value) for value in values]
    if any(value < 0 for value in boosts):
        raise ValueError("pair_confusion_boost values must be non-negative")
    if len({_boost_name(value) for value in boosts}) != len(boosts):
        raise ValueError("pair_confusion_boost values must map to unique run names")
    return boosts


def _config_to_payload(config) -> dict[str, object]:
    return {
        "data": {
            "train_dir": None if config.data.train_dir is None else str(config.data.train_dir),
            "train_csv": None if config.data.train_csv is None else str(config.data.train_csv),
            "image_root": None if config.data.image_root is None else str(config.data.image_root),
            "test_dir": None if config.data.test_dir is None else str(config.data.test_dir),
            "test_csv": None if config.data.test_csv is None else str(config.data.test_csv),
            "image_column": config.data.image_column,
            "label_column": config.data.label_column,
            "val_fraction": config.data.val_fraction,
            "folds": config.data.folds,
            "seed": config.data.seed,
            "num_workers": config.data.num_workers,
            "augment_policy": config.data.augment_policy,
            "transform_backend": config.data.transform_backend,
        },
        "model": {
            "name": config.model.name,
            "image_size": config.model.image_size,
            "pretrained": config.model.pretrained,
            "dropout": config.model.dropout,
        },
        "train": {
            "epochs": config.train.epochs,
            "batch_size": config.train.batch_size,
            "lr": config.train.lr,
            "weight_decay": config.train.weight_decay,
            "label_smoothing": config.train.label_smoothing,
            "loss_name": config.train.loss_name,
            "focal_gamma": config.train.focal_gamma,
            "class_balanced_beta": config.train.class_balanced_beta,
            "ldam_max_margin": config.train.ldam_max_margin,
            "ldam_scale": config.train.ldam_scale,
            "sampler_mode": config.train.sampler_mode,
            "sample_weight_usage": config.train.sample_weight_usage,
            "amp": config.train.amp,
            "mixup_alpha": config.train.mixup_alpha,
            "cutmix_alpha": config.train.cutmix_alpha,
            "jsd_weight": config.train.jsd_weight,
            "ema_decay": config.train.ema_decay,
            "output_dir": str(config.train.output_dir),
        },
        "infer": {
            "batch_size": config.infer.batch_size,
            "tta": config.infer.tta,
            "amp": config.infer.amp,
            "output_csv": str(config.infer.output_csv),
        },
    }


def build_confusion_hard_ablation_configs(
    base_config_path: Path,
    train_csv: Path,
    oof_csv: Path,
    output_root: Path,
    pair_confusion_boosts: Sequence[float],
    error_boost: float = 0.8,
    low_margin_boost: float = 0.3,
    high_loss_boost: float = 0.3,
    pair_min_support: int = 2,
    pair_min_error_share: float = 0.35,
    low_margin_threshold: float = 0.1,
    high_loss_quantile: float = 0.75,
    max_weight: float = 2.2,
    folds: int | None = None,
    epochs: int | None = None,
    image_root: Path | None = None,
) -> list[ConfusionHardAblationConfig]:
    base_config = load_config(base_config_path)
    boosts = _validate_boosts(pair_confusion_boosts)
    output_root.mkdir(parents=True, exist_ok=True)
    outputs: list[ConfusionHardAblationConfig] = []
    for boost in boosts:
        name = _boost_name(boost)
        run_root = output_root / name
        train_output_dir = run_root / "train"
        weighted_csv = run_root / "train_confusion_hard_weighted.csv"
        hard_csv = run_root / "confusion_hard_samples.csv"
        config_path = run_root / "config.yaml"
        stats = apply_hard_mining_weights(
            train_csv=train_csv,
            oof_csv=oof_csv,
            output_csv=weighted_csv,
            hard_csv=hard_csv,
            error_boost=error_boost,
            low_margin_boost=low_margin_boost,
            high_loss_boost=high_loss_boost,
            pair_confusion_boost=boost,
            pair_min_support=pair_min_support,
            pair_min_error_share=pair_min_error_share,
            low_margin_threshold=low_margin_threshold,
            high_loss_quantile=high_loss_quantile,
            max_weight=max_weight,
        )
        (run_root / "pair_stats.json").write_text(
            json.dumps(
                {
                    **stats,
                    "pair_confusion_boost": boost,
                    "pair_min_support": pair_min_support,
                    "pair_min_error_share": pair_min_error_share,
                    "max_weight": max_weight,
                },
                indent=2,
                ensure_ascii=False,
            )
            + "\n",
            encoding="utf-8",
        )
        payload = _config_to_payload(base_config)
        payload["data"]["train_dir"] = None
        payload["data"]["train_csv"] = str(weighted_csv)
        if image_root is not None:
            payload["data"]["image_root"] = str(image_root)
        if folds is not None:
            payload["data"]["folds"] = int(folds)
        if epochs is not None:
            payload["train"]["epochs"] = int(epochs)
        payload["train"]["sampler_mode"] = "sample_weighted"
        payload["train"]["sample_weight_usage"] = "sampler"
        payload["train"]["output_dir"] = str(train_output_dir)
        config_path.parent.mkdir(parents=True, exist_ok=True)
        config_path.write_text(
            yaml.safe_dump(payload, allow_unicode=True, sort_keys=False),
            encoding="utf-8",
        )
        outputs.append(
            ConfusionHardAblationConfig(
                name=name,
                pair_confusion_boost=boost,
                config_path=config_path,
                weighted_csv=weighted_csv,
                hard_csv=hard_csv,
                output_dir=train_output_dir,
            )
        )
    return outputs


def _read_training_summary(run_dir: Path) -> list[dict[str, object]]:
    summary_path = run_dir / "training_summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing training summary: {summary_path}")
    data = json.loads(summary_path.read_text(encoding="utf-8"))
    if not isinstance(data, list) or not data:
        raise ValueError(f"training_summary.json must be a non-empty list: {summary_path}")
    return [dict(item) for item in data]


def _read_hard_confusion_pairs(hard_csv: Path) -> dict[str, int]:
    if not hard_csv.exists():
        return {}
    with hard_csv.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or "confusion_pair" not in reader.fieldnames:
            return {}
        counts = Counter(
            str(row.get("confusion_pair", "")).strip()
            for row in reader
            if str(row.get("confusion_pair", "")).strip()
        )
    return dict(sorted(counts.items()))


def _mean(values: Sequence[float]) -> float:
    if not values:
        raise ValueError("Cannot average an empty sequence")
    return float(sum(values) / len(values))


def summarize_confusion_hard_ablation(
    output_root: Path,
    names: Sequence[str],
    baseline_name: str,
    min_per_class_f1: float = 0.0,
    min_delta_macro_f1: float = 0.0,
    tie_epsilon: float = 0.0,
) -> dict[str, object]:
    if not names:
        raise ValueError("At least one ablation run name is required")
    if baseline_name not in names:
        raise ValueError("baseline_name must be included in names")
    if not 0 <= min_per_class_f1 <= 1:
        raise ValueError("min_per_class_f1 must be in [0, 1]")
    if tie_epsilon < 0:
        raise ValueError("tie_epsilon must be non-negative")

    results: list[dict[str, object]] = []
    baseline_macro_f1: float | None = None
    for name in names:
        run_root = output_root / name
        folds = _read_training_summary(run_root / "train")
        macro_scores = [float(item["best_macro_f1"]) for item in folds]
        per_class_values: dict[str, list[float]] = {}
        for item in folds:
            per_class = item.get("per_class_f1", {})
            if not isinstance(per_class, dict):
                raise ValueError(f"per_class_f1 must be a mapping for name={name}")
            for class_name, score in per_class.items():
                per_class_values.setdefault(str(class_name), []).append(float(score))
        pair_stats_path = run_root / "pair_stats.json"
        pair_stats = json.loads(pair_stats_path.read_text(encoding="utf-8")) if pair_stats_path.exists() else {}
        result = {
            "name": name,
            "output_dir": str(run_root / "train"),
            "folds": len(folds),
            "macro_f1_mean": _mean(macro_scores),
            "macro_f1_by_fold": macro_scores,
            "pair_stats": pair_stats,
            "hard_confusion_pairs": _read_hard_confusion_pairs(run_root / "confusion_hard_samples.csv"),
            "per_class_f1_mean": {
                class_name: _mean(scores)
                for class_name, scores in sorted(per_class_values.items())
            },
        }
        results.append(result)
        if name == baseline_name:
            baseline_macro_f1 = float(result["macro_f1_mean"])
    if baseline_macro_f1 is None:
        raise ValueError("baseline result was not found")

    for item in results:
        per_class_scores = item["per_class_f1_mean"]
        min_class_f1 = min(per_class_scores.values()) if per_class_scores else 0.0
        delta = float(item["macro_f1_mean"]) - baseline_macro_f1
        item["min_per_class_f1"] = float(min_class_f1)
        item["delta_macro_f1_vs_baseline"] = float(delta)
        if min_class_f1 < min_per_class_f1:
            item["selection_status"] = "rejected_min_per_class_f1"
        elif item["name"] != baseline_name and delta < min_delta_macro_f1:
            item["selection_status"] = "rejected_min_delta_macro_f1"
        else:
            item["selection_status"] = "candidate"
    candidates = [item for item in results if item["selection_status"] == "candidate"]
    if not candidates:
        candidates = [item for item in results if item["name"] == baseline_name]
    top_score = max(float(item["macro_f1_mean"]) for item in candidates)
    near_top = [item for item in candidates if top_score - float(item["macro_f1_mean"]) <= tie_epsilon]
    best = min(
        near_top,
        key=lambda item: (
            0 if item["name"] == baseline_name else 1,
            -float(item["macro_f1_mean"]),
            str(item["name"]),
        ),
    )
    return {
        "best_name": best["name"],
        "best_macro_f1_mean": best["macro_f1_mean"],
        "baseline_name": baseline_name,
        "baseline_macro_f1_mean": baseline_macro_f1,
        "selection": {
            "min_per_class_f1": min_per_class_f1,
            "min_delta_macro_f1": min_delta_macro_f1,
            "tie_epsilon": tie_epsilon,
        },
        "results": results,
    }


def write_summary(output_root: Path, summary: dict[str, object]) -> tuple[Path, Path]:
    output_root.mkdir(parents=True, exist_ok=True)
    json_path = output_root / "confusion_hard_ablation_summary.json"
    csv_path = output_root / "confusion_hard_ablation_summary.csv"
    json_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["name", "macro_f1_mean", "delta_macro_f1_vs_baseline", "min_per_class_f1", "folds", "output_dir"])
        for item in summary["results"]:
            writer.writerow(
                [
                    item["name"],
                    f"{float(item['macro_f1_mean']):.6f}",
                    f"{float(item['delta_macro_f1_vs_baseline']):.6f}",
                    f"{float(item['min_per_class_f1']):.6f}",
                    item["folds"],
                    item["output_dir"],
                ]
            )
    return json_path, csv_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate and summarize OOF confusion-pair hard fine-tune ablations.")
    parser.add_argument("--base-config", type=Path, default=Path("configs/convnextv2_384_hard_finetune.yaml"))
    parser.add_argument("--train-csv", type=Path, required=True)
    parser.add_argument("--oof-csv", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=Path("outputs/confusion_hard_ablation"))
    parser.add_argument("--pair-confusion-boost", type=float, nargs="+", default=[0.0, 0.4, 0.8])
    parser.add_argument("--error-boost", type=float, default=0.8)
    parser.add_argument("--low-margin-boost", type=float, default=0.3)
    parser.add_argument("--high-loss-boost", type=float, default=0.3)
    parser.add_argument("--pair-min-support", type=int, default=2)
    parser.add_argument("--pair-min-error-share", type=float, default=0.35)
    parser.add_argument("--low-margin-threshold", type=float, default=0.1)
    parser.add_argument("--high-loss-quantile", type=float, default=0.75)
    parser.add_argument("--max-weight", type=float, default=2.2)
    parser.add_argument("--folds", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--image-root", type=Path, default=None)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--baseline-name", type=str, default=None)
    parser.add_argument("--min-per-class-f1", type=float, default=0.0)
    parser.add_argument("--min-delta-macro-f1", type=float, default=0.003)
    parser.add_argument("--tie-epsilon", type=float, default=0.0)
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--summarize", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outputs = build_confusion_hard_ablation_configs(
        base_config_path=args.base_config,
        train_csv=args.train_csv,
        oof_csv=args.oof_csv,
        output_root=args.output_root,
        pair_confusion_boosts=args.pair_confusion_boost,
        error_boost=args.error_boost,
        low_margin_boost=args.low_margin_boost,
        high_loss_boost=args.high_loss_boost,
        pair_min_support=args.pair_min_support,
        pair_min_error_share=args.pair_min_error_share,
        low_margin_threshold=args.low_margin_threshold,
        high_loss_quantile=args.high_loss_quantile,
        max_weight=args.max_weight,
        folds=args.folds,
        epochs=args.epochs,
        image_root=args.image_root,
    )
    if args.run:
        for item in outputs:
            train_config(load_config(item.config_path), device_request=args.device)
    if args.run or args.summarize:
        names = [item.name for item in outputs]
        baseline_name = args.baseline_name or names[0]
        summary = summarize_confusion_hard_ablation(
            output_root=args.output_root,
            names=names,
            baseline_name=baseline_name,
            min_per_class_f1=args.min_per_class_f1,
            min_delta_macro_f1=args.min_delta_macro_f1,
            tie_epsilon=args.tie_epsilon,
        )
        json_path, csv_path = write_summary(args.output_root, summary)
        print(json.dumps({**summary, "json": str(json_path), "csv": str(csv_path)}, indent=2, ensure_ascii=False))
        return
    print(json.dumps({"configs": [str(item.config_path) for item in outputs]}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
