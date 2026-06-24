from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import yaml

from src.weather_net.config import AppConfig, load_config
from src.weather_net.training import train_config


DEFAULT_USAGES = ["loss", "sampler", "both"]
USAGE_RISK_ORDER = {"loss": 0, "sampler": 1, "both": 2}


@dataclass(frozen=True)
class AblationConfig:
    usage: str
    config_path: Path
    output_dir: Path


def _config_to_payload(config: AppConfig) -> dict[str, object]:
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


def _validate_usages(usages: Sequence[str]) -> list[str]:
    if not usages:
        raise ValueError("At least one sample_weight_usage is required")
    invalid = sorted(set(usages) - {"loss", "sampler", "both"})
    if invalid:
        raise ValueError(f"Invalid sample_weight_usage values: {invalid}")
    return [str(usage) for usage in usages]


def build_ablation_configs(
    base_config: AppConfig,
    output_root: Path,
    usages: Sequence[str] = DEFAULT_USAGES,
) -> list[AblationConfig]:
    output_root.mkdir(parents=True, exist_ok=True)
    usage_values = _validate_usages(usages)
    outputs: list[AblationConfig] = []
    base_payload = _config_to_payload(base_config)
    for usage in usage_values:
        run_dir = output_root / usage
        config_path = output_root / f"{usage}.yaml"
        payload = json.loads(json.dumps(base_payload))
        payload["train"]["sample_weight_usage"] = usage
        payload["train"]["sampler_mode"] = "sample_weighted" if usage in {"sampler", "both"} else "auto"
        payload["train"]["output_dir"] = str(run_dir)
        config_path.write_text(
            yaml.safe_dump(payload, allow_unicode=True, sort_keys=False),
            encoding="utf-8",
        )
        outputs.append(AblationConfig(usage=usage, config_path=config_path, output_dir=run_dir))
    return outputs


def _read_training_summary(run_dir: Path) -> list[dict[str, object]]:
    summary_path = run_dir / "training_summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing training summary: {summary_path}")
    data = json.loads(summary_path.read_text(encoding="utf-8"))
    if not isinstance(data, list) or not data:
        raise ValueError(f"training_summary.json must be a non-empty list: {summary_path}")
    return [dict(item) for item in data]


def _mean(values: Sequence[float]) -> float:
    if not values:
        raise ValueError("Cannot average an empty sequence")
    return float(sum(values) / len(values))


def summarize_ablation_results(
    output_root: Path,
    usages: Sequence[str] = DEFAULT_USAGES,
    baseline_usage: str = "loss",
    min_per_class_f1: float = 0.0,
    min_delta_macro_f1: float = 0.0,
    tie_epsilon: float = 0.0,
) -> dict[str, object]:
    if not 0 <= min_per_class_f1 <= 1:
        raise ValueError("min_per_class_f1 must be in [0, 1]")
    if tie_epsilon < 0:
        raise ValueError("tie_epsilon must be non-negative")
    results: list[dict[str, object]] = []
    usage_values = _validate_usages(usages)
    if baseline_usage not in usage_values:
        raise ValueError("baseline_usage must be included in usages")
    baseline_macro_f1: float | None = None
    for usage in usage_values:
        run_dir = output_root / usage
        folds = _read_training_summary(run_dir)
        if any("best_macro_f1" not in item for item in folds):
            raise ValueError(f"training_summary.json missing best_macro_f1 for usage={usage}")
        macro_scores = [float(item["best_macro_f1"]) for item in folds]
        per_class_values: dict[str, list[float]] = {}
        for item in folds:
            per_class = item.get("per_class_f1", {})
            if not isinstance(per_class, dict):
                raise ValueError(f"per_class_f1 must be a mapping for usage={usage}")
            for class_name, score in per_class.items():
                per_class_values.setdefault(str(class_name), []).append(float(score))
        results.append(
            {
                "usage": usage,
                "output_dir": str(run_dir),
                "folds": len(folds),
                "macro_f1_mean": _mean(macro_scores),
                "macro_f1_by_fold": macro_scores,
                "per_class_f1_mean": {
                    class_name: _mean(scores)
                    for class_name, scores in sorted(per_class_values.items())
                },
            }
        )
        if usage == baseline_usage:
            baseline_macro_f1 = float(results[-1]["macro_f1_mean"])
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
        elif item["usage"] != baseline_usage and delta < min_delta_macro_f1:
            item["selection_status"] = "rejected_min_delta_macro_f1"
        else:
            item["selection_status"] = "candidate"
    candidates = [item for item in results if item["selection_status"] == "candidate"]
    if not candidates:
        candidates = [item for item in results if item["usage"] == baseline_usage]
    top_score = max(float(item["macro_f1_mean"]) for item in candidates)
    near_top = [item for item in candidates if top_score - float(item["macro_f1_mean"]) <= tie_epsilon]
    best = min(
        near_top,
        key=lambda item: (
            USAGE_RISK_ORDER.get(str(item["usage"]), 99),
            -float(item["macro_f1_mean"]),
        ),
    )
    return {
        "best_usage": best["usage"],
        "best_macro_f1_mean": best["macro_f1_mean"],
        "baseline_usage": baseline_usage,
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
    json_path = output_root / "ablation_summary.json"
    csv_path = output_root / "ablation_summary.csv"
    json_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["usage", "macro_f1_mean", "folds", "output_dir"])
        for item in summary["results"]:
            writer.writerow(
                [
                    item["usage"],
                    f"{float(item['macro_f1_mean']):.6f}",
                    item["folds"],
                    item["output_dir"],
                ]
            )
    return json_path, csv_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate and summarize hard fine-tune sample-weight ablations.")
    parser.add_argument("--base-config", type=Path, default=Path("configs/convnextv2_384_hard_finetune.yaml"))
    parser.add_argument("--output-root", type=Path, default=Path("outputs/hard_finetune_ablation"))
    parser.add_argument("--usage", choices=["loss", "sampler", "both"], nargs="+", default=DEFAULT_USAGES)
    parser.add_argument("--train-csv", type=Path, default=None)
    parser.add_argument("--image-root", type=Path, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--folds", type=int, default=None)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--baseline-usage", choices=["loss", "sampler", "both"], default="loss")
    parser.add_argument("--min-per-class-f1", type=float, default=0.0)
    parser.add_argument("--min-delta-macro-f1", type=float, default=0.0)
    parser.add_argument("--tie-epsilon", type=float, default=0.0)
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--summarize", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.base_config)
    if args.train_csv is not None:
        config.data.train_csv = args.train_csv
    if args.image_root is not None:
        config.data.image_root = args.image_root
    if args.epochs is not None:
        config.train.epochs = args.epochs
    if args.folds is not None:
        config.data.folds = args.folds

    configs = build_ablation_configs(config, output_root=args.output_root, usages=args.usage)
    if args.run:
        for item in configs:
            train_config(load_config(item.config_path), device_request=args.device)
    if args.summarize or args.run:
        summary = summarize_ablation_results(
            args.output_root,
            usages=args.usage,
            baseline_usage=args.baseline_usage,
            min_per_class_f1=args.min_per_class_f1,
            min_delta_macro_f1=args.min_delta_macro_f1,
            tie_epsilon=args.tie_epsilon,
        )
        json_path, csv_path = write_summary(args.output_root, summary)
        print(json.dumps({**summary, "json": str(json_path), "csv": str(csv_path)}, indent=2, ensure_ascii=False))
    else:
        print(json.dumps({"configs": [str(item.config_path) for item in configs]}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
