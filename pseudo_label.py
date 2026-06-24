from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from src.weather_net.config import load_config, resolve_device
from src.weather_net.inference import load_unlabeled_rows, predict_probabilities
from src.weather_net.pseudo_label import (
    count_pseudo_labels_by_class,
    estimate_per_class_thresholds,
    select_pseudo_labels,
    write_pseudo_labels,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate high-confidence pseudo labels.")
    parser.add_argument("--config", type=Path, default=Path("configs/convnext_tiny.yaml"))
    parser.add_argument("--checkpoint", type=Path, nargs="+", default=None)
    parser.add_argument("--test-dir", type=Path, default=None)
    parser.add_argument("--test-csv", type=Path, default=None)
    parser.add_argument("--image-root", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=Path("pseudo_labels.csv"))
    parser.add_argument("--threshold", type=float, default=0.95)
    parser.add_argument("--per-class-thresholds", type=Path, default=None)
    parser.add_argument("--per-class-max-count", type=Path, default=None)
    parser.add_argument("--estimate-thresholds-from-oof", type=Path, default=None)
    parser.add_argument("--threshold-output", type=Path, default=Path("per_class_thresholds.json"))
    parser.add_argument("--target-precision", type=float, default=0.95)
    parser.add_argument("--min-class-threshold", type=float, default=0.8)
    parser.add_argument("--fallback-class-threshold", type=float, default=0.99)
    parser.add_argument("--min-margin", type=float, default=0.0)
    parser.add_argument("--require-tta-agreement", action="store_true")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--weights", type=float, nargs="+", default=None)
    parser.add_argument("--tta", action="store_true")
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()
    if args.estimate_thresholds_from_oof is None and args.checkpoint is None:
        parser.error("--checkpoint is required unless --estimate-thresholds-from-oof is used")
    return args


def _load_json_mapping(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Mapping JSON must be an object: {path}")
    return {str(key): value for key, value in payload.items()}


def load_class_float_mapping(path: Path | None) -> dict[str, float] | None:
    if path is None:
        return None
    return {key: float(value) for key, value in _load_json_mapping(path).items()}


def load_class_int_mapping(path: Path | None) -> dict[str, int] | None:
    if path is None:
        return None
    result: dict[str, int] = {}
    for key, value in _load_json_mapping(path).items():
        if isinstance(value, bool):
            raise ValueError(f"Class max count must be an integer: {key}")
        if isinstance(value, int):
            parsed = value
        elif isinstance(value, str) and value.strip().lstrip("-").isdigit():
            parsed = int(value)
        else:
            raise ValueError(f"Class max count must be an integer: {key}")
        if parsed < 0:
            raise ValueError(f"Class max count must be non-negative: {key}")
        result[key] = parsed
    return result


def generate_thresholds_from_oof(
    oof_path: Path,
    output_path: Path,
    target_precision: float,
    min_threshold: float,
    fallback_threshold: float,
) -> dict[str, float]:
    data = np.load(oof_path, allow_pickle=True)
    required = {"y_true", "class_names"}
    missing = required - set(data.files)
    if missing:
        raise ValueError(f"OOF file missing fields {sorted(missing)}: {oof_path}")
    if "probs" in data.files:
        probabilities = np.asarray(data["probs"], dtype=np.float64)
    elif "logits" in data.files:
        logits = np.asarray(data["logits"], dtype=np.float64)
        if logits.ndim != 2:
            raise ValueError(f"OOF logits must be a 2D array: {oof_path}")
        if not np.isfinite(logits).all():
            raise ValueError(f"OOF logits must be finite: {oof_path}")
        shifted = logits - logits.max(axis=1, keepdims=True)
        exp = np.exp(shifted)
        probabilities = exp / exp.sum(axis=1, keepdims=True)
    else:
        raise ValueError(f"OOF file must contain probs or logits: {oof_path}")
    y_true = np.asarray(data["y_true"], dtype=np.int64)
    class_names_array = np.asarray(data["class_names"], dtype=object)
    if class_names_array.ndim != 1:
        raise ValueError(f"OOF class_names must be a 1D array: {oof_path}")
    class_names = [str(value) for value in class_names_array.tolist()]
    if probabilities.ndim != 2:
        raise ValueError(f"OOF probabilities must be a 2D array: {oof_path}")
    if y_true.ndim != 1:
        raise ValueError(f"OOF y_true must be a 1D array: {oof_path}")
    if probabilities.shape[0] != y_true.shape[0]:
        raise ValueError(f"OOF probabilities and y_true must have the same number of rows: {oof_path}")
    if probabilities.shape[1] != len(class_names):
        raise ValueError(f"OOF probability class dimension must match class_names: {oof_path}")
    if not class_names or len(set(class_names)) != len(class_names):
        raise ValueError(f"OOF class_names must be non-empty and unique: {oof_path}")
    if not np.isfinite(probabilities).all():
        raise ValueError(f"OOF probabilities must be finite: {oof_path}")
    if y_true.size and (int(y_true.min()) < 0 or int(y_true.max()) >= len(class_names)):
        raise ValueError(f"OOF y_true contains class ids outside class_names: {oof_path}")
    thresholds = estimate_per_class_thresholds(
        y_true=y_true.tolist(),
        probabilities=probabilities.tolist(),
        class_names=class_names,
        target_precision=target_precision,
        min_threshold=min_threshold,
        fallback_threshold=fallback_threshold,
    )
    thresholds = {label: round(float(value), 6) for label, value in thresholds.items()}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(thresholds, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return thresholds


def main() -> None:
    args = parse_args()
    if args.estimate_thresholds_from_oof is not None:
        thresholds = generate_thresholds_from_oof(
            oof_path=args.estimate_thresholds_from_oof,
            output_path=args.threshold_output,
            target_precision=args.target_precision,
            min_threshold=args.min_class_threshold,
            fallback_threshold=args.fallback_class_threshold,
        )
        print(json.dumps({"thresholds": thresholds, "output": str(args.threshold_output)}, indent=2, ensure_ascii=False))
        return
    config = load_config(args.config if args.config.exists() else None)
    device = resolve_device(args.device)
    rows = load_unlabeled_rows(
        test_dir=args.test_dir or config.data.test_dir,
        test_csv=args.test_csv or config.data.test_csv,
        image_root=args.image_root or config.data.image_root,
        image_column=config.data.image_column,
    )
    use_tta = args.tta or config.infer.tta
    image_paths, probabilities, class_names, image_ids, stats = predict_probabilities(
        checkpoints=args.checkpoint,
        rows=rows,
        batch_size=args.batch_size or config.infer.batch_size,
        device=device,
        tta=use_tta,
        num_workers=config.data.num_workers,
        checkpoint_weights=args.weights,
        transform_backend=config.data.transform_backend,
        amp=config.infer.amp,
    )
    agreement_probabilities = None
    if args.require_tta_agreement:
        _, agreement_probabilities, _, _, agreement_stats = predict_probabilities(
            checkpoints=args.checkpoint,
            rows=rows,
            batch_size=args.batch_size or config.infer.batch_size,
            device=device,
            tta=not use_tta,
            num_workers=config.data.num_workers,
            checkpoint_weights=args.weights,
            transform_backend=config.data.transform_backend,
            amp=config.infer.amp,
        )
        stats["agreement_pass"] = agreement_stats
    per_class_thresholds = load_class_float_mapping(args.per_class_thresholds)
    per_class_max_count = load_class_int_mapping(args.per_class_max_count)
    pseudo_rows = select_pseudo_labels(
        image_paths=image_paths,
        probabilities=probabilities,
        class_names=class_names,
        threshold=args.threshold,
        min_margin=args.min_margin,
        require_tta_agreement=args.require_tta_agreement,
        agreement_probabilities=agreement_probabilities,
        image_ids=image_ids,
        per_class_thresholds=per_class_thresholds,
        per_class_max_count=per_class_max_count,
    )
    write_pseudo_labels(args.output, pseudo_rows)
    selected_by_class = count_pseudo_labels_by_class(pseudo_rows, class_names)
    payload = {
        **stats,
        "threshold": args.threshold,
        "per_class_thresholds": per_class_thresholds,
        "per_class_max_count": per_class_max_count,
        "min_margin": args.min_margin,
        "require_tta_agreement": args.require_tta_agreement,
        "selected": len(pseudo_rows),
        "selected_by_class": selected_by_class,
        "selected_ratio": len(pseudo_rows) / len(rows) if rows else 0.0,
        "output": str(args.output),
    }
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
