from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import numpy as np

from src.weather_net.postprocess import (
    apply_decision_params,
    greedy_search_ensemble_weights,
    macro_f1_from_logits,
    search_per_class_bias,
    search_temperature,
    weighted_sum_logits,
    write_decision_params,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tune OOF decision parameters for macro F1.")
    parser.add_argument("--oof", type=Path, nargs="+", required=True, help="One or more OOF .npz files.")
    parser.add_argument("--checkpoint", type=Path, nargs="*", default=None, help="Checkpoint order matching --oof.")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/decision/default"))
    parser.add_argument("--temperature", type=float, nargs="*", default=None)
    parser.add_argument("--max-ensemble-steps", type=int, default=12)
    return parser.parse_args()


def _load_oof_npz(path: Path) -> tuple[np.ndarray, np.ndarray, list[str], list[str]]:
    data = np.load(path, allow_pickle=True)
    required = {"logits", "y_true", "class_names"}
    missing = required - set(data.files)
    if missing:
        raise ValueError(f"OOF file missing fields {sorted(missing)}: {path}")
    logits = np.asarray(data["logits"], dtype=np.float64)
    y_true = np.asarray(data["y_true"], dtype=np.int64)
    class_names = [str(name) for name in data["class_names"].tolist()]
    if "image_id" not in data.files:
        raise ValueError(f"OOF file missing image_id for alignment: {path}")
    image_ids = [str(value) for value in data["image_id"].tolist()]
    return logits, y_true, class_names, image_ids


def _load_aligned_oofs(paths: Sequence[Path]) -> tuple[np.ndarray, np.ndarray, list[str]]:
    logits_list: list[np.ndarray] = []
    expected_y_true: np.ndarray | None = None
    expected_class_names: list[str] | None = None
    expected_image_ids: list[str] | None = None
    for path in paths:
        logits, y_true, class_names, image_ids = _load_oof_npz(path)
        if expected_y_true is None:
            expected_y_true = y_true
            expected_class_names = class_names
            expected_image_ids = image_ids
        else:
            if not np.array_equal(y_true, expected_y_true):
                raise ValueError(f"OOF y_true differs: {path}")
            if class_names != expected_class_names:
                raise ValueError(f"OOF class_names differ: {path}")
            if expected_image_ids and image_ids and image_ids != expected_image_ids:
                raise ValueError(f"OOF image_id order differs: {path}")
        logits_list.append(logits)
    assert expected_y_true is not None
    assert expected_class_names is not None
    return np.stack(logits_list, axis=0), expected_y_true, expected_class_names


def _write_report(
    path: Path,
    oof_paths: Sequence[Path],
    class_names: Sequence[str],
    weights: Sequence[float],
    temperature: float,
    bias: Sequence[float],
    scores: dict[str, float],
) -> None:
    lines = [
        "# OOF Decision Report",
        "",
        "## Inputs",
        *[f"- `{oof_path}`" for oof_path in oof_paths],
        "",
        "## Decision",
        f"- Classes: {', '.join(class_names)}",
        f"- Weights: {', '.join(f'{weight:.6f}' for weight in weights)}",
        f"- Temperature: {temperature:.6f}",
        f"- Bias: {', '.join(f'{value:.6f}' for value in bias)}",
        "",
        "## Scores",
        *[f"- {key}: {value:.6f}" for key, value in scores.items()],
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def run_oof_decision(
    oof_paths: Sequence[Path],
    output_dir: Path,
    temperature_candidates: Sequence[float] | None = None,
    max_ensemble_steps: int = 12,
    checkpoints: Sequence[Path] | None = None,
) -> dict[str, object]:
    if checkpoints is not None and len(checkpoints) != len(oof_paths):
        raise ValueError("checkpoint count must match OOF count")
    logits_by_model, y_true, class_names = _load_aligned_oofs(oof_paths)
    output_dir.mkdir(parents=True, exist_ok=True)

    baseline_scores = [
        macro_f1_from_logits(logits_by_model[idx], y_true, class_names)
        for idx in range(logits_by_model.shape[0])
    ]
    ensemble_result = greedy_search_ensemble_weights(
        logits_by_model,
        y_true,
        class_names,
        max_steps=max_ensemble_steps,
    )
    ensemble_logits = weighted_sum_logits(logits_by_model, ensemble_result.weights)
    temperature, temperature_score = search_temperature(
        ensemble_logits,
        y_true,
        class_names,
        candidates=temperature_candidates,
    )
    bias_result = search_per_class_bias(
        ensemble_logits,
        y_true,
        class_names,
        temperature=temperature,
    )
    tuned_logits = apply_decision_params(
        ensemble_logits,
        temperature=temperature,
        bias=bias_result.bias,
    )
    tuned_score = macro_f1_from_logits(tuned_logits, y_true, class_names)
    scores = {
        "best_single_macro_f1": float(max(baseline_scores)),
        "ensemble_macro_f1": float(ensemble_result.score),
        "temperature_macro_f1": float(temperature_score),
        "tuned_macro_f1": float(tuned_score),
    }
    write_decision_params(
        output_dir / "decision_params.json",
        class_names=class_names,
        temperature=temperature,
        bias=bias_result.bias,
        weights=ensemble_result.weights,
        scores=scores,
        checkpoints=[str(path) for path in (checkpoints or oof_paths)],
    )
    _write_report(
        output_dir / "decision_report.md",
        oof_paths=oof_paths,
        class_names=class_names,
        weights=ensemble_result.weights,
        temperature=temperature,
        bias=bias_result.bias,
        scores=scores,
    )
    (output_dir / "oof_scorecard.json").write_text(
        json.dumps(
            {
                "oof": [str(path) for path in oof_paths],
                "num_models": len(oof_paths),
                "class_names": class_names,
                "scores": scores,
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    return {
        "num_models": len(oof_paths),
        "class_names": class_names,
        "weights": ensemble_result.weights,
        "temperature": temperature,
        "bias": bias_result.bias,
        "scores": scores,
    }


def main() -> None:
    args = parse_args()
    stats = run_oof_decision(
        oof_paths=args.oof,
        output_dir=args.output_dir,
        temperature_candidates=args.temperature,
        max_ensemble_steps=args.max_ensemble_steps,
        checkpoints=args.checkpoint,
    )
    print(json.dumps(stats, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
