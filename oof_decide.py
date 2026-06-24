from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Sequence

import numpy as np

from src.weather_net.postprocess import (
    apply_decision_params,
    bootstrap_macro_f1_summary,
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
    parser.add_argument("--bootstrap-rounds", type=int, default=0)
    parser.add_argument("--bootstrap-sample-fraction", type=float, default=1.0)
    parser.add_argument("--bootstrap-seed", type=int, default=42)
    parser.add_argument("--bootstrap-min-delta-q05", type=float, default=0.0)
    return parser.parse_args()


def _load_oof_npz(path: Path) -> tuple[np.ndarray, np.ndarray, list[str], list[str], list[str]]:
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
    if logits.ndim != 2:
        raise ValueError(f"OOF logits must be a 2D array: {path}")
    if y_true.ndim != 1:
        raise ValueError(f"OOF y_true must be a 1D array: {path}")
    if logits.shape[0] != y_true.shape[0] or logits.shape[0] != len(image_ids):
        raise ValueError(f"OOF logits, y_true, and image_id must have the same number of rows: {path}")
    if logits.shape[1] != len(class_names):
        raise ValueError(f"OOF logits class dimension must match class_names: {path}")
    if not class_names or len(set(class_names)) != len(class_names):
        raise ValueError(f"OOF class_names must be non-empty and unique: {path}")
    if y_true.size and (int(y_true.min()) < 0 or int(y_true.max()) >= len(class_names)):
        raise ValueError(f"OOF y_true contains class ids outside class_names: {path}")
    duplicates = sorted(image_id for image_id, count in Counter(image_ids).items() if count > 1)
    if duplicates:
        preview = ", ".join(duplicates[:5])
        raise ValueError(f"Duplicate OOF image ids are not safe: {preview}")
    checkpoint_values = [str(value) for value in data["checkpoint"].tolist()] if "checkpoint" in data.files else []
    if checkpoint_values and len(checkpoint_values) != logits.shape[0]:
        raise ValueError(f"OOF checkpoint values must match logits rows: {path}")
    return logits, y_true, class_names, image_ids, checkpoint_values


def _unique_in_order(values: Sequence[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            ordered.append(value)
    return ordered


def _load_aligned_oofs(paths: Sequence[Path]) -> tuple[np.ndarray, np.ndarray, list[str], list[list[str]]]:
    logits_list: list[np.ndarray] = []
    checkpoint_groups: list[list[str]] = []
    expected_y_true: np.ndarray | None = None
    expected_class_names: list[str] | None = None
    expected_image_ids: list[str] | None = None
    for path in paths:
        logits, y_true, class_names, image_ids, checkpoint_values = _load_oof_npz(path)
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
        checkpoint_groups.append(_unique_in_order(checkpoint_values))
    assert expected_y_true is not None
    assert expected_class_names is not None
    return np.stack(logits_list, axis=0), expected_y_true, expected_class_names, checkpoint_groups


def _resolve_decision_checkpoints_and_weights(
    oof_paths: Sequence[Path],
    checkpoint_groups: Sequence[Sequence[str]],
    checkpoints: Sequence[Path] | None,
    oof_weights: Sequence[float],
) -> tuple[list[str], list[float]]:
    grouped_checkpoint_names = [list(group) for group in checkpoint_groups]
    has_checkpoint_metadata = all(grouped_checkpoint_names)
    if any(grouped_checkpoint_names) and not has_checkpoint_metadata:
        raise ValueError("OOF checkpoint metadata must be present in all OOF files or none of them")
    if checkpoints is None:
        if has_checkpoint_metadata:
            checkpoint_names = _unique_in_order(
                checkpoint
                for group in grouped_checkpoint_names
                for checkpoint in group
            )
            checkpoints_by_name = set(checkpoint_names)
            expanded_weights = [0.0 for _ in checkpoint_names]
            for group_idx, group in enumerate(grouped_checkpoint_names):
                group_indices = [idx for idx, checkpoint in enumerate(checkpoint_names) if checkpoint in set(group)]
                if len(group_indices) != len(set(group)):
                    raise ValueError("OOF checkpoint groups must map to unique checkpoint names")
                per_checkpoint_weight = float(oof_weights[group_idx]) / len(group_indices)
                for checkpoint_idx in group_indices:
                    expanded_weights[checkpoint_idx] += per_checkpoint_weight
            if not checkpoints_by_name:
                raise ValueError("OOF checkpoint metadata is empty")
            return checkpoint_names, expanded_weights
        return [str(path) for path in oof_paths], list(oof_weights)

    checkpoint_names = [str(path) for path in checkpoints]
    if not has_checkpoint_metadata:
        if len(checkpoints) != len(oof_paths):
            raise ValueError("checkpoint count must match OOF count unless OOF files contain checkpoint metadata")
        return checkpoint_names, list(oof_weights)

    expanded_weights = [0.0 for _ in checkpoint_names]
    represented = [False for _ in checkpoint_names]
    for group_idx, group in enumerate(grouped_checkpoint_names):
        group_set = set(group)
        group_indices = [idx for idx, checkpoint in enumerate(checkpoint_names) if checkpoint in group_set]
        missing = sorted(set(group) - set(checkpoint_names))
        if missing:
            preview = ", ".join(missing[:5])
            raise ValueError(f"OOF checkpoint values are missing from --checkpoint: {preview}")
        if not group_indices:
            raise ValueError("OOF checkpoint group has no matching --checkpoint entries")
        per_checkpoint_weight = float(oof_weights[group_idx]) / len(group_indices)
        for checkpoint_idx in group_indices:
            expanded_weights[checkpoint_idx] += per_checkpoint_weight
            represented[checkpoint_idx] = True

    if not all(represented):
        raise ValueError("Each --checkpoint must be represented by at least one OOF checkpoint group")
    return checkpoint_names, expanded_weights


def _write_report(
    path: Path,
    oof_paths: Sequence[Path],
    class_names: Sequence[str],
    oof_weights: Sequence[float],
    decision_checkpoints: Sequence[str],
    decision_weights: Sequence[float],
    temperature: float,
    bias: Sequence[float],
    scores: dict[str, float],
    bootstrap: dict[str, float | int] | None = None,
) -> None:
    lines = [
        "# OOF Decision Report",
        "",
        "## Inputs",
        *[f"- `{oof_path}`" for oof_path in oof_paths],
        "",
        "## Decision",
        f"- Classes: {', '.join(class_names)}",
        f"- OOF weights: {', '.join(f'{weight:.6f}' for weight in oof_weights)}",
        "- Inference checkpoints:",
        *[
            f"  - `{checkpoint}`: {weight:.6f}"
            for checkpoint, weight in zip(decision_checkpoints, decision_weights)
        ],
        f"- Temperature: {temperature:.6f}",
        f"- Bias: {', '.join(f'{value:.6f}' for value in bias)}",
        "",
        "## Scores",
        *[f"- {key}: {value:.6f}" for key, value in scores.items()],
        "",
    ]
    if bootstrap is not None:
        lines.extend(
            [
                "## Bootstrap Stability",
                *[
                    f"- {key}: {value:.6f}" if isinstance(value, float) else f"- {key}: {value}"
                    for key, value in bootstrap.items()
                ],
                "",
            ]
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def run_oof_decision(
    oof_paths: Sequence[Path],
    output_dir: Path,
    temperature_candidates: Sequence[float] | None = None,
    max_ensemble_steps: int = 12,
    checkpoints: Sequence[Path] | None = None,
    bootstrap_rounds: int = 0,
    bootstrap_sample_fraction: float = 1.0,
    bootstrap_seed: int = 42,
    bootstrap_min_delta_q05: float = 0.0,
) -> dict[str, object]:
    if bootstrap_rounds < 0:
        raise ValueError("bootstrap_rounds must be non-negative")
    if not 0 < bootstrap_sample_fraction <= 1:
        raise ValueError("bootstrap_sample_fraction must be in (0, 1]")
    logits_by_model, y_true, class_names, checkpoint_groups = _load_aligned_oofs(oof_paths)
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
    temperature_logits = apply_decision_params(
        ensemble_logits,
        temperature=temperature,
    )
    tuned_logits = apply_decision_params(
        ensemble_logits,
        temperature=temperature,
        bias=bias_result.bias,
    )
    tuned_score = macro_f1_from_logits(tuned_logits, y_true, class_names)
    bootstrap_summary = None
    candidate_bias = [float(value) for value in bias_result.bias]
    bias_accepted = True
    if bootstrap_rounds > 0:
        bootstrap_summary = bootstrap_macro_f1_summary(
            baseline_logits=temperature_logits,
            candidate_logits=tuned_logits,
            y_true=y_true,
            class_names=class_names,
            rounds=bootstrap_rounds,
            sample_fraction=bootstrap_sample_fraction,
            seed=bootstrap_seed,
        )
        bias_accepted = float(bootstrap_summary["delta_q05_macro_f1"]) >= bootstrap_min_delta_q05
        if not bias_accepted:
            bias_result = type(bias_result)(
                bias=[0.0 for _ in class_names],
                score=float(temperature_score),
                baseline_score=bias_result.baseline_score,
            )
            tuned_logits = temperature_logits
            tuned_score = float(temperature_score)
        bootstrap_summary = {
            **bootstrap_summary,
            "seed": int(bootstrap_seed),
            "gate_threshold_delta_q05": float(bootstrap_min_delta_q05),
            "bias_accepted": 1 if bias_accepted else 0,
            "candidate_bias": candidate_bias,
            "accepted_bias": [float(value) for value in bias_result.bias],
        }
    scores = {
        "best_single_macro_f1": float(max(baseline_scores)),
        "ensemble_macro_f1": float(ensemble_result.score),
        "temperature_macro_f1": float(temperature_score),
        "tuned_macro_f1": float(tuned_score),
        "bias_accepted": 1.0 if bias_accepted else 0.0,
    }
    if bootstrap_summary is not None:
        scores.update(
            {
                "bootstrap_rounds": float(bootstrap_rounds),
                "bootstrap_sample_fraction": float(bootstrap_sample_fraction),
                "bootstrap_seed": float(bootstrap_seed),
                "bootstrap_gate_threshold_delta_q05": float(bootstrap_min_delta_q05),
                "bootstrap_delta_q05_macro_f1": float(bootstrap_summary["delta_q05_macro_f1"]),
                "bootstrap_delta_mean_macro_f1": float(bootstrap_summary["delta_mean_macro_f1"]),
            }
        )
    decision_checkpoints, decision_weights = _resolve_decision_checkpoints_and_weights(
        oof_paths=oof_paths,
        checkpoint_groups=checkpoint_groups,
        checkpoints=checkpoints,
        oof_weights=ensemble_result.weights,
    )
    write_decision_params(
        output_dir / "decision_params.json",
        class_names=class_names,
        temperature=temperature,
        bias=bias_result.bias,
        weights=decision_weights,
        scores=scores,
        checkpoints=decision_checkpoints,
    )
    _write_report(
        output_dir / "decision_report.md",
        oof_paths=oof_paths,
        class_names=class_names,
        oof_weights=ensemble_result.weights,
        decision_checkpoints=decision_checkpoints,
        decision_weights=decision_weights,
        temperature=temperature,
        bias=bias_result.bias,
        scores=scores,
        bootstrap=bootstrap_summary,
    )
    (output_dir / "oof_scorecard.json").write_text(
        json.dumps(
            {
                "oof": [str(path) for path in oof_paths],
                "num_models": len(oof_paths),
                "class_names": class_names,
                "oof_weights": ensemble_result.weights,
                "decision_checkpoints": decision_checkpoints,
                "decision_weights": decision_weights,
                "scores": scores,
                "bootstrap": bootstrap_summary,
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
        "weights": decision_weights,
        "oof_weights": ensemble_result.weights,
        "temperature": temperature,
        "bias": bias_result.bias,
        "scores": scores,
        "bootstrap": bootstrap_summary,
    }


def main() -> None:
    args = parse_args()
    stats = run_oof_decision(
        oof_paths=args.oof,
        output_dir=args.output_dir,
        temperature_candidates=args.temperature,
        max_ensemble_steps=args.max_ensemble_steps,
        checkpoints=args.checkpoint,
        bootstrap_rounds=args.bootstrap_rounds,
        bootstrap_sample_fraction=args.bootstrap_sample_fraction,
        bootstrap_seed=args.bootstrap_seed,
        bootstrap_min_delta_q05=args.bootstrap_min_delta_q05,
    )
    print(json.dumps(stats, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
