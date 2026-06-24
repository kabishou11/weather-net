from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import dataclass
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


@dataclass(frozen=True)
class DecisionFit:
    weights: list[float]
    baseline_scores: list[float]
    ensemble_score: float
    ensemble_baseline_score: float
    temperature: float
    temperature_score: float
    bias: list[float]
    bias_baseline_score: float
    tuned_score: float
    ensemble_logits: np.ndarray
    temperature_logits: np.ndarray
    tuned_logits: np.ndarray


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
    parser.add_argument("--nested", action="store_true", help="Evaluate decision tuning with fold-held-out OOF splits.")
    parser.add_argument("--nested-min-delta-macro-f1", type=float, default=0.0)
    return parser.parse_args()


def _fit_decision(
    logits_by_model: np.ndarray,
    y_true: np.ndarray,
    class_names: Sequence[str],
    temperature_candidates: Sequence[float] | None,
    max_ensemble_steps: int,
) -> DecisionFit:
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
    return DecisionFit(
        weights=[float(value) for value in ensemble_result.weights],
        baseline_scores=[float(value) for value in baseline_scores],
        ensemble_score=float(ensemble_result.score),
        ensemble_baseline_score=float(ensemble_result.baseline_score),
        temperature=float(temperature),
        temperature_score=float(temperature_score),
        bias=[float(value) for value in bias_result.bias],
        bias_baseline_score=float(bias_result.baseline_score),
        tuned_score=float(tuned_score),
        ensemble_logits=ensemble_logits,
        temperature_logits=temperature_logits,
        tuned_logits=tuned_logits,
    )


def _apply_fit_to_logits(logits_by_model: np.ndarray, fit: DecisionFit) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ensemble_logits = weighted_sum_logits(logits_by_model, fit.weights)
    temperature_logits = apply_decision_params(
        ensemble_logits,
        temperature=fit.temperature,
    )
    tuned_logits = apply_decision_params(
        ensemble_logits,
        temperature=fit.temperature,
        bias=fit.bias,
    )
    return ensemble_logits, temperature_logits, tuned_logits


def _load_oof_npz(
    path: Path,
    require_fold: bool = False,
    require_labeled_source: bool = False,
) -> tuple[np.ndarray, np.ndarray, list[str], list[str], list[str], np.ndarray | None, list[str] | None]:
    data = np.load(path, allow_pickle=True)
    required = {"logits", "y_true", "class_names"}
    if require_fold:
        required.add("fold")
    if require_labeled_source:
        required.add("source")
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
    folds = None
    if "fold" in data.files:
        raw_folds = np.asarray(data["fold"])
        if raw_folds.ndim != 1 or raw_folds.shape[0] != logits.shape[0]:
            raise ValueError(f"OOF fold values must match logits rows: {path}")
        numeric_folds = np.asarray(raw_folds, dtype=np.float64)
        if not np.all(np.isfinite(numeric_folds)):
            raise ValueError(f"OOF fold values must be finite integers: {path}")
        if not np.all(np.equal(numeric_folds, np.floor(numeric_folds))):
            raise ValueError(f"OOF fold values must be integers: {path}")
        folds = numeric_folds.astype(np.int64)
        if folds.size and int(folds.min()) < 0:
            raise ValueError(f"OOF fold values must be non-negative: {path}")
    sources = None
    if "source" in data.files:
        sources = [str(value) for value in data["source"].tolist()]
        if len(sources) != logits.shape[0]:
            raise ValueError(f"OOF source values must match logits rows: {path}")
        non_labeled = sorted({source for source in sources if source != "labeled"})
        if require_labeled_source and non_labeled:
            raise ValueError(f"OOF source values must be labeled for nested decision calibration: {non_labeled}")
    return logits, y_true, class_names, image_ids, checkpoint_values, folds, sources


def _unique_in_order(values: Sequence[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            ordered.append(value)
    return ordered


def _load_aligned_oofs(
    paths: Sequence[Path],
    require_fold: bool = False,
    require_labeled_source: bool = False,
) -> tuple[np.ndarray, np.ndarray, list[str], list[list[str]], np.ndarray | None, list[str] | None]:
    logits_list: list[np.ndarray] = []
    checkpoint_groups: list[list[str]] = []
    expected_y_true: np.ndarray | None = None
    expected_class_names: list[str] | None = None
    expected_image_ids: list[str] | None = None
    expected_folds: np.ndarray | None = None
    expected_sources: list[str] | None = None
    for path in paths:
        logits, y_true, class_names, image_ids, checkpoint_values, folds, sources = _load_oof_npz(
            path,
            require_fold=require_fold,
            require_labeled_source=require_labeled_source,
        )
        if expected_y_true is None:
            expected_y_true = y_true
            expected_class_names = class_names
            expected_image_ids = image_ids
            expected_folds = folds
            expected_sources = sources
        else:
            if not np.array_equal(y_true, expected_y_true):
                raise ValueError(f"OOF y_true differs: {path}")
            if class_names != expected_class_names:
                raise ValueError(f"OOF class_names differ: {path}")
            if expected_image_ids and image_ids and image_ids != expected_image_ids:
                raise ValueError(f"OOF image_id order differs: {path}")
            if require_fold and not np.array_equal(folds, expected_folds):
                raise ValueError(f"OOF fold order differs: {path}")
            if require_labeled_source and sources != expected_sources:
                raise ValueError(f"OOF source order differs: {path}")
        logits_list.append(logits)
        checkpoint_groups.append(_unique_in_order(checkpoint_values))
    assert expected_y_true is not None
    assert expected_class_names is not None
    return np.stack(logits_list, axis=0), expected_y_true, expected_class_names, checkpoint_groups, expected_folds, expected_sources


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


def _class_counts(y_true: np.ndarray, class_names: Sequence[str]) -> dict[str, int]:
    return {
        class_name: int((y_true == class_idx).sum())
        for class_idx, class_name in enumerate(class_names)
    }


def _run_nested_decision(
    logits_by_model: np.ndarray,
    y_true: np.ndarray,
    class_names: Sequence[str],
    folds: np.ndarray,
    temperature_candidates: Sequence[float] | None,
    max_ensemble_steps: int,
    min_delta_macro_f1: float,
) -> dict[str, object]:
    unique_folds = sorted(int(value) for value in np.unique(folds).tolist())
    if len(unique_folds) < 2:
        raise ValueError("nested OOF decision calibration requires at least two folds")

    pooled_ensemble_logits = np.zeros_like(logits_by_model[0], dtype=np.float64)
    pooled_temperature_logits = np.zeros_like(logits_by_model[0], dtype=np.float64)
    pooled_tuned_logits = np.zeros_like(logits_by_model[0], dtype=np.float64)
    fold_payloads: list[dict[str, object]] = []

    for fold in unique_folds:
        eval_mask = folds == fold
        train_mask = ~eval_mask
        if not eval_mask.any() or not train_mask.any():
            raise ValueError(f"nested fold {fold} must have both train and eval rows")
        train_y = y_true[train_mask]
        eval_y = y_true[eval_mask]
        if len(np.unique(train_y)) < len(class_names):
            missing = [
                class_name
                for class_idx, class_name in enumerate(class_names)
                if not np.any(train_y == class_idx)
            ]
            raise ValueError(f"nested train split for fold {fold} is missing classes: {missing}")
        fit = _fit_decision(
            logits_by_model[:, train_mask, :],
            train_y,
            class_names,
            temperature_candidates=temperature_candidates,
            max_ensemble_steps=max_ensemble_steps,
        )
        ensemble_eval, temperature_eval, tuned_eval = _apply_fit_to_logits(
            logits_by_model[:, eval_mask, :],
            fit,
        )
        pooled_ensemble_logits[eval_mask] = ensemble_eval
        pooled_temperature_logits[eval_mask] = temperature_eval
        pooled_tuned_logits[eval_mask] = tuned_eval

        ensemble_eval_score = macro_f1_from_logits(ensemble_eval, eval_y, class_names)
        temperature_eval_score = macro_f1_from_logits(temperature_eval, eval_y, class_names)
        tuned_eval_score = macro_f1_from_logits(tuned_eval, eval_y, class_names)
        fold_payloads.append(
            {
                "fold": int(fold),
                "train_rows": int(train_mask.sum()),
                "eval_rows": int(eval_mask.sum()),
                "train_class_counts": _class_counts(train_y, class_names),
                "eval_class_counts": _class_counts(eval_y, class_names),
                "missing_eval_classes": [
                    class_name
                    for class_idx, class_name in enumerate(class_names)
                    if not np.any(eval_y == class_idx)
                ],
                "weights": fit.weights,
                "temperature": fit.temperature,
                "bias": fit.bias,
                "train_scores": {
                    "best_single_macro_f1": float(max(fit.baseline_scores)),
                    "ensemble_macro_f1": fit.ensemble_score,
                    "temperature_macro_f1": fit.temperature_score,
                    "tuned_macro_f1": fit.tuned_score,
                },
                "eval_scores": {
                    "ensemble_macro_f1": float(ensemble_eval_score),
                    "temperature_macro_f1": float(temperature_eval_score),
                    "tuned_macro_f1": float(tuned_eval_score),
                    "delta_tuned_vs_temperature_macro_f1": float(tuned_eval_score - temperature_eval_score),
                },
            }
        )

    pooled_ensemble_score = macro_f1_from_logits(pooled_ensemble_logits, y_true, class_names)
    pooled_temperature_score = macro_f1_from_logits(pooled_temperature_logits, y_true, class_names)
    pooled_tuned_score = macro_f1_from_logits(pooled_tuned_logits, y_true, class_names)
    delta = float(pooled_tuned_score - pooled_temperature_score)
    bias_accepted = delta >= min_delta_macro_f1
    return {
        "folds": fold_payloads,
        "pooled_scores": {
            "ensemble_macro_f1": float(pooled_ensemble_score),
            "temperature_macro_f1": float(pooled_temperature_score),
            "tuned_macro_f1": float(pooled_tuned_score),
            "delta_tuned_vs_temperature_macro_f1": delta,
        },
        "gate_threshold_delta_macro_f1": float(min_delta_macro_f1),
        "bias_accepted": 1 if bias_accepted else 0,
    }


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
    nested: dict[str, object] | None = None,
) -> None:
    final_bias_accepted = float(scores.get("bias_accepted", 0.0))
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
        f"- Final bias accepted: {final_bias_accepted:.0f}",
        "",
        "## Scores",
        *[f"- {key}: {value:.6f}" for key, value in scores.items()],
        "",
    ]
    if bootstrap is not None:
        lines.extend(
            [
                "## Bootstrap Full-OOF Stability",
                *[
                    f"- {key}: {value:.6f}" if isinstance(value, float) else f"- {key}: {value}"
                    for key, value in bootstrap.items()
                ],
                "",
            ]
        )
    if nested is not None:
        pooled = nested["pooled_scores"]
        lines.extend(
            [
                "## Nested OOF",
                f"- Folds: {len(nested['folds'])}",
                f"- Gate threshold delta macro F1: {nested['gate_threshold_delta_macro_f1']:.6f}",
                f"- Nested bias accepted: {nested['bias_accepted']}",
                f"- Final bias accepted: {final_bias_accepted:.0f}",
                *[f"- {key}: {value:.6f}" for key, value in pooled.items()],
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
    nested: bool = False,
    nested_min_delta_macro_f1: float = 0.0,
) -> dict[str, object]:
    if bootstrap_rounds < 0:
        raise ValueError("bootstrap_rounds must be non-negative")
    if not 0 < bootstrap_sample_fraction <= 1:
        raise ValueError("bootstrap_sample_fraction must be in (0, 1]")
    logits_by_model, y_true, class_names, checkpoint_groups, folds, _sources = _load_aligned_oofs(
        oof_paths,
        require_fold=nested,
        require_labeled_source=nested,
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    full_fit = _fit_decision(
        logits_by_model,
        y_true,
        class_names,
        temperature_candidates=temperature_candidates,
        max_ensemble_steps=max_ensemble_steps,
    )
    temperature = full_fit.temperature
    temperature_score = full_fit.temperature_score
    temperature_logits = full_fit.temperature_logits
    bootstrap_summary = None
    nested_summary = None
    candidate_bias = [float(value) for value in full_fit.bias]
    nested_bias_accepted = True
    bootstrap_bias_accepted = True
    if nested:
        assert folds is not None
        nested_summary = _run_nested_decision(
            logits_by_model=logits_by_model,
            y_true=y_true,
            class_names=class_names,
            folds=folds,
            temperature_candidates=temperature_candidates,
            max_ensemble_steps=max_ensemble_steps,
            min_delta_macro_f1=nested_min_delta_macro_f1,
        )
        nested_bias_accepted = int(nested_summary["bias_accepted"]) == 1
    if bootstrap_rounds > 0:
        bootstrap_summary = bootstrap_macro_f1_summary(
            baseline_logits=temperature_logits,
            candidate_logits=full_fit.tuned_logits,
            y_true=y_true,
            class_names=class_names,
            rounds=bootstrap_rounds,
            sample_fraction=bootstrap_sample_fraction,
            seed=bootstrap_seed,
        )
        bootstrap_bias_accepted = float(bootstrap_summary["delta_q05_macro_f1"]) >= bootstrap_min_delta_q05
        bootstrap_summary = {
            **bootstrap_summary,
            "seed": int(bootstrap_seed),
            "gate_threshold_delta_q05": float(bootstrap_min_delta_q05),
            "bias_accepted": 1 if bootstrap_bias_accepted else 0,
            "candidate_bias": candidate_bias,
            "accepted_bias": [
                float(value)
                for value in (full_fit.bias if nested_bias_accepted and bootstrap_bias_accepted else [0.0 for _ in class_names])
            ],
        }
    bias_accepted = nested_bias_accepted and bootstrap_bias_accepted
    final_bias = full_fit.bias if bias_accepted else [0.0 for _ in class_names]
    if bias_accepted:
        tuned_score = full_fit.tuned_score
    else:
        tuned_score = float(temperature_score)
    scores = {
        "best_single_macro_f1": float(max(full_fit.baseline_scores)),
        "ensemble_macro_f1": float(full_fit.ensemble_score),
        "temperature_macro_f1": float(temperature_score),
        "tuned_macro_f1": float(tuned_score),
        "bias_accepted": 1.0 if bias_accepted else 0.0,
    }
    if nested_summary is not None:
        nested_pooled = nested_summary["pooled_scores"]
        scores.update(
            {
                "nested_ensemble_macro_f1": float(nested_pooled["ensemble_macro_f1"]),
                "nested_temperature_macro_f1": float(nested_pooled["temperature_macro_f1"]),
                "nested_tuned_macro_f1": float(nested_pooled["tuned_macro_f1"]),
                "nested_delta_tuned_vs_temperature_macro_f1": float(
                    nested_pooled["delta_tuned_vs_temperature_macro_f1"]
                ),
                "nested_gate_threshold_delta_macro_f1": float(nested_min_delta_macro_f1),
                "nested_bias_accepted": 1.0 if nested_bias_accepted else 0.0,
            }
        )
    if bootstrap_summary is not None:
        scores.update(
            {
                "bootstrap_rounds": float(bootstrap_rounds),
                "bootstrap_sample_fraction": float(bootstrap_sample_fraction),
                "bootstrap_seed": float(bootstrap_seed),
                "bootstrap_gate_threshold_delta_q05": float(bootstrap_min_delta_q05),
                "bootstrap_delta_q05_macro_f1": float(bootstrap_summary["delta_q05_macro_f1"]),
                "bootstrap_delta_mean_macro_f1": float(bootstrap_summary["delta_mean_macro_f1"]),
                "bootstrap_full_oof_bias_accepted": 1.0 if bootstrap_bias_accepted else 0.0,
                "bootstrap_full_oof_gate_threshold_delta_q05": float(bootstrap_min_delta_q05),
                "bootstrap_full_oof_delta_q05_macro_f1": float(bootstrap_summary["delta_q05_macro_f1"]),
                "bootstrap_full_oof_delta_mean_macro_f1": float(bootstrap_summary["delta_mean_macro_f1"]),
            }
        )
    decision_checkpoints, decision_weights = _resolve_decision_checkpoints_and_weights(
        oof_paths=oof_paths,
        checkpoint_groups=checkpoint_groups,
        checkpoints=checkpoints,
        oof_weights=full_fit.weights,
    )
    write_decision_params(
        output_dir / "decision_params.json",
        class_names=class_names,
        temperature=temperature,
        bias=final_bias,
        weights=decision_weights,
        scores=scores,
        checkpoints=decision_checkpoints,
    )
    _write_report(
        output_dir / "decision_report.md",
        oof_paths=oof_paths,
        class_names=class_names,
        oof_weights=full_fit.weights,
        decision_checkpoints=decision_checkpoints,
        decision_weights=decision_weights,
        temperature=temperature,
        bias=final_bias,
        scores=scores,
        bootstrap=bootstrap_summary,
        nested=nested_summary,
    )
    if nested_summary is not None:
        (output_dir / "nested_oof_scorecard.json").write_text(
            json.dumps(nested_summary, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    (output_dir / "oof_scorecard.json").write_text(
        json.dumps(
            {
                "oof": [str(path) for path in oof_paths],
                "num_models": len(oof_paths),
                "class_names": class_names,
                "oof_weights": full_fit.weights,
                "decision_checkpoints": decision_checkpoints,
                "decision_weights": decision_weights,
                "scores": scores,
                "bootstrap": bootstrap_summary,
                "nested": nested_summary,
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
        "oof_weights": full_fit.weights,
        "temperature": temperature,
        "bias": final_bias,
        "scores": scores,
        "bootstrap": bootstrap_summary,
        "nested": nested_summary,
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
        nested=args.nested,
        nested_min_delta_macro_f1=args.nested_min_delta_macro_f1,
    )
    print(json.dumps(stats, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
