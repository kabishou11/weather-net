from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from .metrics import classification_report


@dataclass(frozen=True)
class BiasSearchResult:
    bias: list[float]
    score: float
    baseline_score: float


@dataclass(frozen=True)
class EnsembleSearchResult:
    weights: list[float]
    score: float
    baseline_score: float
    logits: np.ndarray


@dataclass(frozen=True)
class DecisionParams:
    class_names: list[str]
    temperature: float
    bias: list[float]
    weights: list[float] | None
    scores: dict[str, float]
    checkpoints: list[str] | None = None


def _as_2d_logits(logits: np.ndarray | Sequence[Sequence[float]]) -> np.ndarray:
    array = np.asarray(logits, dtype=np.float64)
    if array.ndim != 2:
        raise ValueError("logits must be a 2D array")
    return array


def predict_from_logits(logits: np.ndarray | Sequence[Sequence[float]]) -> list[int]:
    return _as_2d_logits(logits).argmax(axis=1).astype(int).tolist()


def macro_f1_from_logits(
    logits: np.ndarray | Sequence[Sequence[float]],
    y_true: np.ndarray | Sequence[int],
    class_names: Sequence[str],
) -> float:
    predictions = predict_from_logits(logits)
    return classification_report(list(y_true), predictions, class_names).macro_f1


def apply_decision_params(
    logits: np.ndarray | Sequence[Sequence[float]],
    temperature: float = 1.0,
    bias: Sequence[float] | None = None,
) -> np.ndarray:
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    adjusted = _as_2d_logits(logits) / float(temperature)
    if bias is not None:
        bias_array = np.asarray(bias, dtype=np.float64)
        if bias_array.shape != (adjusted.shape[1],):
            raise ValueError("bias length must match logits class dimension")
        adjusted = adjusted + bias_array
    return adjusted


def normalize_weights(weights: Sequence[float] | None, n_models: int) -> list[float]:
    if n_models <= 0:
        raise ValueError("n_models must be positive")
    if weights is None:
        return [1.0 / n_models] * n_models
    if len(weights) != n_models:
        raise ValueError("weights length must match number of models")
    if any(weight < 0 for weight in weights):
        raise ValueError("weights must be non-negative")
    total = float(sum(weights))
    if total <= 0:
        raise ValueError("weights must have a positive sum")
    return [float(weight) / total for weight in weights]


def weighted_sum_logits(
    logits_by_model: np.ndarray | Sequence[Sequence[Sequence[float]]],
    weights: Sequence[float] | None = None,
) -> np.ndarray:
    logits_array = np.asarray(logits_by_model, dtype=np.float64)
    if logits_array.ndim != 3:
        raise ValueError("logits_by_model must have shape M x N x C")
    normalized = np.asarray(normalize_weights(weights, logits_array.shape[0]), dtype=np.float64)
    return np.tensordot(normalized, logits_array, axes=(0, 0))


def search_temperature(
    logits: np.ndarray | Sequence[Sequence[float]],
    y_true: np.ndarray | Sequence[int],
    class_names: Sequence[str],
    candidates: Sequence[float] | None = None,
) -> tuple[float, float]:
    candidates = candidates or [0.75, 1.0, 1.25, 1.5, 2.0]
    best_temperature = 1.0
    best_score = -1.0
    for temperature in candidates:
        temperature = float(temperature)
        score = macro_f1_from_logits(
            apply_decision_params(logits, temperature=temperature),
            y_true,
            class_names,
        )
        if score > best_score or (
            score == best_score and abs(temperature - 1.0) < abs(best_temperature - 1.0)
        ):
            best_score = score
            best_temperature = temperature
    return best_temperature, best_score


def search_per_class_bias(
    logits: np.ndarray | Sequence[Sequence[float]],
    y_true: np.ndarray | Sequence[int],
    class_names: Sequence[str],
    temperature: float = 1.0,
    steps: Sequence[float] | None = None,
    max_passes: int = 8,
    max_abs_bias: float = 3.0,
    eps: float = 1e-12,
) -> BiasSearchResult:
    adjusted = apply_decision_params(logits, temperature=temperature)
    num_classes = adjusted.shape[1]
    bias = np.zeros(num_classes, dtype=np.float64)
    baseline = macro_f1_from_logits(adjusted, y_true, class_names)
    best_score = baseline
    steps = steps or [1.0, 0.5, 0.25, 0.1, 0.05]

    for delta in steps:
        for _ in range(max_passes):
            improved = False
            for class_idx in range(num_classes):
                best_local_score = best_score
                best_local_bias = bias
                for direction in (-1.0, 1.0):
                    trial = bias.copy()
                    trial[class_idx] = np.clip(
                        trial[class_idx] + (direction * float(delta)),
                        -max_abs_bias,
                        max_abs_bias,
                    )
                    trial = trial - trial.mean()
                    score = macro_f1_from_logits(adjusted + trial, y_true, class_names)
                    if score > best_local_score + eps:
                        best_local_score = score
                        best_local_bias = trial
                if best_local_score > best_score + eps:
                    bias = best_local_bias
                    best_score = best_local_score
                    improved = True
            if not improved:
                break
    return BiasSearchResult(
        bias=[float(value) for value in bias],
        score=float(best_score),
        baseline_score=float(baseline),
    )


def greedy_search_ensemble_weights(
    logits_by_model: np.ndarray | Sequence[Sequence[Sequence[float]]],
    y_true: np.ndarray | Sequence[int],
    class_names: Sequence[str],
    max_steps: int = 12,
    eps: float = 1e-12,
) -> EnsembleSearchResult:
    logits_array = np.asarray(logits_by_model, dtype=np.float64)
    if logits_array.ndim != 3:
        raise ValueError("logits_by_model must have shape M x N x C")
    n_models = logits_array.shape[0]
    single_scores = [
        macro_f1_from_logits(logits_array[idx], y_true, class_names)
        for idx in range(n_models)
    ]
    best_model = int(np.argmax(single_scores))
    counts = np.zeros(n_models, dtype=np.float64)
    counts[best_model] = 1.0
    weights = counts / counts.sum()
    current_logits = weighted_sum_logits(logits_array, weights)
    best_score = macro_f1_from_logits(current_logits, y_true, class_names)
    baseline_score = max(single_scores)

    for _ in range(max_steps):
        best_trial_score = best_score
        best_trial_counts = counts
        best_trial_logits = current_logits
        for model_idx in range(n_models):
            trial_counts = counts.copy()
            trial_counts[model_idx] += 1.0
            trial_weights = trial_counts / trial_counts.sum()
            trial_logits = weighted_sum_logits(logits_array, trial_weights)
            score = macro_f1_from_logits(trial_logits, y_true, class_names)
            if score > best_trial_score + eps:
                best_trial_score = score
                best_trial_counts = trial_counts
                best_trial_logits = trial_logits
        if best_trial_score <= best_score + eps:
            break
        counts = best_trial_counts
        weights = counts / counts.sum()
        current_logits = best_trial_logits
        best_score = best_trial_score

    return EnsembleSearchResult(
        weights=[float(value) for value in weights],
        score=float(best_score),
        baseline_score=float(baseline_score),
        logits=current_logits,
    )


def write_decision_params(
    path: Path,
    class_names: Sequence[str],
    temperature: float,
    bias: Sequence[float],
    weights: Sequence[float] | None,
    scores: dict[str, float],
    checkpoints: Sequence[str] | None = None,
) -> None:
    if len(bias) != len(class_names):
        raise ValueError("bias length must match class_names")
    payload = {
        "version": 1,
        "class_names": list(class_names),
        "temperature": float(temperature),
        "bias": [float(value) for value in bias],
        "weights": None if weights is None else normalize_weights(weights, len(weights)),
        "checkpoints": None if checkpoints is None else [str(path) for path in checkpoints],
        "scores": {key: float(value) for key, value in scores.items()},
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def load_decision_params(
    path: Path,
    expected_class_names: Sequence[str] | None = None,
) -> DecisionParams:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if int(payload.get("version", 0)) != 1:
        raise ValueError(f"Unsupported decision params version: {payload.get('version')}")
    class_names = [str(name) for name in payload["class_names"]]
    if expected_class_names is not None and class_names != list(expected_class_names):
        raise ValueError("decision params class_names do not match checkpoint class mapping")
    temperature = float(payload.get("temperature", 1.0))
    bias = [float(value) for value in payload.get("bias", [0.0] * len(class_names))]
    if len(bias) != len(class_names):
        raise ValueError("decision params bias length must match class_names")
    weights = payload.get("weights")
    normalized_weights = None if weights is None else normalize_weights([float(value) for value in weights], len(weights))
    checkpoints = payload.get("checkpoints")
    scores = {str(key): float(value) for key, value in payload.get("scores", {}).items()}
    return DecisionParams(
        class_names=class_names,
        temperature=temperature,
        bias=bias,
        weights=normalized_weights,
        scores=scores,
        checkpoints=None if checkpoints is None else [str(path) for path in checkpoints],
    )
