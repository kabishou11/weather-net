from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import torch

from .inference import normalize_checkpoint_weights
from .postprocess import greedy_search_ensemble_weights


StateDict = Mapping[str, torch.Tensor]


def average_state_dicts(
    state_dicts: Sequence[StateDict],
    weights: Sequence[float] | None = None,
) -> dict[str, torch.Tensor]:
    if not state_dicts:
        raise ValueError("At least one state_dict is required")
    normalized_weights = normalize_checkpoint_weights(
        [Path(str(index)) for index in range(len(state_dicts))],
        weights,
    )
    reference_keys = set(state_dicts[0].keys())
    for state_dict in state_dicts[1:]:
        if set(state_dict.keys()) != reference_keys:
            raise ValueError("state_dict keys must match for model soup")

    averaged: dict[str, torch.Tensor] = {}
    for key in state_dicts[0].keys():
        first = state_dicts[0][key]
        if not torch.is_tensor(first):
            raise TypeError(f"state_dict value must be a tensor: {key}")
        for state_dict in state_dicts[1:]:
            tensor = state_dict[key]
            if first.shape != tensor.shape:
                raise ValueError(f"state_dict tensor shape mismatch: {key}")

        if first.is_floating_point():
            accumulator = torch.zeros_like(first, dtype=torch.float64)
            for state_dict, weight in zip(state_dicts, normalized_weights):
                accumulator = accumulator + state_dict[key].detach().to(dtype=torch.float64) * weight
            averaged[key] = accumulator.to(dtype=first.dtype)
        else:
            for state_dict in state_dicts[1:]:
                if not torch.equal(first, state_dict[key]):
                    raise ValueError(f"non-floating state_dict tensor mismatch: {key}")
            averaged[key] = first.detach().clone()
    return averaged


def _load_checkpoint(path: Path, map_location: str) -> dict[str, object]:
    checkpoint = torch.load(path, map_location=map_location)
    if not isinstance(checkpoint, dict):
        raise ValueError(f"Checkpoint must be a dict: {path}")
    for key in ("model_state", "class_to_idx", "model_name", "image_size"):
        if key not in checkpoint:
            raise ValueError(f"Checkpoint missing {key}: {path}")
    return checkpoint


def _assert_same_metadata(
    checkpoints: Sequence[dict[str, object]],
    paths: Sequence[Path],
    key: str,
) -> None:
    first = checkpoints[0][key]
    for checkpoint, path in zip(checkpoints[1:], paths[1:]):
        if checkpoint[key] != first:
            raise ValueError(f"{key} mismatch in {path}")


def _class_names_from_mapping(class_to_idx: object) -> list[str]:
    if not isinstance(class_to_idx, dict):
        raise ValueError("class_to_idx must be a mapping")
    return [str(name) for name, _idx in sorted(class_to_idx.items(), key=lambda item: int(item[1]))]


def build_model_soup(
    checkpoints: Sequence[Path],
    weights: Sequence[float] | None = None,
    map_location: str = "cpu",
) -> dict[str, object]:
    if not checkpoints:
        raise ValueError("At least one checkpoint is required")

    loaded = [_load_checkpoint(path, map_location=map_location) for path in checkpoints]
    for key in ("class_to_idx", "model_name", "image_size"):
        _assert_same_metadata(loaded, checkpoints, key)

    state_dicts = [checkpoint["model_state"] for checkpoint in loaded]
    averaged_state = average_state_dicts(state_dicts, weights=weights)  # type: ignore[arg-type]
    normalized_weights = normalize_checkpoint_weights(checkpoints, weights)

    soup = dict(loaded[0])
    soup["model_state"] = averaged_state
    soup["macro_f1"] = max(float(checkpoint.get("macro_f1", 0.0)) for checkpoint in loaded)
    soup["soup"] = {
        "checkpoints": [str(path) for path in checkpoints],
        "weights": normalized_weights,
        "macro_f1_values": [float(checkpoint.get("macro_f1", 0.0)) for checkpoint in loaded],
    }
    return soup


def search_oof_gated_soup_weights(
    logits_by_checkpoint: np.ndarray | Sequence[Sequence[Sequence[float]]],
    y_true: np.ndarray | Sequence[int],
    class_names: Sequence[str],
    max_steps: int = 12,
    min_delta: float = 0.0,
) -> dict[str, object]:
    logits_array = np.asarray(logits_by_checkpoint, dtype=np.float64)
    labels = np.asarray(y_true, dtype=np.int64)
    if logits_array.ndim != 3:
        raise ValueError("OOF logits must have shape checkpoint x image x class")
    if labels.ndim != 1:
        raise ValueError("OOF y_true must be a 1D array")
    if logits_array.shape[1] != labels.shape[0]:
        raise ValueError("OOF logits and y_true must have the same number of images")
    if logits_array.shape[2] != len(class_names):
        raise ValueError("OOF logits class dimension must match class_names")
    if min_delta < 0:
        raise ValueError("min_delta must be non-negative")
    result = greedy_search_ensemble_weights(
        logits_array,
        labels,
        class_names,
        max_steps=max_steps,
    )
    if result.score + 1e-12 < result.baseline_score + float(min_delta):
        raise ValueError("OOF-gated soup did not meet the required min_delta over the best checkpoint")
    selected_indices = [idx for idx, weight in enumerate(result.weights) if weight > 0]
    return {
        "weights": result.weights,
        "selected_indices": selected_indices,
        "score": result.score,
        "baseline_score": result.baseline_score,
    }


def build_oof_gated_model_soup(
    checkpoints: Sequence[Path],
    logits_by_checkpoint: np.ndarray | Sequence[Sequence[Sequence[float]]],
    y_true: np.ndarray | Sequence[int],
    class_names: Sequence[str],
    max_steps: int = 12,
    min_delta: float = 0.0,
    map_location: str = "cpu",
) -> dict[str, object]:
    if not checkpoints:
        raise ValueError("At least one checkpoint is required")
    logits_array = np.asarray(logits_by_checkpoint, dtype=np.float64)
    if logits_array.ndim != 3:
        raise ValueError("OOF logits must have shape checkpoint x image x class")
    if logits_array.shape[0] != len(checkpoints):
        raise ValueError("checkpoint count must match OOF logits model dimension")

    loaded = [_load_checkpoint(path, map_location=map_location) for path in checkpoints]
    for key in ("class_to_idx", "model_name", "image_size"):
        _assert_same_metadata(loaded, checkpoints, key)
    checkpoint_class_names = _class_names_from_mapping(loaded[0]["class_to_idx"])
    if checkpoint_class_names != list(class_names):
        raise ValueError("OOF class_names must match checkpoint class_to_idx order")

    search = search_oof_gated_soup_weights(
        logits_by_checkpoint=logits_array,
        y_true=y_true,
        class_names=class_names,
        max_steps=max_steps,
        min_delta=min_delta,
    )
    selected_indices = [int(index) for index in search["selected_indices"]]  # type: ignore[index]
    selected_loaded = [loaded[index] for index in selected_indices]
    selected_checkpoints = [checkpoints[index] for index in selected_indices]
    selected_weights = [float(search["weights"][index]) for index in selected_indices]  # type: ignore[index]
    state_dicts = [checkpoint["model_state"] for checkpoint in selected_loaded]
    averaged_state = average_state_dicts(state_dicts, weights=selected_weights)  # type: ignore[arg-type]
    soup = dict(selected_loaded[0])
    soup["model_state"] = averaged_state
    soup["macro_f1"] = float(search["score"])
    soup["soup"] = {
        "checkpoints": [str(path) for path in selected_checkpoints],
        "weights": [weight / sum(selected_weights) for weight in selected_weights],
        "candidate_checkpoints": [str(path) for path in checkpoints],
        "candidate_weights": search["weights"],
        "macro_f1_values": [float(checkpoint.get("macro_f1", 0.0)) for checkpoint in loaded],
        "mode": "oof_gated_greedy",
        "selected_indices": selected_indices,
        "oof_score": search["score"],
        "gate_oof_ensemble_score": search["score"],
        "baseline_oof_score": search["baseline_score"],
        "min_delta": float(min_delta),
        "max_steps": int(max_steps),
        "note": "OOF score gates logits ensemble weights; rerun validation on the saved soup checkpoint for final score.",
    }
    return soup


def save_model_soup(
    checkpoints: Sequence[Path],
    output_path: Path,
    weights: Sequence[float] | None = None,
    map_location: str = "cpu",
) -> dict[str, object]:
    soup = build_model_soup(checkpoints, weights=weights, map_location=map_location)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(soup, output_path)
    return soup


def save_oof_gated_model_soup(
    checkpoints: Sequence[Path],
    output_path: Path,
    logits_by_checkpoint: np.ndarray | Sequence[Sequence[Sequence[float]]],
    y_true: np.ndarray | Sequence[int],
    class_names: Sequence[str],
    max_steps: int = 12,
    min_delta: float = 0.0,
    map_location: str = "cpu",
) -> dict[str, object]:
    soup = build_oof_gated_model_soup(
        checkpoints=checkpoints,
        logits_by_checkpoint=logits_by_checkpoint,
        y_true=y_true,
        class_names=class_names,
        max_steps=max_steps,
        min_delta=min_delta,
        map_location=map_location,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(soup, output_path)
    return soup
