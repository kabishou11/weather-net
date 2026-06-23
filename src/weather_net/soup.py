from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

import torch

from .inference import normalize_checkpoint_weights


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
