from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from src.weather_net.soup import save_model_soup, save_oof_gated_model_soup


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Average same-architecture checkpoints into a model soup.")
    parser.add_argument("--checkpoints", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--weights", type=float, nargs="*", default=None)
    parser.add_argument("--map-location", type=str, default="cpu")
    parser.add_argument("--oof", type=Path, nargs="*", default=None)
    parser.add_argument("--min-delta", type=float, default=0.0)
    parser.add_argument("--max-steps", type=int, default=12)
    return parser.parse_args()


def _load_aligned_oof_logits(paths: list[Path]) -> tuple[np.ndarray, np.ndarray, list[str]]:
    logits_list: list[np.ndarray] = []
    expected_y_true: np.ndarray | None = None
    expected_class_names: list[str] | None = None
    expected_image_ids: list[str] | None = None
    for path in paths:
        data = np.load(path, allow_pickle=True)
        required = {"logits", "y_true", "class_names", "image_id"}
        missing = required - set(data.files)
        if missing:
            raise ValueError(f"OOF file missing fields {sorted(missing)}: {path}")
        logits = np.asarray(data["logits"], dtype=np.float64)
        y_true = np.asarray(data["y_true"], dtype=np.int64)
        class_names = [str(value) for value in data["class_names"].tolist()]
        image_ids = [str(value) for value in data["image_id"].tolist()]
        if logits.ndim != 2:
            raise ValueError(f"OOF logits must be a 2D array: {path}")
        if logits.shape[0] != y_true.shape[0] or logits.shape[0] != len(image_ids):
            raise ValueError(f"OOF logits, y_true, and image_id must have the same rows: {path}")
        if logits.shape[1] != len(class_names):
            raise ValueError(f"OOF logits class dimension must match class_names: {path}")
        if expected_y_true is None:
            expected_y_true = y_true
            expected_class_names = class_names
            expected_image_ids = image_ids
        else:
            if not np.array_equal(y_true, expected_y_true):
                raise ValueError(f"OOF y_true differs: {path}")
            if class_names != expected_class_names:
                raise ValueError(f"OOF class_names differ: {path}")
            if image_ids != expected_image_ids:
                raise ValueError(f"OOF image_id order differs: {path}")
        logits_list.append(logits)
    assert expected_y_true is not None
    assert expected_class_names is not None
    return np.stack(logits_list, axis=0), expected_y_true, expected_class_names


def run_soup(
    checkpoints: list[Path],
    output: Path,
    weights: list[float] | None,
    map_location: str,
    oof_paths: list[Path] | None = None,
    min_delta: float = 0.0,
    max_steps: int = 12,
) -> dict[str, object]:
    if oof_paths:
        if weights is not None:
            raise ValueError("--weights cannot be combined with --oof; OOF-gated soup searches weights")
        if len(oof_paths) != len(checkpoints):
            raise ValueError("--oof count must match --checkpoints count")
        logits_by_checkpoint, y_true, class_names = _load_aligned_oof_logits(oof_paths)
        soup = save_oof_gated_model_soup(
            checkpoints=checkpoints,
            output_path=output,
            logits_by_checkpoint=logits_by_checkpoint,
            y_true=y_true,
            class_names=class_names,
            max_steps=max_steps,
            min_delta=min_delta,
            map_location=map_location,
        )
    else:
        soup = save_model_soup(
            checkpoints=checkpoints,
            output_path=output,
            weights=weights,
            map_location=map_location,
        )
    return {
        "output": str(output),
        "model_name": soup["model_name"],
        "image_size": soup["image_size"],
        "num_classes": len(soup["class_to_idx"]),
        "soup": soup["soup"],
    }


def main() -> None:
    args = parse_args()
    stats = run_soup(
        checkpoints=args.checkpoints,
        output=args.output,
        weights=args.weights,
        map_location=args.map_location,
        oof_paths=args.oof,
        min_delta=args.min_delta,
        max_steps=args.max_steps,
    )
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
