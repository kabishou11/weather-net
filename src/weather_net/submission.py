from __future__ import annotations

import csv
from pathlib import Path
from typing import Sequence


def write_submission(
    output_path: Path,
    image_paths: Sequence[Path],
    predictions: Sequence[str],
    image_ids: Sequence[str] | None = None,
    image_column: str = "image",
    label_column: str = "label",
) -> None:
    if len(image_paths) != len(predictions):
        raise ValueError("image_paths and predictions must have the same length")
    if image_ids is not None and len(image_ids) != len(predictions):
        raise ValueError("image_ids and predictions must have the same length")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow([image_column, label_column])
        ids = image_ids if image_ids is not None else [Path(path).name for path in image_paths]
        for image_id, label in zip(ids, predictions):
            writer.writerow([image_id, label])
