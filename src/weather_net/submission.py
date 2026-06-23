from __future__ import annotations

import csv
from pathlib import Path
from typing import Sequence


def _read_sample_submission_schema(sample_submission_path: Path) -> tuple[str, str, list[str]]:
    with sample_submission_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames
        if fieldnames is None or len(fieldnames) < 2:
            raise ValueError(f"Sample submission must have at least two columns: {sample_submission_path}")
        image_column, label_column = fieldnames[0], fieldnames[1]
        ids: list[str] = []
        for row in reader:
            ids.append(str(row[image_column]).strip())
    return image_column, label_column, ids


def write_submission(
    output_path: Path,
    image_paths: Sequence[Path],
    predictions: Sequence[str],
    image_ids: Sequence[str] | None = None,
    image_column: str = "image",
    label_column: str = "label",
    sample_submission_path: Path | None = None,
) -> None:
    if len(image_paths) != len(predictions):
        raise ValueError("image_paths and predictions must have the same length")
    if image_ids is not None and len(image_ids) != len(predictions):
        raise ValueError("image_ids and predictions must have the same length")

    ids = list(image_ids) if image_ids is not None else [Path(path).name for path in image_paths]
    duplicate_ids = sorted({image_id for image_id in ids if ids.count(image_id) > 1})
    if duplicate_ids:
        preview = ", ".join(duplicate_ids[:5])
        raise ValueError(f"Duplicate image ids are not safe for submission: {preview}")
    labels_by_id = dict(zip(ids, predictions))
    output_ids = ids
    if sample_submission_path is not None:
        image_column, label_column, output_ids = _read_sample_submission_schema(sample_submission_path)
        missing_ids = [image_id for image_id in output_ids if image_id not in labels_by_id]
        if missing_ids:
            preview = ", ".join(missing_ids[:5])
            raise ValueError(f"Predictions missing sample submission ids: {preview}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow([image_column, label_column])
        for image_id in output_ids:
            writer.writerow([image_id, labels_by_id[image_id]])
