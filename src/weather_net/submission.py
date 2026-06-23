from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path
from typing import Sequence


def _read_sample_submission_rows(sample_submission_path: Path) -> tuple[list[str], str, str, list[dict[str, str]]]:
    with sample_submission_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames
        if fieldnames is None or len(fieldnames) < 2:
            raise ValueError(f"Sample submission must have at least two columns: {sample_submission_path}")
        image_column, label_column = fieldnames[0], fieldnames[1]
        rows = [dict(row) for row in reader]
    return list(fieldnames), image_column, label_column, rows


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
    duplicate_ids = sorted(image_id for image_id, count in Counter(ids).items() if count > 1)
    if duplicate_ids:
        preview = ", ".join(duplicate_ids[:5])
        raise ValueError(f"Duplicate image ids are not safe for submission: {preview}")
    labels_by_id = dict(zip(ids, predictions))
    output_fieldnames = [image_column, label_column]
    output_rows = [{image_column: image_id, label_column: labels_by_id[image_id]} for image_id in ids]
    if sample_submission_path is not None:
        output_fieldnames, image_column, label_column, sample_rows = _read_sample_submission_rows(
            sample_submission_path
        )
        sample_ids = [str(row[image_column]).strip() for row in sample_rows]
        duplicate_sample_ids = sorted(image_id for image_id, count in Counter(sample_ids).items() if count > 1)
        if duplicate_sample_ids:
            preview = ", ".join(duplicate_sample_ids[:5])
            raise ValueError(f"Duplicate sample submission ids are not safe: {preview}")
        missing_ids = [image_id for image_id in sample_ids if image_id not in labels_by_id]
        if missing_ids:
            preview = ", ".join(missing_ids[:5])
            raise ValueError(f"Predictions missing sample submission ids: {preview}")
        output_rows = []
        for row in sample_rows:
            image_id = str(row[image_column]).strip()
            output_row = {fieldname: row.get(fieldname, "") for fieldname in output_fieldnames}
            output_row[image_column] = image_id
            output_row[label_column] = labels_by_id[image_id]
            output_rows.append(output_row)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=output_fieldnames)
        writer.writeheader()
        writer.writerows(output_rows)
