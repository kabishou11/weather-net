from __future__ import annotations

import csv
import math
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from .data import build_manifest_from_csv, idx_to_class, is_validation_source


def _read_csv_rows(csv_path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header: {csv_path}")
        return list(reader.fieldnames), [dict(row) for row in reader]


def _load_oof_probabilities(oof_npz: Path) -> tuple[list[str], dict[str, list[float]]]:
    data = np.load(oof_npz, allow_pickle=True)
    required = {"image_id", "class_names"}
    missing = required - set(data.files)
    if missing:
        raise ValueError(f"OOF file missing fields {sorted(missing)}: {oof_npz}")
    if "probs" in data.files:
        probabilities = np.asarray(data["probs"], dtype=np.float64)
    elif "logits" in data.files:
        logits = np.asarray(data["logits"], dtype=np.float64)
        if logits.ndim != 2:
            raise ValueError(f"OOF logits must be a 2D array: {oof_npz}")
        if not np.isfinite(logits).all():
            raise ValueError(f"OOF logits must be finite: {oof_npz}")
        shifted = logits - logits.max(axis=1, keepdims=True)
        exp = np.exp(shifted)
        probabilities = exp / exp.sum(axis=1, keepdims=True)
    else:
        raise ValueError(f"OOF file must contain probs or logits: {oof_npz}")

    class_names = [str(value) for value in np.asarray(data["class_names"], dtype=object).tolist()]
    image_ids = [str(value) for value in np.asarray(data["image_id"], dtype=object).tolist()]
    if probabilities.ndim != 2:
        raise ValueError(f"OOF probabilities must be a 2D array: {oof_npz}")
    if probabilities.shape[0] != len(image_ids):
        raise ValueError(f"OOF probabilities and image_id must have the same number of rows: {oof_npz}")
    if probabilities.shape[1] != len(class_names):
        raise ValueError(f"OOF probability class dimension must match class_names: {oof_npz}")
    if not class_names or len(set(class_names)) != len(class_names):
        raise ValueError(f"OOF class_names must be non-empty and unique: {oof_npz}")
    if not np.isfinite(probabilities).all():
        raise ValueError(f"OOF probabilities must be finite: {oof_npz}")
    if (probabilities < 0).any():
        raise ValueError(f"OOF probabilities must be non-negative: {oof_npz}")
    row_sums = probabilities.sum(axis=1)
    if not np.allclose(row_sums, 1.0, rtol=1e-4, atol=1e-4):
        raise ValueError(f"OOF probabilities must sum to 1: {oof_npz}")
    duplicate_ids = sorted(image_id for image_id, count in Counter(image_ids).items() if count > 1)
    if duplicate_ids:
        raise ValueError(f"duplicate OOF image_id values: {duplicate_ids[:5]}")
    if "source" in data.files:
        sources = [str(value) for value in np.asarray(data["source"], dtype=object).tolist()]
        if len(sources) != len(image_ids):
            raise ValueError(f"OOF source and image_id must have the same number of rows: {oof_npz}")
        non_labeled = sorted({source for source in sources if source != "labeled"})
        if non_labeled:
            raise ValueError(f"OOF teacher probabilities must come from labeled rows only: {non_labeled}")
    probabilities = np.round(probabilities, 6)
    return class_names, {
        image_id: [float(value) for value in probabilities[idx].tolist()]
        for idx, image_id in enumerate(image_ids)
    }


def write_teacher_csv_from_oof(
    train_csv: Path,
    image_root: Path | None,
    oof_npz: Path,
    output_csv: Path,
    image_column: str | None = None,
    label_column: str | None = None,
    max_top1_mismatch_rate: float | None = None,
    min_mean_true_probability: float | None = None,
    min_samples_per_class: int | None = None,
) -> dict[str, Any]:
    fieldnames, raw_rows = _read_csv_rows(train_csv)
    manifest, class_to_idx = build_manifest_from_csv(
        train_csv,
        image_root=image_root,
        image_column=image_column,
        label_column=label_column,
    )
    if len(raw_rows) != len(manifest):
        raise ValueError("CSV raw rows and manifest rows must have the same length")
    non_labeled_sources = sorted({row.source for row in manifest if not is_validation_source(row)})
    if non_labeled_sources:
        raise ValueError(f"teacher CSV must be built from labeled rows only: {non_labeled_sources}")

    class_names = idx_to_class(class_to_idx)
    oof_class_names, oof_by_image_id = _load_oof_probabilities(oof_npz)
    if oof_class_names != class_names:
        raise ValueError(f"OOF class_names {oof_class_names} do not match training class order {class_names}")

    missing_ids = [
        row.image_id or str(row.path)
        for row in manifest
        if (row.image_id or str(row.path)) not in oof_by_image_id
    ]
    if missing_ids:
        raise ValueError(f"missing OOF teacher probabilities for rows: {missing_ids[:5]}")

    label_counts = Counter(str(row.label_name) for row in manifest)
    if min_samples_per_class is not None:
        if min_samples_per_class <= 0:
            raise ValueError("min_samples_per_class must be positive")
        low_support = sorted(label for label, count in label_counts.items() if count < min_samples_per_class)
        if low_support:
            raise ValueError(f"teacher per-class coverage below min_samples_per_class: {low_support}")

    true_probabilities: list[float] = []
    top1_mismatches = 0
    for row in manifest:
        if row.label is None:
            raise ValueError("teacher rows must be labeled")
        image_id = row.image_id or str(row.path)
        probabilities = oof_by_image_id[image_id]
        true_probabilities.append(float(probabilities[row.label]))
        if max(range(len(probabilities)), key=lambda idx: probabilities[idx]) != int(row.label):
            top1_mismatches += 1
    top1_mismatch_rate = top1_mismatches / max(1, len(manifest))
    mean_true_probability = sum(true_probabilities) / max(1, len(true_probabilities))
    if max_top1_mismatch_rate is not None:
        if max_top1_mismatch_rate < 0 or max_top1_mismatch_rate > 1:
            raise ValueError("max_top1_mismatch_rate must be in [0, 1]")
        if top1_mismatch_rate > max_top1_mismatch_rate:
            raise ValueError(
                f"teacher top1 mismatch rate {top1_mismatch_rate:.6f} exceeds limit "
                f"{max_top1_mismatch_rate:.6f}"
            )
    if min_mean_true_probability is not None:
        if min_mean_true_probability < 0 or min_mean_true_probability > 1:
            raise ValueError("min_mean_true_probability must be in [0, 1]")
        if mean_true_probability < min_mean_true_probability:
            raise ValueError(
                f"mean true-class teacher probability {mean_true_probability:.6f} below limit "
                f"{min_mean_true_probability:.6f}"
            )

    teacher_fields = [f"teacher_{class_name}" for class_name in class_names]
    existing_teacher_fields = [name for name in fieldnames if name.lower().startswith("teacher_")]
    output_fields = [name for name in fieldnames if name not in existing_teacher_fields] + teacher_fields

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=output_fields)
        writer.writeheader()
        for raw_row, manifest_row in zip(raw_rows, manifest):
            image_id = manifest_row.image_id or str(manifest_row.path)
            probabilities = oof_by_image_id[image_id]
            total = sum(probabilities)
            if not math.isclose(total, 1.0, rel_tol=1e-4, abs_tol=1e-4):
                raise ValueError(f"teacher probabilities must sum to 1: {image_id}")
            row = {name: raw_row.get(name, "") for name in output_fields if not name.startswith("teacher_")}
            row.update(
                {
                    f"teacher_{class_name}": f"{float(probabilities[class_idx]):.8f}"
                    for class_idx, class_name in enumerate(class_names)
                }
            )
            writer.writerow(row)

    return {
        "rows": len(manifest),
        "classes": class_names,
        "teacher_quality": {
            "top1_mismatch_rate": float(top1_mismatch_rate),
            "mean_true_probability": float(mean_true_probability),
            "per_class_counts": {label: int(label_counts.get(label, 0)) for label in class_names},
        },
        "output": str(output_csv),
    }
