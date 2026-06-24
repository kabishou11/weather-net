from __future__ import annotations

import csv
import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class HardSample:
    image_id: str
    sample_weight: float
    hard_reason: str
    confidence: float
    margin: float
    loss: float
    true_label: str
    pred_label: str


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header: {path}")
        return [dict(row) for row in reader]


def _truthy(value: str) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def _validate_oof_records(records: Sequence[Mapping[str, str]]) -> None:
    required = {"image_id", "true_label", "pred_label", "correct", "confidence", "margin", "loss"}
    if not records:
        raise ValueError("OOF CSV contains no rows")
    missing = required - set(records[0])
    if missing:
        raise ValueError(f"OOF CSV missing columns: {sorted(missing)}")
    counts = Counter(str(record["image_id"]).strip() for record in records)
    duplicates = sorted(image_id for image_id, count in counts.items() if count > 1)
    if duplicates:
        preview = ", ".join(duplicates[:5])
        raise ValueError(f"Duplicate OOF image ids are not safe: {preview}")


def compute_hard_sample_weights(
    records: Sequence[Mapping[str, str]],
    base_weight: float = 1.0,
    error_boost: float = 1.0,
    low_margin_boost: float = 0.5,
    high_loss_boost: float = 0.5,
    low_margin_threshold: float = 0.1,
    high_loss_quantile: float = 0.75,
    max_weight: float = 2.5,
) -> dict[str, HardSample]:
    _validate_oof_records(records)
    if base_weight <= 0:
        raise ValueError("base_weight must be positive")
    if error_boost < 0 or low_margin_boost < 0 or high_loss_boost < 0:
        raise ValueError("boost values must be non-negative")
    if low_margin_threshold < 0:
        raise ValueError("low_margin_threshold must be non-negative")
    if not 0 <= high_loss_quantile <= 1:
        raise ValueError("high_loss_quantile must be in [0, 1]")
    if max_weight < base_weight:
        raise ValueError("max_weight must be >= base_weight")

    losses = np.asarray([float(record["loss"]) for record in records], dtype=np.float64)
    if not np.isfinite(losses).all():
        raise ValueError("OOF losses must be finite")
    loss_threshold = float(np.quantile(losses, high_loss_quantile))

    weights: dict[str, HardSample] = {}
    for record in records:
        image_id = str(record["image_id"]).strip()
        confidence = float(record["confidence"])
        margin = float(record["margin"])
        loss = float(record["loss"])
        if not all(math.isfinite(value) for value in [confidence, margin, loss]):
            raise ValueError(f"OOF numeric values must be finite: {image_id}")
        reasons: list[str] = []
        sample_weight = base_weight
        if not _truthy(record["correct"]):
            sample_weight += error_boost
            reasons.append("error")
        if margin <= low_margin_threshold:
            sample_weight += low_margin_boost
            reasons.append("low_margin")
        if loss >= loss_threshold:
            sample_weight += high_loss_boost
            reasons.append("high_loss")
        sample_weight = min(max_weight, sample_weight)
        weights[image_id] = HardSample(
            image_id=image_id,
            sample_weight=sample_weight,
            hard_reason="+".join(reasons) if reasons else "easy",
            confidence=confidence,
            margin=margin,
            loss=loss,
            true_label=str(record["true_label"]).strip(),
            pred_label=str(record["pred_label"]).strip(),
        )
    return weights


def _resolve_hard_sample(image_id: str, hard_samples: Mapping[str, HardSample]) -> HardSample | None:
    if image_id in hard_samples:
        return hard_samples[image_id]
    basename = Path(image_id).name
    basename_matches = [sample for key, sample in hard_samples.items() if Path(key).name == basename]
    if len(basename_matches) == 1:
        return basename_matches[0]
    return None


def _write_weighted_train_csv(output_csv: Path, rows: Sequence[dict[str, str]], fieldnames: Sequence[str]) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_hard_csv(output_csv: Path, rows: Sequence[dict[str, str]]) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "image",
        "label",
        "sample_weight",
        "hard_reason",
        "confidence",
        "margin",
        "loss",
        "prediction",
    ]
    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def apply_hard_mining_weights(
    train_csv: Path,
    oof_csv: Path,
    output_csv: Path,
    hard_csv: Path,
    error_boost: float = 1.0,
    low_margin_boost: float = 0.5,
    high_loss_boost: float = 0.5,
    low_margin_threshold: float = 0.1,
    high_loss_quantile: float = 0.75,
    max_weight: float = 2.5,
) -> dict[str, int]:
    train_rows = _read_csv(train_csv)
    if not train_rows:
        raise ValueError("Training CSV contains no rows")
    fieldnames = list(train_rows[0])
    if "image" not in fieldnames or "label" not in fieldnames:
        raise ValueError("Training CSV must contain image and label columns")
    if "sample_weight" not in fieldnames:
        fieldnames.append("sample_weight")
    if "source" not in fieldnames:
        fieldnames.append("source")

    oof_records = _read_csv(oof_csv)
    hard_samples = compute_hard_sample_weights(
        oof_records,
        error_boost=error_boost,
        low_margin_boost=low_margin_boost,
        high_loss_boost=high_loss_boost,
        low_margin_threshold=low_margin_threshold,
        high_loss_quantile=high_loss_quantile,
        max_weight=max_weight,
    )

    updated = 0
    skipped_pseudo = 0
    hard_rows: list[dict[str, str]] = []
    for row in train_rows:
        image_id = str(row["image"]).strip()
        source = str(row.get("source", "labeled")).strip() or "labeled"
        row["source"] = source
        if source == "pseudo":
            skipped_pseudo += 1
            row["sample_weight"] = f"{float(row.get('sample_weight') or 1.0):.6f}"
            continue
        hard_sample = _resolve_hard_sample(image_id, hard_samples)
        if hard_sample is None:
            row["sample_weight"] = f"{float(row.get('sample_weight') or 1.0):.6f}"
            continue
        row["sample_weight"] = f"{hard_sample.sample_weight:.6f}"
        if hard_sample.sample_weight > 1.0:
            updated += 1
            hard_rows.append(
                {
                    "image": image_id,
                    "label": str(row["label"]).strip(),
                    "sample_weight": f"{hard_sample.sample_weight:.6f}",
                    "hard_reason": hard_sample.hard_reason,
                    "confidence": f"{hard_sample.confidence:.6f}",
                    "margin": f"{hard_sample.margin:.6f}",
                    "loss": f"{hard_sample.loss:.6f}",
                    "prediction": hard_sample.pred_label,
                }
            )

    hard_rows.sort(key=lambda row: (-float(row["sample_weight"]), row["image"]))
    _write_weighted_train_csv(output_csv, train_rows, fieldnames)
    _write_hard_csv(hard_csv, hard_rows)
    return {
        "input": len(train_rows),
        "updated": updated,
        "hard": len(hard_rows),
        "skipped_pseudo": skipped_pseudo,
    }
