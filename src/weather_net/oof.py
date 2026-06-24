from __future__ import annotations

import csv
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from .metrics import classification_report


@dataclass(frozen=True)
class OofRecord:
    image_id: str
    image_path: str
    fold: int
    source: str
    true_idx: int
    true_label: str
    logits: Sequence[float]
    checkpoint: str
    model_name: str


@dataclass(frozen=True)
class OofArtifactPaths:
    csv_path: Path
    npz_path: Path
    metrics_path: Path


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


def _sanitize_class_name(name: str) -> str:
    safe = "".join(char if char.isalnum() else "_" for char in name.strip())
    return safe or "class"


def _class_column_name(class_idx: int, class_name: str) -> str:
    return f"{class_idx}_{_sanitize_class_name(class_name)}"


def _validate_class_mapping(class_names: Sequence[str], class_to_idx: dict[str, int]) -> None:
    expected = [name for name, _ in sorted(class_to_idx.items(), key=lambda item: item[1])]
    if expected != list(class_names):
        raise ValueError("class_to_idx order must match class_names")


def _validate_records(records: Sequence[OofRecord], class_names: Sequence[str]) -> None:
    if not records:
        raise ValueError("OOF records cannot be empty")
    if len(set(class_names)) != len(class_names):
        raise ValueError("class_names must be unique")
    counts = Counter(record.image_id for record in records)
    duplicates = sorted(image_id for image_id, count in counts.items() if count > 1)
    if duplicates:
        preview = ", ".join(duplicates[:5])
        raise ValueError(f"Duplicate OOF image ids are not safe: {preview}")
    expected_classes = len(class_names)
    for record in records:
        if record.source == "pseudo":
            raise ValueError(f"Pseudo rows must not be written as OOF validation records: {record.image_id}")
        if len(record.logits) != expected_classes:
            raise ValueError("Each OOF record must have one logit per class")
        if record.true_idx < 0 or record.true_idx >= expected_classes:
            raise ValueError("OOF true_idx must be within the class range")
        if record.true_label != class_names[record.true_idx]:
            raise ValueError("OOF true_label must match class_names[true_idx]")


def write_oof_artifacts(
    output_dir: Path,
    records: Sequence[OofRecord],
    class_names: Sequence[str],
    class_to_idx: dict[str, int],
    metadata: dict[str, object],
) -> OofArtifactPaths:
    _validate_records(records, class_names)
    _validate_class_mapping(class_names, class_to_idx)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "oof_predictions.csv"
    npz_path = output_dir / "oof_probabilities.npz"
    metrics_path = output_dir / "oof_metrics.json"

    logits = np.asarray([record.logits for record in records], dtype=np.float32)
    probs = _softmax(logits).astype(np.float32)
    y_true = np.asarray([record.true_idx for record in records], dtype=np.int64)
    folds = np.asarray([record.fold for record in records], dtype=np.int64)
    pred = probs.argmax(axis=1).astype(np.int64)
    confidence = probs.max(axis=1)
    sorted_probs = np.sort(probs, axis=1)
    margins = confidence - (sorted_probs[:, -2] if probs.shape[1] > 1 else 0.0)
    entropy = -(probs * np.log(np.clip(probs, 1e-12, 1.0))).sum(axis=1)
    true_probs = probs[np.arange(len(records)), y_true]
    losses = -np.log(np.clip(true_probs, 1e-12, 1.0))

    class_columns = [_class_column_name(idx, name) for idx, name in enumerate(class_names)]
    fieldnames = [
        "image_id",
        "image_path",
        "fold",
        "source",
        "true_idx",
        "true_label",
        "pred_idx",
        "pred_label",
        "correct",
        "confidence",
        "margin",
        "entropy",
        "loss",
        "checkpoint",
        "model_name",
    ]
    fieldnames.extend(f"logit_{name}" for name in class_columns)
    fieldnames.extend(f"prob_{name}" for name in class_columns)

    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for idx, record in enumerate(records):
            row = {
                "image_id": record.image_id,
                "image_path": record.image_path,
                "fold": record.fold,
                "source": record.source,
                "true_idx": record.true_idx,
                "true_label": record.true_label,
                "pred_idx": int(pred[idx]),
                "pred_label": class_names[int(pred[idx])],
                "correct": bool(pred[idx] == record.true_idx),
                "confidence": f"{float(confidence[idx]):.8f}",
                "margin": f"{float(margins[idx]):.8f}",
                "entropy": f"{float(entropy[idx]):.8f}",
                "loss": f"{float(losses[idx]):.8f}",
                "checkpoint": record.checkpoint,
                "model_name": record.model_name,
            }
            for class_idx, column_name in enumerate(class_columns):
                row[f"logit_{column_name}"] = f"{float(logits[idx, class_idx]):.8f}"
                row[f"prob_{column_name}"] = f"{float(probs[idx, class_idx]):.8f}"
            writer.writerow(row)

    np.savez_compressed(
        npz_path,
        logits=logits,
        probs=probs,
        y_true=y_true,
        pred=pred,
        fold=folds,
        image_id=np.asarray([record.image_id for record in records], dtype=object),
        image_path=np.asarray([record.image_path for record in records], dtype=object),
        source=np.asarray([record.source for record in records], dtype=object),
        checkpoint=np.asarray([record.checkpoint for record in records], dtype=object),
        model_name=np.asarray([record.model_name for record in records], dtype=object),
        class_names=np.asarray(list(class_names), dtype=object),
    )

    report = classification_report(y_true.tolist(), pred.tolist(), class_names)
    metrics = {
        "macro_f1": report.macro_f1,
        "accuracy": report.accuracy,
        "per_class_f1": report.per_class_f1,
        "confusion_matrix": report.confusion_matrix,
        "images": len(records),
        "class_to_idx": class_to_idx,
        "metadata": metadata,
    }
    metrics_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return OofArtifactPaths(csv_path=csv_path, npz_path=npz_path, metrics_path=metrics_path)
