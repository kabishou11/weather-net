from __future__ import annotations

import csv
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


@dataclass(frozen=True)
class PredictionRecord:
    image: Path
    label: str
    prediction: str
    confidence: float
    correct: bool


def build_prediction_records(
    image_paths: Sequence[Path],
    y_true: Sequence[int],
    probabilities: Sequence[Sequence[float]],
    class_names: Sequence[str],
) -> list[PredictionRecord]:
    if len(image_paths) != len(y_true) or len(y_true) != len(probabilities):
        raise ValueError("image_paths, y_true, and probabilities must have the same length")

    records: list[PredictionRecord] = []
    for image_path, true_idx, probs in zip(image_paths, y_true, probabilities):
        if len(probs) != len(class_names):
            raise ValueError("Each probability row must match class_names length")
        pred_idx = max(range(len(probs)), key=lambda idx: probs[idx])
        label = class_names[int(true_idx)]
        prediction = class_names[pred_idx]
        records.append(
            PredictionRecord(
                image=Path(image_path),
                label=label,
                prediction=prediction,
                confidence=float(probs[pred_idx]),
                correct=int(true_idx) == pred_idx,
            )
        )
    return records


def _hard_reason(record: PredictionRecord, low_confidence_threshold: float) -> str | None:
    if not record.correct:
        return "error"
    if record.confidence < low_confidence_threshold:
        return "low_confidence"
    return None


def write_error_analysis_csv(
    output_path: Path,
    records: Sequence[PredictionRecord],
    low_confidence_threshold: float = 0.6,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    hard_records = [
        (record, reason)
        for record in records
        if (reason := _hard_reason(record, low_confidence_threshold)) is not None
    ]
    hard_records.sort(key=lambda item: (item[1] != "error", item[0].confidence, item[0].image.name))

    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["image", "label", "prediction", "confidence", "correct", "hard_reason"])
        for record, reason in hard_records:
            writer.writerow(
                [
                    record.image.name,
                    record.label,
                    record.prediction,
                    f"{record.confidence:.6f}",
                    int(record.correct),
                    reason,
                ]
            )


def confusion_pairs(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    class_names: Sequence[str],
    top_k: int | None = None,
) -> list[dict[str, object]]:
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length")
    counter: Counter[tuple[int, int]] = Counter()
    for true_idx, pred_idx in zip(y_true, y_pred):
        true_idx = int(true_idx)
        pred_idx = int(pred_idx)
        if true_idx != pred_idx:
            counter[(true_idx, pred_idx)] += 1

    items = sorted(
        counter.items(),
        key=lambda item: (-item[1], class_names[item[0][0]], class_names[item[0][1]]),
    )
    if top_k is not None:
        items = items[:top_k]
    return [
        {
            "label": class_names[true_idx],
            "prediction": class_names[pred_idx],
            "count": count,
        }
        for (true_idx, pred_idx), count in items
    ]
