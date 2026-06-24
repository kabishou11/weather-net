from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence


@dataclass(frozen=True)
class PseudoLabelRow:
    image: Path
    label: str
    confidence: float
    image_id: str | None = None


def select_pseudo_labels(
    image_paths: Sequence[Path],
    probabilities: Sequence[Sequence[float]],
    class_names: Sequence[str],
    threshold: float,
    min_margin: float = 0.0,
    require_tta_agreement: bool = False,
    agreement_probabilities: Sequence[Sequence[float]] | None = None,
    image_ids: Sequence[str] | None = None,
    per_class_thresholds: Mapping[str, float] | None = None,
    per_class_max_count: Mapping[str, int] | None = None,
) -> list[PseudoLabelRow]:
    if not 0 < threshold <= 1:
        raise ValueError("threshold must be in (0, 1]")
    if min_margin < 0:
        raise ValueError("min_margin must be non-negative")
    if len(image_paths) != len(probabilities):
        raise ValueError("image_paths and probabilities must have the same length")
    if image_ids is not None and len(image_ids) != len(image_paths):
        raise ValueError("image_ids and image_paths must have the same length")
    if require_tta_agreement:
        if agreement_probabilities is None:
            raise ValueError("agreement_probabilities is required when require_tta_agreement=True")
        if len(agreement_probabilities) != len(probabilities):
            raise ValueError("agreement_probabilities and probabilities must have the same length")
    class_set = set(class_names)
    if per_class_thresholds is not None:
        unknown = sorted(set(per_class_thresholds) - class_set)
        if unknown:
            raise ValueError(f"per-class thresholds contain unknown classes: {unknown}")
        for label, value in per_class_thresholds.items():
            if not 0 < float(value) <= 1:
                raise ValueError(f"per-class threshold must be in (0, 1]: {label}")
    if per_class_max_count is not None:
        unknown = sorted(set(per_class_max_count) - class_set)
        if unknown:
            raise ValueError(f"per-class max count contains unknown classes: {unknown}")
        for label, value in per_class_max_count.items():
            if int(value) < 0:
                raise ValueError(f"per-class max count must be non-negative: {label}")

    candidates: list[tuple[int, PseudoLabelRow]] = []
    for index, (image_path, probs) in enumerate(zip(image_paths, probabilities)):
        if len(probs) != len(class_names):
            raise ValueError("Each probability row must match class_names length")
        ranked = sorted(range(len(probs)), key=lambda idx: probs[idx], reverse=True)
        best_idx = ranked[0]
        second_confidence = float(probs[ranked[1]]) if len(ranked) > 1 else 0.0
        confidence = float(probs[best_idx])
        margin = confidence - second_confidence
        if margin < min_margin:
            continue
        if require_tta_agreement and agreement_probabilities is not None:
            agreement_probs = agreement_probabilities[index]
            if len(agreement_probs) != len(class_names):
                raise ValueError("Each agreement probability row must match class_names length")
            agreement_best_idx = max(range(len(agreement_probs)), key=lambda idx: agreement_probs[idx])
            if agreement_best_idx != best_idx:
                continue
        label = class_names[best_idx]
        class_threshold = (
            float(per_class_thresholds[label])
            if per_class_thresholds is not None and label in per_class_thresholds
            else threshold
        )
        if confidence >= class_threshold:
            candidates.append(
                (
                    index,
                    PseudoLabelRow(
                        image=Path(image_path),
                        label=label,
                        confidence=confidence,
                        image_id=image_ids[index] if image_ids is not None else None,
                    ),
                )
            )
    if per_class_max_count is None:
        return [row for _index, row in candidates]

    allowed_indices: set[int] = set()
    for label in class_names:
        class_candidates = [(index, row) for index, row in candidates if row.label == label]
        limit = int(per_class_max_count[label]) if label in per_class_max_count else len(class_candidates)
        if limit == 0:
            continue
        ranked = sorted(class_candidates, key=lambda item: (-item[1].confidence, item[0]))
        allowed_indices.update(index for index, _row in ranked[:limit])
    return [
        row
        for index, row in candidates
        if index in allowed_indices
    ]


def write_pseudo_labels(output_path: Path, rows: Sequence[PseudoLabelRow]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["image", "label", "confidence"])
        for row in rows:
            writer.writerow([row.image_id or row.image.name, row.label, f"{row.confidence:.6f}"])


def count_pseudo_labels_by_class(
    rows: Sequence[PseudoLabelRow],
    class_names: Sequence[str],
) -> dict[str, int]:
    counts = {class_name: 0 for class_name in class_names}
    for row in rows:
        if row.label not in counts:
            raise ValueError(f"Pseudo label row contains unknown class: {row.label}")
        counts[row.label] += 1
    return counts


def estimate_per_class_thresholds(
    y_true: Sequence[int],
    probabilities: Sequence[Sequence[float]],
    class_names: Sequence[str],
    target_precision: float = 0.95,
    min_threshold: float = 0.5,
    fallback_threshold: float = 0.99,
) -> dict[str, float]:
    if not 0 < target_precision <= 1:
        raise ValueError("target_precision must be in (0, 1]")
    if not 0 < min_threshold <= 1:
        raise ValueError("min_threshold must be in (0, 1]")
    if not 0 < fallback_threshold <= 1:
        raise ValueError("fallback_threshold must be in (0, 1]")
    if len(y_true) != len(probabilities):
        raise ValueError("y_true and probabilities must have the same length")
    if not class_names:
        raise ValueError("class_names must be non-empty")

    per_class: dict[str, list[tuple[float, bool]]] = {class_name: [] for class_name in class_names}
    for true_label, probs in zip(y_true, probabilities):
        if len(probs) != len(class_names):
            raise ValueError("Each probability row must match class_names length")
        if int(true_label) < 0 or int(true_label) >= len(class_names):
            raise ValueError("Each y_true label must be a valid class index")
        pred_idx = max(range(len(probs)), key=lambda idx: probs[idx])
        label = class_names[pred_idx]
        per_class[label].append((float(probs[pred_idx]), int(true_label) == pred_idx))

    thresholds: dict[str, float] = {}
    for label, candidates in per_class.items():
        ranked = sorted(candidates, key=lambda item: item[0], reverse=True)
        correct = 0
        selected = 0
        best_threshold: float | None = None
        position = 0
        while position < len(ranked):
            confidence = ranked[position][0]
            if confidence < min_threshold:
                break
            same_confidence_correct = 0
            same_confidence_count = 0
            while position < len(ranked) and ranked[position][0] == confidence:
                same_confidence_correct += 1 if ranked[position][1] else 0
                same_confidence_count += 1
                position += 1
            selected += same_confidence_count
            correct += same_confidence_correct
            precision = correct / selected
            if precision >= target_precision:
                best_threshold = confidence
        thresholds[label] = float(best_threshold if best_threshold is not None else fallback_threshold)
    return thresholds
