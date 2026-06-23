from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


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

    rows: list[PseudoLabelRow] = []
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
        if confidence >= threshold:
            rows.append(
                PseudoLabelRow(
                    image=Path(image_path),
                    label=class_names[best_idx],
                    confidence=confidence,
                    image_id=image_ids[index] if image_ids is not None else None,
                )
            )
    return rows


def write_pseudo_labels(output_path: Path, rows: Sequence[PseudoLabelRow]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["image", "label", "confidence"])
        for row in rows:
            writer.writerow([row.image_id or row.image.name, row.label, f"{row.confidence:.6f}"])
