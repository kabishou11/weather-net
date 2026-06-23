from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

from .data import build_manifest_from_csv, build_manifest_from_image_folder


@dataclass(frozen=True)
class MergedTrainingRow:
    image: Path
    label: str
    source: str
    confidence: float


def _read_pseudo_rows(
    pseudo_csv: Path,
    pseudo_image_root: Path | None,
    min_confidence: float,
) -> list[MergedTrainingRow]:
    if not 0 < min_confidence <= 1:
        raise ValueError("min_confidence must be in (0, 1]")
    root = (pseudo_image_root or pseudo_csv.parent).expanduser().resolve()
    rows: list[MergedTrainingRow] = []
    with pseudo_csv.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"image", "label", "confidence"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Pseudo label CSV missing columns: {sorted(missing)}")
        for item in reader:
            confidence = float(item["confidence"])
            if confidence < min_confidence:
                continue
            rows.append(
                MergedTrainingRow(
                    image=(root / item["image"]).resolve(),
                    label=item["label"],
                    source="pseudo",
                    confidence=confidence,
                )
            )
    return rows


def _load_labeled_rows(
    train_dir: Path | None,
    train_csv: Path | None,
    image_root: Path | None,
) -> list[MergedTrainingRow]:
    if train_dir is not None:
        manifest, _ = build_manifest_from_image_folder(train_dir)
    elif train_csv is not None:
        manifest, _ = build_manifest_from_csv(train_csv, image_root=image_root)
    else:
        raise ValueError("Set train_dir or train_csv")
    rows: list[MergedTrainingRow] = []
    for row in manifest:
        if row.label_name is None:
            raise ValueError(f"Labeled row is missing label_name: {row.path}")
        rows.append(
            MergedTrainingRow(
                image=row.path.resolve(),
                label=row.label_name,
                source="labeled",
                confidence=1.0,
            )
        )
    return rows


def write_merged_training_csv(output_csv: Path, rows: list[MergedTrainingRow]) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["image", "label", "source", "confidence"])
        for row in rows:
            writer.writerow([str(row.image), row.label, row.source, f"{row.confidence:.6f}"])


def merge_training_with_pseudo_labels(
    train_dir: Path | None,
    train_csv: Path | None,
    image_root: Path | None,
    pseudo_csv: Path,
    pseudo_image_root: Path | None,
    output_csv: Path,
    min_confidence: float,
) -> dict[str, int]:
    labeled_rows = _load_labeled_rows(train_dir=train_dir, train_csv=train_csv, image_root=image_root)
    pseudo_rows = _read_pseudo_rows(
        pseudo_csv=pseudo_csv,
        pseudo_image_root=pseudo_image_root,
        min_confidence=min_confidence,
    )
    merged_rows = labeled_rows + pseudo_rows
    write_merged_training_csv(output_csv, merged_rows)
    return {
        "labeled": len(labeled_rows),
        "pseudo": len(pseudo_rows),
        "total": len(merged_rows),
    }
