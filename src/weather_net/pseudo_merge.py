from __future__ import annotations

import csv
import json
import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from .data import build_manifest_from_csv, build_manifest_from_image_folder, idx_to_class


@dataclass(frozen=True)
class MergedTrainingRow:
    image: Path
    label: str
    source: str
    confidence: float
    sample_weight: float
    teacher_probs: tuple[float, ...] | None = None


def _read_pseudo_rows(
    pseudo_csv: Path,
    pseudo_image_root: Path | None,
    min_confidence: float,
    class_names: list[str],
    allow_pseudo_teacher: bool,
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
        teacher_columns = {
            name[len("teacher_") :]: name
            for name in (reader.fieldnames or [])
            if name.lower().startswith("teacher_")
        }
        if teacher_columns and not allow_pseudo_teacher:
            raise ValueError("pseudo teacher columns require explicit allowance")
        if allow_pseudo_teacher:
            missing_teacher = [name for name in class_names if name not in teacher_columns]
            if missing_teacher:
                raise ValueError(f"Pseudo teacher columns missing classes: {missing_teacher}")
            extra_teacher = sorted(set(teacher_columns) - set(class_names))
            if extra_teacher:
                raise ValueError(f"Pseudo teacher columns contain unknown classes: {extra_teacher}")
        for item in reader:
            confidence = float(item["confidence"])
            if not math.isfinite(confidence) or not 0 <= confidence <= 1:
                raise ValueError(f"Pseudo label confidence must be finite and in [0, 1]: {item['confidence']}")
            if confidence < min_confidence:
                continue
            label = str(item["label"]).strip()
            teacher_probs: tuple[float, ...] | None = None
            if allow_pseudo_teacher:
                values = [float(item[teacher_columns[class_name]]) for class_name in class_names]
                if not all(math.isfinite(value) for value in values):
                    raise ValueError(f"Pseudo teacher probabilities must be finite: {item['image']}")
                if any(value < 0 for value in values):
                    raise ValueError(f"Pseudo teacher probabilities must be non-negative: {item['image']}")
                total = sum(values)
                if not math.isclose(total, 1.0, rel_tol=1e-4, abs_tol=1e-4):
                    raise ValueError(f"Pseudo teacher probabilities must sum to 1: {item['image']}")
                top1_idx = max(range(len(values)), key=lambda idx: values[idx])
                if class_names[top1_idx] != label:
                    raise ValueError("teacher top1 must agree with pseudo label")
                teacher_probs = tuple(float(value) for value in values)
            rows.append(
                MergedTrainingRow(
                    image=(root / item["image"]).resolve(),
                    label=label,
                    source="pseudo",
                    confidence=confidence,
                    sample_weight=0.3 + (0.7 * confidence),
                    teacher_probs=teacher_probs,
                )
            )
    return rows


def _dedupe_pseudo_rows(rows: list[MergedTrainingRow]) -> tuple[list[MergedTrainingRow], int]:
    by_path: dict[Path, MergedTrainingRow] = {}
    duplicate_count = 0
    for row in rows:
        key = row.image.resolve()
        existing = by_path.get(key)
        if existing is None:
            by_path[key] = row
            continue
        duplicate_count += 1
        if existing.label != row.label:
            raise ValueError(
                "Conflicting pseudo labels for duplicate image "
                f"{row.image}: {existing.label!r} vs {row.label!r}"
            )
        if row.confidence > existing.confidence:
            by_path[key] = row
    return list(by_path.values()), duplicate_count


def _load_labeled_rows(
    train_dir: Path | None,
    train_csv: Path | None,
    image_root: Path | None,
) -> tuple[list[MergedTrainingRow], list[str]]:
    if train_dir is not None:
        manifest, class_to_idx = build_manifest_from_image_folder(train_dir)
    elif train_csv is not None:
        manifest, class_to_idx = build_manifest_from_csv(train_csv, image_root=image_root)
    else:
        raise ValueError("Set train_dir or train_csv")
    class_names = idx_to_class(class_to_idx)
    rows: list[MergedTrainingRow] = []
    for row in manifest:
        if row.label_name is None:
            raise ValueError(f"Labeled row is missing label_name: {row.path}")
        teacher_probs = row.teacher_probs
        if teacher_probs is None and row.label is not None:
            teacher_probs = tuple(1.0 if idx == row.label else 0.0 for idx, _name in enumerate(class_names))
        rows.append(
            MergedTrainingRow(
                image=row.path.resolve(),
                label=row.label_name,
                source="labeled",
                confidence=1.0,
                sample_weight=row.sample_weight,
                teacher_probs=teacher_probs,
            )
        )
    return rows, class_names


def write_merged_training_csv(output_csv: Path, rows: list[MergedTrainingRow], class_names: list[str]) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    include_teacher = any(row.teacher_probs is not None for row in rows)
    fieldnames = ["image", "label", "source", "confidence", "sample_weight"]
    if include_teacher:
        fieldnames.extend(f"teacher_{class_name}" for class_name in class_names)
    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(fieldnames)
        for row in rows:
            values = [
                str(row.image),
                row.label,
                row.source,
                f"{row.confidence:.6f}",
                f"{row.sample_weight:.6f}",
            ]
            if include_teacher:
                if row.teacher_probs is None or len(row.teacher_probs) != len(class_names):
                    raise ValueError("all rows must have teacher probabilities when teacher columns are written")
                values.extend(f"{float(value):.8f}" for value in row.teacher_probs)
            writer.writerow(values)


def _pseudo_teacher_summary(rows: list[MergedTrainingRow], class_names: list[str]) -> dict[str, object]:
    teacher_rows = [row for row in rows if row.source == "pseudo" and row.teacher_probs is not None]
    if not teacher_rows:
        return {
            "rows": 0,
            "mean_top1_probability": None,
            "mean_margin": None,
        }
    top1_probabilities: list[float] = []
    margins: list[float] = []
    per_class_top1: Counter[str] = Counter()
    for row in teacher_rows:
        values = list(row.teacher_probs or ())
        ranked = sorted(values, reverse=True)
        top1_idx = max(range(len(values)), key=lambda idx: values[idx])
        top1_probabilities.append(float(ranked[0]))
        margins.append(float(ranked[0] - ranked[1]) if len(ranked) > 1 else float(ranked[0]))
        per_class_top1[class_names[top1_idx]] += 1
    return {
        "rows": len(teacher_rows),
        "mean_top1_probability": sum(top1_probabilities) / len(top1_probabilities),
        "mean_margin": sum(margins) / len(margins),
        "top1_per_class": dict(sorted(per_class_top1.items())),
    }


def write_merge_audit_json(
    audit_json: Path,
    rows: list[MergedTrainingRow],
    class_names: list[str],
    stats: dict[str, int],
    min_confidence: float,
    allow_pseudo_teacher: bool,
) -> None:
    pseudo_counts = Counter(row.label for row in rows if row.source == "pseudo")
    source_counts = Counter(row.source for row in rows)
    audit = {
        "stats": stats,
        "class_names": class_names,
        "source_counts": dict(sorted(source_counts.items())),
        "pseudo_per_class": dict(sorted(pseudo_counts.items())),
        "min_confidence": min_confidence,
        "allow_pseudo_teacher": allow_pseudo_teacher,
        "pseudo_teacher": _pseudo_teacher_summary(rows, class_names),
    }
    audit_json.parent.mkdir(parents=True, exist_ok=True)
    audit_json.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def merge_training_with_pseudo_labels(
    train_dir: Path | None,
    train_csv: Path | None,
    image_root: Path | None,
    pseudo_csv: Path,
    pseudo_image_root: Path | None,
    output_csv: Path,
    min_confidence: float,
    allow_pseudo_teacher: bool = False,
    audit_json: Path | None = None,
) -> dict[str, int]:
    labeled_rows, class_names = _load_labeled_rows(train_dir=train_dir, train_csv=train_csv, image_root=image_root)
    pseudo_rows = _read_pseudo_rows(
        pseudo_csv=pseudo_csv,
        pseudo_image_root=pseudo_image_root,
        min_confidence=min_confidence,
        class_names=class_names,
        allow_pseudo_teacher=allow_pseudo_teacher,
    )
    labeled_paths = {row.image.resolve() for row in labeled_rows}
    labeled_labels = {row.label for row in labeled_rows}
    unknown_labels = sorted({row.label for row in pseudo_rows} - labeled_labels)
    if unknown_labels:
        raise ValueError(f"Pseudo labels contain unknown labels: {unknown_labels}")
    deduped_internal_pseudo_rows, skipped_duplicate_pseudo = _dedupe_pseudo_rows(pseudo_rows)
    deduped_pseudo_rows = [
        row for row in deduped_internal_pseudo_rows if row.image.resolve() not in labeled_paths
    ]
    skipped_labeled_duplicates = len(deduped_internal_pseudo_rows) - len(deduped_pseudo_rows)
    skipped_duplicates = skipped_labeled_duplicates + skipped_duplicate_pseudo
    merged_rows = labeled_rows + deduped_pseudo_rows
    if not allow_pseudo_teacher:
        merged_rows = [
            MergedTrainingRow(
                image=row.image,
                label=row.label,
                source=row.source,
                confidence=row.confidence,
                sample_weight=row.sample_weight,
                teacher_probs=None,
            )
            for row in merged_rows
        ]
    write_merged_training_csv(output_csv, merged_rows, class_names=class_names)
    stats = {
        "labeled": len(labeled_rows),
        "pseudo": len(deduped_pseudo_rows),
        "total": len(merged_rows),
        "skipped_duplicates": skipped_duplicates,
        "skipped_labeled_duplicates": skipped_labeled_duplicates,
        "skipped_duplicate_pseudo": skipped_duplicate_pseudo,
    }
    if audit_json is not None:
        write_merge_audit_json(
            audit_json=audit_json,
            rows=merged_rows,
            class_names=class_names,
            stats=stats,
            min_confidence=min_confidence,
            allow_pseudo_teacher=allow_pseudo_teacher,
        )
    return stats
