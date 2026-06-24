from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.weather_net.data import ManifestRow, build_manifest_from_csv, build_manifest_from_image_folder, idx_to_class


RESERVED_SPLIT_TOKENS = {"test_dataset", "test_images", "alien_test"}


@dataclass(frozen=True)
class TrainingMergeRow:
    image: Path
    label: str
    source: str
    confidence: float
    sample_weight: float
    original_image: str
    content_sha256: str


@dataclass(frozen=True)
class RejectedExternalRow:
    image: str
    label: str
    source: str
    sample_weight: str
    reason: str
    kept_image: str = ""
    kept_source: str = ""
    content_sha256: str = ""


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _has_reserved_split_token(value: str | Path) -> bool:
    return bool({part.lower() for part in Path(value).parts} & RESERVED_SPLIT_TOKENS)


def _is_external_source(source: str) -> bool:
    return source == "external" or source.startswith("external_")


def _validate_finite_gate(name: str, value: float | None, minimum: float | None, maximum: float | None) -> None:
    if value is None:
        return
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    if minimum is not None and value < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    if maximum is not None and value > maximum:
        raise ValueError(f"{name} must be <= {maximum}")


def _format_image_path(path: Path, output_image_root: Path | None) -> str:
    path = path.expanduser().resolve()
    if output_image_root is None:
        return str(path)
    root = output_image_root.expanduser().resolve()
    try:
        return path.relative_to(root).as_posix()
    except ValueError as error:
        raise ValueError(f"image is not under --output-image-root: {path}") from error


def _load_labeled_rows(
    train_csv: Path | None,
    train_dir: Path | None,
    image_root: Path | None,
) -> tuple[list[ManifestRow], dict[str, int]]:
    if train_csv is not None and train_dir is not None:
        raise ValueError("merge requires either --train-csv or --train-dir, not both")
    if train_dir is not None:
        return build_manifest_from_image_folder(train_dir)
    if train_csv is not None:
        return build_manifest_from_csv(train_csv, image_root=image_root)
    raise ValueError("merge requires --train-csv or --train-dir")


def _load_external_rows(external_csv: Path, external_image_root: Path | None) -> list[ManifestRow]:
    rows, _class_to_idx = build_manifest_from_csv(external_csv, image_root=external_image_root)
    return rows


def _reject_from_external_row(
    row: ManifestRow,
    reason: str,
    content_hash: str = "",
    kept: TrainingMergeRow | None = None,
) -> RejectedExternalRow:
    return RejectedExternalRow(
        image=row.image_id or str(row.path),
        label=row.label_name or "",
        source=row.source,
        sample_weight=f"{row.sample_weight:.6f}",
        reason=reason,
        kept_image=kept.original_image if kept is not None else "",
        kept_source=kept.source if kept is not None else "",
        content_sha256=content_hash,
    )


def _validate_external_row(row: ManifestRow) -> None:
    image_id = row.image_id or str(row.path)
    if not _is_external_source(row.source):
        raise ValueError(f"external rows must use source=external or external_*: {image_id}")
    if not row.has_explicit_sample_weight:
        raise ValueError(f"external rows require explicit sample_weight: {image_id}")
    if _has_reserved_split_token(image_id) or _has_reserved_split_token(row.path):
        raise ValueError(f"external row references reserved test split path: {image_id}")
    if row.label_name is None:
        raise ValueError(f"external row missing label: {image_id}")
    if not row.path.exists():
        raise ValueError(f"external image does not exist: {image_id}")
    if not math.isfinite(row.sample_weight) or row.sample_weight <= 0:
        raise ValueError(f"external sample_weight must be finite and positive: {image_id}")


def _write_training_csv(
    output_csv: Path,
    rows: list[TrainingMergeRow],
    output_image_root: Path | None,
) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "image",
        "label",
        "source",
        "confidence",
        "sample_weight",
        "original_image",
        "content_sha256",
    ]
    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "image": _format_image_path(row.image, output_image_root),
                    "label": row.label,
                    "source": row.source,
                    "confidence": f"{row.confidence:.6f}",
                    "sample_weight": f"{row.sample_weight:.6f}",
                    "original_image": row.original_image,
                    "content_sha256": row.content_sha256,
                }
            )


def _row_digest(
    image: str,
    label: str,
    source: str,
    confidence: float,
    sample_weight: float,
    content_sha256: str,
) -> str:
    payload = {
        "image": image,
        "label": label,
        "source": source,
        "confidence": f"{confidence:.6f}",
        "sample_weight": f"{sample_weight:.6f}",
        "content_sha256": content_sha256,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def _training_row_digest(row: TrainingMergeRow, output_image_root: Path | None) -> str:
    return _row_digest(
        image=_format_image_path(row.image, output_image_root),
        label=row.label,
        source=row.source,
        confidence=row.confidence,
        sample_weight=row.sample_weight,
        content_sha256=row.content_sha256,
    )


def _write_rejected_csv(rejected_csv: Path, rows: list[RejectedExternalRow]) -> None:
    rejected_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "image",
        "label",
        "source",
        "sample_weight",
        "reason",
        "kept_image",
        "kept_source",
        "content_sha256",
    ]
    with rejected_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "image": row.image,
                    "label": row.label,
                    "source": row.source,
                    "sample_weight": row.sample_weight,
                    "reason": row.reason,
                    "kept_image": row.kept_image,
                    "kept_source": row.kept_source,
                    "content_sha256": row.content_sha256,
                }
            )


def merge_labeled_with_external_data(
    train_csv: Path | None,
    train_dir: Path | None,
    image_root: Path | None,
    external_csv: Path,
    external_image_root: Path | None,
    output_csv: Path,
    rejected_csv: Path,
    audit_json: Path,
    output_image_root: Path | None = None,
    max_external_per_labeled_class_ratio: float | None = None,
    max_external_sample_weight: float | None = None,
    max_external_effective_weight_share: float | None = None,
    clip_external_sample_weight: bool = False,
    drop_unmapped_external: bool = True,
) -> dict[str, Any]:
    labeled_manifest, class_to_idx = _load_labeled_rows(train_csv=train_csv, train_dir=train_dir, image_root=image_root)
    class_names = idx_to_class(class_to_idx)
    allowed_labels = set(class_names)
    labeled_counts = Counter(row.label_name for row in labeled_manifest)
    _validate_finite_gate(
        "max_external_per_labeled_class_ratio",
        max_external_per_labeled_class_ratio,
        minimum=0.0,
        maximum=None,
    )
    _validate_finite_gate("max_external_sample_weight", max_external_sample_weight, minimum=0.0, maximum=None)
    _validate_finite_gate(
        "max_external_effective_weight_share",
        max_external_effective_weight_share,
        minimum=0.0,
        maximum=1.0,
    )
    if max_external_per_labeled_class_ratio is not None:
        external_quota_by_label = {
            label: int(math.floor(count * max_external_per_labeled_class_ratio))
            for label, count in labeled_counts.items()
            if label is not None
        }
    else:
        external_quota_by_label = None

    kept_rows: list[TrainingMergeRow] = []
    rejected_rows: list[RejectedExternalRow] = []
    by_hash: dict[str, TrainingMergeRow] = {}

    for row in labeled_manifest:
        if row.label_name is None:
            raise ValueError(f"labeled row missing label: {row.image_id or row.path}")
        if row.source != "labeled":
            raise ValueError(
                "base training rows must be labeled; run external merge from the official labeled "
                f"training CSV/ImageFolder, not a pseudo/external merged CSV: {row.image_id or row.path}"
            )
        if not row.path.exists():
            raise ValueError(f"labeled image does not exist: {row.image_id or row.path}")
        content_hash = _hash_file(row.path)
        merged = TrainingMergeRow(
            image=row.path.resolve(),
            label=row.label_name,
            source="labeled",
            confidence=1.0,
            sample_weight=row.sample_weight,
            original_image=row.image_id or str(row.path),
            content_sha256=content_hash,
        )
        existing = by_hash.get(content_hash)
        if existing is not None:
            raise ValueError(
                "labeled training data contains duplicate image content: "
                f"{merged.original_image} duplicates {existing.original_image}"
            )
        by_hash[content_hash] = merged
        kept_rows.append(merged)

    kept_external_counts: Counter[str] = Counter()
    clipped_external_sample_weight = 0
    external_manifest = _load_external_rows(external_csv=external_csv, external_image_root=external_image_root)

    for row in external_manifest:
        _validate_external_row(row)
        label = row.label_name or ""
        if label not in allowed_labels:
            if drop_unmapped_external:
                rejected_rows.append(_reject_from_external_row(row, "label_not_in_labeled_classes"))
                continue
            raise ValueError(f"external label is not in labeled classes: {label}")

        sample_weight = row.sample_weight
        if max_external_sample_weight is not None:
            if max_external_sample_weight <= 0:
                raise ValueError("max_external_sample_weight must be positive")
            if sample_weight > max_external_sample_weight:
                if not clip_external_sample_weight:
                    raise ValueError(
                        "external sample_weight exceeds max_external_sample_weight: "
                        f"{row.image_id or row.path}"
                    )
                sample_weight = max_external_sample_weight
                clipped_external_sample_weight += 1

        content_hash = _hash_file(row.path)
        existing = by_hash.get(content_hash)
        if existing is not None:
            rejected_rows.append(
                _reject_from_external_row(row, "duplicate_hash", content_hash=content_hash, kept=existing)
            )
            continue

        if external_quota_by_label is not None:
            quota = external_quota_by_label.get(label, 0)
            if kept_external_counts[label] >= quota:
                rejected_rows.append(_reject_from_external_row(row, "class_quota_exceeded"))
                continue

        merged = TrainingMergeRow(
            image=row.path.resolve(),
            label=label,
            source=row.source,
            confidence=row.confidence,
            sample_weight=sample_weight,
            original_image=row.image_id or str(row.path),
            content_sha256=content_hash,
        )
        by_hash[content_hash] = merged
        kept_rows.append(merged)
        kept_external_counts[label] += 1

    labeled_weight = sum(row.sample_weight for row in kept_rows if row.source == "labeled")
    external_weight = sum(row.sample_weight for row in kept_rows if row.source.startswith("external"))
    total_effective_weight = labeled_weight + external_weight
    external_effective_weight_share = (
        external_weight / total_effective_weight if total_effective_weight > 0 else 0.0
    )
    if max_external_effective_weight_share is not None:
        if external_effective_weight_share > max_external_effective_weight_share:
            raise ValueError(
                "external effective weight share "
                f"{external_effective_weight_share:.6f} exceeds limit "
                f"{max_external_effective_weight_share:.6f}"
            )

    _write_training_csv(output_csv=output_csv, rows=kept_rows, output_image_root=output_image_root)
    _write_rejected_csv(rejected_csv=rejected_csv, rows=rejected_rows)

    source_counts = Counter(row.source for row in kept_rows)
    rejected_by_reason = Counter(row.reason for row in rejected_rows)
    row_digests = [_training_row_digest(row, output_image_root) for row in kept_rows]
    summary: dict[str, Any] = {
        "labeled_rows": len(labeled_manifest),
        "input_external_rows": len(external_manifest),
        "kept_external_rows": sum(kept_external_counts.values()),
        "rejected_external_rows": len(rejected_rows),
        "clipped_external_sample_weight": clipped_external_sample_weight,
        "total_rows": len(kept_rows),
        "classes": class_names,
        "labeled_by_label": dict(sorted((str(label), count) for label, count in labeled_counts.items())),
        "kept_external_by_label": dict(sorted(kept_external_counts.items())),
        "source_counts": dict(sorted(source_counts.items())),
        "rejected_by_reason": dict(sorted(rejected_by_reason.items())),
        "row_digest_sha256": row_digests,
        "effective_weight": {
            "labeled": labeled_weight,
            "external": external_weight,
            "external_share": external_effective_weight_share,
        },
        "output_csv": str(output_csv.expanduser().resolve()),
        "rejected_csv": str(rejected_csv.expanduser().resolve()),
        "output_image_root": str(output_image_root.expanduser().resolve()) if output_image_root is not None else None,
        "max_external_per_labeled_class_ratio": max_external_per_labeled_class_ratio,
        "max_external_sample_weight": max_external_sample_weight,
        "max_external_effective_weight_share": max_external_effective_weight_share,
        "clip_external_sample_weight": clip_external_sample_weight,
        "drop_unmapped_external": drop_unmapped_external,
    }
    audit_json.parent.mkdir(parents=True, exist_ok=True)
    audit_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge official labeled training data with deduplicated, low-weight external weather data."
    )
    parser.add_argument("--train-csv", type=Path, default=None)
    parser.add_argument("--train-dir", type=Path, default=None)
    parser.add_argument("--image-root", type=Path, default=None)
    parser.add_argument("--external-csv", type=Path, required=True)
    parser.add_argument("--external-image-root", type=Path, default=None)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--rejected-csv", type=Path, required=True)
    parser.add_argument("--audit-json", type=Path, required=True)
    parser.add_argument("--output-image-root", type=Path, default=None)
    parser.add_argument("--max-external-per-labeled-class-ratio", type=float, default=None)
    parser.add_argument("--max-external-sample-weight", type=float, default=None)
    parser.add_argument("--max-external-effective-weight-share", type=float, default=None)
    parser.add_argument("--clip-external-sample-weight", action="store_true")
    parser.add_argument("--keep-unmapped-external", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = merge_labeled_with_external_data(
        train_csv=args.train_csv,
        train_dir=args.train_dir,
        image_root=args.image_root,
        external_csv=args.external_csv,
        external_image_root=args.external_image_root,
        output_csv=args.output_csv,
        rejected_csv=args.rejected_csv,
        audit_json=args.audit_json,
        output_image_root=args.output_image_root,
        max_external_per_labeled_class_ratio=args.max_external_per_labeled_class_ratio,
        max_external_sample_weight=args.max_external_sample_weight,
        max_external_effective_weight_share=args.max_external_effective_weight_share,
        clip_external_sample_weight=args.clip_external_sample_weight,
        drop_unmapped_external=not args.keep_unmapped_external,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
