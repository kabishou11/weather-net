from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any


RESERVED_SPLIT_TOKENS = {"test_dataset", "test_images", "alien_test"}


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _has_reserved_split(path: str) -> bool:
    return bool({part.lower() for part in Path(path).parts} & RESERVED_SPLIT_TOKENS)


def _read_manifest(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header: {path}")
        required = {"image", "label", "source", "sample_weight"}
        missing = required - set(reader.fieldnames)
        if missing:
            raise ValueError(f"external manifest missing columns {sorted(missing)}: {path}")
        return list(reader.fieldnames), [dict(row) for row in reader]


def _validate_row(row: dict[str, str], manifest_path: Path, image_root: Path) -> tuple[Path, float]:
    image = row["image"].strip()
    if not image:
        raise ValueError(f"empty image field in {manifest_path}")
    if _has_reserved_split(image):
        raise ValueError(f"external manifest references reserved test split: {manifest_path}:{image}")
    if not row["source"].strip().startswith("external"):
        raise ValueError(f"external manifest source must start with external: {manifest_path}:{image}")
    sample_weight = float(row["sample_weight"])
    if not math.isfinite(sample_weight) or sample_weight <= 0:
        raise ValueError(f"sample_weight must be finite and positive: {manifest_path}:{image}")
    image_path = (image_root / image).resolve()
    if not image_path.exists():
        raise ValueError(f"manifest image does not exist: {manifest_path}:{image}")
    return image_path, sample_weight


def dedupe_external_manifests(
    manifest_roots: list[tuple[Path, Path]],
    output_csv: Path,
    rejected_csv: Path,
    audit_json: Path,
    output_image_root: Path | None = None,
) -> dict[str, Any]:
    if not manifest_roots:
        raise ValueError("at least one manifest/root pair is required")

    base_fieldnames: list[str] | None = None
    kept_rows: list[dict[str, str]] = []
    rejected_rows: list[dict[str, str]] = []
    by_hash: dict[str, dict[str, str]] = {}
    input_counts: Counter[str] = Counter()
    kept_counts: Counter[str] = Counter()
    rejected_counts: Counter[str] = Counter()

    output_root = output_image_root.expanduser().resolve() if output_image_root is not None else None

    for priority, (manifest_path, image_root) in enumerate(manifest_roots):
        manifest_path = manifest_path.expanduser().resolve()
        image_root = image_root.expanduser().resolve()
        fieldnames, rows = _read_manifest(manifest_path)
        if base_fieldnames is None:
            base_fieldnames = list(fieldnames)
        else:
            extra = [field for field in fieldnames if field not in base_fieldnames]
            base_fieldnames.extend(extra)
        for row in rows:
            image_path, _sample_weight = _validate_row(row, manifest_path, image_root)
            source = row["source"].strip()
            input_counts[source] += 1
            content_hash = _hash_file(image_path)
            enriched = dict(row)
            original_image = enriched["image"]
            if output_root is not None:
                try:
                    enriched["image"] = image_path.relative_to(output_root).as_posix()
                except ValueError as error:
                    raise ValueError(f"image is not under output_image_root: {image_path}") from error
            enriched["original_image"] = original_image
            enriched["manifest_path"] = str(manifest_path)
            enriched["image_root"] = str(image_root)
            enriched["content_sha256"] = content_hash
            enriched["priority"] = str(priority)
            existing = by_hash.get(content_hash)
            if existing is None:
                by_hash[content_hash] = enriched
                kept_rows.append(enriched)
                kept_counts[source] += 1
                continue
            rejected = dict(enriched)
            rejected["reason"] = "duplicate_hash"
            rejected["kept_image"] = existing["image"]
            rejected["kept_source"] = existing["source"]
            rejected["kept_manifest_path"] = existing["manifest_path"]
            rejected_rows.append(rejected)
            rejected_counts[source] += 1

    if base_fieldnames is None:
        raise ValueError("no manifests were read")
    if not kept_rows:
        raise ValueError("no rows kept after dedupe")

    output_fields = list(
        dict.fromkeys(base_fieldnames + ["original_image", "manifest_path", "image_root", "content_sha256", "priority"])
    )
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=output_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(kept_rows)

    rejected_fields = output_fields + ["reason", "kept_image", "kept_source", "kept_manifest_path"]
    rejected_csv.parent.mkdir(parents=True, exist_ok=True)
    with rejected_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rejected_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rejected_rows)

    summary: dict[str, Any] = {
        "input_rows": sum(input_counts.values()),
        "kept_rows": len(kept_rows),
        "rejected_duplicate_hash": len(rejected_rows),
        "input_by_source": dict(sorted(input_counts.items())),
        "kept_by_source": dict(sorted(kept_counts.items())),
        "rejected_by_source": dict(sorted(rejected_counts.items())),
        "output_csv": str(output_csv),
        "rejected_csv": str(rejected_csv),
        "output_image_root": str(output_root) if output_root is not None else None,
        "manifest_priority": [
            {"manifest": str(manifest.expanduser().resolve()), "image_root": str(root.expanduser().resolve())}
            for manifest, root in manifest_roots
        ],
    }
    audit_json.parent.mkdir(parents=True, exist_ok=True)
    audit_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge external manifests and remove duplicate image content.")
    parser.add_argument("--manifest", type=Path, action="append", required=True)
    parser.add_argument("--image-root", type=Path, action="append", required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--rejected-csv", type=Path, required=True)
    parser.add_argument("--audit-json", type=Path, required=True)
    parser.add_argument("--output-image-root", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if len(args.manifest) != len(args.image_root):
        raise ValueError("--manifest and --image-root must be provided the same number of times")
    summary = dedupe_external_manifests(
        manifest_roots=list(zip(args.manifest, args.image_root)),
        output_csv=args.output_csv,
        rejected_csv=args.rejected_csv,
        audit_json=args.audit_json,
        output_image_root=args.output_image_root,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
