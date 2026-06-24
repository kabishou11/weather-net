from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

from PIL import Image

from inference_budget import check_inference_budget
from src.weather_net.config import AppConfig, load_config
from src.weather_net.data import ManifestRow, build_manifest_from_csv, build_manifest_from_image_folder, idx_to_class, load_class_mapping
from src.weather_net.training import validate_external_merge_audit


def _load_rows(
    config: AppConfig,
    train_csv: Path | None,
    train_dir: Path | None,
    image_root: Path | None,
    class_map: Path | None,
    require_class_map_for_csv: bool = False,
) -> tuple[list[ManifestRow], dict[str, int]]:
    if train_csv is not None:
        config.data.train_csv = train_csv
    if train_dir is not None:
        config.data.train_dir = train_dir
    if image_root is not None:
        config.data.image_root = image_root
    if class_map is not None:
        config.data.class_map = class_map

    class_to_idx = load_class_mapping(config.data.class_map) if config.data.class_map is not None else None

    if config.data.train_csv is not None and config.data.train_dir is not None:
        raise ValueError("preflight requires either --train-csv or --train-dir, not both")
    if config.data.train_dir is not None:
        return build_manifest_from_image_folder(config.data.train_dir, class_to_idx=class_to_idx)
    if config.data.train_csv is not None:
        if require_class_map_for_csv and class_to_idx is None:
            raise ValueError(
                "server-strict preflight requires data.class_map or --class-map for CSV training"
            )
        return build_manifest_from_csv(
            config.data.train_csv,
            image_root=config.data.image_root,
            class_to_idx=class_to_idx,
            image_column=config.data.image_column,
            label_column=config.data.label_column,
        )
    raise ValueError("preflight requires --train-csv or --train-dir")


def _count_by_label(rows: list[ManifestRow]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for row in rows:
        counts[str(row.label_name)] += 1
    return dict(sorted(counts.items()))


def _count_by_label_for_source(rows: list[ManifestRow], source: str) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for row in rows:
        if row.source == source:
            counts[str(row.label_name)] += 1
    return dict(sorted(counts.items()))


def _validate_teacher_distillation(
    rows: list[ManifestRow],
    distillation_alpha: float,
    class_names: list[str],
    allow_pseudo_teacher_distillation: bool,
    pseudo_min_confidence: float | None,
    pseudo_max_ratio: float | None,
) -> dict[str, object]:
    has_teacher = [row.teacher_probs is not None for row in rows]
    if distillation_alpha > 0 and not all(has_teacher):
        missing = sum(1 for value in has_teacher if not value)
        raise ValueError(f"teacher distillation is enabled but {missing} rows are missing teacher probabilities")
    if any(has_teacher) and not all(has_teacher):
        raise ValueError("teacher probability columns must be present for every row or no rows")
    bad_teacher_width = [
        row.image_id or str(row.path)
        for row in rows
        if row.teacher_probs is not None and len(row.teacher_probs) != len(class_names)
    ]
    if bad_teacher_width:
        raise ValueError(f"teacher probability width differs from class count for rows: {bad_teacher_width[:5]}")
    if distillation_alpha > 0:
        if allow_pseudo_teacher_distillation and (pseudo_min_confidence is None or pseudo_max_ratio is None):
            raise ValueError(
                "pseudo teacher distillation requires --pseudo-min-confidence and --pseudo-max-ratio"
            )
        allowed_sources = {"labeled", "pseudo"} if allow_pseudo_teacher_distillation else {"labeled"}
        non_labeled_sources = sorted({row.source for row in rows if row.source not in allowed_sources})
        if non_labeled_sources:
            raise ValueError(
                "teacher distillation preflight found disallowed sources; "
                f"found sources: {non_labeled_sources}"
            )
        if allow_pseudo_teacher_distillation:
            mismatched = []
            for row in rows:
                if row.source != "pseudo" or row.teacher_probs is None:
                    continue
                top1_idx = max(range(len(row.teacher_probs)), key=lambda idx: row.teacher_probs[idx])
                if row.label is not None and top1_idx != row.label:
                    mismatched.append(row.image_id or str(row.path))
            if mismatched:
                raise ValueError(f"pseudo teacher top1 differs from pseudo label for rows: {mismatched[:5]}")
    return {
        "enabled": distillation_alpha > 0,
        "rows_with_teacher": sum(1 for value in has_teacher if value),
    }


def _is_external_source(source: str) -> bool:
    return source == "external" or source.startswith("external_")


def _has_reserved_test_split_token(row: ManifestRow) -> bool:
    tokens: set[str] = set()
    if row.image_id:
        tokens.update(part.lower() for part in Path(row.image_id).parts)
    tokens.update(part.lower() for part in row.path.parts)
    return bool(tokens & {"test_dataset", "test_images", "alien_test"})


def _validate_sources(
    rows: list[ManifestRow],
    allow_external_data: bool,
    external_max_ratio: float | None,
    external_max_sample_weight: float | None,
    pseudo_min_confidence: float | None,
    pseudo_max_ratio: float | None,
) -> dict[str, int]:
    source_counts = Counter(row.source for row in rows)
    external_rows = sum(count for source, count in source_counts.items() if _is_external_source(source))
    if external_rows and not allow_external_data:
        raise ValueError("external data is present but --allow-external-data was not set")
    if external_rows:
        reserved_test_rows = [
            row.image_id or str(row.path)
            for row in rows
            if _is_external_source(row.source) and _has_reserved_test_split_token(row)
        ]
        if reserved_test_rows:
            preview = reserved_test_rows[:5]
            raise ValueError(f"external rows reference reserved test split paths: {preview}")
        implicit_weight = [
            row.image_id or str(row.path)
            for row in rows
            if _is_external_source(row.source) and not row.has_explicit_sample_weight
        ]
        if implicit_weight:
            preview = implicit_weight[:5]
            raise ValueError(f"external rows require explicit sample_weight values: {preview}")
    if external_rows and external_max_ratio is not None:
        if external_max_ratio < 0 or external_max_ratio > 1:
            raise ValueError("external_max_ratio must be in [0, 1]")
        ratio = external_rows / max(1, len(rows))
        if ratio > external_max_ratio:
            raise ValueError(f"external ratio {ratio:.6f} exceeds limit {external_max_ratio:.6f}")
    if external_rows and external_max_sample_weight is not None:
        if external_max_sample_weight <= 0:
            raise ValueError("external_max_sample_weight must be positive")
        overweight = [
            row.image_id or str(row.path)
            for row in rows
            if _is_external_source(row.source) and row.sample_weight > external_max_sample_weight
        ]
        if overweight:
            preview = overweight[:5]
            raise ValueError(f"external sample_weight exceeds limit for rows: {preview}")
    pseudo_rows = source_counts.get("pseudo", 0)
    invalid_pseudo_confidence = [
        row.image_id or str(row.path)
        for row in rows
        if row.source == "pseudo" and (not math.isfinite(row.confidence) or row.confidence < 0 or row.confidence > 1)
    ]
    if invalid_pseudo_confidence:
        raise ValueError(f"pseudo confidence must be finite and in [0, 1] for rows: {invalid_pseudo_confidence[:5]}")
    if pseudo_rows and external_rows:
        raise ValueError("pseudo labels and external data should not be mixed in the same first-pass training CSV")
    if pseudo_rows and pseudo_min_confidence is not None:
        if pseudo_min_confidence < 0 or pseudo_min_confidence > 1:
            raise ValueError("pseudo_min_confidence must be in [0, 1]")
        low_confidence = [
            row.image_id or str(row.path)
            for row in rows
            if row.source == "pseudo" and row.confidence < pseudo_min_confidence
        ]
        if low_confidence:
            raise ValueError(f"pseudo confidence below limit for rows: {low_confidence[:5]}")
    if pseudo_rows and pseudo_max_ratio is not None:
        if pseudo_max_ratio < 0 or pseudo_max_ratio > 1:
            raise ValueError("pseudo_max_ratio must be in [0, 1]")
        labeled_rows = sum(1 for row in rows if row.source != "pseudo")
        ratio = pseudo_rows / max(1, labeled_rows)
        if ratio > pseudo_max_ratio:
            raise ValueError(f"pseudo ratio {ratio:.6f} exceeds limit {pseudo_max_ratio:.6f}")
    return dict(sorted(source_counts.items()))


def _validate_class_support(
    rows: list[ManifestRow],
    class_to_idx: dict[str, int],
    min_images_per_class: int,
    min_labeled_images_per_class: int | None = None,
    min_labeled_images_reason: str | None = None,
    expected_classes: list[str] | None = None,
    require_all_expected_classes: bool = False,
) -> dict[str, object]:
    if min_images_per_class <= 0:
        raise ValueError("min_images_per_class must be positive")
    if min_labeled_images_per_class is not None and min_labeled_images_per_class <= 0:
        raise ValueError("min_labeled_images_per_class must be positive")
    if expected_classes is not None:
        expected = list(dict.fromkeys(str(name).strip() for name in expected_classes if str(name).strip()))
        if not expected:
            raise ValueError("expected_classes must not be empty")
        unexpected = sorted(set(class_to_idx) - set(expected))
        if unexpected:
            raise ValueError(f"unexpected classes outside expected set: {unexpected}")
        if require_all_expected_classes:
            missing_expected = sorted(set(expected) - set(class_to_idx))
            if missing_expected:
                raise ValueError(f"expected classes missing from manifest: {missing_expected}")
    label_counts = _count_by_label(rows)
    missing = [name for name in idx_to_class(class_to_idx) if label_counts.get(name, 0) < min_images_per_class]
    if missing:
        raise ValueError(f"classes below min_images_per_class={min_images_per_class}: {missing}")
    labeled_counts = _count_by_label_for_source(rows, "labeled")
    if min_labeled_images_per_class is not None:
        missing_labeled = [
            name
            for name in idx_to_class(class_to_idx)
            if labeled_counts.get(name, 0) < min_labeled_images_per_class
        ]
        if missing_labeled:
            if min_labeled_images_reason is not None:
                raise ValueError(
                    f"labeled classes below {min_labeled_images_reason} "
                    f"(min_labeled_images_per_class={min_labeled_images_per_class}): {missing_labeled}"
                )
            raise ValueError(
                "labeled classes below min_labeled_images_per_class="
                f"{min_labeled_images_per_class}: {missing_labeled}"
            )
    return {
        "all": label_counts,
        "labeled": labeled_counts,
    }


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_image_integrity(
    rows: list[ManifestRow],
    check_image_exists: bool,
    check_readable_images: bool,
    check_unique_image_id: bool,
    check_unique_realpath: bool,
    check_unique_image_hash: bool,
) -> dict[str, object]:
    requires_existing_files = check_image_exists or check_readable_images or check_unique_image_hash
    if requires_existing_files:
        missing = [row.image_id or str(row.path) for row in rows if not row.path.exists()]
        if missing:
            raise ValueError(f"missing image files: {missing[:5]}")
    if check_readable_images:
        unreadable = []
        for row in rows:
            try:
                with Image.open(row.path) as image:
                    image.verify()
            except Exception:
                unreadable.append(row.image_id or str(row.path))
        if unreadable:
            raise ValueError(f"unreadable image files: {unreadable[:5]}")
    if check_unique_image_id:
        ids = [row.image_id for row in rows if row.image_id]
        duplicates = sorted(name for name, count in Counter(ids).items() if count > 1)
        if duplicates:
            raise ValueError(f"duplicate image_id values: {duplicates[:5]}")
    if check_unique_realpath:
        realpaths = [str(row.path.resolve()) for row in rows]
        duplicates = sorted(name for name, count in Counter(realpaths).items() if count > 1)
        if duplicates:
            raise ValueError(f"duplicate real image paths: {duplicates[:5]}")
    hash_duplicates: list[list[str]] = []
    if check_unique_image_hash:
        by_hash: dict[str, list[str]] = {}
        for row in rows:
            image_hash = _hash_file(row.path)
            by_hash.setdefault(image_hash, []).append(row.image_id or str(row.path))
        hash_duplicates = sorted(values for values in by_hash.values() if len(values) > 1)
        if hash_duplicates:
            raise ValueError(f"duplicate image content hashes: {hash_duplicates[:5]}")
    return {
        "checked_exists": check_image_exists,
        "checked_readable_images": check_readable_images,
        "checked_unique_image_id": check_unique_image_id,
        "checked_unique_realpath": check_unique_realpath,
        "checked_unique_image_hash": check_unique_image_hash,
    }


def _validate_inference_budget(
    inference_stats: Path | None,
    max_seconds_per_image: float | None,
    max_checkpoints: int,
    allow_tta: bool,
) -> dict[str, object] | None:
    if inference_stats is None:
        return None
    if max_seconds_per_image is None:
        raise ValueError("--max-seconds-per-image is required when --inference-stats is provided")
    result = check_inference_budget(
        stats_path=inference_stats,
        max_seconds_per_image=max_seconds_per_image,
        max_checkpoints=max_checkpoints,
        allow_tta=allow_tta,
    )
    if result["status"] != "pass":
        raise ValueError(f"inference budget failed: {result['violations']}")
    return result


def run_preflight_checks(
    config_path: Path | None,
    config: AppConfig | None = None,
    profile: str = "custom",
    train_csv: Path | None = None,
    train_dir: Path | None = None,
    image_root: Path | None = None,
    class_map: Path | None = None,
    expected_classes: list[str] | None = None,
    require_all_expected_classes: bool = False,
    check_image_exists: bool = False,
    check_readable_images: bool = False,
    check_unique_image_id: bool = False,
    check_unique_realpath: bool = False,
    check_unique_image_hash: bool = False,
    allow_external_data: bool = False,
    external_max_ratio: float | None = None,
    external_max_sample_weight: float | None = None,
    pseudo_min_confidence: float | None = None,
    pseudo_max_ratio: float | None = None,
    allow_pseudo_teacher_distillation: bool = False,
    min_images_per_class: int = 1,
    min_labeled_images_per_class: int | None = None,
    inference_stats: Path | None = None,
    max_seconds_per_image: float | None = None,
    max_checkpoints: int = 1,
    allow_tta: bool = False,
) -> dict[str, Any]:
    if profile not in {"custom", "server-strict"}:
        raise ValueError("profile must be one of: custom, server-strict")
    if config is None:
        config = load_config(config_path if config_path is not None else None)
    min_labeled_images_reason = None
    if profile == "server-strict":
        check_image_exists = True
        check_readable_images = True
        check_unique_image_id = True
        check_unique_realpath = True
        check_unique_image_hash = True
        if external_max_ratio is None:
            external_max_ratio = 0.5
        if external_max_sample_weight is None:
            external_max_sample_weight = 0.5
        if pseudo_min_confidence is None:
            pseudo_min_confidence = 0.95
        if pseudo_max_ratio is None:
            pseudo_max_ratio = 0.5
        if min_labeled_images_per_class is None:
            min_labeled_images_per_class = 2
        requested_folds = int(config.data.folds)
        if requested_folds > min_labeled_images_per_class:
            min_labeled_images_reason = "requested folds"
        min_labeled_images_per_class = max(min_labeled_images_per_class, requested_folds)
        if inference_stats is not None and max_seconds_per_image is None:
            max_seconds_per_image = 0.05

    rows, class_to_idx = _load_rows(
        config,
        train_csv=train_csv,
        train_dir=train_dir,
        image_root=image_root,
        class_map=class_map,
        require_class_map_for_csv=profile == "server-strict",
    )
    label_counts = _validate_class_support(
        rows,
        class_to_idx,
        min_images_per_class=min_images_per_class,
        min_labeled_images_per_class=min_labeled_images_per_class,
        min_labeled_images_reason=min_labeled_images_reason,
        expected_classes=expected_classes,
        require_all_expected_classes=require_all_expected_classes,
    )
    image_integrity = _validate_image_integrity(
        rows,
        check_image_exists=check_image_exists,
        check_readable_images=check_readable_images,
        check_unique_image_id=check_unique_image_id,
        check_unique_realpath=check_unique_realpath,
        check_unique_image_hash=check_unique_image_hash,
    )
    source_counts = _validate_sources(
        rows,
        allow_external_data=allow_external_data,
        external_max_ratio=external_max_ratio,
        external_max_sample_weight=external_max_sample_weight,
        pseudo_min_confidence=pseudo_min_confidence,
        pseudo_max_ratio=pseudo_max_ratio,
    )
    external_audit = validate_external_merge_audit(rows, config) if profile == "server-strict" else None
    teacher = _validate_teacher_distillation(
        rows,
        config.train.distillation_alpha,
        class_names=idx_to_class(class_to_idx),
        allow_pseudo_teacher_distillation=allow_pseudo_teacher_distillation,
        pseudo_min_confidence=pseudo_min_confidence,
        pseudo_max_ratio=pseudo_max_ratio,
    )
    inference = _validate_inference_budget(
        inference_stats=inference_stats,
        max_seconds_per_image=max_seconds_per_image,
        max_checkpoints=max_checkpoints,
        allow_tta=allow_tta,
    )

    return {
        "status": "pass",
        "profile": profile,
        "config": str(config_path) if config_path is not None else None,
        "rows": len(rows),
        "classes": idx_to_class(class_to_idx),
        "label_counts": label_counts["all"],
        "labeled_label_counts": label_counts["labeled"],
        "source_counts": source_counts,
        "external_audit": external_audit,
        "image_integrity": image_integrity,
        "teacher_distillation": teacher,
        "inference_budget": inference,
        "recommendation": "ready_for_server_training",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run server-training preflight checks before expensive weather runs.")
    parser.add_argument("--config", type=Path, default=Path("configs/convnextv2_384.yaml"))
    parser.add_argument("--profile", choices=["custom", "server-strict"], default="custom")
    parser.add_argument("--train-csv", type=Path, default=None)
    parser.add_argument("--train-dir", type=Path, default=None)
    parser.add_argument("--image-root", type=Path, default=None)
    parser.add_argument("--class-map", type=Path, default=None)
    parser.add_argument("--expected-classes", nargs="+", default=None)
    parser.add_argument("--require-all-expected-classes", action="store_true")
    parser.add_argument("--check-image-exists", action="store_true")
    parser.add_argument("--check-readable-images", action="store_true")
    parser.add_argument("--check-unique-image-id", action="store_true")
    parser.add_argument("--check-unique-realpath", action="store_true")
    parser.add_argument("--check-unique-image-hash", action="store_true")
    parser.add_argument("--allow-external-data", action="store_true")
    parser.add_argument("--external-max-ratio", type=float, default=None)
    parser.add_argument("--external-max-sample-weight", type=float, default=None)
    parser.add_argument("--pseudo-min-confidence", type=float, default=None)
    parser.add_argument("--pseudo-max-ratio", type=float, default=None)
    parser.add_argument("--allow-pseudo-teacher-distillation", action="store_true")
    parser.add_argument("--min-images-per-class", type=int, default=1)
    parser.add_argument("--min-labeled-images-per-class", type=int, default=None)
    parser.add_argument("--inference-stats", type=Path, default=None)
    parser.add_argument("--max-seconds-per-image", type=float, default=None)
    parser.add_argument("--max-checkpoints", type=int, default=1)
    parser.add_argument("--allow-tta", action="store_true")
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_preflight_checks(
        config_path=args.config,
        profile=args.profile,
        train_csv=args.train_csv,
        train_dir=args.train_dir,
        image_root=args.image_root,
        class_map=args.class_map,
        expected_classes=args.expected_classes,
        require_all_expected_classes=args.require_all_expected_classes,
        check_image_exists=args.check_image_exists,
        check_readable_images=args.check_readable_images,
        check_unique_image_id=args.check_unique_image_id,
        check_unique_realpath=args.check_unique_realpath,
        check_unique_image_hash=args.check_unique_image_hash,
        allow_external_data=args.allow_external_data,
        external_max_ratio=args.external_max_ratio,
        external_max_sample_weight=args.external_max_sample_weight,
        pseudo_min_confidence=args.pseudo_min_confidence,
        pseudo_max_ratio=args.pseudo_max_ratio,
        allow_pseudo_teacher_distillation=args.allow_pseudo_teacher_distillation,
        min_images_per_class=args.min_images_per_class,
        min_labeled_images_per_class=args.min_labeled_images_per_class,
        inference_stats=args.inference_stats,
        max_seconds_per_image=args.max_seconds_per_image,
        max_checkpoints=args.max_checkpoints,
        allow_tta=args.allow_tta,
    )
    text = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text, end="")


if __name__ == "__main__":
    main()
