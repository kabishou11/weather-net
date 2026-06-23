from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


@dataclass(frozen=True)
class ManifestRow:
    path: Path
    label: int | None = None
    label_name: str | None = None
    image_id: str | None = None
    source: str = "labeled"
    confidence: float = 1.0
    sample_weight: float = 1.0


def is_image_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS


def list_images(root: Path) -> list[Path]:
    root = root.expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"Image root does not exist: {root}")
    if root.is_file():
        return [root] if is_image_file(root) else []
    return sorted(path.resolve() for path in root.rglob("*") if is_image_file(path))


def build_manifest_from_image_folder(
    root: Path,
    class_to_idx: dict[str, int] | None = None,
) -> tuple[list[ManifestRow], dict[str, int]]:
    root = root.expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"ImageFolder root does not exist: {root}")
    classes = sorted(path.name for path in root.iterdir() if path.is_dir())
    if not classes:
        raise ValueError(f"No class folders found under {root}")

    if class_to_idx is None:
        class_to_idx = {name: idx for idx, name in enumerate(classes)}
    else:
        missing = sorted(set(classes) - set(class_to_idx))
        if missing:
            raise ValueError(f"ImageFolder classes missing from class mapping: {missing}")
        classes = sorted(classes, key=lambda name: class_to_idx[name])
    rows: list[ManifestRow] = []
    for class_name in classes:
        class_dir = root / class_name
        for image_path in list_images(class_dir):
            rows.append(
                ManifestRow(
                    path=image_path,
                    label=class_to_idx[class_name],
                    label_name=class_name,
                    image_id=str(image_path.relative_to(root)),
                    source="labeled",
                    confidence=1.0,
                    sample_weight=1.0,
                )
            )
    if not rows:
        raise ValueError(f"No images found under {root}")
    return rows, class_to_idx


def _read_csv_rows(csv_path: Path) -> list[dict[str, str]]:
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header: {csv_path}")
        return [dict(row) for row in reader]


def _pick_column(fieldnames: Sequence[str], candidates: Sequence[str], role: str) -> str:
    normalized = {name.lower(): name for name in fieldnames}
    for candidate in candidates:
        if candidate.lower() in normalized:
            return normalized[candidate.lower()]
    raise ValueError(f"CSV is missing a {role} column. Tried: {', '.join(candidates)}")


def build_manifest_from_csv(
    csv_path: Path,
    image_root: Path | None = None,
    class_to_idx: dict[str, int] | None = None,
    image_column: str | None = None,
    label_column: str | None = None,
) -> tuple[list[ManifestRow], dict[str, int]]:
    csv_path = csv_path.expanduser().resolve()
    rows = _read_csv_rows(csv_path)
    if not rows:
        raise ValueError(f"CSV contains no rows: {csv_path}")

    fieldnames = list(rows[0].keys())
    image_column = image_column or _pick_column(
        fieldnames,
        ["image", "filename", "file", "path", "img_path", "image_path"],
        "image path",
    )
    label_column = label_column or _pick_column(
        fieldnames,
        ["label", "class", "category", "weather", "target"],
        "label",
    )
    image_root = (image_root or csv_path.parent).expanduser().resolve()

    labels = [str(row[label_column]).strip() for row in rows]
    if class_to_idx is None:
        class_to_idx = {name: idx for idx, name in enumerate(sorted(set(labels)))}
    else:
        missing = sorted(set(labels) - set(class_to_idx))
        if missing:
            raise ValueError(f"CSV labels missing from class mapping: {missing}")

    source_column = next((name for name in fieldnames if name.lower() == "source"), None)
    confidence_column = next((name for name in fieldnames if name.lower() == "confidence"), None)
    sample_weight_column = next((name for name in fieldnames if name.lower() == "sample_weight"), None)

    manifest: list[ManifestRow] = []
    for row in rows:
        image_id = str(row[image_column]).strip()
        label_name = str(row[label_column]).strip()
        source = str(row.get(source_column, "labeled")).strip() if source_column else "labeled"
        confidence = float(row.get(confidence_column, 1.0)) if confidence_column else 1.0
        if sample_weight_column:
            sample_weight = float(row[sample_weight_column])
        elif source == "pseudo":
            sample_weight = 0.3 + (0.7 * confidence)
        else:
            sample_weight = 1.0
        manifest.append(
            ManifestRow(
                path=(image_root / image_id).resolve(),
                label=class_to_idx[label_name],
                label_name=label_name,
                image_id=image_id,
                source=source,
                confidence=confidence,
                sample_weight=sample_weight,
            )
        )
    return manifest, dict(class_to_idx)


def build_unlabeled_manifest_from_csv(
    csv_path: Path,
    image_root: Path | None = None,
    image_column: str | None = None,
) -> list[ManifestRow]:
    csv_path = csv_path.expanduser().resolve()
    rows = _read_csv_rows(csv_path)
    if not rows:
        raise ValueError(f"CSV contains no rows: {csv_path}")

    fieldnames = list(rows[0].keys())
    image_column = image_column or _pick_column(
        fieldnames,
        ["image", "filename", "file", "path", "img_path", "image_path"],
        "image path",
    )
    image_root = (image_root or csv_path.parent).expanduser().resolve()
    return [
        ManifestRow(
            path=(image_root / str(row[image_column]).strip()).resolve(),
            image_id=str(row[image_column]).strip(),
            source="unlabeled",
            confidence=1.0,
            sample_weight=1.0,
        )
        for row in rows
    ]


def save_class_mapping(class_to_idx: dict[str, int], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ordered = {name: class_to_idx[name] for name in sorted(class_to_idx, key=class_to_idx.get)}
    output_path.write_text(
        json.dumps(ordered, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def load_class_mapping(path: Path) -> dict[str, int]:
    data = json.loads(path.read_text(encoding="utf-8"))
    mapping = {str(key): int(value) for key, value in data.items()}
    values = sorted(mapping.values())
    if values != list(range(len(values))):
        raise ValueError(f"Class mapping must use contiguous ids from 0: {path}")
    return mapping


def idx_to_class(class_to_idx: dict[str, int]) -> list[str]:
    return [name for name, _ in sorted(class_to_idx.items(), key=lambda item: item[1])]


def split_train_val(
    rows: Sequence[ManifestRow],
    val_fraction: float,
    seed: int,
) -> tuple[list[ManifestRow], list[ManifestRow]]:
    if not 0 < val_fraction < 1:
        raise ValueError("val_fraction must be between 0 and 1")

    import random

    by_label: dict[int, list[ManifestRow]] = {}
    for row in rows:
        if row.label is None:
            raise ValueError("Cannot split unlabeled rows")
        by_label.setdefault(row.label, []).append(row)

    rng = random.Random(seed)
    train_rows: list[ManifestRow] = []
    val_rows: list[ManifestRow] = []
    for label_rows in by_label.values():
        shuffled = list(label_rows)
        rng.shuffle(shuffled)
        if len(shuffled) == 1:
            train_rows.extend(shuffled)
            continue
        val_count = max(1, round(len(shuffled) * val_fraction))
        val_count = min(val_count, len(shuffled) - 1)
        val_rows.extend(shuffled[:val_count])
        train_rows.extend(shuffled[val_count:])
    return sorted(train_rows, key=lambda row: str(row.path)), sorted(
        val_rows,
        key=lambda row: str(row.path),
    )


def iter_kfold_splits(
    rows: Sequence[ManifestRow],
    folds: int,
    seed: int,
) -> Iterable[tuple[int, list[ManifestRow], list[ManifestRow]]]:
    if folds < 2:
        raise ValueError("folds must be at least 2")

    try:
        from sklearn.model_selection import StratifiedKFold
    except Exception as exc:  # pragma: no cover - exercised in lean environments.
        raise RuntimeError("Install scikit-learn to use k-fold training") from exc

    labels = [row.label for row in rows]
    if any(label is None for label in labels):
        raise ValueError("Cannot split unlabeled rows")

    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    indices = list(range(len(rows)))
    for fold_idx, (train_idx, val_idx) in enumerate(splitter.split(indices, labels)):
        yield (
            fold_idx,
            [rows[idx] for idx in train_idx],
            [rows[idx] for idx in val_idx],
        )
