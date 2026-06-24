from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

from src.weather_net.data import idx_to_class, is_image_file, load_class_mapping


DEFAULT_LABEL_MAPS: dict[str, dict[str, str]] = {
    "mwd": {
        "cloudy": "cloudy",
        "rain": "rain",
        "shine": "sunny",
        "sunrise": "sunny",
    },
    "weapd": {
        "dew": "dew",
        "fogsmog": "fog",
        "fog_smog": "fog",
        "frost": "snow",
        "glaze": "snow",
        "hail": "snow",
        "lightning": "lightning",
        "rain": "rain",
        "rainbow": "rainbow",
        "rime": "snow",
        "sandstorm": "sandstorm",
        "snow": "snow",
    },
}


def _normalize_label(label: str) -> str:
    return label.strip().lower().replace("/", "_").replace(" ", "_").replace("-", "_")


def _load_label_map(path: Path | None, preset: str | None) -> dict[str, str]:
    mapping: dict[str, str] = {}
    if preset:
        if preset not in DEFAULT_LABEL_MAPS:
            raise ValueError(f"Unknown label map preset: {preset}")
        mapping.update(DEFAULT_LABEL_MAPS[preset])
    if path is not None:
        with path.open("r", encoding="utf-8") as handle:
            raw = json.load(handle)
        if not isinstance(raw, dict):
            raise ValueError("label map JSON must be an object")
        mapping.update({_normalize_label(str(key)): str(value).strip() for key, value in raw.items()})
    return mapping


def load_allowed_labels(allowed_labels: list[str] | None, class_map: Path | None) -> list[str] | None:
    if class_map is not None:
        class_labels = idx_to_class(load_class_mapping(class_map))
        if allowed_labels is None:
            return class_labels
        requested = sorted(dict.fromkeys(str(label).strip() for label in allowed_labels if str(label).strip()))
        outside = sorted(set(requested) - set(class_labels))
        if outside:
            raise ValueError(f"allowed labels outside class map: {outside}")
        return [label for label in class_labels if label in set(requested)]
    labels: list[str] = []
    if allowed_labels is not None:
        labels.extend(str(label).strip() for label in allowed_labels if str(label).strip())
    if not labels:
        return None
    return sorted(dict.fromkeys(labels))


def _label_from_relative_path(path: Path, label_map: dict[str, str]) -> tuple[str, str, bool]:
    parts = [_normalize_label(part) for part in path.parts[:-1]]
    if not parts:
        raise ValueError(f"Image path has no label directory: {path}")
    for end in range(len(parts), 0, -1):
        candidate = "_".join(parts[:end])
        if candidate in label_map:
            return candidate, label_map[candidate], True
    original_label = parts[0]
    return original_label, original_label, False


def build_external_manifest(
    image_root: Path,
    output_csv: Path,
    summary_json: Path,
    dataset_name: str,
    label_map: dict[str, str],
    drop_unmapped: bool = False,
    dataset_url: str = "",
    license_name: str = "",
    doi: str = "",
    allowed_labels: list[str] | None = None,
) -> dict[str, object]:
    image_root = image_root.expanduser().resolve()
    if not dataset_name.strip().startswith("external_"):
        raise ValueError("dataset_name must start with external_")
    rows: list[tuple[str, str, str, str, str, str, str]] = []
    skipped: Counter[str] = Counter()
    allowed_label_set = set(allowed_labels) if allowed_labels is not None else None
    normalized_label_map = {str(key): str(value) for key, value in sorted(label_map.items())}
    label_map_sha256 = hashlib.sha256(
        json.dumps(normalized_label_map, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()

    for path in sorted(image_root.rglob("*")):
        if not path.is_file() or not is_image_file(path):
            continue
        relative_path = path.relative_to(image_root)
        original_label, mapped_label, is_mapped = _label_from_relative_path(relative_path, label_map)
        if is_mapped and not mapped_label:
            raise ValueError(f"empty mapped label for original label: {original_label}")
        if drop_unmapped and not is_mapped:
            skipped[original_label] += 1
            continue
        if allowed_label_set is not None and mapped_label not in allowed_label_set:
            raise ValueError(f"mapped labels outside allowed labels: {[mapped_label]}")
        image = relative_path.as_posix()
        rows.append((image, mapped_label, dataset_name, original_label, dataset_url, license_name, doi))

    if not rows:
        raise ValueError(f"No image files found under {image_root}")

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["image", "label", "source", "original_label", "dataset_url", "license", "doi"])
        writer.writerows(rows)

    counts = Counter(label for _image, label, _source, _original_label, _url, _license, _doi in rows)
    original_counts = Counter(
        original_label for _image, _label, _source, original_label, _url, _license, _doi in rows
    )
    summary: dict[str, object] = {
        "dataset": dataset_name,
        "dataset_url": dataset_url,
        "license": license_name,
        "doi": doi,
        "image_root": str(image_root),
        "output_csv": str(output_csv),
        "total": len(rows),
        "label_counts": dict(sorted(counts.items())),
        "original_label_counts": dict(sorted(original_counts.items())),
        "skipped_unmapped": dict(sorted(skipped.items())),
        "label_map_sha256": label_map_sha256,
        "allowed_labels": list(allowed_labels) if allowed_labels is not None else None,
    }
    summary_json.parent.mkdir(parents=True, exist_ok=True)
    summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create CSV manifests for external ImageFolder weather datasets.")
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--summary-json", type=Path, required=True)
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--dataset-url", default="")
    parser.add_argument("--license", dest="license_name", default="")
    parser.add_argument("--doi", default="")
    parser.add_argument("--label-map-preset", choices=sorted(DEFAULT_LABEL_MAPS), default=None)
    parser.add_argument("--label-map-json", type=Path, default=None)
    parser.add_argument("--allowed-labels", nargs="+", default=None)
    parser.add_argument("--class-map", type=Path, default=None)
    parser.add_argument("--allow-unbounded-labels", action="store_true")
    parser.add_argument("--drop-unmapped", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    label_map = _load_label_map(args.label_map_json, args.label_map_preset)
    allowed_labels = load_allowed_labels(args.allowed_labels, args.class_map)
    if allowed_labels is None and not args.allow_unbounded_labels:
        raise ValueError("external manifest generation requires --class-map or --allowed-labels")
    summary = build_external_manifest(
        image_root=args.image_root,
        output_csv=args.output_csv,
        summary_json=args.summary_json,
        dataset_name=args.dataset_name,
        label_map=label_map,
        drop_unmapped=args.drop_unmapped,
        dataset_url=args.dataset_url,
        license_name=args.license_name,
        doi=args.doi,
        allowed_labels=allowed_labels,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
