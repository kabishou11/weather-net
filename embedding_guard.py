from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.weather_net.data import build_manifest_from_csv, build_manifest_from_image_folder
from src.weather_net.embedding_guard import filter_pseudo_labels_with_embedding_guard, load_embedding_index


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Filter pseudo labels with offline image embeddings.")
    parser.add_argument("--train-dir", type=Path, default=None)
    parser.add_argument("--train-csv", type=Path, default=None)
    parser.add_argument("--image-root", type=Path, default=None)
    parser.add_argument("--pseudo-csv", type=Path, required=True)
    parser.add_argument("--embeddings", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("pseudo_labels_filtered.csv"))
    parser.add_argument("--rejected-output", type=Path, default=Path("pseudo_labels_rejected.csv"))
    parser.add_argument("--min-similarity", type=float, default=0.8)
    parser.add_argument("--min-margin", type=float, default=0.1)
    args = parser.parse_args()
    if args.train_dir is None and args.train_csv is None:
        parser.error("Set --train-dir or --train-csv")
    if args.train_dir is not None and args.train_csv is not None:
        parser.error("Use only one of --train-dir or --train-csv")
    return args


def load_training_rows(train_dir: Path | None, train_csv: Path | None, image_root: Path | None):
    if train_dir is not None:
        rows, _class_to_idx = build_manifest_from_image_folder(train_dir)
        return rows
    if train_csv is not None:
        rows, _class_to_idx = build_manifest_from_csv(train_csv, image_root=image_root)
        return rows
    raise ValueError("Set train_dir or train_csv")


def main() -> None:
    args = parse_args()
    train_rows = load_training_rows(args.train_dir, args.train_csv, args.image_root)
    embeddings = load_embedding_index(args.embeddings)
    stats = filter_pseudo_labels_with_embedding_guard(
        train_rows=train_rows,
        pseudo_csv=args.pseudo_csv,
        embeddings=embeddings,
        output_csv=args.output,
        rejected_csv=args.rejected_output,
        min_similarity=args.min_similarity,
        min_margin=args.min_margin,
    )
    print(json.dumps({**stats, "output": str(args.output), "rejected_output": str(args.rejected_output)}, indent=2))


if __name__ == "__main__":
    main()
