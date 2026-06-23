from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from src.weather_net.data import build_manifest_from_csv, build_manifest_from_image_folder, idx_to_class
from src.weather_net.datasets import WeatherImageDataset, build_transforms
from src.weather_net.error_analysis import (
    build_prediction_records,
    confusion_pairs,
    write_error_analysis_csv,
)
from src.weather_net.inference import load_checkpoint
from src.weather_net.metrics import classification_report
from src.weather_net.config import resolve_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate a weather classifier checkpoint.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--val-dir", type=Path, default=None)
    parser.add_argument("--val-csv", type=Path, default=None)
    parser.add_argument("--image-root", type=Path, default=None)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--errors-csv", type=Path, default=None)
    parser.add_argument("--low-confidence-threshold", type=float, default=0.6)
    return parser.parse_args()


@torch.no_grad()
def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    model, class_to_idx, image_size = load_checkpoint(args.checkpoint, device=device)

    if args.val_dir is not None:
        rows, _ = build_manifest_from_image_folder(args.val_dir, class_to_idx=class_to_idx)
    elif args.val_csv is not None:
        rows, _ = build_manifest_from_csv(
            args.val_csv,
            image_root=args.image_root,
            class_to_idx=class_to_idx,
        )
    else:
        raise SystemExit("Set --val-dir or --val-csv")

    dataset = WeatherImageDataset(rows, transform=build_transforms(image_size, train=False), return_path=True)
    loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    y_true: list[int] = []
    y_pred: list[int] = []
    probabilities: list[list[float]] = []
    image_paths: list[Path] = []
    for images, targets, paths, _ in loader:
        logits = model(images.to(device))
        probs = torch.softmax(logits, dim=1).cpu()
        y_true.extend(targets.tolist())
        y_pred.extend(logits.argmax(dim=1).cpu().tolist())
        probabilities.extend([[float(value) for value in row] for row in probs.tolist()])
        image_paths.extend(Path(path) for path in paths)

    class_names = idx_to_class(class_to_idx)
    report = classification_report(y_true, y_pred, class_names)
    payload = {
        "macro_f1": report.macro_f1,
        "accuracy": report.accuracy,
        "per_class_f1": report.per_class_f1,
        "confusion_matrix": report.confusion_matrix,
        "top_confusions": confusion_pairs(y_true, y_pred, class_names, top_k=20),
    }
    text = json.dumps(payload, indent=2, ensure_ascii=False)
    print(text)
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(text + "\n", encoding="utf-8")
    if args.errors_csv is not None:
        records = build_prediction_records(
            image_paths=image_paths,
            y_true=y_true,
            probabilities=probabilities,
            class_names=class_names,
        )
        write_error_analysis_csv(
            args.errors_csv,
            records,
            low_confidence_threshold=args.low_confidence_threshold,
        )


if __name__ == "__main__":
    main()
