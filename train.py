from __future__ import annotations

import argparse
from pathlib import Path

from src.weather_net.config import load_config
from src.weather_net.training import train_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train weather image classifiers.")
    parser.add_argument("--config", type=Path, default=Path("configs/convnext_tiny.yaml"))
    parser.add_argument("--train-dir", type=Path, default=None)
    parser.add_argument("--train-csv", type=Path, default=None)
    parser.add_argument("--image-root", type=Path, default=None)
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--folds", type=int, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--device", type=str, default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config if args.config.exists() else None)
    if args.train_dir is not None:
        config.data.train_dir = args.train_dir
    if args.train_csv is not None:
        config.data.train_csv = args.train_csv
    if args.image_root is not None:
        config.data.image_root = args.image_root
    if args.model is not None:
        config.model.name = args.model
    if args.epochs is not None:
        config.train.epochs = args.epochs
    if args.batch_size is not None:
        config.train.batch_size = args.batch_size
    if args.folds is not None:
        config.data.folds = args.folds
    if args.output_dir is not None:
        config.train.output_dir = args.output_dir

    checkpoints = train_config(config, device_request=args.device)
    print("Saved checkpoints:")
    for checkpoint in checkpoints:
        print(checkpoint)


if __name__ == "__main__":
    main()
