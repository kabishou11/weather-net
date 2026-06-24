from __future__ import annotations

import argparse
from pathlib import Path

from src.weather_net.config import load_config, validate_config
from src.weather_net.training import train_config
from training_preflight import run_preflight_checks


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train weather image classifiers.")
    parser.add_argument("--config", type=Path, default=Path("configs/convnext_tiny.yaml"))
    parser.add_argument("--train-dir", type=Path, default=None)
    parser.add_argument("--train-csv", type=Path, default=None)
    parser.add_argument("--image-root", type=Path, default=None)
    parser.add_argument("--class-map", type=Path, default=None)
    parser.add_argument("--preflight", choices=["off", "custom", "server-strict"], default="server-strict")
    parser.add_argument("--i-understand-preflight-off", action="store_true")
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
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--folds", type=int, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--device", type=str, default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.config.exists():
        raise FileNotFoundError(f"Config file does not exist: {args.config}")
    config = load_config(args.config)
    if args.train_dir is not None:
        config.data.train_dir = args.train_dir
    if args.train_csv is not None:
        config.data.train_csv = args.train_csv
    if args.image_root is not None:
        config.data.image_root = args.image_root
    if args.class_map is not None:
        config.data.class_map = args.class_map
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

    validate_config(config)
    if config.data.train_csv is not None and config.data.train_dir is not None:
        raise ValueError("train.py requires either --train-csv or --train-dir, not both")
    if args.preflight == "off" and not args.i_understand_preflight_off:
        raise ValueError("--preflight off requires --i-understand-preflight-off")

    if args.preflight != "off":
        run_preflight_checks(
            config_path=args.config,
            config=config,
            profile=args.preflight,
            train_csv=config.data.train_csv,
            train_dir=config.data.train_dir,
            image_root=config.data.image_root,
            class_map=config.data.class_map,
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

    checkpoints = train_config(config, device_request=args.device)
    print("Saved checkpoints:")
    for checkpoint in checkpoints:
        print(checkpoint)


if __name__ == "__main__":
    main()
