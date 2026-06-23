from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.weather_net.config import load_config, resolve_device
from src.weather_net.inference import load_unlabeled_rows, predict_probabilities
from src.weather_net.pseudo_label import select_pseudo_labels, write_pseudo_labels


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate high-confidence pseudo labels.")
    parser.add_argument("--config", type=Path, default=Path("configs/convnext_tiny.yaml"))
    parser.add_argument("--checkpoint", type=Path, nargs="+", required=True)
    parser.add_argument("--test-dir", type=Path, default=None)
    parser.add_argument("--test-csv", type=Path, default=None)
    parser.add_argument("--image-root", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=Path("pseudo_labels.csv"))
    parser.add_argument("--threshold", type=float, default=0.95)
    parser.add_argument("--min-margin", type=float, default=0.0)
    parser.add_argument("--require-tta-agreement", action="store_true")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--weights", type=float, nargs="+", default=None)
    parser.add_argument("--tta", action="store_true")
    parser.add_argument("--device", type=str, default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config if args.config.exists() else None)
    device = resolve_device(args.device)
    rows = load_unlabeled_rows(
        test_dir=args.test_dir or config.data.test_dir,
        test_csv=args.test_csv or config.data.test_csv,
        image_root=args.image_root or config.data.image_root,
        image_column=config.data.image_column,
    )
    use_tta = args.tta or config.infer.tta
    image_paths, probabilities, class_names, image_ids, stats = predict_probabilities(
        checkpoints=args.checkpoint,
        rows=rows,
        batch_size=args.batch_size or config.infer.batch_size,
        device=device,
        tta=use_tta,
        num_workers=config.data.num_workers,
        checkpoint_weights=args.weights,
        transform_backend=config.data.transform_backend,
    )
    agreement_probabilities = None
    if args.require_tta_agreement:
        _, agreement_probabilities, _, _, agreement_stats = predict_probabilities(
            checkpoints=args.checkpoint,
            rows=rows,
            batch_size=args.batch_size or config.infer.batch_size,
            device=device,
            tta=not use_tta,
            num_workers=config.data.num_workers,
            checkpoint_weights=args.weights,
            transform_backend=config.data.transform_backend,
        )
        stats["agreement_pass"] = agreement_stats
    pseudo_rows = select_pseudo_labels(
        image_paths=image_paths,
        probabilities=probabilities,
        class_names=class_names,
        threshold=args.threshold,
        min_margin=args.min_margin,
        require_tta_agreement=args.require_tta_agreement,
        agreement_probabilities=agreement_probabilities,
        image_ids=image_ids,
    )
    write_pseudo_labels(args.output, pseudo_rows)
    payload = {
        **stats,
        "threshold": args.threshold,
        "min_margin": args.min_margin,
        "require_tta_agreement": args.require_tta_agreement,
        "selected": len(pseudo_rows),
        "selected_ratio": len(pseudo_rows) / len(rows) if rows else 0.0,
        "output": str(args.output),
    }
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
