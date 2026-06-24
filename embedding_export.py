from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.weather_net.config import resolve_device
from src.weather_net.embedding_export import (
    build_embedding_loader,
    collect_embeddings,
    create_timm_encoder,
    resolve_encoder_model_name,
    write_embedding_npz,
)
from src.weather_net.inference import load_unlabeled_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export offline image embeddings for semantic pseudo-label guards.")
    parser.add_argument("--image-dir", type=Path, default=None)
    parser.add_argument("--image-csv", type=Path, default=None)
    parser.add_argument("--image-root", type=Path, default=None)
    parser.add_argument("--image-column", type=str, default=None)
    parser.add_argument("--backend", choices=["timm"], default="timm")
    parser.add_argument("--model-name", type=str, default="convnext_tiny")
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--transform-backend", choices=["auto", "albumentations", "torchvision"], default="auto")
    parser.add_argument("--pretrained", dest="pretrained", action="store_true", default=True)
    parser.add_argument("--no-pretrained", dest="pretrained", action="store_false")
    args = parser.parse_args()
    if args.image_dir is None and args.image_csv is None:
        parser.error("Set --image-dir or --image-csv")
    if args.image_dir is not None and args.image_csv is not None:
        parser.error("Use only one of --image-dir or --image-csv")
    return args


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    rows = load_unlabeled_rows(
        test_dir=args.image_dir,
        test_csv=args.image_csv,
        image_root=args.image_root,
        image_column=args.image_column,
    )
    if args.backend != "timm":
        raise ValueError(f"Unsupported embedding backend: {args.backend}")

    encoder_model_name = resolve_encoder_model_name(args.model_name)
    model = create_timm_encoder(
        model_name=args.model_name,
        pretrained=args.pretrained,
        checkpoint_path=args.checkpoint,
        device=device,
    )
    loader = build_embedding_loader(
        rows=rows,
        image_size=args.image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        transform_backend=args.transform_backend,
    )
    image_ids, embeddings = collect_embeddings(
        model=model,
        loader=loader,
        device=device,
        amp=args.amp,
    )
    write_embedding_npz(
        args.output,
        image_ids=image_ids,
        embeddings=embeddings,
        metadata={
            "backend": args.backend,
            "model_name": args.model_name,
            "encoder_model_name": encoder_model_name,
            "image_size": args.image_size,
            "pretrained": bool(args.pretrained),
            "checkpoint": "" if args.checkpoint is None else str(args.checkpoint),
        },
    )
    print(
        json.dumps(
            {
                "images": len(image_ids),
                "embedding_dim": int(embeddings.shape[1]),
                "output": str(args.output),
                "backend": args.backend,
                "model_name": args.model_name,
                "encoder_model_name": encoder_model_name,
                "image_size": args.image_size,
                "device": device,
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
