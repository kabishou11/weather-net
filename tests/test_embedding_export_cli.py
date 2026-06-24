import sys
from pathlib import Path


def test_embedding_export_parse_args_accepts_timm_backend(monkeypatch) -> None:
    from embedding_export import parse_args

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "embedding_export.py",
            "--image-dir",
            "images",
            "--backend",
            "timm",
            "--model-name",
            "resnet18",
            "--output",
            "embeddings.npz",
            "--batch-size",
            "16",
        ],
    )

    args = parse_args()

    assert args.image_dir == Path("images")
    assert args.backend == "timm"
    assert args.model_name == "resnet18"
    assert args.output == Path("embeddings.npz")
    assert args.batch_size == 16
