from pathlib import Path


def test_train_parse_args_accepts_class_map(monkeypatch) -> None:
    from train import parse_args

    monkeypatch.setattr(
        "sys.argv",
        [
            "train.py",
            "--train-csv",
            "data/train.csv",
            "--image-root",
            "data/images",
            "--class-map",
            "outputs/convnextv2_384/class_to_idx.json",
        ],
    )

    args = parse_args()

    assert args.class_map == Path("outputs/convnextv2_384/class_to_idx.json")
