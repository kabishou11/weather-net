import sys
from pathlib import Path


def test_hard_mining_parse_args_accepts_expected_files(monkeypatch) -> None:
    from hard_mining import parse_args

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "hard_mining.py",
            "--train-csv",
            "train.csv",
            "--oof-csv",
            "oof_predictions.csv",
            "--output",
            "weighted_train.csv",
            "--hard-output",
            "hard.csv",
            "--pair-confusion-boost",
            "0.4",
            "--pair-min-support",
            "2",
            "--pair-min-error-share",
            "0.6",
        ],
    )

    args = parse_args()

    assert args.train_csv == Path("train.csv")
    assert args.oof_csv == Path("oof_predictions.csv")
    assert args.output == Path("weighted_train.csv")
    assert args.hard_output == Path("hard.csv")
    assert args.pair_confusion_boost == 0.4
    assert args.pair_min_support == 2
    assert args.pair_min_error_share == 0.6


def test_hard_mining_main_writes_weighted_outputs(monkeypatch, tmp_path: Path, capsys) -> None:
    from hard_mining import main

    train_csv = tmp_path / "train.csv"
    train_csv.write_text(
        "image,label\n"
        "easy.jpg,rain\n"
        "wrong.jpg,sunny\n",
        encoding="utf-8",
    )
    oof_csv = tmp_path / "oof.csv"
    oof_csv.write_text(
        "image_id,true_label,pred_label,correct,confidence,margin,loss\n"
        "easy.jpg,rain,rain,True,0.950000,0.800000,0.050000\n"
        "wrong.jpg,sunny,rain,False,0.900000,0.700000,2.000000\n",
        encoding="utf-8",
    )
    output = tmp_path / "weighted.csv"
    hard_output = tmp_path / "hard.csv"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "hard_mining.py",
            "--train-csv",
            str(train_csv),
            "--oof-csv",
            str(oof_csv),
            "--output",
            str(output),
            "--hard-output",
            str(hard_output),
        ],
    )

    main()

    assert '"updated": 1' in capsys.readouterr().out
    assert "wrong.jpg,sunny,2.500000,labeled" in output.read_text(encoding="utf-8")
    assert "wrong.jpg,sunny,2.500000,error+high_loss" in hard_output.read_text(encoding="utf-8")


def test_hard_mining_main_passes_pair_confusion_options(monkeypatch, tmp_path: Path, capsys) -> None:
    from hard_mining import main

    train_csv = tmp_path / "train.csv"
    train_csv.write_text(
        "image,label\n"
        "rain_1.jpg,rain\n"
        "rain_2.jpg,rain\n",
        encoding="utf-8",
    )
    oof_csv = tmp_path / "oof.csv"
    oof_csv.write_text(
        "image_id,true_label,pred_label,correct,confidence,margin,loss\n"
        "rain_1.jpg,rain,fog,False,0.900000,0.700000,2.000000\n"
        "rain_2.jpg,rain,fog,False,0.850000,0.650000,1.900000\n",
        encoding="utf-8",
    )
    output = tmp_path / "weighted.csv"
    hard_output = tmp_path / "hard.csv"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "hard_mining.py",
            "--train-csv",
            str(train_csv),
            "--oof-csv",
            str(oof_csv),
            "--output",
            str(output),
            "--hard-output",
            str(hard_output),
            "--low-margin-boost",
            "0",
            "--high-loss-boost",
            "0",
            "--pair-confusion-boost",
            "0.4",
            "--pair-min-support",
            "2",
        ],
    )

    main()

    assert '"pair_confusion_boosted": 2' in capsys.readouterr().out
    assert "pair_confusion:rain->fog" in hard_output.read_text(encoding="utf-8")
