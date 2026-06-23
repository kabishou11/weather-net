from pathlib import Path


def test_write_submission_outputs_header_and_predictions(tmp_path: Path) -> None:
    from src.weather_net.submission import write_submission

    output_path = tmp_path / "submission.csv"

    write_submission(
        output_path=output_path,
        image_paths=[Path("/data/test/b.jpg"), Path("/data/test/a.jpg")],
        predictions=["rain", "sunny"],
    )

    assert output_path.read_text(encoding="utf-8") == (
        "image,label\n"
        "b.jpg,rain\n"
        "a.jpg,sunny\n"
    )


def test_write_submission_can_preserve_original_image_ids(tmp_path: Path) -> None:
    from src.weather_net.submission import write_submission

    output_path = tmp_path / "submission.csv"

    write_submission(
        output_path=output_path,
        image_paths=[Path("/data/test/a.jpg"), Path("/data/test/b.jpg")],
        predictions=["rain", "sunny"],
        image_ids=["folder/a.jpg", "folder/b.jpg"],
    )

    assert output_path.read_text(encoding="utf-8") == (
        "image,label\n"
        "folder/a.jpg,rain\n"
        "folder/b.jpg,sunny\n"
    )


def test_write_submission_rejects_length_mismatch(tmp_path: Path) -> None:
    from src.weather_net.submission import write_submission

    try:
        write_submission(
            output_path=tmp_path / "bad.csv",
            image_paths=[Path("a.jpg")],
            predictions=[],
        )
    except ValueError as exc:
        assert "same length" in str(exc)
    else:
        raise AssertionError("write_submission should reject mismatched inputs")
