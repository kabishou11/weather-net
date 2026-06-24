from pathlib import Path

from PIL import Image


def _make_image(path: Path, color: tuple[int, int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (8, 8), color).save(path)


def test_dedupe_external_manifests_keeps_first_priority_and_audits_rejects(tmp_path: Path) -> None:
    import csv

    from dedupe_external_manifests import dedupe_external_manifests

    root_a = tmp_path / "a"
    root_b = tmp_path / "b"
    _make_image(root_a / "rain.jpg", (10, 20, 30))
    _make_image(root_b / "rain_copy.jpg", (10, 20, 30))
    _make_image(root_b / "sunny.jpg", (240, 220, 80))
    manifest_a = tmp_path / "a.csv"
    manifest_a.write_text(
        "image,label,source,original_label,dataset_url,license,doi,sample_weight\n"
        "rain.jpg,rain,external_a,rain,,,,0.5\n",
        encoding="utf-8",
    )
    manifest_b = tmp_path / "b.csv"
    manifest_b.write_text(
        "image,label,source,original_label,dataset_url,license,doi,sample_weight\n"
        "rain_copy.jpg,rain,external_b,rain,,,,0.2\n"
        "sunny.jpg,sunny,external_b,sunny,,,,0.2\n",
        encoding="utf-8",
    )

    summary = dedupe_external_manifests(
        manifest_roots=[(manifest_a, root_a), (manifest_b, root_b)],
        output_csv=tmp_path / "merged.csv",
        rejected_csv=tmp_path / "rejected.csv",
        audit_json=tmp_path / "audit.json",
    )

    kept = list(csv.DictReader((tmp_path / "merged.csv").open("r", encoding="utf-8")))
    rejected = list(csv.DictReader((tmp_path / "rejected.csv").open("r", encoding="utf-8")))
    assert summary["input_rows"] == 3
    assert summary["kept_rows"] == 2
    assert summary["rejected_duplicate_hash"] == 1
    assert [row["source"] for row in kept] == ["external_a", "external_b"]
    assert rejected[0]["image"] == "rain_copy.jpg"
    assert rejected[0]["kept_source"] == "external_a"


def test_dedupe_external_manifests_can_rebase_image_paths(tmp_path: Path) -> None:
    import csv

    from dedupe_external_manifests import dedupe_external_manifests

    root = tmp_path / "external" / "dataset"
    _make_image(root / "rain.jpg", (10, 20, 30))
    manifest = tmp_path / "manifest.csv"
    manifest.write_text(
        "image,label,source,sample_weight\n"
        "rain.jpg,rain,external_a,0.5\n",
        encoding="utf-8",
    )

    dedupe_external_manifests(
        manifest_roots=[(manifest, root)],
        output_csv=tmp_path / "merged.csv",
        rejected_csv=tmp_path / "rejected.csv",
        audit_json=tmp_path / "audit.json",
        output_image_root=tmp_path,
    )

    rows = list(csv.DictReader((tmp_path / "merged.csv").open("r", encoding="utf-8")))
    assert rows[0]["image"] == "external/dataset/rain.jpg"
    assert rows[0]["original_image"] == "rain.jpg"


def test_dedupe_external_manifests_rejects_reserved_rows(tmp_path: Path) -> None:
    import pytest

    from dedupe_external_manifests import dedupe_external_manifests

    root = tmp_path / "images"
    _make_image(root / "alien_test" / "x.jpg", (1, 2, 3))
    manifest = tmp_path / "bad.csv"
    manifest.write_text(
        "image,label,source,sample_weight\n"
        "alien_test/x.jpg,rain,external_bad,0.2\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="reserved test split"):
        dedupe_external_manifests(
            manifest_roots=[(manifest, root)],
            output_csv=tmp_path / "merged.csv",
            rejected_csv=tmp_path / "rejected.csv",
            audit_json=tmp_path / "audit.json",
        )
