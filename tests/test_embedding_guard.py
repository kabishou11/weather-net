from pathlib import Path
import csv

import numpy as np
import pytest


def test_load_embedding_index_normalizes_vectors_and_rejects_duplicate_ids(tmp_path: Path) -> None:
    from src.weather_net.embedding_guard import load_embedding_index

    path = tmp_path / "embeddings.npz"
    np.savez_compressed(
        path,
        image_id=np.array(["a.jpg", "a.jpg"], dtype=object),
        embedding=np.array([[3.0, 4.0], [1.0, 0.0]], dtype=np.float32),
    )

    with pytest.raises(ValueError, match="Duplicate"):
        load_embedding_index(path)

    np.savez_compressed(
        path,
        image_id=np.array(["a.jpg"], dtype=object),
        embedding=np.array([[3.0, 4.0]], dtype=np.float32),
    )

    index = load_embedding_index(path)

    assert np.allclose(index["a.jpg"], np.array([0.6, 0.8]))


def test_build_class_prototypes_from_labeled_manifest(tmp_path: Path) -> None:
    from src.weather_net.data import ManifestRow
    from src.weather_net.embedding_guard import build_class_prototypes

    rows = [
        ManifestRow(path=tmp_path / "rain_a.jpg", label_name="rain", image_id="rain_a.jpg"),
        ManifestRow(path=tmp_path / "rain_b.jpg", label_name="rain", image_id="rain_b.jpg"),
        ManifestRow(path=tmp_path / "sunny.jpg", label_name="sunny", image_id="sunny.jpg"),
    ]
    embeddings = {
        "rain_a.jpg": np.array([1.0, 0.0]),
        "rain_b.jpg": np.array([1.0, 0.0]),
        "sunny.jpg": np.array([0.0, 1.0]),
    }

    prototypes = build_class_prototypes(rows, embeddings)

    assert np.allclose(prototypes["rain"], np.array([1.0, 0.0]))
    assert np.allclose(prototypes["sunny"], np.array([0.0, 1.0]))


def test_build_class_prototypes_ignores_pseudo_source_rows(tmp_path: Path) -> None:
    from src.weather_net.data import ManifestRow
    from src.weather_net.embedding_guard import build_class_prototypes

    rows = [
        ManifestRow(path=tmp_path / "rain_real.jpg", label_name="rain", image_id="rain_real.jpg", source="labeled"),
        ManifestRow(path=tmp_path / "rain_pseudo.jpg", label_name="rain", image_id="rain_pseudo.jpg", source="pseudo"),
    ]
    embeddings = {
        "rain_real.jpg": np.array([1.0, 0.0]),
        "rain_pseudo.jpg": np.array([0.0, 1.0]),
    }

    prototypes = build_class_prototypes(rows, embeddings)

    assert np.allclose(prototypes["rain"], np.array([1.0, 0.0]))


def test_filter_pseudo_labels_with_embedding_guard_keeps_semantic_matches(tmp_path: Path) -> None:
    from src.weather_net.data import ManifestRow
    from src.weather_net.embedding_guard import filter_pseudo_labels_with_embedding_guard

    train_rows = [
        ManifestRow(path=tmp_path / "rain_ref.jpg", label_name="rain", image_id="rain_ref.jpg"),
        ManifestRow(path=tmp_path / "sunny_ref.jpg", label_name="sunny", image_id="sunny_ref.jpg"),
    ]
    pseudo_csv = tmp_path / "pseudo.csv"
    pseudo_csv.write_text(
        "image,label,confidence\n"
        "u_rain.jpg,rain,0.970000\n"
        "u_bad.jpg,rain,0.990000\n"
        "u_sunny.jpg,sunny,0.960000\n",
        encoding="utf-8",
    )
    embeddings = {
        "rain_ref.jpg": np.array([1.0, 0.0]),
        "sunny_ref.jpg": np.array([0.0, 1.0]),
        "u_rain.jpg": np.array([0.95, 0.05]),
        "u_bad.jpg": np.array([0.05, 0.95]),
        "u_sunny.jpg": np.array([0.1, 0.9]),
    }
    filtered_csv = tmp_path / "filtered.csv"
    rejected_csv = tmp_path / "rejected.csv"

    stats = filter_pseudo_labels_with_embedding_guard(
        train_rows=train_rows,
        pseudo_csv=pseudo_csv,
        embeddings=embeddings,
        output_csv=filtered_csv,
        rejected_csv=rejected_csv,
        min_similarity=0.8,
        min_margin=0.1,
    )

    assert stats == {
        "input": 3,
        "kept": 2,
        "rejected": 1,
        "rejected_low_similarity": 1,
        "rejected_low_margin": 0,
    }
    with filtered_csv.open("r", encoding="utf-8", newline="") as handle:
        kept_rows = list(csv.DictReader(handle))
    assert [row["image"] for row in kept_rows] == ["u_rain.jpg", "u_sunny.jpg"]
    assert [row["label"] for row in kept_rows] == ["rain", "sunny"]
    assert float(kept_rows[0]["semantic_similarity"]) == pytest.approx(0.998618, abs=1e-6)
    assert float(kept_rows[0]["semantic_margin"]) == pytest.approx(0.946059, abs=1e-6)
    assert float(kept_rows[1]["semantic_similarity"]) == pytest.approx(0.993884, abs=1e-6)
    assert float(kept_rows[1]["semantic_margin"]) == pytest.approx(0.883452, abs=1e-6)
    with rejected_csv.open("r", encoding="utf-8", newline="") as handle:
        rejected_rows = list(csv.DictReader(handle))
    assert rejected_rows[0]["image"] == "u_bad.jpg"
    assert rejected_rows[0]["reason"] == "low_similarity"


def test_filter_pseudo_labels_rejects_missing_pseudo_embeddings(tmp_path: Path) -> None:
    from src.weather_net.data import ManifestRow
    from src.weather_net.embedding_guard import filter_pseudo_labels_with_embedding_guard

    pseudo_csv = tmp_path / "pseudo.csv"
    pseudo_csv.write_text("image,label,confidence\nu0.jpg,rain,0.970000\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Missing embeddings"):
        filter_pseudo_labels_with_embedding_guard(
            train_rows=[ManifestRow(path=tmp_path / "rain_ref.jpg", label_name="rain", image_id="rain_ref.jpg")],
            pseudo_csv=pseudo_csv,
            embeddings={"rain_ref.jpg": np.array([1.0, 0.0])},
            output_csv=tmp_path / "filtered.csv",
            rejected_csv=tmp_path / "rejected.csv",
            min_similarity=0.8,
            min_margin=0.1,
        )


def test_filter_pseudo_labels_rejects_unknown_labels(tmp_path: Path) -> None:
    from src.weather_net.data import ManifestRow
    from src.weather_net.embedding_guard import filter_pseudo_labels_with_embedding_guard

    pseudo_csv = tmp_path / "pseudo.csv"
    pseudo_csv.write_text("image,label,confidence\nu0.jpg,snow,0.970000\n", encoding="utf-8")

    with pytest.raises(ValueError, match="unknown labels"):
        filter_pseudo_labels_with_embedding_guard(
            train_rows=[ManifestRow(path=tmp_path / "rain_ref.jpg", label_name="rain", image_id="rain_ref.jpg")],
            pseudo_csv=pseudo_csv,
            embeddings={
                "rain_ref.jpg": np.array([1.0, 0.0]),
                "u0.jpg": np.array([1.0, 0.0]),
            },
            output_csv=tmp_path / "filtered.csv",
            rejected_csv=tmp_path / "rejected.csv",
            min_similarity=0.8,
            min_margin=0.1,
        )


def test_filter_pseudo_labels_rejects_mismatched_embedding_dimensions(tmp_path: Path) -> None:
    from src.weather_net.data import ManifestRow
    from src.weather_net.embedding_guard import filter_pseudo_labels_with_embedding_guard

    pseudo_csv = tmp_path / "pseudo.csv"
    pseudo_csv.write_text("image,label,confidence\nu0.jpg,rain,0.970000\n", encoding="utf-8")

    with pytest.raises(ValueError, match="dimension"):
        filter_pseudo_labels_with_embedding_guard(
            train_rows=[ManifestRow(path=tmp_path / "rain_ref.jpg", label_name="rain", image_id="rain_ref.jpg")],
            pseudo_csv=pseudo_csv,
            embeddings={
                "rain_ref.jpg": np.array([1.0, 0.0]),
                "u0.jpg": np.array([1.0, 0.0, 0.0]),
            },
            output_csv=tmp_path / "filtered.csv",
            rejected_csv=tmp_path / "rejected.csv",
            min_similarity=0.8,
            min_margin=0.1,
        )
