from pathlib import Path

import numpy as np
import pytest
import torch


def test_write_embedding_npz_preserves_ids_and_normalizes_vectors(tmp_path: Path) -> None:
    from src.weather_net.embedding_export import write_embedding_npz

    output = tmp_path / "embeddings.npz"
    write_embedding_npz(
        output,
        image_ids=["a.jpg", "b.jpg"],
        embeddings=np.array([[3.0, 4.0], [0.0, 2.0]], dtype=np.float32),
        metadata={"backend": "test"},
    )

    data = np.load(output, allow_pickle=True)

    assert data["image_id"].tolist() == ["a.jpg", "b.jpg"]
    assert np.allclose(data["embedding"], np.array([[0.6, 0.8], [0.0, 1.0]], dtype=np.float32))
    assert data["backend"].item() == "test"


def test_write_embedding_npz_rejects_duplicate_ids(tmp_path: Path) -> None:
    from src.weather_net.embedding_export import write_embedding_npz

    with pytest.raises(ValueError, match="Duplicate"):
        write_embedding_npz(
            tmp_path / "embeddings.npz",
            image_ids=["a.jpg", "a.jpg"],
            embeddings=np.ones((2, 3), dtype=np.float32),
        )


def test_collect_embeddings_uses_dataset_image_ids() -> None:
    from torch.utils.data import DataLoader, Dataset

    from src.weather_net.embedding_export import collect_embeddings

    class TinyDataset(Dataset):
        def __len__(self) -> int:
            return 2

        def __getitem__(self, index: int):
            image = torch.full((3, 2, 2), float(index + 1))
            return image, -1, f"/tmp/{index}.jpg", f"id_{index}.jpg"

    class TinyEncoder(torch.nn.Module):
        def forward_features(self, images: torch.Tensor) -> torch.Tensor:
            return images.mean(dim=(2, 3))

    image_ids, embeddings = collect_embeddings(
        TinyEncoder(),
        DataLoader(TinyDataset(), batch_size=2),
        device="cpu",
        amp=False,
    )

    assert image_ids == ["id_0.jpg", "id_1.jpg"]
    assert embeddings.shape == (2, 3)
    assert np.allclose(embeddings[0], np.array([1.0, 1.0, 1.0]))
    assert np.allclose(embeddings[1], np.array([2.0, 2.0, 2.0]))


def test_prepare_encoder_state_drops_classifier_keys_and_strips_backbone_prefix() -> None:
    from src.weather_net.embedding_export import prepare_encoder_state

    state = {
        "backbone.stem.weight": torch.ones(1),
        "backbone.stages.0.weight": torch.ones(1) * 2,
        "classifier.weight": torch.ones(2, 3),
        "classifier.bias": torch.ones(2),
        "head.weight": torch.ones(2, 3),
        "head.bias": torch.ones(2),
        "head.norm.weight": torch.ones(3),
        "head.fc.weight": torch.ones(2, 3),
    }

    prepared = prepare_encoder_state(state)

    assert prepared == {
        "stem.weight": state["backbone.stem.weight"],
        "stages.0.weight": state["backbone.stages.0.weight"],
        "head.norm.weight": state["head.norm.weight"],
    }


def test_select_compatible_encoder_state_rejects_low_match_ratio() -> None:
    from src.weather_net.embedding_export import select_compatible_encoder_state

    prepared = {
        "stem.weight": torch.ones(3, 3),
        "block.weight": torch.ones(4, 4),
        "other.weight": torch.ones(5, 5),
    }
    model_state = {
        "stem.weight": torch.zeros(3, 3),
        "block.weight": torch.zeros(9, 9),
    }

    with pytest.raises(ValueError, match="matched too few"):
        select_compatible_encoder_state(prepared, model_state, min_match_ratio=0.5)


def test_resolve_encoder_model_name_uses_weather_expert_backbone() -> None:
    from src.weather_net.embedding_export import resolve_encoder_model_name

    assert resolve_encoder_model_name("weather_expert:convnext_tiny") == "convnext_tiny"
    assert resolve_encoder_model_name("efficientnet_b0") == "efficientnet_b0"
