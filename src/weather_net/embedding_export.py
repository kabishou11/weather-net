from __future__ import annotations

import math
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader

from .data import ManifestRow
from .datasets import WeatherImageDataset, build_transforms
from .inference import _loader_kwargs


CLASSIFIER_KEY_PREFIXES = (
    "classifier.",
    "fc.",
    "head.fc.",
    "head.classifier.",
    "context_head.",
)
CLASSIFIER_EXACT_KEYS = {"head.weight", "head.bias"}


def normalize_embeddings(embeddings: np.ndarray) -> np.ndarray:
    vectors = np.asarray(embeddings, dtype=np.float32)
    if vectors.ndim != 2:
        raise ValueError("embeddings must be a 2D array")
    if vectors.shape[1] == 0:
        raise ValueError("embedding dimension must be positive")
    if not np.isfinite(vectors).all():
        raise ValueError("embeddings must be finite")
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    if not np.isfinite(norms).all() or np.any(norms <= 0):
        raise ValueError("embeddings must have finite non-zero norms")
    return (vectors / norms).astype(np.float32, copy=False)


def write_embedding_npz(
    output: Path,
    image_ids: Sequence[str],
    embeddings: np.ndarray,
    metadata: Mapping[str, object] | None = None,
) -> None:
    ids = [str(image_id) for image_id in image_ids]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate image_id values are not allowed")
    vectors = normalize_embeddings(embeddings)
    if len(ids) != vectors.shape[0]:
        raise ValueError("image_ids and embeddings must have the same row count")

    payload: dict[str, object] = {
        "image_id": np.asarray(ids, dtype=object),
        "embedding": vectors,
    }
    for key, value in (metadata or {}).items():
        if key in payload:
            raise ValueError(f"metadata key conflicts with reserved field: {key}")
        payload[str(key)] = np.asarray(value)

    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **payload)


def prepare_encoder_state(state: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    prepared: dict[str, torch.Tensor] = {}
    for raw_key, value in state.items():
        if not torch.is_tensor(value):
            continue
        key = str(raw_key)
        if key.startswith("module."):
            key = key.removeprefix("module.")
        if key in CLASSIFIER_EXACT_KEYS or any(key.startswith(prefix) for prefix in CLASSIFIER_KEY_PREFIXES):
            continue
        if key.startswith("backbone."):
            key = key.removeprefix("backbone.")
        if key in CLASSIFIER_EXACT_KEYS or any(key.startswith(prefix) for prefix in CLASSIFIER_KEY_PREFIXES):
            continue
        prepared[key] = value
    return prepared


def select_compatible_encoder_state(
    prepared_state: Mapping[str, torch.Tensor],
    model_state: Mapping[str, torch.Tensor],
    min_match_ratio: float = 0.5,
) -> dict[str, torch.Tensor]:
    if not 0 < min_match_ratio <= 1:
        raise ValueError("min_match_ratio must be in (0, 1]")
    compatible_state = {
        key: value
        for key, value in prepared_state.items()
        if key in model_state and tuple(value.shape) == tuple(model_state[key].shape)
    }
    if not compatible_state:
        raise ValueError("Checkpoint did not match the embedding encoder")
    match_ratio = len(compatible_state) / max(1, len(prepared_state))
    if match_ratio < min_match_ratio:
        raise ValueError(
            "Checkpoint matched too few encoder parameters; "
            f"matched {len(compatible_state)} of {len(prepared_state)}"
        )
    return compatible_state


def resolve_encoder_model_name(model_name: str) -> str:
    if model_name.startswith("weather_expert:"):
        return model_name.split(":", 1)[1]
    return model_name


def _pool_features(features: torch.Tensor | Sequence[torch.Tensor] | Mapping[str, torch.Tensor]) -> torch.Tensor:
    if isinstance(features, Mapping):
        if not features:
            raise ValueError("feature mapping is empty")
        features = next(reversed(features.values()))
    if isinstance(features, (list, tuple)):
        if not features:
            raise ValueError("feature sequence is empty")
        features = features[-1]
    if not torch.is_tensor(features):
        raise TypeError("encoder features must be a torch.Tensor")
    if features.ndim == 4:
        return features.mean(dim=(2, 3))
    if features.ndim == 3:
        return features.mean(dim=1)
    if features.ndim == 2:
        return features
    if features.ndim == 1:
        return features.unsqueeze(0)
    raise ValueError(f"Unsupported feature shape: {tuple(features.shape)}")


def _forward_features(model: torch.nn.Module, images: torch.Tensor) -> torch.Tensor:
    if hasattr(model, "forward_features"):
        features = model.forward_features(images)
    else:
        features = model(images)
    return _pool_features(features)


@torch.no_grad()
def collect_embeddings(
    model: torch.nn.Module,
    loader: DataLoader,
    device: str,
    amp: bool = False,
) -> tuple[list[str], np.ndarray]:
    model.to(device)
    model.eval()
    use_amp = amp and device == "cuda"

    image_ids: list[str] = []
    batches: list[np.ndarray] = []
    with torch.inference_mode():
        for images, _labels, _paths, ids in loader:
            images = images.to(device, non_blocking=True)
            with torch.autocast(device_type="cuda", enabled=use_amp):
                features = _forward_features(model, images)
            batches.append(features.detach().float().cpu().numpy())
            image_ids.extend(str(image_id) for image_id in ids)

    if not batches:
        raise ValueError("No images were provided for embedding export")
    embeddings = np.concatenate(batches, axis=0).astype(np.float32, copy=False)
    if len(image_ids) != embeddings.shape[0]:
        raise ValueError("Collected image ids and embeddings have different row counts")
    return image_ids, embeddings


def create_timm_encoder(
    model_name: str,
    pretrained: bool = True,
    checkpoint_path: Path | None = None,
    device: str = "cpu",
) -> torch.nn.Module:
    import timm

    encoder_name = resolve_encoder_model_name(model_name)
    model = timm.create_model(encoder_name, pretrained=pretrained, num_classes=0)
    if checkpoint_path is not None:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        state = checkpoint.get("model_state", checkpoint)
        if not isinstance(state, Mapping):
            raise ValueError("Embedding checkpoint must be a state dict or contain model_state")
        prepared = prepare_encoder_state(state)
        model_state = model.state_dict()
        compatible_state = select_compatible_encoder_state(prepared, model_state)
        model.load_state_dict(compatible_state, strict=False)
    return model


def build_embedding_loader(
    rows: Sequence[ManifestRow],
    image_size: int,
    batch_size: int,
    num_workers: int,
    transform_backend: str = "auto",
) -> DataLoader:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if image_size <= 0 or not math.isfinite(float(image_size)):
        raise ValueError("image_size must be positive")
    dataset = WeatherImageDataset(
        rows,
        transform=build_transforms(
            image_size=image_size,
            train=False,
            policy="standard",
            backend=transform_backend,
        ),
        return_path=True,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        **_loader_kwargs(num_workers),
    )
