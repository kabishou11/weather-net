from PIL import Image
import pytest


def test_weather_augmix_transform_returns_tensor_shape() -> None:
    from src.weather_net.datasets import build_transforms

    transform = build_transforms(
        image_size=32,
        train=True,
        policy="weather_augmix",
        backend="torchvision",
    )

    tensor = transform(Image.new("RGB", (40, 40), (120, 140, 160)))

    assert tuple(tensor.shape) == (3, 32, 32)


def test_build_transforms_rejects_unknown_backend() -> None:
    from src.weather_net.datasets import build_transforms

    with pytest.raises(ValueError, match="backend"):
        build_transforms(image_size=32, train=True, policy="standard", backend="mystery")


def test_build_transforms_rejects_unknown_policy() -> None:
    from src.weather_net.datasets import build_transforms

    with pytest.raises(ValueError, match="policy"):
        build_transforms(image_size=32, train=True, policy="unknown", backend="torchvision")
