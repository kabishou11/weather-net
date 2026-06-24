from __future__ import annotations

import io
import random
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter
from torch.utils.data import Dataset

from .data import ManifestRow


def load_rgb_image(path: Path) -> Image.Image:
    with Image.open(path) as image:
        return image.convert("RGB")


class WeatherImageDataset(Dataset):
    def __init__(
        self,
        rows: Sequence[ManifestRow],
        transform: Callable | None = None,
        return_path: bool = False,
    ) -> None:
        self.rows = list(rows)
        self.transform = transform
        self.return_path = return_path

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        row = self.rows[index]
        image = load_rgb_image(row.path)
        if self.transform is not None:
            image = self.transform(image)
        label = -1 if row.label is None else row.label
        sample_weight = float(row.sample_weight)
        if self.return_path:
            return image, label, str(row.path), row.image_id or row.path.name
        teacher_probs = torch.tensor(row.teacher_probs or (), dtype=torch.float32)
        return image, label, sample_weight, teacher_probs


class AlbumentationsAdapter:
    def __init__(self, transform) -> None:
        self.transform = transform

    def __call__(self, image: Image.Image) -> torch.Tensor:
        array = np.asarray(image)
        return self.transform(image=array)["image"]


class WeatherAugMixPIL:
    def __init__(self, p: float = 0.75) -> None:
        self.p = p

    def __call__(self, image: Image.Image) -> Image.Image:
        if random.random() > self.p:
            return image
        ops = [
            self._fog,
            self._rain,
            self._low_light,
            self._glare,
            self._jpeg,
            self._blur,
        ]
        for op in random.sample(ops, k=random.randint(1, 3)):
            image = op(image)
        return image

    def _fog(self, image: Image.Image) -> Image.Image:
        fog = Image.new("RGB", image.size, (220, 225, 230))
        return Image.blend(image, fog, random.uniform(0.12, 0.32))

    def _rain(self, image: Image.Image) -> Image.Image:
        image = image.copy()
        draw = ImageDraw.Draw(image)
        width, height = image.size
        for _ in range(max(8, width * height // 1800)):
            x = random.randint(0, width)
            y = random.randint(0, height)
            length = random.randint(5, 14)
            draw.line((x, y, x + 2, y + length), fill=(210, 220, 230), width=1)
        return image.filter(ImageFilter.GaussianBlur(radius=0.25))

    def _low_light(self, image: Image.Image) -> Image.Image:
        image = ImageEnhance.Brightness(image).enhance(random.uniform(0.45, 0.85))
        return ImageEnhance.Contrast(image).enhance(random.uniform(0.9, 1.25))

    def _glare(self, image: Image.Image) -> Image.Image:
        image = image.copy()
        overlay = Image.new("RGB", image.size, (0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        width, height = image.size
        cx = random.randint(0, width)
        cy = random.randint(0, height)
        radius = random.randint(max(4, min(width, height) // 8), max(5, min(width, height) // 3))
        draw.ellipse((cx - radius, cy - radius, cx + radius, cy + radius), fill=(255, 245, 220))
        return Image.blend(image, overlay, random.uniform(0.08, 0.18))

    def _jpeg(self, image: Image.Image) -> Image.Image:
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=random.randint(35, 75))
        buffer.seek(0)
        return Image.open(buffer).convert("RGB")

    def _blur(self, image: Image.Image) -> Image.Image:
        return image.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.2, 1.1)))


class ThreeViewTransform:
    def __init__(self, clean_transform: Callable, aug_transform: Callable) -> None:
        self.clean_transform = clean_transform
        self.aug_transform = aug_transform

    def __call__(self, image: Image.Image) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            self.clean_transform(image),
            self.aug_transform(image),
            self.aug_transform(image),
        )


def _validate_transform_options(policy: str, backend: str) -> None:
    if policy not in {"standard", "weather_augmix", "heavy", "augmix_jsd"}:
        raise ValueError("policy must be one of: standard, weather_augmix, heavy, augmix_jsd")
    if backend not in {"auto", "albumentations", "torchvision"}:
        raise ValueError("backend must be one of: auto, albumentations, torchvision")


def _build_torchvision_transforms(image_size: int, train: bool, policy: str) -> Callable:
    from torchvision import transforms

    if train:
        ops: list[Callable] = [
            transforms.RandomResizedCrop(image_size, scale=(0.65 if policy == "heavy" else 0.72, 1.0)),
            transforms.RandomHorizontalFlip(),
        ]
        if policy in {"weather_augmix", "heavy"}:
            ops.append(WeatherAugMixPIL(p=0.9 if policy == "heavy" else 0.75))
        ops.extend(
            [
                transforms.ColorJitter(
                    brightness=0.25 if policy == "heavy" else 0.2,
                    contrast=0.25 if policy == "heavy" else 0.2,
                    saturation=0.18,
                    hue=0.04,
                ),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225),
                ),
                transforms.RandomErasing(p=0.28 if policy == "heavy" else 0.2),
            ]
        )
        return transforms.Compose(ops)
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            ),
        ]
    )


def _build_albumentations_transforms(image_size: int, train: bool, policy: str) -> Callable:
    try:
        import albumentations as A
        from albumentations.pytorch import ToTensorV2

        if train:
            blur_noise = A.OneOf(
                [
                    A.MotionBlur(blur_limit=5),
                    A.GaussianBlur(blur_limit=5),
                    A.GaussNoise(var_limit=(5.0, 25.0)),
                ],
                p=0.25,
            )
            weather_ops = []
            if policy in {"weather_augmix", "heavy"}:
                weather_ops = [
                    A.RandomFog(fog_coef_lower=0.08, fog_coef_upper=0.22, alpha_coef=0.08, p=0.22),
                    A.RandomRain(blur_value=3, brightness_coefficient=0.9, p=0.18),
                    A.RandomSunFlare(src_radius=32, p=0.08),
                    A.ImageCompression(quality_lower=35, quality_upper=80, p=0.18),
                ]
            transform = A.Compose(
                [
                    A.RandomResizedCrop(
                        size=(image_size, image_size),
                        scale=(0.65 if policy == "heavy" else 0.72, 1.0),
                        ratio=(0.75, 1.33),
                    ),
                    A.HorizontalFlip(p=0.5),
                    A.RandomBrightnessContrast(p=0.55),
                    A.HueSaturationValue(p=0.25),
                    *weather_ops,
                    blur_noise,
                    A.CoarseDropout(
                        max_holes=8,
                        max_height=max(8, image_size // 12),
                        max_width=max(8, image_size // 12),
                        p=0.25,
                    ),
                    A.Normalize(),
                    ToTensorV2(),
                ]
            )
        else:
            transform = A.Compose(
                [
                    A.Resize(image_size, image_size),
                    A.Normalize(),
                    ToTensorV2(),
                ]
            )
        return AlbumentationsAdapter(transform)
    except ImportError:
        raise


def build_transforms(
    image_size: int,
    train: bool,
    policy: str = "standard",
    backend: str = "auto",
) -> Callable:
    _validate_transform_options(policy, backend)
    if policy == "augmix_jsd":
        if not train:
            policy = "standard"
        elif backend == "torchvision":
            return ThreeViewTransform(
                clean_transform=_build_torchvision_transforms(image_size=image_size, train=True, policy="standard"),
                aug_transform=_build_torchvision_transforms(image_size=image_size, train=True, policy="weather_augmix"),
            )
        elif backend == "albumentations":
            return ThreeViewTransform(
                clean_transform=_build_albumentations_transforms(image_size=image_size, train=True, policy="standard"),
                aug_transform=_build_albumentations_transforms(image_size=image_size, train=True, policy="weather_augmix"),
            )
        else:
            try:
                return ThreeViewTransform(
                    clean_transform=_build_albumentations_transforms(
                        image_size=image_size,
                        train=True,
                        policy="standard",
                    ),
                    aug_transform=_build_albumentations_transforms(
                        image_size=image_size,
                        train=True,
                        policy="weather_augmix",
                    ),
                )
            except ImportError:
                return ThreeViewTransform(
                    clean_transform=_build_torchvision_transforms(
                        image_size=image_size,
                        train=True,
                        policy="standard",
                    ),
                    aug_transform=_build_torchvision_transforms(
                        image_size=image_size,
                        train=True,
                        policy="weather_augmix",
                    ),
                )
    if backend == "torchvision":
        return _build_torchvision_transforms(image_size=image_size, train=train, policy=policy)
    if backend == "albumentations":
        return _build_albumentations_transforms(image_size=image_size, train=train, policy=policy)
    try:
        return _build_albumentations_transforms(image_size=image_size, train=train, policy=policy)
    except ImportError:
        return _build_torchvision_transforms(image_size=image_size, train=train, policy=policy)
