from __future__ import annotations

from typing import Any

import torch
from torch import nn


class SmallConvNet(nn.Module):
    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 24, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(24),
            nn.SiLU(inplace=True),
            nn.Conv2d(24, 48, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(48),
            nn.SiLU(inplace=True),
            nn.Conv2d(48, 96, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(96),
            nn.SiLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Linear(96, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return self.classifier(torch.flatten(x, 1))


class GeMPool2d(nn.Module):
    def __init__(self, p: float = 3.0, eps: float = 1e-6) -> None:
        super().__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.clamp(min=self.eps).pow(self.p)
        x = torch.nn.functional.adaptive_avg_pool2d(x, 1).pow(1.0 / self.p)
        return torch.flatten(x, 1)


class WeatherExpertSmallNet(nn.Module):
    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.stage1 = nn.Sequential(
            nn.Conv2d(3, 24, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(24),
            nn.SiLU(inplace=True),
        )
        self.stage2 = nn.Sequential(
            nn.Conv2d(24, 48, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(48),
            nn.SiLU(inplace=True),
        )
        self.stage3 = nn.Sequential(
            nn.Conv2d(48, 96, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(96),
            nn.SiLU(inplace=True),
        )
        self.pool = GeMPool2d()
        self.attention = nn.Sequential(
            nn.Linear(48 + 96, 64),
            nn.SiLU(inplace=True),
            nn.Linear(64, 2),
            nn.Softmax(dim=1),
        )
        self.classifier = nn.Linear(48 + 96, num_classes)
        self.context_head = nn.Linear(48 + 96, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stage1(x)
        mid = self.stage2(x)
        high = self.stage3(mid)
        mid_vec = self.pool(mid)
        high_vec = self.pool(high)
        fused = torch.cat([mid_vec, high_vec], dim=1)
        weights = self.attention(fused)
        fused = torch.cat([mid_vec * weights[:, :1], high_vec * weights[:, 1:]], dim=1)
        return self.classifier(fused)


class WeatherExpertTimm(nn.Module):
    def __init__(self, backbone_name: str, num_classes: int, pretrained: bool, **kwargs: Any) -> None:
        super().__init__()
        import timm

        self.backbone = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            features_only=True,
            out_indices=(-2, -1),
            **kwargs,
        )
        channels = self.backbone.feature_info.channels()
        self.pool = GeMPool2d()
        fused_dim = int(sum(channels))
        self.attention = nn.Sequential(
            nn.Linear(fused_dim, max(64, fused_dim // 4)),
            nn.SiLU(inplace=True),
            nn.Linear(max(64, fused_dim // 4), len(channels)),
            nn.Softmax(dim=1),
        )
        self.classifier = nn.Linear(fused_dim, num_classes)
        self.context_head = nn.Linear(fused_dim, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        pooled = [self.pool(feature) for feature in features]
        fused = torch.cat(pooled, dim=1)
        weights = self.attention(fused)
        weighted = torch.cat(
            [feature * weights[:, idx : idx + 1] for idx, feature in enumerate(pooled)],
            dim=1,
        )
        return self.classifier(weighted)


def create_weather_expert(
    backbone_name: str,
    num_classes: int,
    pretrained: bool,
    **kwargs: Any,
) -> nn.Module:
    if backbone_name == "small_cnn":
        return WeatherExpertSmallNet(num_classes=num_classes)
    return WeatherExpertTimm(
        backbone_name=backbone_name,
        num_classes=num_classes,
        pretrained=pretrained,
        **kwargs,
    )


def create_classifier(
    model_name: str,
    num_classes: int,
    pretrained: bool = True,
    **kwargs: Any,
) -> nn.Module:
    if num_classes <= 0:
        raise ValueError("num_classes must be positive")
    if model_name.startswith("weather_expert:"):
        return create_weather_expert(
            backbone_name=model_name.split(":", 1)[1],
            num_classes=num_classes,
            pretrained=pretrained,
            **kwargs,
        )

    try:
        import timm

        return timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=num_classes,
            **kwargs,
        )
    except Exception as timm_exc:
        try:
            from torchvision import models

            if model_name == "resnet18":
                weights = models.ResNet18_Weights.DEFAULT if pretrained else None
                model = models.resnet18(weights=weights)
                model.fc = nn.Linear(model.fc.in_features, num_classes)
                return model
        except Exception:
            pass

        if model_name in {"small_cnn", "resnet18"}:
            return SmallConvNet(num_classes=num_classes)
        raise RuntimeError(
            "Unable to create model. Install timm for competition models such as "
            "convnext_tiny, efficientnet_b3, or swin_tiny."
        ) from timm_exc
