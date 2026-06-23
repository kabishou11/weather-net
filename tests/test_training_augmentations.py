import torch


def test_apply_batch_augmentations_uses_cutmix_when_enabled() -> None:
    from src.weather_net.training import apply_batch_augmentations

    images = torch.zeros(2, 3, 8, 8)
    images[1] = 1.0
    targets = torch.tensor([0, 1])

    mixed_images, soft_targets = apply_batch_augmentations(
        images,
        targets,
        num_classes=2,
        mixup_alpha=0.0,
        cutmix_alpha=1.0,
        force_mode="cutmix",
    )

    assert mixed_images.shape == images.shape
    assert soft_targets.shape == (2, 2)
    assert torch.allclose(soft_targets.sum(dim=1), torch.ones(2))
    assert not torch.equal(mixed_images, images)


def test_apply_batch_augmentations_returns_one_hot_without_mix() -> None:
    from src.weather_net.training import apply_batch_augmentations

    images = torch.randn(2, 3, 4, 4)
    targets = torch.tensor([1, 0])

    mixed_images, soft_targets = apply_batch_augmentations(
        images,
        targets,
        num_classes=2,
        mixup_alpha=0.0,
        cutmix_alpha=0.0,
    )

    assert torch.equal(mixed_images, images)
    assert torch.equal(soft_targets, torch.tensor([[0.0, 1.0], [1.0, 0.0]]))
