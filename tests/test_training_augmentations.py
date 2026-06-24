from pathlib import Path

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


def test_apply_batch_augmentations_mixes_sample_weights_for_cutmix() -> None:
    from src.weather_net.training import apply_batch_augmentations

    torch.manual_seed(0)
    images = torch.zeros(2, 3, 8, 8)
    images[1] = 1.0
    targets = torch.tensor([0, 1])
    sample_weights = torch.tensor([1.0, 0.4])

    _mixed_images, soft_targets, weighted_targets = apply_batch_augmentations(
        images,
        targets,
        num_classes=2,
        mixup_alpha=0.0,
        cutmix_alpha=1.0,
        sample_weights=sample_weights,
        force_mode="cutmix",
    )

    assert soft_targets.shape == (2, 2)
    assert weighted_targets.shape == soft_targets.shape
    assert torch.all(weighted_targets <= soft_targets)
    assert not torch.allclose(weighted_targets, soft_targets)
    assert torch.any(weighted_targets.sum(dim=1) < soft_targets.sum(dim=1))


def test_apply_batch_augmentations_uses_component_sample_weights_for_mixup(monkeypatch) -> None:
    from src.weather_net import training

    images = torch.zeros(2, 3, 4, 4)
    images[1] = 1.0
    targets = torch.tensor([0, 1])
    sample_weights = torch.tensor([1.0, 0.4])

    monkeypatch.setattr(training.np.random, "beta", lambda _alpha, _beta: 0.25)
    monkeypatch.setattr(
        training.torch,
        "randperm",
        lambda _size, device=None: torch.tensor([1, 0], device=device),
    )

    _mixed_images, soft_targets, weighted_targets = training.apply_batch_augmentations(
        images,
        targets,
        num_classes=2,
        mixup_alpha=1.0,
        cutmix_alpha=0.0,
        sample_weights=sample_weights,
        force_mode="mixup",
    )

    assert torch.allclose(soft_targets, torch.tensor([[0.25, 0.75], [0.75, 0.25]]))
    assert torch.allclose(weighted_targets, torch.tensor([[0.25, 0.30], [0.75, 0.10]]))


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


def test_compute_class_weights_boosts_tail_classes() -> None:
    from src.weather_net.training import compute_class_weights

    weights = compute_class_weights(labels=[0, 0, 0, 1], num_classes=2, loss_name="class_balanced", beta=0.99)

    assert weights.shape == (2,)
    assert weights[1] > weights[0]
    assert torch.isclose(weights.mean(), torch.tensor(1.0), atol=1e-6)


def test_make_sampler_auto_disables_weighted_sampling_for_class_balanced_loss(tmp_path: Path) -> None:
    from src.weather_net.data import ManifestRow
    from src.weather_net.training import make_sampler

    rows = [
        ManifestRow(path=tmp_path / "a.jpg", label=0),
        ManifestRow(path=tmp_path / "b.jpg", label=0),
        ManifestRow(path=tmp_path / "c.jpg", label=1),
    ]

    assert make_sampler(rows, sampler_mode="auto", loss_name="class_balanced_focal") is None
    assert make_sampler(rows, sampler_mode="weighted", loss_name="class_balanced_focal") is not None


def test_weighted_soft_cross_entropy_supports_focal_tail_weighting() -> None:
    from src.weather_net.training import weighted_soft_cross_entropy

    logits = torch.tensor([[4.0, 0.0], [0.2, 0.0]])
    soft_targets = torch.tensor([[1.0, 0.0], [1.0, 0.0]])
    no_focal = weighted_soft_cross_entropy(logits, soft_targets, focal_gamma=0.0, reduction="none")
    focal = weighted_soft_cross_entropy(logits, soft_targets, focal_gamma=2.0, reduction="none")

    assert focal[0] < no_focal[0]
    assert focal[1] > focal[0]


def test_weighted_soft_cross_entropy_applies_class_weights_to_soft_targets() -> None:
    from src.weather_net.training import weighted_soft_cross_entropy

    logits = torch.tensor([[2.0, 0.0]])
    soft_targets = torch.tensor([[0.75, 0.25]])
    unweighted = weighted_soft_cross_entropy(logits, soft_targets)
    weighted = weighted_soft_cross_entropy(
        logits,
        soft_targets,
        class_weights=torch.tensor([1.0, 3.0]),
    )

    assert weighted > unweighted


def test_weighted_soft_cross_entropy_normalizes_class_weight_scale() -> None:
    from src.weather_net.training import weighted_soft_cross_entropy

    logits = torch.tensor([[0.0, 0.0]])
    soft_targets = torch.tensor([[1.0, 0.0]])

    base = weighted_soft_cross_entropy(logits, soft_targets)
    scaled = weighted_soft_cross_entropy(logits, soft_targets, class_weights=torch.tensor([10.0, 10.0]))

    assert torch.isclose(base, scaled, atol=1e-6)


def test_weighted_soft_cross_entropy_normalizes_embedded_target_weights() -> None:
    from src.weather_net.training import weighted_soft_cross_entropy

    logits = torch.tensor([[0.0, 0.0]])
    weighted_targets = torch.tensor([[0.4, 0.0]])

    loss = weighted_soft_cross_entropy(logits, weighted_targets)

    assert torch.isclose(loss, torch.tensor(0.6931472), atol=1e-6)
