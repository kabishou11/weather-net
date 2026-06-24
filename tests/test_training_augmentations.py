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


def test_make_sampler_sample_weighted_uses_manifest_sample_weights(tmp_path: Path) -> None:
    from src.weather_net.data import ManifestRow
    from src.weather_net.training import make_sampler

    rows = [
        ManifestRow(path=tmp_path / "easy.jpg", label=0, sample_weight=1.0),
        ManifestRow(path=tmp_path / "hard.jpg", label=0, sample_weight=2.5),
        ManifestRow(path=tmp_path / "tail.jpg", label=1, sample_weight=1.0),
    ]

    sampler = make_sampler(rows, sampler_mode="sample_weighted", loss_name="class_balanced_focal")

    assert sampler is not None
    assert list(sampler.weights.tolist()) == [1.0, 2.5, 1.0]


def test_make_sampler_sample_weighted_rejects_invalid_sample_weights(tmp_path: Path) -> None:
    import pytest

    from src.weather_net.data import ManifestRow
    from src.weather_net.training import make_sampler

    rows = [
        ManifestRow(path=tmp_path / "a.jpg", label=0, sample_weight=1.0),
        ManifestRow(path=tmp_path / "b.jpg", label=1, sample_weight=0.0),
    ]

    with pytest.raises(ValueError, match="sample_weight"):
        make_sampler(rows, sampler_mode="sample_weighted")


def test_make_loaders_can_use_sample_weights_for_sampler_only(tmp_path: Path) -> None:
    from src.weather_net.data import ManifestRow
    from src.weather_net.training import make_loaders

    rows = [
        ManifestRow(path=tmp_path / "easy.jpg", label=0, sample_weight=1.0),
        ManifestRow(path=tmp_path / "hard.jpg", label=1, sample_weight=2.5),
    ]

    train_loader, _val_loader = make_loaders(
        train_rows=rows,
        val_rows=rows,
        image_size=8,
        batch_size=2,
        num_workers=0,
        sampler_mode="sample_weighted",
        sample_weight_usage="sampler",
        loss_name="class_balanced_focal",
    )

    assert train_loader.sampler is not None
    assert list(train_loader.sampler.weights.tolist()) == [1.0, 2.5]
    assert train_loader.dataset.rows[1].sample_weight == 1.0


def test_make_loaders_sample_weight_usage_loss_disables_sample_weight_sampler(tmp_path: Path) -> None:
    from src.weather_net.data import ManifestRow
    from src.weather_net.training import make_loaders

    rows = [
        ManifestRow(path=tmp_path / "easy.jpg", label=0, sample_weight=1.0),
        ManifestRow(path=tmp_path / "hard.jpg", label=1, sample_weight=2.5),
    ]

    train_loader, _val_loader = make_loaders(
        train_rows=rows,
        val_rows=rows,
        image_size=8,
        batch_size=2,
        num_workers=0,
        sampler_mode="sample_weighted",
        sample_weight_usage="loss",
        loss_name="class_balanced_focal",
    )

    assert train_loader.batch_sampler.sampler.__class__.__name__ == "RandomSampler"
    assert train_loader.dataset.rows[1].sample_weight == 2.5


def test_make_loaders_sample_weight_usage_both_keeps_sampler_and_loss_weights(tmp_path: Path) -> None:
    from src.weather_net.data import ManifestRow
    from src.weather_net.training import make_loaders

    rows = [
        ManifestRow(path=tmp_path / "easy.jpg", label=0, sample_weight=1.0),
        ManifestRow(path=tmp_path / "hard.jpg", label=1, sample_weight=2.5),
    ]

    train_loader, _val_loader = make_loaders(
        train_rows=rows,
        val_rows=rows,
        image_size=8,
        batch_size=2,
        num_workers=0,
        sampler_mode="sample_weighted",
        sample_weight_usage="both",
        loss_name="class_balanced_focal",
    )

    assert train_loader.sampler is not None
    assert list(train_loader.sampler.weights.tolist()) == [1.0, 2.5]
    assert train_loader.dataset.rows[1].sample_weight == 2.5


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


def test_augmix_jsd_loss_is_zero_for_identical_predictions() -> None:
    from src.weather_net.training import augmix_jsd_loss

    logits = torch.tensor([[3.0, 0.0], [0.0, 3.0]])

    loss = augmix_jsd_loss(logits, logits.clone(), logits.clone())

    assert torch.isclose(loss, torch.tensor(0.0), atol=1e-7)


def test_augmix_jsd_loss_is_positive_for_different_predictions() -> None:
    from src.weather_net.training import augmix_jsd_loss

    clean = torch.tensor([[4.0, 0.0], [0.0, 4.0]])
    aug1 = torch.tensor([[0.0, 4.0], [0.0, 4.0]])
    aug2 = torch.tensor([[4.0, 0.0], [4.0, 0.0]])

    loss = augmix_jsd_loss(clean, aug1, aug2)

    assert torch.isfinite(loss)
    assert loss > 0


def test_augmix_jsd_loss_matches_augmix_reference_direction() -> None:
    from src.weather_net.training import augmix_jsd_loss

    clean = torch.tensor([[3.0, 0.0]])
    aug1 = torch.tensor([[0.0, 2.0]])
    aug2 = torch.tensor([[1.0, 1.0]])
    clean_probs = torch.softmax(clean, dim=1)
    aug1_probs = torch.softmax(aug1, dim=1)
    aug2_probs = torch.softmax(aug2, dim=1)
    mixture_log_probs = ((clean_probs + aug1_probs + aug2_probs) / 3.0).clamp_min(1e-7).log()
    expected = (
        torch.nn.functional.kl_div(mixture_log_probs, clean_probs, reduction="batchmean")
        + torch.nn.functional.kl_div(mixture_log_probs, aug1_probs, reduction="batchmean")
        + torch.nn.functional.kl_div(mixture_log_probs, aug2_probs, reduction="batchmean")
    ) / 3.0

    loss = augmix_jsd_loss(clean, aug1, aug2)

    assert torch.isclose(loss, expected, atol=1e-7)


def test_augmix_jsd_loss_respects_sample_weights() -> None:
    from src.weather_net.training import augmix_jsd_loss

    clean = torch.tensor([[4.0, 0.0], [4.0, 0.0]])
    aug1 = torch.tensor([[0.0, 4.0], [4.0, 0.0]])
    aug2 = torch.tensor([[0.0, 4.0], [4.0, 0.0]])

    unweighted = augmix_jsd_loss(clean, aug1, aug2)
    weighted = augmix_jsd_loss(clean, aug1, aug2, sample_weights=torch.tensor([0.1, 1.0]))

    assert weighted < unweighted


def test_split_augmix_jsd_batch_unpacks_three_views() -> None:
    from src.weather_net.training import split_augmix_jsd_batch

    clean = torch.zeros(2, 3, 4, 4)
    aug1 = torch.ones(2, 3, 4, 4)
    aug2 = torch.full((2, 3, 4, 4), 2.0)

    base, extra_views = split_augmix_jsd_batch([clean, aug1, aug2])

    assert torch.equal(base, clean)
    assert len(extra_views) == 2
    assert torch.equal(extra_views[0], aug1)
    assert torch.equal(extra_views[1], aug2)


def test_train_one_epoch_consumes_augmix_jsd_batches() -> None:
    from torch import nn
    from torch.utils.data import DataLoader, Dataset

    from src.weather_net.training import train_one_epoch

    clean = torch.randn(4, 3, 4, 4)
    aug1 = clean + 0.05
    aug2 = clean - 0.05
    targets = torch.tensor([0, 1, 0, 1])
    sample_weights = torch.ones(4)

    class ThreeViewDataset(Dataset):
        def __len__(self) -> int:
            return len(targets)

        def __getitem__(self, index: int):
            return (clean[index], aug1[index], aug2[index]), targets[index], sample_weights[index]

    loader = DataLoader(ThreeViewDataset(), batch_size=2)

    class TinyClassifier(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.net = nn.Sequential(nn.Flatten(), nn.Linear(3 * 4 * 4, 2))

        def forward(self, images: torch.Tensor) -> torch.Tensor:
            return self.net(images)

    model = TinyClassifier()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    scaler = torch.amp.GradScaler("cuda", enabled=False)

    loss = train_one_epoch(
        model,
        loader,
        optimizer,
        scaler,
        device="cpu",
        num_classes=2,
        label_smoothing=0.0,
        mixup_alpha=0.0,
        cutmix_alpha=0.0,
        amp=False,
        jsd_weight=1.0,
    )

    assert loss > 0


def test_train_one_epoch_rejects_jsd_weight_for_single_view_batches() -> None:
    import pytest
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset

    from src.weather_net.training import train_one_epoch

    loader = DataLoader(
        TensorDataset(
            torch.randn(2, 3, 4, 4),
            torch.tensor([0, 1]),
            torch.ones(2),
        ),
        batch_size=2,
    )
    model = nn.Sequential(nn.Flatten(), nn.Linear(3 * 4 * 4, 2))
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    scaler = torch.amp.GradScaler("cuda", enabled=False)

    with pytest.raises(ValueError, match="augmix_jsd"):
        train_one_epoch(
            model,
            loader,
            optimizer,
            scaler,
            device="cpu",
            num_classes=2,
            label_smoothing=0.0,
            mixup_alpha=0.0,
            cutmix_alpha=0.0,
            amp=False,
            jsd_weight=1.0,
        )
