import torch


def test_combine_tta_logits_rejects_lower_confidence_view() -> None:
    from src.weather_net.inference import combine_tta_logits

    base_logits = torch.tensor([[5.0, 1.0, 0.0]])
    augmented_logits = torch.tensor([[2.0, 1.8, 0.1]])

    combined = combine_tta_logits(base_logits, augmented_logits)

    assert torch.allclose(combined, base_logits)


def test_combine_tta_logits_rejects_top1_change_with_higher_entropy() -> None:
    from src.weather_net.inference import combine_tta_logits

    base_logits = torch.tensor([[3.5, 1.0, 0.0]])
    augmented_logits = torch.tensor([[2.0, 2.2, 1.8]])

    combined = combine_tta_logits(base_logits, augmented_logits)

    assert torch.allclose(combined, base_logits)


def test_combine_tta_logits_averages_consistent_stronger_view() -> None:
    from src.weather_net.inference import combine_tta_logits

    base_logits = torch.tensor([[3.0, 1.0, 0.0]])
    augmented_logits = torch.tensor([[4.0, 0.5, -0.5]])

    combined = combine_tta_logits(base_logits, augmented_logits)

    assert torch.allclose(combined, (base_logits + augmented_logits) / 2)
