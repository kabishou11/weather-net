import torch


def test_create_classifier_outputs_expected_shape() -> None:
    from src.weather_net.models import create_classifier

    model = create_classifier(
        model_name="resnet18",
        num_classes=4,
        pretrained=False,
    )

    model.eval()
    with torch.no_grad():
        logits = model(torch.randn(2, 3, 64, 64))

    assert logits.shape == (2, 4)


def test_weather_expert_classifier_outputs_expected_shape() -> None:
    from src.weather_net.models import create_classifier

    model = create_classifier(
        model_name="weather_expert:small_cnn",
        num_classes=3,
        pretrained=False,
    )

    model.eval()
    with torch.no_grad():
        logits = model(torch.randn(2, 3, 64, 64))

    assert logits.shape == (2, 3)
