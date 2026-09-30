from src.models import get_model


def test_supported_models():
    for name in ("resnet50", "mobilenet_v2"):
        model = get_model(name)
        assert model is not None
