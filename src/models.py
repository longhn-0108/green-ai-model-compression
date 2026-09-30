import torch.nn as nn
import torchvision.models as models


def get_model(model_name: str, num_classes: int = 100) -> nn.Module:
    """Build a CIFAR-100-compatible ResNet-50 or MobileNetV2."""
    if model_name == "resnet50":
        model = models.resnet50(weights="IMAGENET1K_V1")
        model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        model.maxpool = nn.Identity()
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model

    if model_name == "mobilenet_v2":
        model = models.mobilenet_v2(weights="IMAGENET1K_V1")
        old_conv = model.features[0][0]
        model.features[0][0] = nn.Conv2d(
            old_conv.in_channels,
            old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=1,
            padding=old_conv.padding,
            bias=False,
        )
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
        return model

    raise ValueError(f"Unsupported model: {model_name}")
