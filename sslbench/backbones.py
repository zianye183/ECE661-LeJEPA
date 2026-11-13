"""Backbones return (module, feature_dim). The module maps images to flat features."""

from __future__ import annotations

from torch import nn
from torchvision.models import resnet18


def resnet18_cifar() -> tuple[nn.Module, int]:
    """ResNet-18 with the usual CIFAR stem: 3x3 conv, stride 1, no max-pool."""
    net = resnet18(weights=None)
    net.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    net.maxpool = nn.Identity()
    net.fc = nn.Identity()
    return net, 512


def build_backbone(name: str) -> tuple[nn.Module, int]:
    if name == "resnet18_cifar":
        return resnet18_cifar()
    if name.startswith("timm:"):  # e.g. timm:vit_tiny_patch16_224 (stretch goal only)
        import timm

        net = timm.create_model(name[5:], pretrained=False, num_classes=0, img_size=32, patch_size=4)
        return net, net.num_features
    raise ValueError(f"Unknown backbone {name!r}")
