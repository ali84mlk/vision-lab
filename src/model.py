"""Models for image classification via TRANSFER LEARNING.

Training a CNN from scratch needs a lot of data and compute. A model pretrained on ImageNet has
already learned general visual features (edges, textures, shapes). We keep that "backbone" and only
replace the last layer (the "head") so it outputs OUR number of classes. That is why a free T4 GPU
(or even a CPU, for small experiments) is enough.
"""

from __future__ import annotations

import torch.nn as nn
from torchvision import models

MODEL_NAMES = ("resnet18", "mobilenet_v3_small", "efficientnet_b0")


def _head(model: nn.Module, name: str) -> nn.Module:
    """The final classification layer(s) of each architecture (what we replace and always train)."""
    return model.fc if name == "resnet18" else model.classifier


def build_model(name: str, num_classes: int, pretrained: bool = True, freeze_backbone: bool = False) -> nn.Module:
    """Create `name`, swap its final layer for `num_classes` outputs, optionally freeze the backbone."""
    if name not in MODEL_NAMES:
        raise ValueError(f"unknown model '{name}', choose one of {MODEL_NAMES}")

    if name == "resnet18":
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        model = models.resnet18(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif name == "mobilenet_v3_small":
        weights = models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        model = models.mobilenet_v3_small(weights=weights)
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
    else:  # efficientnet_b0
        weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        model = models.efficientnet_b0(weights=weights)
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)

    if freeze_backbone:
        set_backbone_trainable(model, name, False)
    return model


def set_backbone_trainable(model: nn.Module, name: str, trainable: bool) -> None:
    """Freeze (False) or unfreeze (True) everything except the head.

    A frozen parameter has requires_grad=False, so autograd computes no gradient for it and the
    optimizer leaves it untouched (PyTorch optimizers skip parameters whose .grad is None).
    A common recipe: train only the head for a few epochs (fast, stable), then unfreeze and
    fine-tune everything with a smaller learning rate.
    """
    head_ids = {id(p) for p in _head(model, name).parameters()}
    for p in model.parameters():
        p.requires_grad = True if id(p) in head_ids else trainable


def count_parameters(model: nn.Module) -> tuple[int, int]:
    """Return (trainable, total) parameter counts."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return trainable, total
