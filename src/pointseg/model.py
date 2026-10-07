"""The segmentation network.

DeepLabV3+ with an ImageNet-pretrained ResNet-50 encoder, via ``segmentation_models_pytorch``.

The architecture is deliberately an ordinary, strong baseline rather than anything novel. Every
experiment here varies supervision, not capacity, so the model has to be held constant and
uninteresting for the comparison to mean anything. The pretrained encoder matters more than
usual under point supervision: with twenty labelled pixels per image there is nowhere near
enough signal to learn good low-level features from scratch, so almost all of the visual
prior has to arrive with the weights.
"""
from __future__ import annotations

import segmentation_models_pytorch as smp
import torch.nn as nn

from pointseg.config import CFG, NC


def build_model(encoder: str | None = None, num_classes: int = NC,
                encoder_weights: str | None = "imagenet") -> nn.Module:
    """DeepLabV3+ / ResNet-50, roughly 26.7 M parameters."""
    return smp.DeepLabV3Plus(
        encoder_name=encoder or CFG["encoder"],
        encoder_weights=encoder_weights,
        in_channels=3,
        classes=num_classes,
    )


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
