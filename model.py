import torch
import torch.nn as nn
from torchvision.models import mobilenet_v2

def get_cifar10_mobilenetv2(num_classes: int = 10, width_mult: float = 1.0, dropout: float = 0.2) -> nn.Module:
    """
    Adapts torchvision MobileNetV2 for 32x32 CIFAR-10 inputs by reducing early
    downsampling strides so the final feature map before pooling is 4x4 instead of 1x1.
    """
    # [expansion_factor, out_channels, num_blocks, stride]
    cifar_inverted_residual_setting = [
        [1, 16, 1, 1],
        [6, 24, 2, 1],  # Changed stride 2 -> 1 for 32x32 input
        [6, 32, 3, 2],  # 32x32 -> 16x16
        [6, 64, 4, 2],  # 16x16 -> 8x8
        [6, 96, 3, 1],
        [6, 160, 3, 2], # 8x8 -> 4x4
        [6, 320, 1, 1],
    ]

    model = mobilenet_v2(
        weights=None,
        num_classes=num_classes,
        width_mult=width_mult,
        dropout=dropout,
        inverted_residual_setting=cifar_inverted_residual_setting,
    )

    # Change stem Conv2d stride from 2 to 1 for 32x32 images
    model.features[0][0].stride = (1, 1)
    return model