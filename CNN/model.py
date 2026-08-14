"""Compact U-Net regression baseline for 32 x 32 colorization."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class DoubleConv(nn.Sequential):
    """Two 3x3 convolution, batch-normalization, ReLU stages."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )


class CompactUNetColorizer(nn.Module):
    """Map normalized CIELAB L images to normalized continuous ab channels."""

    def __init__(self) -> None:
        super().__init__()
        self.encoder1 = DoubleConv(1, 32)
        self.encoder2 = DoubleConv(32, 64)
        self.encoder3 = DoubleConv(64, 128)
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = DoubleConv(128, 256)
        self.decoder1 = DoubleConv(256 + 128, 128)
        self.decoder2 = DoubleConv(128 + 64, 64)
        self.decoder3 = DoubleConv(64 + 32, 32)
        self.output_conv = nn.Conv2d(32, 2, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.encoder1(x)
        e2 = self.encoder2(self.pool(e1))
        e3 = self.encoder3(self.pool(e2))
        bottleneck = self.bottleneck(self.pool(e3))

        d1 = F.interpolate(bottleneck, scale_factor=2, mode="bilinear", align_corners=False)
        d1 = self.decoder1(torch.cat((d1, e3), dim=1))
        d2 = F.interpolate(d1, scale_factor=2, mode="bilinear", align_corners=False)
        d2 = self.decoder2(torch.cat((d2, e2), dim=1))
        d3 = F.interpolate(d2, scale_factor=2, mode="bilinear", align_corners=False)
        d3 = self.decoder3(torch.cat((d3, e1), dim=1))
        return torch.tanh(self.output_conv(d3))


def count_trainable_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


if __name__ == "__main__":
    colorizer = CompactUNetColorizer()
    output = colorizer(torch.randn(4, 1, 32, 32))
    assert output.shape == (4, 2, 32, 32)
    print(f"Output shape: {tuple(output.shape)}")
    print(f"Trainable parameters: {count_trainable_parameters(colorizer):,}")
