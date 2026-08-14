import torch.nn as nn


class Patch4Discriminator(nn.Module):
    """RF = 4: each logit classifies a 4x4 neighborhood. Channels 64 -> 128 -> 1."""

    def __init__(self):
        super().__init__()
        self.patch_size = 4
        self.model = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=4, stride=1, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, kernel_size=1, stride=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 1, kernel_size=1, stride=1),
        )

    def forward(self, x):
        return self.model(x)


class Patch16Discriminator(nn.Module):
    """RF ~= 16: each logit classifies a ~16x16 neighborhood. Channels 64 -> 128 -> 1."""

    def __init__(self):
        super().__init__()
        self.patch_size = 16
        self.model = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, kernel_size=4, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 1, kernel_size=4, stride=1, padding=1),
        )

    def forward(self, x):
        return self.model(x)


class ImageDiscriminator(nn.Module):
    """RF = 32: one score for the whole 32x32 image.

    Deeper pix2pix-style stack: 64 -> 128 -> 256 -> 512 -> 1.
    """

    def __init__(self):
        super().__init__()
        self.patch_size = 32
        self.model = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, kernel_size=4, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=1),
            nn.AdaptiveAvgPool2d(1), 
        )

    def forward(self, x):
        return self.model(x)


def build_discriminator(patch_size=4):
    table = {
        4: Patch4Discriminator,
        16: Patch16Discriminator,
        32: ImageDiscriminator,
    }
    return table[patch_size]()
