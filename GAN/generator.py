import torch
import torch.nn as nn


class Down(nn.Module):
    def __init__(self, in_ch, out_ch, use_norm=True):
        super().__init__()
        layers = [nn.Conv2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1)]
        if use_norm:
            layers += [nn.BatchNorm2d(out_ch)]
        layers += [nn.LeakyReLU(0.2, inplace=True)]
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class Up(nn.Module):
    def __init__(self, in_ch, out_ch, use_dropout=False):
        super().__init__()
        layers = [
            nn.ConvTranspose2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        ]
        if use_dropout:
            layers += [nn.Dropout(0.5)]
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class UNetGenerator(nn.Module):
    """U-Net generator for 32x32 Lab colorization: L (1 ch) -> ab (2 ch).

    first conv stride 1 (stay 32), bottleneck at 2x2,
    final 1x1 conv + tanh
    """

    def __init__(self):
        super().__init__()
        self.down1 = nn.Conv2d(1, 64, kernel_size=4, stride=1, padding="same") 
        self.down2 = Down(64, 128)
        self.down3 = Down(128, 256)
        self.down4 = Down(256, 512)
        self.down5 = Down(512, 512, use_norm=False)  

        self.up1 = Up(512, 512, use_dropout=True)
        self.up2 = Up(1024, 256, use_dropout=True)
        self.up3 = Up(512, 128)
        self.up4 = Up(256, 64)
        self.out = nn.Sequential(
            nn.Conv2d(128, 2, kernel_size=1, stride=1, padding=0),
            nn.Tanh(),
        )
        self.act = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        d1 = self.act(self.down1(x))
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)

        u1 = self.up1(d5)
        u2 = self.up2(torch.cat([u1, d4], 1))
        u3 = self.up3(torch.cat([u2, d3], 1))
        u4 = self.up4(torch.cat([u3, d2], 1))
        return self.out(torch.cat([u4, d1], 1))
