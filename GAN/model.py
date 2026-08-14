import torch
import torch.nn as nn

from .generator import UNetGenerator
from .discriminator import build_discriminator


class CGAN(nn.Module):
    def __init__(self, patch_size=4, lambda_l1=100.0, label_smoothing=0.0):
        super().__init__()
        self.G = UNetGenerator()
        self.D = build_discriminator(patch_size=patch_size)
        self.criterion_gan = nn.BCEWithLogitsLoss()
        self.criterion_l1 = nn.L1Loss()
        self.lambda_l1 = lambda_l1
        self.label_smoothing = float(label_smoothing)

    def forward(self, L):
        return self.G(L)

    def discriminator_loss(self, L, ab_real, ab_fake):
        pred_real = self.D(torch.cat([L, ab_real], 1))
        real_tgt = torch.ones_like(pred_real) * (1.0 - self.label_smoothing)
        loss_real = self.criterion_gan(pred_real, real_tgt)

        pred_fake = self.D(torch.cat([L, ab_fake.detach()], 1))
        loss_fake = self.criterion_gan(pred_fake, torch.zeros_like(pred_fake))

        return 0.5 * (loss_real + loss_fake)

    def generator_loss(self, L, ab_real, ab_fake):
        pred_fake = self.D(torch.cat([L, ab_fake], 1))
        loss_gan = self.criterion_gan(pred_fake, torch.ones_like(pred_fake))
        loss_l1 = self.criterion_l1(ab_fake, ab_real) * self.lambda_l1
        return loss_gan + loss_l1
