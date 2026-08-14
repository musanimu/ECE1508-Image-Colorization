"""CVAE V2: conditional-prior Lab colorization (from CIFAR10_CVAE_V2 notebook)."""
#Note: This file was generated using AI based on the ipynb file.
from __future__ import annotations

import torch
from torch import nn


class CVAEV2(nn.Module):
    """Map L (+ latent from p(z|L)) to ab; posterior q(z|L,ab) used only in training."""

    def __init__(self, latent_dim: int = 64) -> None:
        super().__init__()
        self.latent_dim = latent_dim

        self.q1 = nn.Sequential(
            nn.Conv2d(3, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.q2 = nn.Sequential(
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.q3 = nn.Sequential(
            nn.Conv2d(128, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.q_mu = nn.Linear(256 * 4 * 4, latent_dim)
        self.q_logvar = nn.Linear(256 * 4 * 4, latent_dim)

        self.p1 = nn.Sequential(
            nn.Conv2d(1, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.p2 = nn.Sequential(
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.p3 = nn.Sequential(
            nn.Conv2d(128, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        self.p_mu = nn.Linear(256 * 4 * 4, latent_dim)
        self.p_logvar = nn.Linear(256 * 4 * 4, latent_dim)

        self.z_projection = nn.Linear(latent_dim, 256 * 4 * 4)
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(384, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.up3 = nn.Sequential(
            nn.ConvTranspose2d(192, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.out = nn.Sequential(
            nn.Conv2d(65, 64, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 2, 3, 1, 1),
            nn.Tanh(),
        )

    def encode_posterior(self, L: torch.Tensor, ab: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.q3(self.q2(self.q1(torch.cat([L, ab], dim=1)))).flatten(start_dim=1)
        return self.q_mu(h), torch.clamp(self.q_logvar(h), -10.0, 10.0)

    def encode_prior(
        self, L: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        f1 = self.p1(L)
        f2 = self.p2(f1)
        f3 = self.p3(f2)
        flat = f3.flatten(start_dim=1)
        return self.p_mu(flat), torch.clamp(self.p_logvar(flat), -10.0, 10.0), (f1, f2, f3)

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        return mu + torch.exp(0.5 * logvar) * torch.randn_like(logvar)

    def decode_from_features(
        self,
        L: torch.Tensor,
        z: torch.Tensor,
        condition_features: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        f1, f2, f3 = condition_features
        z_features = self.z_projection(z).view(z.size(0), 256, 4, 4)
        h = self.up1(torch.cat([f3, z_features], dim=1))
        h = self.up2(torch.cat([h, f2], dim=1))
        h = self.up3(torch.cat([h, f1], dim=1))
        return self.out(torch.cat([h, L], dim=1))

    def decode_prior(
        self,
        L: torch.Tensor,
        sample: bool = False,
        temperature: float = 1.0,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Inference: default uses prior mean (deterministic), matching notebook eval."""
        mu_p, logvar_p, features = self.encode_prior(L)
        if sample:
            z = mu_p + temperature * torch.exp(0.5 * logvar_p) * torch.randn_like(logvar_p)
        else:
            z = mu_p
        return self.decode_from_features(L, z, features), mu_p, logvar_p

    def forward(
        self, L: torch.Tensor, ab: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        mu_q, logvar_q = self.encode_posterior(L, ab)
        mu_p, logvar_p, features = self.encode_prior(L)
        z = self.reparameterize(mu_q, logvar_q)
        pred_ab = self.decode_from_features(L, z, features)
        return pred_ab, mu_q, logvar_q, mu_p, logvar_p
