"""Shared training, device, reproducibility, and visualization utilities."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from skimage.color import lab2rgb


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def select_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def normalized_lab_to_rgb(L: torch.Tensor | np.ndarray, ab: torch.Tensor | np.ndarray) -> np.ndarray:
    """Reverse the shared normalization and convert one CHW Lab sample to RGB."""
    L_array = L.detach().cpu().numpy() if isinstance(L, torch.Tensor) else np.asarray(L)
    ab_array = ab.detach().cpu().numpy() if isinstance(ab, torch.Tensor) else np.asarray(ab)
    if L_array.shape[0] != 1 or ab_array.shape[0] != 2 or L_array.shape[1:] != ab_array.shape[1:]:
        raise ValueError(f"Expected L [1,H,W] and ab [2,H,W], got {L_array.shape}, {ab_array.shape}")
    lab = np.concatenate(((L_array + 1.0) * 50.0, ab_array * 128.0), axis=0)
    return np.clip(lab2rgb(np.transpose(lab, (1, 2, 0))), 0.0, 1.0)


def save_json(data: Any, path: Path) -> None:
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def save_loss_curve(history: dict[str, Any], path: Path) -> None:
    epochs = range(1, len(history["train_loss"]) + 1)
    figure, axis = plt.subplots(figsize=(7, 5))
    axis.plot(epochs, history["train_loss"], label="Training")
    axis.plot(epochs, history["validation_loss"], label="Validation")
    axis.set(xlabel="Epoch", ylabel="Reconstruction loss", title="CNN training history")
    axis.grid(alpha=0.3)
    axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)
