"""Train conditional GAN Lab colorization on the preprocessed CIFAR-10 H5 splits."""
# Note: This file was mainly generated using AI to enable parameter knobs to be tuned.

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import h5py
import lpips
import matplotlib.pyplot as plt
import numpy as np
import torch
from skimage.color import lab2rgb
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

from .model import CGAN


PATCH_SIZE = 32
D_LR_MULT = 0.1
BETA1 = 0.0
NUM_WORKERS = 2
SEED = 42
NUM_SAMPLES = 16
CGAN_CKPT = Path(__file__).resolve().parent / "CGAN.pt"

DATA_DIR = None  # resolved at startup via download_data.resolve_lab_data_dir()


def get_data_dir() -> Path:
    global DATA_DIR
    if DATA_DIR is None:
        try:
            from download_data import resolve_lab_data_dir
        except ImportError:
            import sys

            sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
            from download_data import resolve_lab_data_dir

        DATA_DIR = resolve_lab_data_dir(Path(__file__).resolve().parent.parent)
    return DATA_DIR


class LabH5Dataset(Dataset):
    """Reads L/ab tensors written by preprocess_image_colorization_data.py."""

    def __init__(self, h5_path: Path):
        self.h5_path = Path(h5_path)
        with h5py.File(self.h5_path, "r") as f:
            self.length = int(f["L"].shape[0])
        self._file = None

    def __len__(self) -> int:
        return self.length

    def _h5(self):
        # reopen per worker process after DataLoader fork
        if self._file is None:
            self._file = h5py.File(self.h5_path, "r")
        return self._file

    def __getitem__(self, index: int):
        f = self._h5()
        L = torch.from_numpy(np.asarray(f["L"][index], dtype=np.float32))
        ab = torch.from_numpy(np.asarray(f["ab"][index], dtype=np.float32))
        return L, ab


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train CGAN colorization on CIFAR-10 Lab.")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--out-dir", type=Path, default=Path(__file__).resolve().parent / "checkpoints")
    p.add_argument("--lr", type=float, default=2e-4, help="Generator Adam LR (D uses 0.1 * lr)")
    p.add_argument("--lambda-l1", type=float, default=100.0)
    p.add_argument("--lr-decay", action="store_true", help="Linearly decay LR to 0 over this run")
    p.add_argument(
        "--label-smoothing",
        type=float,
        default=0.0,
        help="One-sided real-label smoothing for D (e.g. 0.1 -> real targets 0.9)",
    )
    p.add_argument(
        "--resume",
        type=Path,
        default=CGAN_CKPT,
        help=f"Load generator weights only (default: {CGAN_CKPT.name}); skipped if file missing",
    )
    p.add_argument(
        "--from-scratch",
        action="store_true",
        help="Ignore --resume / CGAN.pt and train a new generator",
    )
    return p.parse_args()


def lab_norm_to_rgb(L_chw: np.ndarray, ab_chw: np.ndarray) -> np.ndarray:
    lab = np.stack([(L_chw[0] + 1.0) * 50.0, ab_chw[0] * 128.0, ab_chw[1] * 128.0], axis=-1)
    return np.clip(lab2rgb(lab.astype(np.float64)), 0, 1)


def lab_batch_to_rgb_m11(L: torch.Tensor, ab: torch.Tensor) -> torch.Tensor:
    """Convert normalized Lab batch to RGB in [-1, 1] for LPIPS (Bx3xHxW)."""
    L_np = L.detach().cpu().numpy()
    ab_np = ab.detach().cpu().numpy()
    rgbs = []
    for i in range(L_np.shape[0]):
        rgb = lab_norm_to_rgb(L_np[i], ab_np[i])  # HWC [0,1]
        rgbs.append(torch.from_numpy(rgb.transpose(2, 0, 1).astype(np.float32)))
    return torch.stack(rgbs, dim=0) * 2.0 - 1.0


def load_generator_weights(model: CGAN, ckpt_path: Path, device: torch.device) -> dict:
    """Load G weights from a G-only or legacy full-model checkpoint."""
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    if isinstance(ckpt, dict) and "G" in ckpt:
        g_state = ckpt["G"]
    elif isinstance(ckpt, dict) and "model" in ckpt:
        g_state = {
            k.replace("G.", "", 1): v
            for k, v in ckpt["model"].items()
            if k.startswith("G.")
        }
        if not g_state:
            raise KeyError(f"{ckpt_path} has 'model' but no G.* keys")
    elif isinstance(ckpt, dict) and all(isinstance(v, torch.Tensor) for v in ckpt.values()):
        g_state = ckpt
    else:
        raise KeyError(f"unrecognized checkpoint format: {ckpt_path}")
    model.G.load_state_dict(g_state, strict=True)
    return ckpt if isinstance(ckpt, dict) else {"G": g_state}


def save_generator_ckpt(
    path: Path,
    model: CGAN,
    epoch: int,
    args: argparse.Namespace,
    metrics: dict[str, float],
) -> None:
    torch.save(
        {
            "epoch": epoch,
            "G": model.G.state_dict(),
            "args": vars(args),
            "val_l1": metrics["val_l1"],
            "val_lpips": metrics["val_lpips"],
            "val_chroma_pred": metrics["val_chroma_pred"],
            "val_chroma_gt": metrics["val_chroma_gt"],
            "val_chroma_l1": metrics["val_chroma_l1"],
        },
        path,
    )


@torch.no_grad()
def eval_metrics(
    model: CGAN,
    loader: DataLoader,
    device: torch.device,
    lpips_fn: lpips.LPIPS,
) -> dict[str, float]:
    """val_L1 (ab), LPIPS (RGB), mean chroma, chroma L1."""
    model.eval()
    sum_l1 = 0.0
    n_ab = 0
    sum_lpips = 0.0
    n_img = 0
    sum_chroma_pred = 0.0
    sum_chroma_gt = 0.0
    sum_chroma_l1 = 0.0
    n_pix = 0

    for L, ab in loader:
        L = L.to(device)
        ab = ab.to(device)
        ab_fake = model(L)

        sum_l1 += torch.nn.functional.l1_loss(ab_fake, ab, reduction="sum").item()
        n_ab += ab.numel()

        chroma_pred = torch.sqrt(ab_fake[:, 0] ** 2 + ab_fake[:, 1] ** 2)
        chroma_gt = torch.sqrt(ab[:, 0] ** 2 + ab[:, 1] ** 2)
        sum_chroma_pred += chroma_pred.sum().item()
        sum_chroma_gt += chroma_gt.sum().item()
        sum_chroma_l1 += (chroma_pred - chroma_gt).abs().sum().item()
        n_pix += chroma_gt.numel()

        rgb_fake = lab_batch_to_rgb_m11(L, ab_fake).to(device)
        rgb_real = lab_batch_to_rgb_m11(L, ab).to(device)
        d = lpips_fn(rgb_fake, rgb_real)
        sum_lpips += d.sum().item()
        n_img += d.numel()

    model.train()
    return {
        "val_l1": sum_l1 / max(n_ab, 1),
        "val_lpips": sum_lpips / max(n_img, 1),
        "val_chroma_pred": sum_chroma_pred / max(n_pix, 1),
        "val_chroma_gt": sum_chroma_gt / max(n_pix, 1),
        "val_chroma_l1": sum_chroma_l1 / max(n_pix, 1),
    }


@torch.no_grad()
def save_epoch_samples(
    model: CGAN,
    sample_L: torch.Tensor,
    sample_ab: torch.Tensor,
    out_path: Path,
    device: torch.device,
) -> None:
    model.eval()
    L = sample_L.to(device)
    ab_fake = model(L).cpu().numpy()
    L_np = sample_L.numpy()
    ab_np = sample_ab.numpy()
    n = L_np.shape[0]

    fig, axes = plt.subplots(n, 3, figsize=(6, 2 * n))
    if n == 1:
        axes = np.expand_dims(axes, 0)
    for row in range(n):
        gray = (L_np[row, 0] + 1.0) / 2.0
        axes[row, 0].imshow(gray, cmap="gray", vmin=0, vmax=1)
        axes[row, 1].imshow(lab_norm_to_rgb(L_np[row], ab_fake[row]))
        axes[row, 2].imshow(lab_norm_to_rgb(L_np[row], ab_np[row]))
        for c in range(3):
            axes[row, c].axis("off")
        if row == 0:
            axes[0, 0].set_title("L")
            axes[0, 1].set_title("pred")
            axes[0, 2].set_title("GT")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    model.train()


def main() -> None:
    args = parse_args()
    torch.manual_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_dir = get_data_dir()
    print(f"device={device}  data={data_dir}")

    train_set = LabH5Dataset(data_dir / "train_lab.h5")
    val_set = LabH5Dataset(data_dir / "validation_lab.h5")
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=device.type == "cuda",
    )

    rng = np.random.default_rng(SEED)
    sample_idxs = rng.choice(len(val_set), size=min(NUM_SAMPLES, len(val_set)), replace=False)
    sample_L = torch.stack([val_set[int(i)][0] for i in sample_idxs])
    sample_ab = torch.stack([val_set[int(i)][1] for i in sample_idxs])
    samples_dir = args.out_dir / "samples"

    model = CGAN(
        patch_size=PATCH_SIZE,
        lambda_l1=args.lambda_l1,
        label_smoothing=args.label_smoothing,
    ).to(device)

    start_epoch = 1
    if not args.from_scratch and args.resume is not None and args.resume.exists():
        ckpt = load_generator_weights(model, args.resume, device)
        print(
            f"resumed G-only from {args.resume}  "
            f"ckpt_epoch={ckpt.get('epoch')}  "
            f"prev_val_L1={ckpt.get('val_l1')}  prev_val_LPIPS={ckpt.get('val_lpips')}  "
            f"(fresh D + optimizers, epoch starts at 1)"
        )
    elif args.from_scratch:
        print("training generator from scratch (--from-scratch)")
    else:
        print(f"no checkpoint at {args.resume}; training generator from scratch")

    betas = (BETA1, 0.999)
    opt_G = torch.optim.Adam(model.G.parameters(), lr=args.lr, betas=betas)
    opt_D = torch.optim.Adam(model.D.parameters(), lr=args.lr * D_LR_MULT, betas=betas)
    end_epoch = start_epoch + args.epochs - 1

    def lr_lambda(step_idx: int) -> float:
        if not args.lr_decay:
            return 1.0
        return max(0.0, 1.0 - step_idx / float(args.epochs))

    sched_G = torch.optim.lr_scheduler.LambdaLR(opt_G, lr_lambda=lr_lambda)
    sched_D = torch.optim.lr_scheduler.LambdaLR(opt_D, lr_lambda=lr_lambda)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    lpips_fn = lpips.LPIPS(net="alex").to(device)
    lpips_fn.eval()
    best_lpips = float("inf")
    best_l1 = float("inf")
    print(
        f"train epochs {start_epoch}..{end_epoch}  lr_G={args.lr}  "
        f"lr_D={args.lr * D_LR_MULT}  beta1={BETA1}  "
        f"lambda_l1={args.lambda_l1}  label_smoothing={args.label_smoothing}  "
        f"patch_size={PATCH_SIZE}  batch_size={args.batch_size}  "
        f"lr_decay={args.lr_decay}  out={args.out_dir}"
    )
    print("checkpointing: G weights only — updates GAN/CGAN.pt on best val_LPIPS; also latest.pt each epoch")

    for epoch in range(start_epoch, end_epoch + 1):
        model.train()
        model.lambda_l1 = args.lambda_l1
        t0 = time.time()
        sum_d = 0.0
        sum_g = 0.0
        steps = 0

        pbar = tqdm(train_loader, desc=f"epoch {epoch}/{end_epoch}", leave=False)
        for L, ab_real in pbar:
            L = L.to(device, non_blocking=True)
            ab_real = ab_real.to(device, non_blocking=True)

            ab_fake = model(L)

            opt_D.zero_grad(set_to_none=True)
            loss_d = model.discriminator_loss(L, ab_real, ab_fake)
            loss_d.backward()
            opt_D.step()

            opt_G.zero_grad(set_to_none=True)
            ab_fake = model(L)
            loss_g = model.generator_loss(L, ab_real, ab_fake)
            loss_g.backward()
            opt_G.step()

            sum_d += loss_d.item()
            sum_g += loss_g.item()
            steps += 1
            pbar.set_postfix(D=f"{loss_d.item():.3f}", G=f"{loss_g.item():.3f}")

        sched_G.step()
        sched_D.step()

        metrics = eval_metrics(model, val_loader, device, lpips_fn)
        avg_d = sum_d / max(steps, 1)
        avg_g = sum_g / max(steps, 1)
        lr_now = opt_G.param_groups[0]["lr"]
        print(
            f"epoch {epoch:03d}  "
            f"loss_D={avg_d:.4f}  loss_G={avg_g:.4f}  "
            f"val_L1={metrics['val_l1']:.6f}  "
            f"val_LPIPS={metrics['val_lpips']:.6f}  "
            f"chroma_pred={metrics['val_chroma_pred']:.4f}  "
            f"chroma_gt={metrics['val_chroma_gt']:.4f}  "
            f"chroma_L1={metrics['val_chroma_l1']:.4f}  "
            f"lr={lr_now:.2e}  time={time.time() - t0:.1f}s"
        )

        metrics_csv = args.out_dir / "metrics.csv"
        row = {
            "epoch": epoch,
            "loss_D": avg_d,
            "loss_G": avg_g,
            "val_L1": metrics["val_l1"],
            "val_LPIPS": metrics["val_lpips"],
            "chroma_pred": metrics["val_chroma_pred"],
            "chroma_gt": metrics["val_chroma_gt"],
            "chroma_L1": metrics["val_chroma_l1"],
            "lr": lr_now,
            "time_s": time.time() - t0,
        }
        write_header = not metrics_csv.exists()
        with metrics_csv.open("a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(row.keys()))
            if write_header:
                w.writeheader()
            w.writerow(row)

        sample_path = samples_dir / f"epoch_{epoch:04d}.png"
        save_epoch_samples(model, sample_L, sample_ab, sample_path, device)
        print(f"  wrote {sample_path}")

        save_generator_ckpt(args.out_dir / "latest.pt", model, epoch, args, metrics)
        if metrics["val_lpips"] < best_lpips:
            best_lpips = metrics["val_lpips"]
            save_generator_ckpt(CGAN_CKPT, model, epoch, args, metrics)
            print(f"  saved {CGAN_CKPT} (val_LPIPS={best_lpips:.6f})")
        if metrics["val_l1"] < best_l1:
            best_l1 = metrics["val_l1"]

    print(
        f"done. best val_LPIPS={best_lpips:.6f}  best val_L1={best_l1:.6f}  "
        f"best weights: {CGAN_CKPT}  latest: {args.out_dir / 'latest.pt'}"
    )


if __name__ == "__main__":
    main()
