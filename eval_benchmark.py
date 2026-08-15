"""Cross-model colorization benchmark: metrics + comparison samples.

Arguments:
  --seed
  --num-samples

Examples (from final_project/):
  python eval_benchmark.py
  python eval_benchmark.py --seed 42 --num-samples 100
"""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch
from skimage.color import lab2rgb
from skimage.metrics import peak_signal_noise_ratio

from CNN.model import CompactUNetColorizer
from CVAE.model import CVAEV2
from GAN.model import CGAN
from GAN.train import CGAN_CKPT, load_generator_weights
from download_data import resolve_lab_data_dir

ROOT = Path(__file__).resolve().parent
DATA_DIR = resolve_lab_data_dir(ROOT)
OUT_DIR = ROOT / "eval_benchmark"
CNN_CKPT = ROOT / "CNN" / "best_model.pt"
CVAE_CKPT = ROOT / "CVAE" / "cvae_v2_final_inference.pt"
SPLIT = "test"


def lab_norm_to_rgb(L_chw: np.ndarray, ab_chw: np.ndarray) -> np.ndarray:
    """Convert normalized Lab (L in [-1,1], ab in [-1,1]) to RGB in [0, 1]."""
    lab = np.stack(
        [(L_chw[0] + 1.0) * 50.0, ab_chw[0] * 128.0, ab_chw[1] * 128.0],
        axis=-1,
    )
    return np.clip(lab2rgb(lab.astype(np.float64)), 0, 1)


def colorfulness_hasler(rgb01: np.ndarray) -> float:
    """Hasler & Süsstrunk (2003) colorfulness. rgb01 is HWC float in [0, 1]."""
    rgb = np.clip(rgb01, 0.0, 1.0).astype(np.float64) * 255.0
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    rg = r - g
    yb = 0.5 * (r + g) - b
    std_rgyb = float(np.sqrt(rg.std() ** 2 + yb.std() ** 2))
    mean_rgyb = float(np.sqrt(rg.mean() ** 2 + yb.mean() ** 2))
    return std_rgyb + 0.3 * mean_rgyb


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Cross-model colorization benchmark.")
    p.add_argument("--seed", type=int, default=45)
    p.add_argument("--num-samples", type=int, default=20, help="Images for metrics and compare grid")
    return p.parse_args()


def choose_indices(n_total: int, k: int, rng: np.random.Generator) -> np.ndarray:
    if k > n_total:
        raise ValueError(f"need {k} samples but split only has {n_total}")
    return np.sort(rng.choice(n_total, size=k, replace=False).astype(np.int64))


def load_lab_batch(h5_path: Path, idxs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Load L/ab; preserve idxs display order (h5py needs increasing fancy index)."""
    order_sort = np.argsort(idxs)
    idxs_sorted = idxs[order_sort]
    inv_sort = np.empty_like(order_sort)
    inv_sort[order_sort] = np.arange(len(order_sort))
    with h5py.File(h5_path, "r") as f:
        L = np.asarray(f["L"][idxs_sorted], dtype=np.float32)[inv_sort]
        ab = np.asarray(f["ab"][idxs_sorted], dtype=np.float32)[inv_sort]
    return L, ab


@torch.no_grad()
def predict_ab(model: torch.nn.Module, L: np.ndarray, device: torch.device) -> np.ndarray:
    """CNN: model(L) -> ab with eval() (BN running stats, dropout off)."""
    model.eval()
    return model(torch.from_numpy(L).to(device)).cpu().numpy()


def enable_dropout(model: torch.nn.Module) -> None:
    """Keep BN/eval behavior but force Dropout layers into train mode."""
    model.eval()
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.train()


@torch.no_grad()
def predict_cgan_ab(model: CGAN, L: np.ndarray, device: torch.device) -> np.ndarray:
    """CGAN: L -> ab with generator dropout enabled (BN still uses running stats)."""
    enable_dropout(model)
    return model(torch.from_numpy(L).to(device)).cpu().numpy()


@torch.no_grad()
def predict_cvae_ab(model: CVAEV2, L: np.ndarray, device: torch.device) -> np.ndarray:
    """CVAE V2 eval mode: decode from conditional prior mean (deterministic)."""
    model.eval()
    pred_ab, _, _ = model.decode_prior(torch.from_numpy(L).to(device), sample=False)
    return pred_ab.cpu().numpy()


def rgb_batch(L: np.ndarray, ab: np.ndarray) -> np.ndarray:
    return np.stack([lab_norm_to_rgb(L[i], ab[i]) for i in range(L.shape[0])], axis=0)


def compute_batch_metrics(
    pred_ab: np.ndarray,
    gt_ab: np.ndarray,
    pred_rgb: np.ndarray,
    gt_rgb: np.ndarray,
) -> dict[str, float]:
    """MAE/MSE on normalized ab [-1,1]; PSNR on RGB [0,1]; colorful Hasler 0-255."""
    diff_ab = pred_ab - gt_ab
    mae = float(np.mean(np.abs(diff_ab)))
    mse = float(np.mean(diff_ab**2))
    psnrs = [
        float(peak_signal_noise_ratio(gt_rgb[i], pred_rgb[i], data_range=1.0))
        for i in range(pred_rgb.shape[0])
    ]
    color_pred = [colorfulness_hasler(pred_rgb[i]) for i in range(pred_rgb.shape[0])]
    color_gt = [colorfulness_hasler(gt_rgb[i]) for i in range(gt_rgb.shape[0])]
    return {
        "mae": mae,
        "mse": mse,
        "psnr": float(np.mean(psnrs)),
        "colorful_pred": float(np.mean(color_pred)),
        "colorful_gt": float(np.mean(color_gt)),
        "n": int(pred_ab.shape[0]),
    }


def format_metrics_txt(
    *,
    model_name: str,
    ckpt: Path,
    extra_header: list[str],
    split: str,
    seed: int,
    device: torch.device,
    idxs: np.ndarray,
    metrics: dict[str, float],
) -> list[str]:
    return [
        f"model={model_name}",
        f"ckpt={ckpt}",
        *extra_header,
        f"split={split}  seed={seed}  n={metrics['n']}  device={device}",
        f"indices={','.join(str(int(i)) for i in idxs)}",
        "",
        "Normalized ab metrics (pred vs GT, range [-1,1]):",
        f"  MAE  = {metrics['mae']:.6f}",
        f"  MSE  = {metrics['mse']:.6f}",
        "",
        "RGB PSNR (pred vs GT, range [0,1]):",
        f"  PSNR = {metrics['psnr']:.4f} dB",
        "",
        "Hasler-Süsstrunk colorfulness (RGB scaled to 0 - 255):",
        f"  colorful_pred = {metrics['colorful_pred']:.4f}",
        f"  colorful_gt   = {metrics['colorful_gt']:.4f}",
    ]


def save_compare_grid(
    out_path: Path,
    L: np.ndarray,
    cgan_rgb: np.ndarray,
    gt_rgb: np.ndarray,
    h5_idxs: np.ndarray,
    title: str,
    cnn_rgb: np.ndarray | None = None,
    cvae_rgb: np.ndarray | None = None,
) -> None:
    """Rows: L | CNN | CVAE | CGAN | GT."""
    n = L.shape[0]
    col_specs = [
        ("L", "gray"),
        ("CNN", "cnn"),
        ("CVAE", "cvae"),
        ("CGAN", "cgan"),
        ("GT", "gt"),
    ]
    n_cols = len(col_specs)
    fig, axes = plt.subplots(n, n_cols, figsize=(2.0 * n_cols, 2.0 * n))
    if n == 1:
        axes = np.expand_dims(axes, 0)

    for row in range(n):
        gray = (L[row, 0] + 1.0) / 2.0
        for c, (name, kind) in enumerate(col_specs):
            ax = axes[row, c]
            if kind == "gray":
                ax.imshow(gray, cmap="gray", vmin=0, vmax=1)
            elif kind == "gt":
                ax.imshow(gt_rgb[row])
            elif kind == "cgan":
                ax.imshow(cgan_rgb[row])
            elif kind == "cnn":
                if cnn_rgb is not None:
                    ax.imshow(cnn_rgb[row])
                else:
                    ax.set_facecolor("#e8e8e8")
                    ax.text(0.5, 0.5, "—", ha="center", va="center", transform=ax.transAxes, fontsize=14)
            elif kind == "cvae":
                if cvae_rgb is not None:
                    ax.imshow(cvae_rgb[row])
                else:
                    ax.set_facecolor("#e8e8e8")
                    ax.text(0.5, 0.5, "—", ha="center", va="center", transform=ax.transAxes, fontsize=14)
            ax.axis("off")
            if row == 0:
                ax.set_title(name, fontsize=10)
        axes[row, 0].set_ylabel(
            f"h5={int(h5_idxs[row])}",
            fontsize=7,
            rotation=0,
            labelpad=36,
            va="center",
        )

    fig.suptitle(title, fontsize=11)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def load_cnn(device: torch.device) -> tuple[CompactUNetColorizer, dict]:
    if not CNN_CKPT.is_file():
        raise FileNotFoundError(f"CNN checkpoint not found: {CNN_CKPT}")
    ckpt = torch.load(CNN_CKPT, map_location=device, weights_only=False)
    model = CompactUNetColorizer().to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, ckpt


def load_cvae(device: torch.device) -> tuple[CVAEV2, dict]:
    if not CVAE_CKPT.is_file():
        raise FileNotFoundError(f"CVAE checkpoint not found: {CVAE_CKPT}")
    ckpt = torch.load(CVAE_CKPT, map_location=device, weights_only=False)
    latent_dim = int(ckpt.get("config", {}).get("latent_dim", 64))
    model = CVAEV2(latent_dim=latent_dim).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, ckpt


def load_cgan(device: torch.device) -> tuple[CGAN, dict]:
    model = CGAN(patch_size=4, lambda_l1=100.0).to(device)
    ckpt = load_generator_weights(model, CGAN_CKPT, device)
    model.eval()
    return model, ckpt


def main() -> None:
    args = parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cnn, cnn_ckpt = load_cnn(device)
    cvae, cvae_ckpt = load_cvae(device)
    cgan, cgan_ckpt = load_cgan(device)

    h5_path = DATA_DIR / f"{SPLIT}_lab.h5"
    rng = np.random.default_rng(args.seed)
    with h5py.File(h5_path, "r") as f:
        n_total = int(f["L"].shape[0])

    idxs = choose_indices(n_total, args.num_samples, rng)
    L, ab_gt = load_lab_batch(h5_path, idxs)
    gt_rgb = rgb_batch(L, ab_gt)

    cnn_ab = predict_ab(cnn, L, device)
    cvae_ab = predict_cvae_ab(cvae, L, device)
    cgan_ab = predict_cgan_ab(cgan, L, device)

    cnn_rgb = rgb_batch(L, cnn_ab)
    cvae_rgb = rgb_batch(L, cvae_ab)
    cgan_rgb = rgb_batch(L, cgan_ab)

    cnn_metrics = compute_batch_metrics(cnn_ab, ab_gt, cnn_rgb, gt_rgb)
    cvae_metrics = compute_batch_metrics(cvae_ab, ab_gt, cvae_rgb, gt_rgb)
    cgan_metrics = compute_batch_metrics(cgan_ab, ab_gt, cgan_rgb, gt_rgb)

    n = args.num_samples
    (OUT_DIR / f"indices_{n:03d}.txt").write_text(
        "\n".join(str(int(i)) for i in idxs) + "\n"
    )

    cnn_lines = format_metrics_txt(
        model_name="CNN",
        ckpt=CNN_CKPT,
        extra_header=[
            f"epoch={cnn_ckpt.get('epoch')}  "
            f"ckpt_train_loss={cnn_ckpt.get('train_loss')}  "
            f"ckpt_val_loss={cnn_ckpt.get('validation_loss')}",
        ],
        split=SPLIT,
        seed=args.seed,
        device=device,
        idxs=idxs,
        metrics=cnn_metrics,
    )
    cnn_metrics_path = OUT_DIR / f"cnn_metrics_{n:03d}.txt"
    cnn_metrics_path.write_text("\n".join(cnn_lines) + "\n")

    cvae_lines = format_metrics_txt(
        model_name="CVAE",
        ckpt=CVAE_CKPT,
        extra_header=[
            f"epoch={cvae_ckpt.get('best_epoch')}  "
            f"ckpt_best_selection={cvae_ckpt.get('best_selection')}  "
            f"mode=decode_prior(sample=False)",
        ],
        split=SPLIT,
        seed=args.seed,
        device=device,
        idxs=idxs,
        metrics=cvae_metrics,
    )
    cvae_metrics_path = OUT_DIR / f"cvae_metrics_{n:03d}.txt"
    cvae_metrics_path.write_text("\n".join(cvae_lines) + "\n")

    cgan_lines = format_metrics_txt(
        model_name="CGAN",
        ckpt=CGAN_CKPT,
        extra_header=[
            f"epoch={cgan_ckpt.get('epoch')}  "
            f"ckpt_val_lpips={cgan_ckpt.get('val_lpips')}  "
            f"ckpt_val_l1={cgan_ckpt.get('val_l1')}",
            "inference=dropout_on (BN eval / Dropout train)",
        ],
        split=SPLIT,
        seed=args.seed,
        device=device,
        idxs=idxs,
        metrics=cgan_metrics,
    )
    cgan_metrics_path = OUT_DIR / f"cgan_metrics_{n:03d}.txt"
    cgan_metrics_path.write_text("\n".join(cgan_lines) + "\n")

    compare_path = OUT_DIR / f"compare_{n:03d}.png"
    save_compare_grid(
        compare_path,
        L,
        cgan_rgb,
        gt_rgb,
        idxs,
        title=f"CNN | CVAE | CGAN  seed={args.seed}  n={n}",
        cnn_rgb=cnn_rgb,
        cvae_rgb=cvae_rgb,
    )

    print(f"wrote {cnn_metrics_path}")
    print(f"wrote {cvae_metrics_path}")
    print(f"wrote {cgan_metrics_path}")
    print(f"wrote {compare_path}")
    print("--- CNN ---")
    for line in cnn_lines:
        print(line)
    print("--- CVAE ---")
    for line in cvae_lines:
        print(line)
    print("--- CGAN ---")
    for line in cgan_lines:
        print(line)


if __name__ == "__main__":
    main()
