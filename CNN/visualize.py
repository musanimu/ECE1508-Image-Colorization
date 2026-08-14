"""Visualize a checkpoint on random train or validation examples without using test data."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

try:
    from CNN.dataset import H5ColorizationDataset
    from CNN.download_data import find_dataset_files
    from CNN.model import CompactUNetColorizer
    from CNN.utils import normalized_lab_to_rgb, select_device, set_seed
except ModuleNotFoundError:
    from dataset import H5ColorizationDataset
    from download_data import find_dataset_files
    from model import CompactUNetColorizer
    from utils import normalized_lab_to_rgb, select_device, set_seed


CLASS_NAMES = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "validation"), required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--num-examples", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.num_examples <= 0:
        raise ValueError("--num-examples must be positive")
    if not args.checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
    files = find_dataset_files(args.data_root)
    h5_path = files["train_lab.h5" if args.split == "train" else "validation_lab.h5"]
    dataset = H5ColorizationDataset(h5_path)
    set_seed(args.seed)
    device = select_device() if args.device == "auto" else torch.device(args.device)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    if args.device == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested but is not available")
    print(f"Selected device: {device}")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model = CompactUNetColorizer().to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    rng = np.random.default_rng(args.seed)
    indices = rng.choice(len(dataset), size=min(args.num_examples, len(dataset)), replace=False)
    records = []
    with torch.no_grad():
        for index in indices:
            sample = dataset[int(index)]
            predicted_ab = model(sample["L"].unsqueeze(0).to(device))[0].cpu()
            records.append({
                "index": int(index),
                "label": int(sample["label"]),
                "L": sample["L"].numpy(),
                "truth": normalized_lab_to_rgb(sample["L"], sample["ab"]),
                "prediction": normalized_lab_to_rgb(sample["L"], predicted_ab),
            })

    columns = 5
    groups = int(np.ceil(len(records) / columns))
    figure, axes = plt.subplots(groups * 3, columns, figsize=(3 * columns, 2.6 * groups * 3), squeeze=False)
    for cell in range(groups * columns):
        for offset in range(3):
            axes[(cell // columns) * 3 + offset, cell % columns].axis("off")
        if cell >= len(records):
            continue
        record = records[cell]
        row = (cell // columns) * 3
        column = cell % columns
        axes[row, column].imshow(
            np.clip((record["L"][0] + 1.0) / 2.0, 0, 1),
            cmap="gray",
            vmin=0,
            vmax=1,
            interpolation="nearest",
        )
        axes[row, column].set_title(f"Input | {CLASS_NAMES[record['label']]} | #{record['index']}", fontsize=8)
        axes[row + 1, column].imshow(record["truth"], interpolation="nearest")
        axes[row + 1, column].set_title("Ground truth", fontsize=8)
        axes[row + 2, column].imshow(record["prediction"], interpolation="nearest")
        axes[row + 2, column].set_title("CNN prediction", fontsize=8)
    figure.suptitle(f"Random {args.split} colorization examples — checkpoint epoch {checkpoint.get('epoch', '?')}")
    figure.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=150, bbox_inches="tight")
    plt.close(figure)
    dataset.close()
    print(f"Saved {len(records)} random {args.split} examples to {args.output}")
    print("The test set was not accessed.")


if __name__ == "__main__":
    main()
