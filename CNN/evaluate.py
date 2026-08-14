"""Evaluate a trained CNN checkpoint on a shared test HDF5 file."""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from torch.utils.data import DataLoader

try:
    from CNN.dataset import H5ColorizationDataset
    from CNN.download_data import find_dataset_files
    from CNN.model import CompactUNetColorizer, count_trainable_parameters
    from CNN.utils import normalized_lab_to_rgb, save_json, select_device, set_seed
except ModuleNotFoundError:
    from dataset import H5ColorizationDataset
    from download_data import find_dataset_files
    from model import CompactUNetColorizer, count_trainable_parameters
    from utils import normalized_lab_to_rgb, save_json, select_device, set_seed


CLASS_NAMES = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--test-h5", type=Path)
    source.add_argument("--data-root", type=Path, help="Search this downloaded Drive directory for test_lab.h5.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--random-examples", type=int, default=20)
    parser.add_argument("--extreme-examples", type=int, default=10)
    return parser.parse_args()


def save_grid(records: list[dict], indices: list[int], path: Path, title: str) -> None:
    if not indices:
        return
    figure, axes = plt.subplots(
        len(indices), 3, figsize=(8, 2.5 * len(indices)), squeeze=False
    )
    column_titles = ("L input", "CNN prediction", "Ground truth")
    for column, column_title in enumerate(column_titles):
        axes[0, column].set_title(column_title, fontsize=12, fontweight="bold")

    for row, index in enumerate(indices):
        record = records[index]
        L = record["L"]
        for axis in axes[row]:
            axis.axis("off")
        axes[row, 0].imshow(
            np.clip((L[0] + 1.0) / 2.0, 0, 1),
            cmap="gray",
            vmin=0,
            vmax=1,
            interpolation="nearest",
        )
        axes[row, 1].imshow(
            record["prediction_rgb"], interpolation="nearest"
        )
        axes[row, 2].imshow(record["truth_rgb"], interpolation="nearest")
        axes[row, 0].set_ylabel(
            f"{CLASS_NAMES[record['label']]} #{index}", fontsize=8
        )
        axes[row, 1].set_xlabel(f"ab MSE {record['mse']:.4f}", fontsize=8)
    figure.suptitle(title, fontsize=14, y=0.998)
    figure.tight_layout(rect=(0, 0, 1, 0.995))
    figure.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(figure)


def save_grid_pages(
    records: list[dict],
    indices: list[int],
    path: Path,
    title: str,
    examples_per_page: int = 20,
) -> None:
    """Save readable pages while preserving the requested total sample count."""
    if len(indices) <= examples_per_page:
        save_grid(records, indices, path, title)
        return
    page_count = int(np.ceil(len(indices) / examples_per_page))
    for page_index in range(page_count):
        start = page_index * examples_per_page
        page_indices = indices[start : start + examples_per_page]
        page_path = path.with_name(
            f"{path.stem}_{page_index + 1:02d}{path.suffix}"
        )
        save_grid(
            records,
            page_indices,
            page_path,
            f"{title} — page {page_index + 1}/{page_count}",
        )


def summary(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {"mean": float(array.mean()), "standard_deviation": float(array.std())}


def synchronize(device: torch.device) -> None:
    """Wait for asynchronous accelerator work before timing boundaries."""
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0 or args.num_workers < 0:
        raise ValueError("batch size must be positive and workers cannot be negative")
    if args.data_root is not None:
        args.test_h5 = find_dataset_files(args.data_root)["test_lab.h5"]
    if not args.checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)
    device = select_device()
    print(f"Selected device: {device}")
    dataset = H5ColorizationDataset(args.test_h5)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=device.type == "cuda")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model = CompactUNetColorizer().to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    records: list[dict] = []
    inference_seconds = 0.0
    with torch.no_grad():
        for batch in loader:
            L_device = batch["L"].to(device, non_blocking=True)
            synchronize(device)
            start = time.perf_counter()
            prediction = model(L_device)
            synchronize(device)
            inference_seconds += time.perf_counter() - start
            predictions = prediction.cpu()
            for L, target, predicted, label in zip(batch["L"], batch["ab"], predictions, batch["label"]):
                truth_rgb = normalized_lab_to_rgb(L, target)
                prediction_rgb = normalized_lab_to_rgb(L, predicted)
                difference = predicted.numpy() - target.numpy()
                records.append({
                    "L": L.numpy(), "label": int(label), "truth_rgb": truth_rgb,
                    "prediction_rgb": prediction_rgb, "mse": float(np.mean(difference ** 2)),
                    "mae": float(np.mean(np.abs(difference))),
                    "psnr": float(peak_signal_noise_ratio(truth_rgb, prediction_rgb, data_range=1.0)),
                    "ssim": float(structural_similarity(truth_rgb, prediction_rgb, data_range=1.0, channel_axis=-1)),
                })

    metrics = {
        "Normalized ab-space MSE": summary([record["mse"] for record in records]),
        "Normalized ab-space MAE": summary([record["mae"] for record in records]),
        "RGB PSNR": summary([record["psnr"] for record in records]),
        "RGB SSIM": summary([record["ssim"] for record in records]),
        "average_inference_time_per_image_seconds": inference_seconds / len(dataset),
        "total_test_images": len(dataset),
        "trainable_parameters": count_trainable_parameters(model),
        "checkpoint_epoch": checkpoint.get("epoch"),
    }
    save_json(metrics, args.output_dir / "metrics.json")
    with (args.output_dir / "metrics.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(("metric", "mean", "standard_deviation"))
        for name in ("Normalized ab-space MSE", "Normalized ab-space MAE", "RGB PSNR", "RGB SSIM"):
            writer.writerow((name, metrics[name]["mean"], metrics[name]["standard_deviation"]))

    rng = np.random.default_rng(args.seed)
    random_indices = rng.choice(len(records), size=min(args.random_examples, len(records)), replace=False).tolist()
    ordered = sorted(range(len(records)), key=lambda index: records[index]["mse"])
    count = min(args.extreme_examples, len(records))
    class_indices = [next(index for index, record in enumerate(records) if record["label"] == label) for label in sorted({record["label"] for record in records})]
    save_grid_pages(
        records,
        random_indices,
        args.output_dir / "random_examples.png",
        "Random test examples",
    )
    save_grid(records, ordered[:count], args.output_dir / "best_examples.png", "Lowest normalized ab-space error")
    save_grid(records, ordered[-count:][::-1], args.output_dir / "failure_examples.png", "Highest normalized ab-space error")
    save_grid(records, class_indices, args.output_dir / "class_examples.png", "Examples across CIFAR-10 classes")
    dataset.close()
    for name, value in metrics.items():
        print(f"{name}: {value}")
    print(f"Outputs saved to {args.output_dir}")


if __name__ == "__main__":
    main()
