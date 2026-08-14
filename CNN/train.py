"""Train the deterministic CNN colorization baseline."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

try:
    from CNN.dataset import H5ColorizationDataset
    from CNN.download_data import find_dataset_files
    from CNN.model import CompactUNetColorizer, count_trainable_parameters
    from CNN.utils import save_json, save_loss_curve, select_device, set_seed
except ModuleNotFoundError:  # Supports: python CNN/train.py
    from dataset import H5ColorizationDataset
    from download_data import find_dataset_files
    from model import CompactUNetColorizer, count_trainable_parameters
    from utils import save_json, save_loss_curve, select_device, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-h5", type=Path)
    parser.add_argument("--val-h5", type=Path)
    parser.add_argument("--data-root", type=Path, help="Search this downloaded Drive directory for the fixed HDF5 filenames.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--loss", choices=("mse", "l1"), default="mse")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--patience", type=int, default=5, help="Use 0 to disable early stopping.")
    return parser.parse_args()


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
) -> float:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_samples = 0
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for batch in loader:
            L = batch["L"].to(device, non_blocking=True)
            ab = batch["ab"].to(device, non_blocking=True)
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(L), ab)
            if optimizer is not None:
                loss.backward()
                optimizer.step()
            batch_size = L.shape[0]
            total_loss += loss.item() * batch_size
            total_samples += batch_size
    return total_loss / total_samples


def main() -> None:
    args = parse_args()
    if args.epochs <= 0 or args.batch_size <= 0 or args.learning_rate <= 0 or args.num_workers < 0:
        raise ValueError("epochs, batch size, and learning rate must be positive; workers cannot be negative")
    if args.data_root is not None:
        if args.train_h5 is not None or args.val_h5 is not None:
            raise ValueError("Use either --data-root or --train-h5/--val-h5, not both")
        files = find_dataset_files(args.data_root)
        args.train_h5 = files["train_lab.h5"]
        args.val_h5 = files["validation_lab.h5"]
    elif args.train_h5 is None or args.val_h5 is None:
        raise ValueError("Provide --data-root, or provide both --train-h5 and --val-h5")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)
    device = select_device()
    print(f"Selected device: {device}")

    train_dataset = H5ColorizationDataset(args.train_h5)
    validation_dataset = H5ColorizationDataset(args.val_h5)
    generator = torch.Generator().manual_seed(args.seed)
    loader_options = dict(batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=device.type == "cuda")
    train_loader = DataLoader(train_dataset, shuffle=True, generator=generator, **loader_options)
    validation_loader = DataLoader(validation_dataset, shuffle=False, **loader_options)

    model = CompactUNetColorizer().to(device)
    parameter_count = count_trainable_parameters(model)
    criterion = nn.MSELoss() if args.loss == "mse" else nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    model_config = {"name": "CompactUNetColorizer", "input_shape": [1, 32, 32], "output_shape": [2, 32, 32], "trainable_parameters": parameter_count}
    training_config = {
        "loss": args.loss, "optimizer": "Adam", "learning_rate": args.learning_rate,
        "batch_size": args.batch_size, "requested_epochs": args.epochs, "seed": args.seed,
        "num_workers": args.num_workers, "patience": args.patience,
        "train_h5": str(args.train_h5), "validation_h5": str(args.val_h5),
        "dataset_split": {"train": len(train_dataset), "validation": len(validation_dataset)},
    }
    print(f"Trainable parameters: {parameter_count:,}")

    history: dict[str, list[float] | float | int] = {"train_loss": [], "validation_loss": [], "epoch_duration_seconds": []}
    best_loss = float("inf")
    stale_epochs = 0
    training_start = time.perf_counter()
    last_checkpoint = None
    for epoch in range(1, args.epochs + 1):
        epoch_start = time.perf_counter()
        train_loss = run_epoch(model, train_loader, criterion, device, optimizer)
        validation_loss = run_epoch(model, validation_loader, criterion, device)
        duration = time.perf_counter() - epoch_start
        history["train_loss"].append(train_loss)  # type: ignore[union-attr]
        history["validation_loss"].append(validation_loss)  # type: ignore[union-attr]
        history["epoch_duration_seconds"].append(duration)  # type: ignore[union-attr]
        checkpoint = {
            "epoch": epoch, "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(), "train_loss": train_loss,
            "validation_loss": validation_loss, "model_config": model_config,
            "training_config": training_config,
        }
        last_checkpoint = checkpoint
        if validation_loss < best_loss:
            best_loss = validation_loss
            stale_epochs = 0
            torch.save(checkpoint, args.output_dir / "best_model.pt")
        else:
            stale_epochs += 1
        print(f"Epoch {epoch:03d}/{args.epochs:03d} | train {train_loss:.6f} | validation {validation_loss:.6f} | {duration:.1f}s")
        if args.patience > 0 and stale_epochs >= args.patience:
            print(f"Early stopping after {stale_epochs} epochs without improvement.")
            break

    history["total_training_duration_seconds"] = time.perf_counter() - training_start
    history["epochs_completed"] = len(history["train_loss"])  # type: ignore[arg-type]
    assert last_checkpoint is not None
    torch.save(last_checkpoint, args.output_dir / "last_model.pt")
    save_json(history, args.output_dir / "history.json")
    save_loss_curve(history, args.output_dir / "loss_curve.png")
    save_json({"model_config": model_config, "training_config": training_config}, args.output_dir / "config.json")
    train_dataset.close()
    validation_dataset.close()
    print(f"Outputs saved to {args.output_dir}")


if __name__ == "__main__":
    main()
