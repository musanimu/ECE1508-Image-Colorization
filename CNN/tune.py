"""Grid-search learning rate and batch size using validation loss only."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

try:
    from CNN.download_data import find_dataset_files
except ModuleNotFoundError:
    from download_data import find_dataset_files


DEFAULT_LEARNING_RATES = (1e-4, 2e-4, 5e-4)
DEFAULT_BATCH_SIZES = (32, 64)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("CNN/outputs/tuning_mse"))
    parser.add_argument("--learning-rates", type=float, nargs="+", default=DEFAULT_LEARNING_RATES)
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=DEFAULT_BATCH_SIZES)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--baseline-run", type=Path, help="Reuse a completed MSE run, such as CNN/outputs/cnn_mse.")
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as file:
        return json.load(file)


def save_json(data: dict[str, Any], path: Path) -> None:
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def result_from_run(run_dir: Path) -> dict[str, Any]:
    history_path = run_dir / "history.json"
    config_path = run_dir / "config.json"
    checkpoint_path = run_dir / "best_model.pt"
    for path in (history_path, config_path, checkpoint_path):
        if not path.is_file():
            raise FileNotFoundError(f"Incomplete training run; missing {path}")
    history = load_json(history_path)
    config = load_json(config_path)["training_config"]
    losses = history["validation_loss"]
    best_index = min(range(len(losses)), key=losses.__getitem__)
    return {
        "learning_rate": float(config["learning_rate"]),
        "batch_size": int(config["batch_size"]),
        "loss": config["loss"],
        "seed": int(config["seed"]),
        "best_epoch": best_index + 1,
        "best_validation_loss": float(losses[best_index]),
        "epochs_completed": int(history["epochs_completed"]),
        "total_training_duration_seconds": float(history["total_training_duration_seconds"]),
        "run_directory": str(run_dir.resolve()),
        "best_checkpoint": str(checkpoint_path.resolve()),
    }


def run_name(learning_rate: float, batch_size: int) -> str:
    rate = f"{learning_rate:.8f}".rstrip("0").rstrip(".").replace(".", "p")
    return f"lr_{rate}_bs_{batch_size}"


def validate_reused_run(result: dict[str, Any], args: argparse.Namespace) -> None:
    pair = (result["learning_rate"], result["batch_size"])
    grid = {(rate, size) for rate in args.learning_rates for size in args.batch_sizes}
    if pair not in grid:
        raise ValueError(f"Reused configuration {pair} is not part of the requested grid")
    if result["loss"] != "mse" or result["seed"] != args.seed:
        raise ValueError("Reused runs must use MSE and the same seed as the tuning search")


def main() -> None:
    args = parse_args()
    if any(rate <= 0 for rate in args.learning_rates) or any(size <= 0 for size in args.batch_sizes):
        raise ValueError("Learning rates and batch sizes must be positive")
    files = find_dataset_files(args.data_root)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results: dict[tuple[float, int], dict[str, Any]] = {}

    if args.baseline_run is not None:
        baseline = result_from_run(args.baseline_run)
        validate_reused_run(baseline, args)
        results[(baseline["learning_rate"], baseline["batch_size"])] = baseline
        print(f"Reusing baseline: lr={baseline['learning_rate']}, batch={baseline['batch_size']}, validation={baseline['best_validation_loss']:.6f}")

    train_script = Path(__file__).with_name("train.py")
    for learning_rate in args.learning_rates:
        for batch_size in args.batch_sizes:
            key = (float(learning_rate), int(batch_size))
            if key in results:
                continue
            run_dir = args.output_dir / run_name(*key)
            if (run_dir / "history.json").is_file():
                existing = result_from_run(run_dir)
                validate_reused_run(existing, args)
                results[key] = existing
                print(f"Reusing completed tuning run: {run_dir}")
                continue
            command = [
                sys.executable, str(train_script), "--train-h5", str(files["train_lab.h5"]),
                "--val-h5", str(files["validation_lab.h5"]), "--output-dir", str(run_dir),
                "--loss", "mse", "--learning-rate", str(learning_rate), "--batch-size", str(batch_size),
                "--epochs", str(args.epochs), "--patience", str(args.patience), "--seed", str(args.seed),
                "--num-workers", str(args.num_workers),
            ]
            print(f"Starting grid run: lr={learning_rate}, batch={batch_size}", flush=True)
            subprocess.run(command, check=True)
            results[key] = result_from_run(run_dir)

    ranked = sorted(results.values(), key=lambda result: result["best_validation_loss"])
    winner = ranked[0]
    summary = {
        "selection_metric": "minimum validation MSE",
        "fixed_settings": {"model": "CompactUNetColorizer", "loss": "mse", "optimizer": "Adam", "maximum_epochs": args.epochs, "patience": args.patience, "seed": args.seed},
        "search_space": {"learning_rates": args.learning_rates, "batch_sizes": args.batch_sizes},
        "ranked_results": ranked,
        "selected_configuration": winner,
    }
    save_json(summary, args.output_dir / "tuning_results.json")
    with (args.output_dir / "tuning_results.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(ranked[0].keys()))
        writer.writeheader()
        writer.writerows(ranked)
    print("\nSelected configuration:")
    print(f"  learning rate: {winner['learning_rate']}")
    print(f"  batch size: {winner['batch_size']}")
    print(f"  best epoch: {winner['best_epoch']}")
    print(f"  validation loss: {winner['best_validation_loss']:.6f}")
    print(f"  checkpoint: {winner['best_checkpoint']}")
    print("The test set was not used during tuning.")


if __name__ == "__main__":
    main()
