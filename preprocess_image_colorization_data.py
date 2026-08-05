#!/usr/bin/env python3
"""
Shared dataset preprocessing for the ECE1508 image colorization project.

1. CIFAR-10
   - 45,000 training images
   - 5,000 validation images
   - 10,000 official test images
   - 32 x 32 resolution

2. Places365
   - balanced subset across 365 categories
   - default: 100 train, 10 validation, and 10 test images per category
   - default: 64 x 64 resolution

For both datasets:
- RGB images are converted to CIE LAB.
- L is stored as the grayscale/lightness input.
- ab is stored as the color prediction target.
- L normalization: L / 50 - 1
- ab normalization: ab / 128
- Outputs are compressed HDF5 files plus metadata, indices, and checksums.

Examples
--------
CIFAR-10:
    python preprocess_image_colorization_data.py \
        --dataset cifar10 \
        --project-root ./ECE1508_Image_Colorization

Places365:
    python preprocess_image_colorization_data.py \
        --dataset places365 \
        --project-root ./ECE1508_Image_Colorization

Install dependencies:
    python -m pip install torch torchvision numpy scikit-image \
        scikit-learn h5py pillow tqdm
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence

import h5py
import numpy as np
import torch
import torchvision
from PIL import Image
from sklearn.model_selection import train_test_split
from skimage.color import rgb2lab
from torch.utils.data import Dataset
from torchvision.datasets import CIFAR10, Places365
from torchvision.transforms import CenterCrop, Compose, InterpolationMode, Resize
from tqdm.auto import tqdm


CIFAR10_CLASS_NAMES = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare shared LAB datasets for image colorization."
    )

    parser.add_argument(
        "--dataset",
        required=True,
        choices=["cifar10", "places365"],
        help="Dataset to preprocess.",
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path.cwd() / "ECE1508_Image_Colorization",
        help=(
            "Project root containing raw_data/ and shared_data/. "
            "Default: ./ECE1508_Image_Colorization"
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used for fixed split selection.",
    )
    parser.add_argument(
        "--compression-level",
        type=int,
        default=4,
        choices=range(0, 10),
        metavar="[0-9]",
        help="Gzip compression level for HDF5 files.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing processed output files.",
    )
    parser.add_argument(
        "--no-download",
        action="store_true",
        help="Do not download source data; use existing raw files.",
    )

    # CIFAR-10 options
    parser.add_argument(
        "--cifar-validation-size",
        type=int,
        default=5000,
        help="Number of validation images taken from official CIFAR-10 training data.",
    )

    # Places365 options
    parser.add_argument(
        "--places-image-size",
        type=int,
        default=64,
        help="Final square image size for Places365.",
    )
    parser.add_argument(
        "--places-resize-short-side",
        type=int,
        default=72,
        help="Resize shorter side before center crop for Places365.",
    )
    parser.add_argument(
        "--places-train-per-category",
        type=int,
        default=100,
        help="Number of training images selected per Places365 category.",
    )
    parser.add_argument(
        "--places-validation-per-category",
        type=int,
        default=10,
        help="Number of validation images selected per Places365 category.",
    )
    parser.add_argument(
        "--places-test-per-category",
        type=int,
        default=10,
        help="Number of test images selected per Places365 category.",
    )

    return parser.parse_args()


def set_reproducibility(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def sha256_file(file_path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()

    with file_path.open("rb") as file:
        while True:
            chunk = file.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)

    return digest.hexdigest()


def ensure_outputs_available(
    output_dir: Path,
    expected_names: Sequence[str],
    overwrite: bool,
) -> None:
    existing = [output_dir / name for name in expected_names if (output_dir / name).exists()]

    if existing and not overwrite:
        formatted = "\n".join(f"  - {path}" for path in existing)
        raise FileExistsError(
            "Processed outputs already exist:\n"
            f"{formatted}\n"
            "Use --overwrite to replace them."
        )

    if overwrite:
        for path in existing:
            path.unlink()


def rgb_uint8_to_normalized_lab(rgb_uint8: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Convert one HWC uint8 RGB image to normalized CHW L and ab arrays."""

    if rgb_uint8.ndim != 3 or rgb_uint8.shape[-1] != 3:
        raise ValueError(f"Expected an HWC RGB image, received shape {rgb_uint8.shape}.")

    rgb = rgb_uint8.astype(np.float32) / 255.0
    lab = rgb2lab(rgb).astype(np.float32)

    L = lab[..., 0:1]
    ab = lab[..., 1:3]

    L_normalized = (L / 50.0) - 1.0
    ab_normalized = ab / 128.0

    L_chw = np.transpose(L_normalized, (2, 0, 1))
    ab_chw = np.transpose(ab_normalized, (2, 0, 1))

    return L_chw, ab_chw


def create_h5_datasets(
    h5_file: h5py.File,
    num_images: int,
    image_size: int,
    compression_level: int,
    label_dtype: np.dtype[Any],
) -> tuple[h5py.Dataset, h5py.Dataset, h5py.Dataset]:
    chunk_size = min(256, num_images)

    L_dataset = h5_file.create_dataset(
        "L",
        shape=(num_images, 1, image_size, image_size),
        dtype=np.float16,
        chunks=(chunk_size, 1, image_size, image_size),
        compression="gzip",
        compression_opts=compression_level,
    )

    ab_dataset = h5_file.create_dataset(
        "ab",
        shape=(num_images, 2, image_size, image_size),
        dtype=np.float16,
        chunks=(chunk_size, 2, image_size, image_size),
        compression="gzip",
        compression_opts=compression_level,
    )

    labels_dataset = h5_file.create_dataset(
        "labels",
        shape=(num_images,),
        dtype=label_dtype,
        compression="gzip",
        compression_opts=compression_level,
    )

    return L_dataset, ab_dataset, labels_dataset


def write_common_h5_attributes(
    h5_file: h5py.File,
    *,
    dataset_name: str,
    dataset_version: str,
    split_name: str,
    source_split: str,
    seed: int,
    image_size: int,
) -> None:
    h5_file.attrs["dataset_name"] = dataset_name
    h5_file.attrs["dataset_version"] = dataset_version
    h5_file.attrs["split_name"] = split_name
    h5_file.attrs["source_split"] = source_split
    h5_file.attrs["random_seed"] = seed
    h5_file.attrs["image_size"] = image_size
    h5_file.attrs["L_normalization"] = "L / 50 - 1"
    h5_file.attrs["ab_normalization"] = "ab / 128"
    h5_file.attrs["storage_dtype"] = "float16"


def validate_h5(
    path: Path,
    *,
    expected_size: int,
    image_size: int,
    num_categories: int | None = None,
    expected_per_category: int | None = None,
) -> dict[str, Any]:
    with h5py.File(path, "r") as h5_file:
        expected_L_shape = (expected_size, 1, image_size, image_size)
        expected_ab_shape = (expected_size, 2, image_size, image_size)

        if h5_file["L"].shape != expected_L_shape:
            raise ValueError(
                f"{path.name}: expected L shape {expected_L_shape}, "
                f"received {h5_file['L'].shape}."
            )

        if h5_file["ab"].shape != expected_ab_shape:
            raise ValueError(
                f"{path.name}: expected ab shape {expected_ab_shape}, "
                f"received {h5_file['ab'].shape}."
            )

        if h5_file["labels"].shape != (expected_size,):
            raise ValueError(
                f"{path.name}: labels shape is {h5_file['labels'].shape}."
            )

        labels = h5_file["labels"][:]

        if num_categories is not None and expected_per_category is not None:
            counts = np.bincount(labels.astype(np.int64), minlength=num_categories)
            if not np.all(counts == expected_per_category):
                raise ValueError(
                    f"{path.name}: category counts do not all equal "
                    f"{expected_per_category}."
                )

        sample_size = min(1000, expected_size)
        L_sample = h5_file["L"][:sample_size]
        ab_sample = h5_file["ab"][:sample_size]

        summary = {
            "L_shape": list(h5_file["L"].shape),
            "ab_shape": list(h5_file["ab"].shape),
            "labels_shape": list(h5_file["labels"].shape),
            "sample_L_min": float(L_sample.min()),
            "sample_L_max": float(L_sample.max()),
            "sample_ab_min": float(ab_sample.min()),
            "sample_ab_max": float(ab_sample.max()),
        }

    return summary


# ---------------------------------------------------------------------------
# CIFAR-10
# ---------------------------------------------------------------------------

def export_cifar10_split(
    dataset: Dataset[Any],
    indices: np.ndarray,
    output_path: Path,
    split_name: str,
    source_split: str,
    seed: int,
    compression_level: int,
) -> None:
    num_images = len(indices)
    image_size = 32
    write_batch_size = 1000

    with h5py.File(output_path, "w") as h5_file:
        L_dataset, ab_dataset, labels_dataset = create_h5_datasets(
            h5_file,
            num_images=num_images,
            image_size=image_size,
            compression_level=compression_level,
            label_dtype=np.int64,
        )

        h5_file.create_dataset(
            "source_indices",
            data=indices.astype(np.int64),
            compression="gzip",
            compression_opts=compression_level,
        )

        write_common_h5_attributes(
            h5_file,
            dataset_name="CIFAR-10",
            dataset_version="cifar10_lab_v1",
            split_name=split_name,
            source_split=source_split,
            seed=seed,
            image_size=image_size,
        )

        for start in tqdm(
            range(0, num_images, write_batch_size),
            desc=f"CIFAR-10 {split_name}",
        ):
            end = min(start + write_batch_size, num_images)
            batch_indices = indices[start:end]

            L_batch = np.empty(
                (len(batch_indices), 1, image_size, image_size),
                dtype=np.float16,
            )
            ab_batch = np.empty(
                (len(batch_indices), 2, image_size, image_size),
                dtype=np.float16,
            )
            labels_batch = np.empty(len(batch_indices), dtype=np.int64)

            for position, source_index in enumerate(batch_indices):
                image, label = dataset[int(source_index)]
                rgb_uint8 = np.asarray(image.convert("RGB"), dtype=np.uint8)
                L_chw, ab_chw = rgb_uint8_to_normalized_lab(rgb_uint8)

                L_batch[position] = L_chw.astype(np.float16)
                ab_batch[position] = ab_chw.astype(np.float16)
                labels_batch[position] = int(label)

            L_dataset[start:end] = L_batch
            ab_dataset[start:end] = ab_batch
            labels_dataset[start:end] = labels_batch


def preprocess_cifar10(args: argparse.Namespace) -> Path:
    project_root = args.project_root.resolve()
    raw_dir = project_root / "raw_data" / "cifar10"
    output_dir = project_root / "shared_data" / "cifar10_lab_v1"

    raw_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    expected_names = [
        "train_lab.h5",
        "validation_lab.h5",
        "test_lab.h5",
        "split_indices.npz",
        "metadata.json",
    ]
    ensure_outputs_available(output_dir, expected_names, args.overwrite)

    print("Loading CIFAR-10...")
    base_train = CIFAR10(
        root=raw_dir,
        train=True,
        download=not args.no_download,
    )
    base_test = CIFAR10(
        root=raw_dir,
        train=False,
        download=not args.no_download,
    )

    validation_size = args.cifar_validation_size
    if not 1 <= validation_size < len(base_train):
        raise ValueError(
            "--cifar-validation-size must be between 1 and "
            f"{len(base_train) - 1}."
        )

    train_labels_all = np.asarray(base_train.targets, dtype=np.int64)
    all_train_indices = np.arange(len(base_train), dtype=np.int64)

    train_indices, validation_indices = train_test_split(
        all_train_indices,
        test_size=validation_size,
        random_state=args.seed,
        shuffle=True,
        stratify=train_labels_all,
    )

    train_indices = np.asarray(train_indices, dtype=np.int64)
    validation_indices = np.asarray(validation_indices, dtype=np.int64)
    test_indices = np.arange(len(base_test), dtype=np.int64)

    if len(np.intersect1d(train_indices, validation_indices)) != 0:
        raise RuntimeError("CIFAR-10 train and validation splits overlap.")

    indices_path = output_dir / "split_indices.npz"
    np.savez_compressed(
        indices_path,
        train_indices=train_indices,
        validation_indices=validation_indices,
        test_indices=test_indices,
    )

    train_path = output_dir / "train_lab.h5"
    validation_path = output_dir / "validation_lab.h5"
    test_path = output_dir / "test_lab.h5"

    export_cifar10_split(
        base_train,
        train_indices,
        train_path,
        "train",
        "official_train",
        args.seed,
        args.compression_level,
    )
    export_cifar10_split(
        base_train,
        validation_indices,
        validation_path,
        "validation",
        "official_train",
        args.seed,
        args.compression_level,
    )
    export_cifar10_split(
        base_test,
        test_indices,
        test_path,
        "test",
        "official_test",
        args.seed,
        args.compression_level,
    )

    validation_summary = {
        "train": validate_h5(
            train_path,
            expected_size=len(train_indices),
            image_size=32,
        ),
        "validation": validate_h5(
            validation_path,
            expected_size=len(validation_indices),
            image_size=32,
        ),
        "test": validate_h5(
            test_path,
            expected_size=len(test_indices),
            image_size=32,
        ),
    }

    class_distribution = {
        "train": np.bincount(
            train_labels_all[train_indices], minlength=10
        ).astype(int).tolist(),
        "validation": np.bincount(
            train_labels_all[validation_indices], minlength=10
        ).astype(int).tolist(),
        "test": np.bincount(
            np.asarray(base_test.targets, dtype=np.int64), minlength=10
        ).astype(int).tolist(),
    }

    files_for_checksum = [indices_path, train_path, validation_path, test_path]
    checksums = {
        path.name: sha256_file(path)
        for path in tqdm(files_for_checksum, desc="CIFAR-10 checksums")
    }

    metadata = {
        "dataset_name": "CIFAR-10",
        "dataset_version": "cifar10_lab_v1",
        "random_seed": args.seed,
        "split_method": "Stratified split of official training set",
        "train_size": len(train_indices),
        "validation_size": len(validation_indices),
        "test_size": len(test_indices),
        "image_size": [3, 32, 32],
        "model_input": "Normalized CIE-LAB L channel",
        "model_target": "Normalized CIE-LAB a and b channels",
        "L_normalization": "L / 50 - 1",
        "ab_normalization": "ab / 128",
        "storage_dtype": "float16",
        "class_names": CIFAR10_CLASS_NAMES,
        "class_distribution": class_distribution,
        "validation_summary": validation_summary,
        "sha256": checksums,
        "torch_version": torch.__version__,
        "torchvision_version": torchvision.__version__,
    }

    metadata_path = output_dir / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print("\nCIFAR-10 preprocessing complete.")
    print(f"Output folder: {output_dir}")
    print(f"Training images: {len(train_indices):,}")
    print(f"Validation images: {len(validation_indices):,}")
    print(f"Test images: {len(test_indices):,}")

    return output_dir


# ---------------------------------------------------------------------------
# Places365
# ---------------------------------------------------------------------------

def group_indices_by_class(targets: Sequence[int | None]) -> dict[int, list[int]]:
    grouped: dict[int, list[int]] = defaultdict(list)

    for index, target in enumerate(targets):
        if target is None:
            continue
        grouped[int(target)].append(index)

    return grouped


def select_balanced_train_validation_indices(
    targets: Sequence[int | None],
    *,
    num_categories: int,
    train_per_category: int,
    validation_per_category: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    grouped = group_indices_by_class(targets)
    rng = np.random.default_rng(seed)

    selected_train: list[int] = []
    selected_validation: list[int] = []

    required = train_per_category + validation_per_category

    for class_id in range(num_categories):
        class_indices = np.asarray(grouped[class_id], dtype=np.int64)

        if len(class_indices) < required:
            raise ValueError(
                f"Places365 class {class_id} has {len(class_indices)} images, "
                f"but {required} are required."
            )

        shuffled = rng.permutation(class_indices)
        selected_train.extend(shuffled[:train_per_category].tolist())
        selected_validation.extend(
            shuffled[train_per_category:required].tolist()
        )

    train_array = rng.permutation(
        np.asarray(selected_train, dtype=np.int64)
    )
    validation_array = rng.permutation(
        np.asarray(selected_validation, dtype=np.int64)
    )

    return train_array, validation_array


def select_balanced_test_indices(
    targets: Sequence[int | None],
    *,
    num_categories: int,
    test_per_category: int,
    seed: int,
) -> np.ndarray:
    grouped = group_indices_by_class(targets)
    rng = np.random.default_rng(seed)

    selected_test: list[int] = []

    for class_id in range(num_categories):
        class_indices = np.asarray(grouped[class_id], dtype=np.int64)

        if len(class_indices) < test_per_category:
            raise ValueError(
                f"Places365 class {class_id} has {len(class_indices)} images, "
                f"but {test_per_category} are required."
            )

        shuffled = rng.permutation(class_indices)
        selected_test.extend(shuffled[:test_per_category].tolist())

    return rng.permutation(np.asarray(selected_test, dtype=np.int64))


def export_places365_split(
    dataset: Places365,
    indices: np.ndarray,
    output_path: Path,
    manifest_path: Path,
    *,
    split_name: str,
    source_split: str,
    seed: int,
    image_size: int,
    resize_short_side: int,
    compression_level: int,
) -> None:
    num_images = len(indices)
    write_batch_size = 256

    spatial_preprocess = Compose(
        [
            Resize(
                resize_short_side,
                interpolation=InterpolationMode.BICUBIC,
                antialias=True,
            ),
            CenterCrop(image_size),
        ]
    )

    with h5py.File(output_path, "w") as h5_file:
        L_dataset, ab_dataset, labels_dataset = create_h5_datasets(
            h5_file,
            num_images=num_images,
            image_size=image_size,
            compression_level=compression_level,
            label_dtype=np.int16,
        )

        h5_file.create_dataset(
            "source_indices",
            data=indices.astype(np.int64),
            compression="gzip",
            compression_opts=compression_level,
        )

        write_common_h5_attributes(
            h5_file,
            dataset_name="Places365-Standard",
            dataset_version=f"places365_lab{image_size}_v1",
            split_name=split_name,
            source_split=source_split,
            seed=seed,
            image_size=image_size,
        )
        h5_file.attrs["resize_short_side"] = resize_short_side
        h5_file.attrs["spatial_preprocessing"] = (
            f"RGB; resize shorter side to {resize_short_side} using bicubic; "
            f"center crop to {image_size}x{image_size}"
        )

        with manifest_path.open("w", newline="", encoding="utf-8") as csv_file:
            writer = csv.DictWriter(
                csv_file,
                fieldnames=[
                    "processed_position",
                    "source_index",
                    "source_split",
                    "source_path",
                    "class_id",
                    "class_name",
                ],
            )
            writer.writeheader()

            for start in tqdm(
                range(0, num_images, write_batch_size),
                desc=f"Places365 {split_name}",
            ):
                end = min(start + write_batch_size, num_images)
                batch_indices = indices[start:end]

                L_batch = np.empty(
                    (len(batch_indices), 1, image_size, image_size),
                    dtype=np.float16,
                )
                ab_batch = np.empty(
                    (len(batch_indices), 2, image_size, image_size),
                    dtype=np.float16,
                )
                labels_batch = np.empty(len(batch_indices), dtype=np.int16)

                manifest_rows: list[dict[str, Any]] = []

                for position, source_index in enumerate(batch_indices):
                    source_index = int(source_index)
                    image, label = dataset[source_index]

                    prepared = spatial_preprocess(image.convert("RGB"))
                    rgb_uint8 = np.asarray(prepared, dtype=np.uint8)
                    L_chw, ab_chw = rgb_uint8_to_normalized_lab(rgb_uint8)

                    L_batch[position] = L_chw.astype(np.float16)
                    ab_batch[position] = ab_chw.astype(np.float16)
                    labels_batch[position] = int(label)

                    source_path = dataset.imgs[source_index][0]
                    manifest_rows.append(
                        {
                            "processed_position": start + position,
                            "source_index": source_index,
                            "source_split": source_split,
                            "source_path": str(source_path),
                            "class_id": int(label),
                            "class_name": dataset.classes[int(label)],
                        }
                    )

                L_dataset[start:end] = L_batch
                ab_dataset[start:end] = ab_batch
                labels_dataset[start:end] = labels_batch
                writer.writerows(manifest_rows)


def preprocess_places365(args: argparse.Namespace) -> Path:
    project_root = args.project_root.resolve()
    raw_dir = project_root / "raw_data" / "places365"

    image_size = args.places_image_size
    resize_short_side = args.places_resize_short_side

    if image_size <= 0:
        raise ValueError("--places-image-size must be positive.")
    if resize_short_side < image_size:
        raise ValueError(
            "--places-resize-short-side must be at least --places-image-size."
        )

    version = f"places365_lab{image_size}_v1"
    output_dir = project_root / "shared_data" / version

    raw_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    expected_names = [
        f"train_lab_{image_size}.h5",
        f"validation_lab_{image_size}.h5",
        f"test_lab_{image_size}.h5",
        f"train_lab_{image_size}.csv",
        f"validation_lab_{image_size}.csv",
        f"test_lab_{image_size}.csv",
        "split_indices.npz",
        "class_names.json",
        "metadata.json",
    ]
    ensure_outputs_available(output_dir, expected_names, args.overwrite)

    print(
        "Loading Places365 source data.\n"
        "This source download is large and requires substantial free disk space."
    )

    download = not args.no_download
    train_source = Places365(
        root=raw_dir,
        split="train-standard",
        small=True,
        download=download,
    )
    test_source = Places365(
        root=raw_dir,
        split="val",
        small=True,
        download=download,
    )

    num_categories = len(train_source.classes)
    if train_source.classes != test_source.classes:
        raise RuntimeError("Places365 train and validation class lists differ.")

    train_indices, validation_indices = select_balanced_train_validation_indices(
        train_source.targets,
        num_categories=num_categories,
        train_per_category=args.places_train_per_category,
        validation_per_category=args.places_validation_per_category,
        seed=args.seed,
    )
    test_indices = select_balanced_test_indices(
        test_source.targets,
        num_categories=num_categories,
        test_per_category=args.places_test_per_category,
        seed=args.seed,
    )

    if len(np.intersect1d(train_indices, validation_indices)) != 0:
        raise RuntimeError("Places365 train and validation splits overlap.")

    indices_path = output_dir / "split_indices.npz"
    np.savez_compressed(
        indices_path,
        train_indices=train_indices,
        validation_indices=validation_indices,
        test_indices=test_indices,
    )

    train_path = output_dir / f"train_lab_{image_size}.h5"
    validation_path = output_dir / f"validation_lab_{image_size}.h5"
    test_path = output_dir / f"test_lab_{image_size}.h5"

    train_manifest = output_dir / f"train_lab_{image_size}.csv"
    validation_manifest = output_dir / f"validation_lab_{image_size}.csv"
    test_manifest = output_dir / f"test_lab_{image_size}.csv"

    export_places365_split(
        train_source,
        train_indices,
        train_path,
        train_manifest,
        split_name="train",
        source_split="train-standard",
        seed=args.seed,
        image_size=image_size,
        resize_short_side=resize_short_side,
        compression_level=args.compression_level,
    )
    export_places365_split(
        train_source,
        validation_indices,
        validation_path,
        validation_manifest,
        split_name="validation",
        source_split="train-standard",
        seed=args.seed,
        image_size=image_size,
        resize_short_side=resize_short_side,
        compression_level=args.compression_level,
    )
    export_places365_split(
        test_source,
        test_indices,
        test_path,
        test_manifest,
        split_name="test",
        source_split="val",
        seed=args.seed,
        image_size=image_size,
        resize_short_side=resize_short_side,
        compression_level=args.compression_level,
    )

    expected_counts = {
        "train": args.places_train_per_category,
        "validation": args.places_validation_per_category,
        "test": args.places_test_per_category,
    }

    validation_summary = {
        "train": validate_h5(
            train_path,
            expected_size=len(train_indices),
            image_size=image_size,
            num_categories=num_categories,
            expected_per_category=expected_counts["train"],
        ),
        "validation": validate_h5(
            validation_path,
            expected_size=len(validation_indices),
            image_size=image_size,
            num_categories=num_categories,
            expected_per_category=expected_counts["validation"],
        ),
        "test": validate_h5(
            test_path,
            expected_size=len(test_indices),
            image_size=image_size,
            num_categories=num_categories,
            expected_per_category=expected_counts["test"],
        ),
    }

    class_names_path = output_dir / "class_names.json"
    class_names_path.write_text(
        json.dumps(
            {
                str(class_id): class_name
                for class_id, class_name in enumerate(train_source.classes)
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    files_for_checksum = [
        indices_path,
        train_path,
        validation_path,
        test_path,
        train_manifest,
        validation_manifest,
        test_manifest,
        class_names_path,
    ]
    checksums = {
        path.name: sha256_file(path)
        for path in tqdm(files_for_checksum, desc="Places365 checksums")
    }

    metadata = {
        "dataset_name": "Places365-Standard",
        "dataset_version": version,
        "source_train_split": "train-standard",
        "source_test_split": "val",
        "source_small_images": True,
        "random_seed": args.seed,
        "number_of_categories": num_categories,
        "train_per_category": args.places_train_per_category,
        "validation_per_category": args.places_validation_per_category,
        "test_per_category": args.places_test_per_category,
        "train_size": len(train_indices),
        "validation_size": len(validation_indices),
        "test_size": len(test_indices),
        "image_size": [3, image_size, image_size],
        "resize_short_side": resize_short_side,
        "model_input": "Normalized CIE-LAB L channel",
        "model_target": "Normalized CIE-LAB a and b channels",
        "L_normalization": "L / 50 - 1",
        "ab_normalization": "ab / 128",
        "storage_dtype": "float16",
        "validation_summary": validation_summary,
        "sha256": checksums,
        "torch_version": torch.__version__,
        "torchvision_version": torchvision.__version__,
    }

    metadata_path = output_dir / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print("\nPlaces365 preprocessing complete.")
    print(f"Output folder: {output_dir}")
    print(f"Training images: {len(train_indices):,}")
    print(f"Validation images: {len(validation_indices):,}")
    print(f"Test images: {len(test_indices):,}")

    return output_dir


def main() -> int:
    args = parse_args()
    args.project_root = args.project_root.expanduser()

    set_reproducibility(args.seed)

    print(f"Dataset: {args.dataset}")
    print(f"Project root: {args.project_root.resolve()}")
    print(f"Seed: {args.seed}")
    print(f"Torch: {torch.__version__}")
    print(f"Torchvision: {torchvision.__version__}")

    try:
        if args.dataset == "cifar10":
            preprocess_cifar10(args)
        elif args.dataset == "places365":
            preprocess_places365(args)
        else:
            raise ValueError(f"Unsupported dataset: {args.dataset}")
    except KeyboardInterrupt:
        print("\nPreprocessing interrupted by user.", file=sys.stderr)
        return 130
    except Exception as error:
        print(f"\nERROR: {error}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
