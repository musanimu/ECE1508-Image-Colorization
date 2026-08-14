"""HDF5 dataset access for the shared, preprocessed colorization data."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import h5py
import torch
from torch.utils.data import Dataset


class H5ColorizationDataset(Dataset[dict[str, torch.Tensor]]):
    """Read normalized L/ab samples with a lazy HDF5 handle per worker process."""

    def __init__(self, h5_path: str | Path) -> None:
        self.h5_path = Path(h5_path).expanduser()
        self._h5: h5py.File | None = None
        if not self.h5_path.is_file():
            raise FileNotFoundError(f"HDF5 dataset not found: {self.h5_path}")

        with h5py.File(self.h5_path, "r") as h5_file:
            missing = {"L", "ab", "labels"}.difference(h5_file.keys())
            if missing:
                raise KeyError(f"{self.h5_path} is missing datasets: {sorted(missing)}")
            lengths = {len(h5_file[key]) for key in ("L", "ab", "labels")}
            if len(lengths) != 1:
                raise ValueError("L, ab, and labels have inconsistent lengths")
            self._length = lengths.pop()
            if h5_file["L"].shape[1:] != (1, 32, 32):
                raise ValueError(f"Expected L shape [N,1,32,32], got {h5_file['L'].shape}")
            if h5_file["ab"].shape[1:] != (2, 32, 32):
                raise ValueError(f"Expected ab shape [N,2,32,32], got {h5_file['ab'].shape}")

    def _file(self) -> h5py.File:
        if self._h5 is None:
            self._h5 = h5py.File(self.h5_path, "r")
        return self._h5

    def __len__(self) -> int:
        return self._length

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        if index < 0:
            index += self._length
        if index < 0 or index >= self._length:
            raise IndexError(index)
        h5_file = self._file()
        return {
            "L": torch.as_tensor(h5_file["L"][index], dtype=torch.float32),
            "ab": torch.as_tensor(h5_file["ab"][index], dtype=torch.float32),
            "label": torch.as_tensor(h5_file["labels"][index], dtype=torch.long),
        }

    def close(self) -> None:
        if self._h5 is not None:
            self._h5.close()
            self._h5 = None

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["_h5"] = None
        return state

    def __del__(self) -> None:
        if hasattr(self, "_h5"):
            try:
                self.close()
            except (AttributeError, TypeError):
                # Python may already have torn down h5py internals at shutdown.
                pass


def validate_dataset(h5_path: str | Path) -> None:
    """Print basic schema, dtype, and normalized-range checks."""
    dataset = H5ColorizationDataset(h5_path)
    sample = dataset[0]
    print(f"Dataset length: {len(dataset)}")
    for key in ("L", "ab"):
        tensor = sample[key]
        print(f"{key} shape: {tuple(tensor.shape)}")
        print(f"{key} dtype: {tensor.dtype}")
        print(f"{key} min/max: {tensor.min().item():.4f} / {tensor.max().item():.4f}")
    dataset.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate a shared colorization HDF5 file.")
    parser.add_argument("h5_path", type=Path)
    validate_dataset(parser.parse_args().h5_path)
