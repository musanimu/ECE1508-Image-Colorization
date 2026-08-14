"""Download the preprocessed colorization data from Google Drive."""

from __future__ import annotations

import argparse
from pathlib import Path


DEFAULT_DRIVE_FOLDER = "https://drive.google.com/drive/folders/10yeI2-kI23M206A4eweQ_jZSWkKj3H7K"
REQUIRED_FILES = ("train_lab.h5", "validation_lab.h5", "test_lab.h5")


def find_dataset_files(root: Path) -> dict[str, Path]:
    """Find the three fixed CIFAR-10 HDF5 artifacts beneath a downloaded folder."""
    found: dict[str, Path] = {}
    for filename in REQUIRED_FILES:
        matches = list(root.rglob(filename))
        if len(matches) != 1:
            raise FileNotFoundError(
                f"Expected exactly one {filename!r} below {root}, found {len(matches)}."
            )
        found[filename] = matches[0].resolve()
    return found


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--folder-url",
        default=DEFAULT_DRIVE_FOLDER,
        help="Public Google Drive folder URL (defaults to the team's shared folder).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("shared_data"),
        help="Local destination; HDF5 files remain ignored by Git.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        import gdown
    except ImportError as error:
        raise SystemExit(
            "Google Drive support requires gdown. Install it with: "
            "python3 -m pip install gdown"
        ) from error

    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Downloading existing preprocessed files to {args.output_dir.resolve()}")
    downloaded = gdown.download_folder(
        url=args.folder_url,
        output=str(args.output_dir),
        quiet=False,
    )
    if not downloaded:
        raise RuntimeError(
            "Google Drive returned no files. Confirm that the folder is shared publicly."
        )

    files = find_dataset_files(args.output_dir)
    print("Dataset ready; no preprocessing was performed:")
    print(f"  train:      {files['train_lab.h5']}")
    print(f"  validation: {files['validation_lab.h5']}")
    print(f"  test:       {files['test_lab.h5']}")


if __name__ == "__main__":
    main()
