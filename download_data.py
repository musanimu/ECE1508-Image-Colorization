"""Download the team's already-preprocessed colorization data from Google Drive.

Run from the repository root (final_project/):

  python3 download_data.py
  python3 download_data.py --output-dir shared_data
"""

from __future__ import annotations

import argparse
from pathlib import Path


DEFAULT_DRIVE_FOLDER = "https://drive.google.com/drive/folders/10yeI2-kI23M206A4eweQ_jZSWkKj3H7K"
REQUIRED_FILES = ("train_lab.h5", "validation_lab.h5", "test_lab.h5")
PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "shared_data"


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


def resolve_lab_data_dir(project_root: Path | None = None) -> Path:
    """Return the directory that contains train/validation/test_lab.h5.

    Search order: shared_data/, then common legacy extract paths, then project root.
    """
    root = (project_root or PROJECT_ROOT).resolve()
    candidates = [
        root / "shared_data",
        root / "cifar10_lab_v1-20260808T045604Z-1-001",
        root,
    ]
    errors: list[str] = []
    for candidate in candidates:
        if not candidate.exists():
            continue
        try:
            files = find_dataset_files(candidate)
            return files["train_lab.h5"].parent
        except FileNotFoundError as exc:
            errors.append(str(exc))
    raise FileNotFoundError(
        "Could not locate train_lab.h5 / validation_lab.h5 / test_lab.h5. "
        "Download with: python3 download_data.py\n" + "\n".join(errors)
    )


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
        default=DEFAULT_OUTPUT_DIR,
        help="Local destination; HDF5 files remain ignored by Git (default: shared_data/).",
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
