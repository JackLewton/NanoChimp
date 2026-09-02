#!/usr/bin/env python3
"""
Create a date-disjoint train/val/test split for YOLO detection images.

Unlike tools/split_dataset.py (which splits by video ID, allowing the same
calendar date in both train and val), this script assigns every image by
the YYYYMMDD date encoded in its filename. No recording date appears in
more than one split, which reduces leakage from shared lighting,
backgrounds, and poses on consecutive clips from the same camera.

Default assignment (ChimpTZ-26 Option 2):

    train  Oct 4, Oct 26, Oct 28, Nov 1, Sept 18  (+ undated files)
    val    Sept 30, Oct 1, Oct 3, Oct 5
    test   June 4–5, Sept 16, Sept 17

The test set therefore covers Camera 1 / Window 1, Window 2, and D2,
each on dates that do not occur in train or val.

Usage:
    python tools/split_dataset_date_disjoint.py --images_dir yolo_dataset/images
"""

from __future__ import annotations

import argparse
import os
import re
from collections import defaultdict
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import yaml

# YYYYMMDD dates assigned to each split. Files with no parseable date
# (UNDATED) are placed in train so they cannot leak into evaluation.
TRAIN_DATES = {
    "20250918",
    "20251004",
    "20251026",
    "20251028",
    "20251101",
}
VAL_DATES = {
    "20250930",
    "20251001",
    "20251003",
    "20251005",
}
TEST_DATES = {
    "20250604",
    "20250605",
    "20250916",
    "20250917",
}
UNDATED = "undated"

DATE_PATTERN = re.compile(r"(20\d{2}[01]\d[0-3]\d)")
IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp")


def extract_date(filename: str) -> str:
    """Return the first YYYYMMDD substring in ``filename``, or ``UNDATED``."""
    match = DATE_PATTERN.search(filename)
    return match.group(1) if match else UNDATED


def camera_group(filename: str) -> str:
    """Coarse camera label used only for the printed summary."""
    name = filename.lower()
    if "window 1" in name or "camera1" in name or "camera 1" in name:
        return "Window 1 / Camera 1"
    if "window 2" in name:
        return "Window 2"
    if "d2" in name:
        return "D2"
    if "d1" in name:
        return "D1"
    return "Other"


def _split_for_date(date: str) -> str:
    if date in TRAIN_DATES or date == UNDATED:
        return "train"
    if date in VAL_DATES:
        return "val"
    if date in TEST_DATES:
        return "test"
    return "unassigned"


def _list_images(images_dir: str, suffixes: Sequence[str]) -> List[str]:
    return sorted(
        name
        for name in os.listdir(images_dir)
        if any(name.lower().endswith(suf) for suf in suffixes)
    )


def _write_list(path: str, lines: Iterable[str]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for line in lines:
            f.write(line + "\n")


def _write_data_yaml(output_dir: str) -> str:
    """Write (or overwrite) data.yaml so Ultralytics can use split='test'."""
    abs_dir = os.path.abspath(output_dir)
    config = {
        "path": abs_dir,
        "train": os.path.join(abs_dir, "train.txt"),
        "val": os.path.join(abs_dir, "val.txt"),
        "test": os.path.join(abs_dir, "test.txt"),
        "nc": 1,
        "names": {0: "chimp"},
    }
    yaml_path = os.path.join(abs_dir, "data.yaml")
    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.dump(config, f, default_flow_style=False)
    return yaml_path


def _print_summary(buckets: Dict[str, List[str]], unassigned: List[str]) -> None:
    total = sum(len(v) for v in buckets.values()) + len(unassigned)
    print(f"\nDate-disjoint split of {total} images:")
    print(f"{'Split':<8} {'Images':>7} {'%':>7}  Dates / cameras")
    print("-" * 72)
    for split in ("train", "val", "test"):
        paths = buckets[split]
        n = len(paths)
        pct = (100.0 * n / total) if total else 0.0
        dates = sorted({extract_date(os.path.basename(p)) for p in paths})
        cams: Dict[str, int] = defaultdict(int)
        for p in paths:
            cams[camera_group(os.path.basename(p))] += 1
        cam_str = ", ".join(f"{k}={v}" for k, v in sorted(cams.items()))
        print(f"{split:<8} {n:7d} {pct:6.1f}%  dates={dates}")
        print(f"{'':8} {'':7} {'':7}  cameras: {cam_str}")
    if unassigned:
        dates = sorted({extract_date(os.path.basename(p)) for p in unassigned})
        print(f"{'skip':<8} {len(unassigned):7d}  dates not in Option 2 map: {dates}")


def create_date_disjoint_split(
    images_dir: str,
    output_dir: Optional[str] = None,
    write_yaml: bool = True,
    image_suffixes: Sequence[str] = IMAGE_SUFFIXES,
) -> Tuple[str, str, str]:
    """Assign images to train/val/test by recording date and write manifests.

    Args:
        images_dir: Directory containing YOLO images.
        output_dir: Directory for train.txt / val.txt / test.txt
            (defaults to the parent of ``images_dir``).
        write_yaml: If True, write data.yaml with a ``test`` key.
        image_suffixes: Image file extensions to include.

    Returns:
        Paths to (train.txt, val.txt, test.txt).
    """
    if not os.path.isdir(images_dir):
        raise FileNotFoundError(f"Images directory not found: {images_dir}")

    output_dir = output_dir or os.path.dirname(os.path.abspath(images_dir))
    os.makedirs(output_dir, exist_ok=True)

    overlap = (TRAIN_DATES & VAL_DATES) | (TRAIN_DATES & TEST_DATES) | (VAL_DATES & TEST_DATES)
    if overlap:
        raise ValueError(f"Date maps overlap (this is a bug): {sorted(overlap)}")

    buckets: Dict[str, List[str]] = {"train": [], "val": [], "test": []}
    unassigned: List[str] = []

    for name in _list_images(images_dir, image_suffixes):
        path = os.path.abspath(os.path.join(images_dir, name))
        split = _split_for_date(extract_date(name))
        if split == "unassigned":
            unassigned.append(path)
        else:
            buckets[split].append(path)

    for split in buckets:
        buckets[split].sort()

    train_list = os.path.join(output_dir, "train.txt")
    val_list = os.path.join(output_dir, "val.txt")
    test_list = os.path.join(output_dir, "test.txt")
    _write_list(train_list, buckets["train"])
    _write_list(val_list, buckets["val"])
    _write_list(test_list, buckets["test"])

    _print_summary(buckets, unassigned)
    print(f"\nWrote:\n  {train_list}\n  {val_list}\n  {test_list}")

    if write_yaml:
        yaml_path = _write_data_yaml(output_dir)
        print(f"  {yaml_path}")

    if unassigned:
        print(
            f"\nWarning: {len(unassigned)} images had dates that are not in the "
            "Option 2 map and were left out of all splits."
        )

    return train_list, val_list, test_list


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Create a date-disjoint train/val/test split (Option 2: "
            "test = June 4–5, Sept 16, Sept 17)."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Example:\n"
            "  python tools/split_dataset_date_disjoint.py \\\n"
            "      --images_dir yolo_dataset/images \\\n"
            "      --output_dir yolo_dataset"
        ),
    )
    parser.add_argument(
        "--images_dir",
        default="yolo_dataset/images",
        help="Directory containing YOLO images.",
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        help="Directory for train.txt/val.txt/test.txt (default: parent of images_dir).",
    )
    parser.add_argument(
        "--no_yaml",
        action="store_true",
        help="Do not write/update data.yaml.",
    )
    args = parser.parse_args()

    create_date_disjoint_split(
        images_dir=args.images_dir,
        output_dir=args.output_dir,
        write_yaml=not args.no_yaml,
    )


if __name__ == "__main__":
    main()
