#!/usr/bin/env python3
"""
Create train/val/test splits for a YOLO images folder based on video stem.

Same grouping as tools/split_dataset.py: frames from the same source video
stay in one split. This script also writes a held-out test list.

Videos (not images) are shuffled with a fixed seed, then partitioned by
clip count. Default ratios are 70% train / 15% val / 15% test.

Usage:
    python tools/split_dataset_with_test.py --images_dir yolo_dataset/images
"""

from __future__ import annotations

import argparse
import os
import random
import sys
from collections import defaultdict
from typing import Dict, List, Optional, Sequence, Tuple

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from tools.split_dataset import _extract_video_id, _list_images


def create_train_val_test_split(
    images_dir: str,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
    output_dir: Optional[str] = None,
    image_suffixes: Sequence[str] = (".jpg", ".jpeg", ".png", ".bmp", ".tiff"),
) -> Tuple[str, str, str, int, int, int]:
    """Split YOLO images by video ID into train, val, and test lists.

    Args:
        images_dir: Directory containing images.
        val_ratio: Fraction of videos assigned to validation.
        test_ratio: Fraction of videos assigned to test.
        seed: RNG seed used after sorting video IDs.
        output_dir: Directory for list files (default: parent of images_dir).
        image_suffixes: Image file extensions to include.

    Returns:
        Paths to train/val/test lists and the image counts in each.
    """
    if not os.path.isdir(images_dir):
        raise FileNotFoundError(f"Images directory not found: {images_dir}")
    if val_ratio < 0 or test_ratio < 0 or val_ratio + test_ratio >= 1:
        raise ValueError("val_ratio and test_ratio must be >= 0 and sum to less than 1.")

    output_dir = output_dir or os.path.dirname(os.path.abspath(images_dir))
    os.makedirs(output_dir, exist_ok=True)

    image_names = _list_images(images_dir, image_suffixes)
    image_paths = [os.path.abspath(os.path.join(images_dir, n)) for n in image_names]

    video_to_images: Dict[str, List[str]] = defaultdict(list)
    for img_path in image_paths:
        video_id = _extract_video_id(os.path.basename(img_path))
        video_to_images[video_id].append(img_path)

    for video_id in video_to_images:
        video_to_images[video_id].sort()

    video_ids = sorted(video_to_images.keys())
    random.Random(seed).shuffle(video_ids)

    n_videos = len(video_ids)
    n_test = max(1, int(n_videos * test_ratio)) if n_videos > 0 and test_ratio > 0 else 0
    n_val = max(1, int(n_videos * val_ratio)) if n_videos > 0 and val_ratio > 0 else 0
    if n_test + n_val >= n_videos:
        raise ValueError(
            f"Not enough videos ({n_videos}) for test={n_test} and val={n_val}."
        )

    test_video_ids = set(video_ids[:n_test])
    val_video_ids = set(video_ids[n_test : n_test + n_val])
    train_video_ids = set(video_ids[n_test + n_val :])

    train_paths, val_paths, test_paths = [], [], []
    for video_id in video_ids:
        images = video_to_images[video_id]
        if video_id in test_video_ids:
            test_paths.extend(images)
        elif video_id in val_video_ids:
            val_paths.extend(images)
        else:
            train_paths.extend(images)

    train_paths.sort()
    val_paths.sort()
    test_paths.sort()

    train_list = os.path.join(output_dir, "train.txt")
    val_list = os.path.join(output_dir, "val.txt")
    test_list = os.path.join(output_dir, "test.txt")
    for path, lines in (
        (train_list, train_paths),
        (val_list, val_paths),
        (test_list, test_paths),
    ):
        with open(path, "w", encoding="utf-8") as f:
            for line in lines:
                f.write(line + "\n")

    print(
        f"Split {n_videos} videos ({len(image_paths)} images) -> "
        f"train: {len(train_video_ids)} videos ({len(train_paths)} images), "
        f"val: {len(val_video_ids)} videos ({len(val_paths)} images), "
        f"test: {len(test_video_ids)} videos ({len(test_paths)} images)"
    )
    print(f"Wrote {train_list}, {val_list} and {test_list}")

    return train_list, val_list, test_list, len(train_paths), len(val_paths), len(test_paths)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create video-disjoint train/val/test splits for a YOLO images folder."
    )
    parser.add_argument(
        "--images_dir",
        required=True,
        help="Path to YOLO images directory (e.g. yolo_dataset/images).",
    )
    parser.add_argument(
        "--val_ratio",
        type=float,
        default=0.15,
        help="Fraction of videos for validation (default: 0.15).",
    )
    parser.add_argument(
        "--test_ratio",
        type=float,
        default=0.15,
        help="Fraction of videos for test (default: 0.15).",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42).")
    parser.add_argument(
        "--output_dir",
        default=None,
        help="Output directory for list files (default: parent of images_dir).",
    )
    args = parser.parse_args()

    create_train_val_test_split(
        images_dir=args.images_dir,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
