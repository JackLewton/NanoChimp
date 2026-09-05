#!/usr/bin/env python3
"""
Video-grouped 5-fold split for a YOLO images folder.

Same grouping as tools/split_dataset.py: frames from one source video stay
in one split. Videos are shuffled, then KFold holds out one test fold each
time (~20% of videos). From the remainder, ~15% of all videos go to val
and the rest to train (~65/15/20 of videos).

Each fold writes train.txt, val.txt, and test.txt under output_dir/fold_N/.

Usage:
    python tools/split_dataset_kfold.py --images_dir yolo_dataset/images
"""

from __future__ import annotations

import argparse
import os
import sys
from collections import defaultdict
from typing import Dict, List, Optional, Sequence

import numpy as np
from sklearn.model_selection import KFold, train_test_split

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from tools.split_dataset import _extract_video_id, _list_images


def create_kfold_splits(
    images_dir: str,
    n_splits: int = 5,
    val_ratio: float = 0.15,
    seed: int = 42,
    output_dir: Optional[str] = None,
    image_suffixes: Sequence[str] = (".jpg", ".jpeg", ".png", ".bmp", ".tiff"),
) -> List[str]:
    """Write one train/val/test split per fold, grouped by video ID.

    Args:
        images_dir: Directory containing images.
        n_splits: Number of folds (each video is test in exactly one fold).
        val_ratio: Fraction of all videos assigned to validation in each fold.
        seed: Random seed for KFold shuffle and the val split.
        output_dir: Directory that will contain fold_1/, fold_2/, ...
        image_suffixes: Image extensions to include.

    Returns:
        Paths to the fold directories that were written.
    """
    images_dir = os.path.abspath(images_dir)
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(images_dir), "folds")
    output_dir = os.path.abspath(output_dir)

    video_to_images: Dict[str, List[str]] = defaultdict(list)
    for filename in _list_images(images_dir, image_suffixes):
        video_to_images[_extract_video_id(filename)].append(
            os.path.join(images_dir, filename)
        )

    video_ids = np.array(sorted(video_to_images.keys()))
    if len(video_ids) < n_splits:
        raise ValueError(
            f"Need at least {n_splits} videos for {n_splits}-fold; "
            f"found {len(video_ids)}"
        )

    val_from_remaining = val_ratio / (1.0 - 1.0 / n_splits)
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    fold_dirs: List[str] = []

    for fold, (trainval_idx, test_idx) in enumerate(kf.split(video_ids), start=1):
        trainval_ids = video_ids[trainval_idx]
        test_ids = video_ids[test_idx]
        train_ids, val_ids = train_test_split(
            trainval_ids,
            test_size=val_from_remaining,
            random_state=seed,
        )

        fold_dir = os.path.join(output_dir, f"fold_{fold}")
        os.makedirs(fold_dir, exist_ok=True)
        for name, ids in (("train", train_ids), ("val", val_ids), ("test", test_ids)):
            paths = sorted(p for vid in ids for p in video_to_images[vid])
            with open(os.path.join(fold_dir, f"{name}.txt"), "w", encoding="utf-8") as f:
                for line in paths:
                    f.write(line + "\n")

        fold_dirs.append(fold_dir)
        print(
            f"Fold {fold}/{n_splits}: "
            f"{len(train_ids)} train / {len(val_ids)} val / {len(test_ids)} test videos "
            f"({sum(len(video_to_images[v]) for v in train_ids)} / "
            f"{sum(len(video_to_images[v]) for v in val_ids)} / "
            f"{sum(len(video_to_images[v]) for v in test_ids)} images)"
        )

    return fold_dirs


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create video-grouped k-fold train/val/test splits."
    )
    parser.add_argument(
        "--images_dir",
        type=str,
        required=True,
        help="Directory containing images",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Parent directory for fold_N/ (default: <images_dir>/../folds)",
    )
    parser.add_argument(
        "--n_splits",
        type=int,
        default=5,
        help="Number of folds (default: 5)",
    )
    parser.add_argument(
        "--val_ratio",
        type=float,
        default=0.15,
        help="Fraction of all videos used for validation in each fold",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    args = parser.parse_args()

    create_kfold_splits(
        images_dir=args.images_dir,
        n_splits=args.n_splits,
        val_ratio=args.val_ratio,
        seed=args.seed,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
