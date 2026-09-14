#!/usr/bin/env python3
"""
Prepare the data for the detection model benchmark scripts.

The idea is to have a shared dataset for the benchmark so that the conditions are the same for the MMDet and YOLO models.

Converts COCO-format annotations to YOLO format, creates a reproducible
video-disjoint train/val split (seed=42), and writes COCO-format split
JSONs required by MMDetection.

Run this script before running these benchmark scripts:
    train_yolo_benchmark.py
    train_mmdet_benchmark.py

Both of the above training scripts read from the same output directory, which guarantees
all models are evaluated on an identical split.

Usage:
    python tools/prepare_benchmark_data.py \\
        --annotation_file data/annotations/annotations.json \\
        --image_dir data/images/ \\
        --output_dir yolo_benchmark_dataset/
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import yaml
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.split_dataset import create_train_val_split
from tools.split_dataset_kfold import create_kfold_splits


def convert_coco_to_yolo(annotation_file: str, image_dir: str, output_dir: str) -> int:
    """Convert COCO annotations to YOLO format and copy images.

    Bounding boxes are converted from absolute COCO [x, y, w, h] to
    YOLO normalised [cx, cy, w, h] format.  All instances are mapped to
    a single class (chimp = 0).

    Args:
        annotation_file: Path to COCO JSON annotation file.
        image_dir: Directory containing the source images.
        output_dir: Root output directory; images/ and labels/ are created here.

    Returns:
        Number of images successfully written.
    """
    print(f"Loading annotations from {annotation_file}...")
    with open(annotation_file) as f:
        data = json.load(f)

    images_dir = os.path.join(output_dir, "images")
    labels_dir = os.path.join(output_dir, "labels")

    for d in (images_dir, labels_dir):
        if os.path.exists(d):
            shutil.rmtree(d)
        os.makedirs(d)

    img_to_anns: dict[int, list] = {}
    for ann in data["annotations"]:
        img_to_anns.setdefault(ann["image_id"], []).append(ann)

    n_written = 0
    print(f"Processing {len(data['images'])} images...")
    for img_info in tqdm(data["images"], desc="Converting to YOLO format"):
        src = os.path.join(image_dir, img_info["file_name"])
        if not os.path.isfile(src):
            continue
        try:
            img = Image.open(src).convert("RGB")
        except Exception as e:
            print(f"Warning: could not open {src}: {e}")
            continue

        img.save(os.path.join(images_dir, img_info["file_name"]), quality=95)

        label_path = os.path.join(
            labels_dir,
            os.path.splitext(img_info["file_name"])[0] + ".txt",
        )
        W, H = img_info["width"], img_info["height"]
        with open(label_path, "w") as f:
            for ann in img_to_anns.get(img_info["id"], []):
                x, y, w, h = ann["bbox"]
                cx = (x + w / 2) / W
                cy = (y + h / 2) / H
                f.write(f"0 {cx:.6f} {cy:.6f} {w/W:.6f} {h/H:.6f}\n")

        n_written += 1

    return n_written


def create_coco_splits(
    annotation_file: str,
    train_txt: str,
    val_txt: str,
    output_dir: str,
    test_txt: str | None = None,
) -> dict[str, str]:
    """Filter a COCO JSON into subsets matching the txt splits.

    Args:
        annotation_file: Original full COCO JSON path.
        train_txt: Path to train.txt (one absolute image path per line).
        val_txt: Path to val.txt (one absolute image path per line).
        output_dir: Directory to write <split>_split.json files.
        test_txt: Optional path to test.txt.

    Returns:
        Mapping of split name to written JSON path.
    """
    with open(annotation_file) as f:
        data = json.load(f)

    def _names(txt: str) -> set[str]:
        with open(txt) as f:
            return {os.path.basename(line.strip()) for line in f if line.strip()}

    split_files = {
        "train": _names(train_txt),
        "val": _names(val_txt),
    }
    if test_txt:
        split_files["test"] = _names(test_txt)

    base = {
        "info": data.get("info", {}),
        "licenses": data.get("licenses", []),
        "categories": data["categories"],
    }
    splits: dict[str, dict] = {
        name: {**base, "images": [], "annotations": []} for name in split_files
    }
    id_to_split: dict[int, str] = {}
    for img in data["images"]:
        fname = os.path.basename(img["file_name"])
        for name, files in split_files.items():
            if fname in files:
                splits[name]["images"].append(img)
                id_to_split[img["id"]] = name
                break

    for ann in data["annotations"]:
        name = id_to_split.get(ann["image_id"])
        if name:
            splits[name]["annotations"].append(ann)

    written: dict[str, str] = {}
    for name, split in splits.items():
        path = os.path.join(output_dir, f"{name}_split.json")
        with open(path, "w") as f:
            json.dump(split, f)
        written[name] = path

    counts = " / ".join(f"{len(splits[n]['images'])} {n}" for n in splits)
    print(f"COCO splits written: {counts} images")
    return written


def write_yolo_yaml(output_dir: str) -> str:
    """Write a data.yaml compatible with Ultralytics YOLO.

    Args:
        output_dir: Directory containing train.txt and val.txt.

    Returns:
        Path to the written data.yaml.
    """
    abs_dir = os.path.abspath(output_dir)
    config = {
        "path": abs_dir,
        "train": "train.txt",
        "val": "val.txt",
        "nc": 1,
        "names": {0: "chimp"},
    }
    if os.path.isfile(os.path.join(abs_dir, "test.txt")):
        config["test"] = "test.txt"
    yaml_path = os.path.join(abs_dir, "data.yaml")
    with open(yaml_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False)
    return yaml_path


def write_fold_splits(annotation_file: str, output_dir: str, n_folds: int) -> list[str]:
    """Write video-grouped k-fold YOLO lists, data.yaml, and COCO JSONs."""
    fold_dirs = create_kfold_splits(
        images_dir=os.path.join(output_dir, "images"),
        n_splits=n_folds,
        val_ratio=0.15,
        seed=42,
        output_dir=os.path.join(output_dir, "folds"),
    )
    for fold_dir in fold_dirs:
        test_txt = os.path.join(fold_dir, "test.txt")
        create_coco_splits(
            annotation_file,
            os.path.join(fold_dir, "train.txt"),
            os.path.join(fold_dir, "val.txt"),
            fold_dir,
            test_txt=test_txt if os.path.isfile(test_txt) else None,
        )
        write_yolo_yaml(fold_dir)
    return fold_dirs


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare the shared benchmark dataset (run once before training).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Example:\n"
            "  python tools/prepare_benchmark_data.py \\\n"
            "      --annotation_file data/annotations/annotations.json \\\n"
            "      --image_dir data/images/ \\\n"
            "      --output_dir yolo_benchmark_dataset/"
        ),
    )
    parser.add_argument(
        "--annotation_file", default="data/annotations/annotations.json",
        help="Path to the COCO JSON annotation file.",
    )
    parser.add_argument(
        "--image_dir", default="data/images/",
        help="Directory containing source images.",
    )
    parser.add_argument(
        "--output_dir", default="yolo_benchmark_dataset/",
        help="Root output directory for the prepared dataset.",
    )
    parser.add_argument(
        "--kfold", action="store_true",
        help="Also write video-grouped k-fold splits under <output_dir>/folds/.",
    )
    parser.add_argument(
        "--n_folds", type=int, default=5,
        help="Number of folds when using --kfold (default: 5).",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Step 1: convert COCO annotations to YOLO format
    n = convert_coco_to_yolo(args.annotation_file, args.image_dir, args.output_dir)
    print(f"Wrote {n} images.\n")

    # Step 2: create a reproducible video-disjoint train/val split
    print("Creating train/val split (seed=42)...")
    train_txt, val_txt, n_train, n_val = create_train_val_split(
        images_dir=os.path.join(args.output_dir, "images"),
        val_ratio=0.2,
        seed=42,
        output_dir=args.output_dir,
    )
    print(f"Split: {n_train} train / {n_val} val images\n")

    # Step 3: write COCO-format split JSONs for MMDetection
    create_coco_splits(args.annotation_file, train_txt, val_txt, args.output_dir)

    # Step 4: write YOLO data.yaml for Ultralytics
    yaml_path = write_yolo_yaml(args.output_dir)

    if args.kfold:
        print(f"\nCreating video-grouped {args.n_folds}-fold splits...")
        write_fold_splits(args.annotation_file, args.output_dir, args.n_folds)

    print("\nDataset preparation complete.")
    print(f"  YOLO config : {yaml_path}")
    print(f"  MMDet train : {os.path.join(args.output_dir, 'train_split.json')}")
    print(f"  MMDet val   : {os.path.join(args.output_dir, 'val_split.json')}")
    if args.kfold:
        print(f"  K-fold dir  : {os.path.join(args.output_dir, 'folds')}")


if __name__ == "__main__":
    main()
