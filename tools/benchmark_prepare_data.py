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
) -> tuple[str, str]:
    """Filter a COCO JSON into train and val subsets matching the txt splits.

    Args:
        annotation_file: Original full COCO JSON path.
        train_txt: Path to train.txt (one absolute image path per line).
        val_txt: Path to val.txt (one absolute image path per line).
        output_dir: Directory to write train_split.json and val_split.json.

    Returns:
        Paths to (train_split.json, val_split.json).
    """
    with open(annotation_file) as f:
        data = json.load(f)

    def _names(txt: str) -> set[str]:
        with open(txt) as f:
            return {os.path.basename(line.strip()) for line in f if line.strip()}

    train_files = _names(train_txt)
    val_files = _names(val_txt)

    base = {
        "info": data.get("info", {}),
        "licenses": data.get("licenses", []),
        "categories": data["categories"],
    }
    train_data: dict = {**base, "images": [], "annotations": []}
    val_data: dict = {**base, "images": [], "annotations": []}

    train_ids: set[int] = set()
    val_ids: set[int] = set()
    for img in data["images"]:
        fname = os.path.basename(img["file_name"])
        if fname in train_files:
            train_data["images"].append(img)
            train_ids.add(img["id"])
        elif fname in val_files:
            val_data["images"].append(img)
            val_ids.add(img["id"])

    for ann in data["annotations"]:
        if ann["image_id"] in train_ids:
            train_data["annotations"].append(ann)
        elif ann["image_id"] in val_ids:
            val_data["annotations"].append(ann)

    train_json = os.path.join(output_dir, "train_split.json")
    val_json = os.path.join(output_dir, "val_split.json")
    for path, split in ((train_json, train_data), (val_json, val_data)):
        with open(path, "w") as f:
            json.dump(split, f)

    print(
        f"COCO splits written: {len(train_data['images'])} train / "
        f"{len(val_data['images'])} val images"
    )
    return train_json, val_json


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
        "train": os.path.join(abs_dir, "train.txt"),
        "val": os.path.join(abs_dir, "val.txt"),
        "nc": 1,
        "names": {0: "chimp"},
    }
    yaml_path = os.path.join(abs_dir, "data.yaml")
    with open(yaml_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False)
    return yaml_path


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

    print("\nDataset preparation complete.")
    print(f"  YOLO config : {yaml_path}")
    print(f"  MMDet train : {os.path.join(args.output_dir, 'train_split.json')}")
    print(f"  MMDet val   : {os.path.join(args.output_dir, 'val_split.json')}")


if __name__ == "__main__":
    main()
