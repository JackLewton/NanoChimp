#!/usr/bin/env python3
"""
Evaluate a YOLO detector separately on side-view and aerial-view images.

Aerial = filenames containing -D1- or -D2-; everything else is side.

Prints Images / Precision / Recall / F1 / mAP50 / mAP50:95 per camera view.

Usage:
    python tools/evaluate_by_camera.py --weights path/to/best.pt \\
        --data yolo_dataset/data.yaml --split val
    python tools/evaluate_by_camera.py --fold 1 --split test
    python tools/evaluate_by_camera.py --split test
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
from typing import Dict, List, Optional, Tuple

import yaml
from ultralytics import YOLO

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from tools.evaluate_det_model_kfold import _resolve_run

CAMERA_ORDER = ("side", "aerial", "all")


def camera_view(image_path: str) -> str:
    """Return 'aerial' for D1/D2 cameras, otherwise 'side'."""
    name = os.path.basename(image_path)
    if "-D1-" in name or "-D2-" in name:
        return "aerial"
    return "side"


def _f1(precision: float, recall: float) -> float:
    if precision + recall <= 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def load_split_images(split_file: str) -> Dict[str, List[str]]:
    groups: Dict[str, List[str]] = {key: [] for key in CAMERA_ORDER}
    with open(split_file, encoding="utf-8") as f:
        for line in f:
            path = line.strip()
            if not path:
                continue
            groups[camera_view(path)].append(path)
            groups["all"].append(path)
    return groups


def resolve_split_file(data_yaml: str, split: str) -> Tuple[str, str]:
    """Return (dataset root, path to the split .txt) from a YOLO data.yaml."""
    with open(data_yaml, encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    yaml_dir = os.path.dirname(os.path.abspath(data_yaml))
    dataset_path = os.path.abspath(cfg.get("path") or yaml_dir)

    rel = cfg.get(split)
    candidates = []
    if rel:
        candidates.append(rel if os.path.isabs(rel) else os.path.join(dataset_path, rel))
    candidates.extend(
        (
            os.path.join(yaml_dir, f"{split}.txt"),
            os.path.join(dataset_path, f"{split}.txt"),
        )
    )
    for path in candidates:
        if os.path.isfile(path):
            return dataset_path, path
    raise FileNotFoundError(
        f"No {split} image list for {data_yaml}. Looked in:\n  "
        + "\n  ".join(candidates)
    )


def write_subset_yaml(
    image_list: List[str],
    data_yaml: str,
    dataset_path: str,
    temp_dir: str,
    subset_name: str,
) -> str:
    with open(data_yaml, encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    split_txt = os.path.join(temp_dir, f"{subset_name}.txt")
    with open(split_txt, "w", encoding="utf-8") as f:
        for path in image_list:
            f.write(path + "\n")

    subset_cfg = {
        "path": dataset_path,
        "train": cfg.get("train", "train.txt"),
        "val": split_txt,
        "nc": cfg.get("nc", 1),
        "names": cfg.get("names", {0: "chimp"}),
    }
    out = os.path.join(temp_dir, f"data_{subset_name}.yaml")
    with open(out, "w", encoding="utf-8") as f:
        yaml.dump(subset_cfg, f, default_flow_style=False)
    return out


def evaluate_subset(
    model: YOLO,
    data_yaml_path: str,
    imgsz: int,
    conf: float,
    iou: float,
    project: str,
    name: str,
) -> Dict[str, float]:
    results = model.val(
        data=data_yaml_path,
        conf=conf,
        iou=iou,
        imgsz=imgsz,
        verbose=True,
        project=project,
        name=name,
        save=False,
        plots=False,
    )
    precision = float(results.box.mp)
    recall = float(results.box.mr)
    return {
        "precision": precision,
        "recall": recall,
        "f1": _f1(precision, recall),
        "mAP50": float(results.box.map50),
        "mAP50_95": float(results.box.map),
    }


def print_table(title: str, results: Dict[str, Dict]) -> None:
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)
    print(
        "{:<12} {:>8} {:>10} {:>10} {:>10} {:>10} {:>12}".format(
            "Camera view", "Images", "Precision", "Recall", "F1", "mAP50", "mAP50:95"
        )
    )
    print("-" * 72)
    for view in CAMERA_ORDER:
        if view not in results:
            continue
        row = results[view]
        print(
            "{:<12} {:>8} {:>10.4f} {:>10.4f} {:>10.4f} {:>10.4f} {:>12.4f}".format(
                view.capitalize(),
                row["num_images"],
                row["precision"],
                row["recall"],
                row["f1"],
                row["mAP50"],
                row["mAP50_95"],
            )
        )
    print("-" * 72)


def print_mean_table(fold_results: List[Dict[str, Dict]]) -> None:
    import numpy as np

    print("\n" + "=" * 72)
    print(f"Mean ± SD over {len(fold_results)} folds")
    print("=" * 72)
    print(
        "{:<12} {:>8} {:>16} {:>16} {:>16} {:>16} {:>16}".format(
            "Camera view", "Images", "Precision", "Recall", "F1", "mAP50", "mAP50:95"
        )
    )
    print("-" * 72)
    for view in CAMERA_ORDER:
        rows = [fold[view] for fold in fold_results if view in fold]
        if not rows:
            continue
        images = int(round(float(np.mean([r["num_images"] for r in rows]))))
        cells = [view.capitalize(), f"{images:>8}"]
        for key in ("precision", "recall", "f1", "mAP50", "mAP50_95"):
            values = np.array([r[key] for r in rows], dtype=float)
            mean = values.mean()
            std = values.std(ddof=1) if len(values) > 1 else 0.0
            cells.append(f"{mean:.4f} ± {std:.4f}")
        print("{:<12} {} {:>16} {:>16} {:>16} {:>16} {:>16}".format(*cells))
    print("-" * 72)


def run_job(
    label: str,
    weights: str,
    data_yaml: str,
    split: str,
    imgsz: int,
    conf: float,
    iou: float,
) -> Dict[str, Dict]:
    dataset_path, split_file = resolve_split_file(data_yaml, split)
    groups = load_split_images(split_file)

    print(f"\n--- {label} ---")
    print(f"weights: {weights}")
    print(f"data:    {data_yaml}")
    print(f"split:   {split_file}")
    print(
        "images:  "
        f"side={len(groups['side'])}  "
        f"aerial={len(groups['aerial'])}  "
        f"all={len(groups['all'])}"
    )

    model = YOLO(weights)
    results: Dict[str, Dict] = {}
    temp_dir = tempfile.mkdtemp(prefix="camera_eval_")
    try:
        for view in CAMERA_ORDER:
            if not groups[view]:
                print(f"Skipping {view}: no images")
                continue
            subset_yaml = write_subset_yaml(
                groups[view], data_yaml, dataset_path, temp_dir, f"{label}_{view}"
            )
            metrics = evaluate_subset(
                model,
                subset_yaml,
                imgsz,
                conf,
                iou,
                temp_dir,
                f"{label}_{view}",
            )
            metrics["num_images"] = len(groups[view])
            results[view] = metrics
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

    print_table(f"{label}  ({split})", results)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Break YOLO detection metrics down by camera view (side vs aerial).",
    )
    parser.add_argument("--weights", default=None, help="Single best.pt (overrides k-fold)")
    parser.add_argument("--data", default=None, help="data.yaml used with --weights")
    parser.add_argument("--fold", type=int, default=None, help="Evaluate one fold only (1-indexed)")
    parser.add_argument("--n_folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42, help="Training seed in the run folder name")
    parser.add_argument("--weights_dir", default="runs/detect/yolo_training")
    parser.add_argument(
        "--data_dir",
        default="yolo_dataset/folds",
        help="Directory containing fold_N/data.yaml (k-fold mode)",
    )
    parser.add_argument("--split", default="test", help="Ultralytics split to evaluate")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument(
        "--conf",
        type=float,
        default=0.001,
        help="Confidence floor for mAP (Ultralytics P/R are reported at max-F1)",
    )
    parser.add_argument("--iou", type=float, default=0.5)
    parser.add_argument(
        "--output",
        default="evaluation_by_camera_results.json",
        help="JSON path for the tabulated metrics",
    )
    parser.add_argument(
        "--runs",
        nargs="+",
        metavar="RUN",
        default=None,
        help="Run folders or best.pt paths in fold order (k-fold mode)",
    )
    args = parser.parse_args()

    jobs: List[Tuple[str, str, str]] = []
    if args.weights:
        data_yaml = args.data or os.path.join("yolo_dataset", "data.yaml")
        if not os.path.isfile(data_yaml):
            parser.error(f"data.yaml not found: {data_yaml} (pass --data)")
        jobs.append(("single", args.weights, data_yaml))
    else:
        run_names = args.runs or [
            f"bounding_box_model_fold_{i}_seed{args.seed}"
            for i in range(1, args.n_folds + 1)
        ]
        folds = [args.fold] if args.fold is not None else list(range(1, len(run_names) + 1))
        for fold in folds:
            data_yaml = os.path.join(args.data_dir, f"fold_{fold}", "data.yaml")
            if not os.path.isfile(data_yaml):
                raise FileNotFoundError(data_yaml)
            run = run_names[fold - 1]
            jobs.append(
                (f"fold_{fold}", _resolve_run(args.weights_dir, run, "best"), data_yaml)
            )

    print("Detection evaluation by camera type")
    print(f"split={args.split}  conf={args.conf}  iou={args.iou}  imgsz={args.imgsz}")
    print("aerial = filenames containing -D1- or -D2-; side = everything else")

    all_results = []
    for label, weights, data_yaml in jobs:
        all_results.append(
            {
                "label": label,
                "weights": weights,
                "data": data_yaml,
                "views": run_job(
                    label, weights, data_yaml, args.split, args.imgsz, args.conf, args.iou
                ),
            }
        )

    if len(all_results) > 1:
        print_mean_table([item["views"] for item in all_results])

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
