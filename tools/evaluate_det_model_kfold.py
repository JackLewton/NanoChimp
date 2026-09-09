#!/usr/bin/env python3
"""
Evaluate each k-fold detection model on that fold's held-out test split.

Prints per-fold Ultralytics metrics and mean ± SD across folds.

Usage:
    python tools/evaluate_det_model_kfold.py
    python tools/evaluate_det_model_kfold.py --runs \\
        bounding_box_model_fold_1_noaug_20260908_1959 \\
        bounding_box_model_fold_2_noaug_20260908_1959 \\
        bounding_box_model_fold_3_noaug_20260908_1959 \\
        bounding_box_model_fold_4_noaug_20260908_1959 \\
        bounding_box_model_fold_5_noaug_20260908_1959
"""

from __future__ import annotations

import argparse
import os
from typing import Dict, List, Optional

import numpy as np
from ultralytics import YOLO

METRIC_KEYS = (
    ("mAP50", "metrics/mAP50(B)"),
    ("mAP50-95", "metrics/mAP50-95(B)"),
    ("Precision", "metrics/precision(B)"),
    ("Recall", "metrics/recall(B)"),
)


def _f1(precision: float, recall: float) -> float:
    if precision + recall <= 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def _extract(results_dict: Dict[str, float]) -> Dict[str, float]:
    out = {name: float(results_dict[key]) for name, key in METRIC_KEYS}
    out["F1"] = _f1(out["Precision"], out["Recall"])
    return out


def _as_best_pt(path: str) -> Optional[str]:
    if os.path.isfile(path) and path.endswith(".pt"):
        return path
    candidate = os.path.join(path, "weights", "best.pt")
    if os.path.isfile(candidate):
        return candidate
    return None


def _resolve_run(weights_dir: str, run: str) -> str:
    """Resolve a run folder name or path to weights/best.pt."""
    direct = _as_best_pt(run)
    if direct:
        return direct
    candidates = [
        os.path.join(weights_dir, run),
        os.path.join("yolo_training", run),
        os.path.join("runs", "detect", "yolo_training", run),
        os.path.join("runs", "detect", weights_dir, run),
    ]
    for base in candidates:
        found = _as_best_pt(base)
        if found:
            return found
    raise FileNotFoundError(
        f"No best.pt for run '{run}'. Looked in:\n  " + "\n  ".join(candidates)
    )


def _resolve_weights(weights_dir: str, fold: int) -> str:
    return _resolve_run(weights_dir, f"bounding_box_model_fold_{fold}")


def evaluate_fold(weights: str, data_yaml: str, split: str, imgsz: int) -> Dict[str, float]:
    model = YOLO(weights)
    metrics = model.val(data=data_yaml, split=split, imgsz=imgsz, plots=False)
    return _extract(metrics.results_dict)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate k-fold YOLO models on the test split and average them."
    )
    parser.add_argument(
        "--weights_dir",
        default="runs/detect/yolo_training",
        help="Parent directory of the five run folders",
    )
    parser.add_argument(
        "--runs",
        nargs=5,
        metavar="RUN",
        default=None,
        help=(
            "Five run folders or best.pt paths in fold order 1–5 "
            "(e.g. bounding_box_model_fold_1_noaug_20260908_1959 ...). "
            "Default: bounding_box_model_fold_1 ... fold_5"
        ),
    )
    parser.add_argument(
        "--data_dir",
        default="yolo_dataset/folds",
        help="Directory containing fold_N/data.yaml",
    )
    parser.add_argument("--n_folds", type=int, default=5, help="Number of folds")
    parser.add_argument("--split", default="test", help="Ultralytics split to evaluate")
    parser.add_argument("--imgsz", type=int, default=640, help="Eval image size")
    args = parser.parse_args()

    if args.n_folds != 5 and args.runs is not None:
        parser.error("--runs lists five folders; use n_folds=5")

    run_names = args.runs or [f"bounding_box_model_fold_{i}" for i in range(1, args.n_folds + 1)]
    n_folds = len(run_names)

    rows: List[Dict[str, float]] = []
    names = ["mAP50", "mAP50-95", "Precision", "Recall", "F1"]

    print(f"Evaluating {n_folds} folds on split='{args.split}'\n")
    for fold, run in enumerate(run_names, start=1):
        weights = _resolve_run(args.weights_dir, run)
        data_yaml = os.path.join(args.data_dir, f"fold_{fold}", "data.yaml")
        if not os.path.isfile(data_yaml):
            raise FileNotFoundError(f"Missing {data_yaml}")

        print(f"--- Fold {fold}/{n_folds} ---")
        print(f"run:     {run}")
        print(f"weights: {weights}")
        print(f"data:    {data_yaml}")
        metrics = evaluate_fold(weights, data_yaml, args.split, args.imgsz)
        rows.append(metrics)
        print(
            "  ".join(f"{name}={metrics[name]:.4f}" for name in names)
        )
        print()

    print("=" * 60)
    print(f"{args.split} mean ± SD over {n_folds} folds")
    print("=" * 60)
    for name in names:
        values = np.array([row[name] for row in rows], dtype=float)
        mean = values.mean()
        std = values.std(ddof=1) if len(values) > 1 else 0.0
        print(f"{name:10s}  {mean:.4f} ± {std:.4f}")


if __name__ == "__main__":
    main()
