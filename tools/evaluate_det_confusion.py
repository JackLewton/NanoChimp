#!/usr/bin/env python3
"""
Confusion matrix for YOLO detection on a held-out split (default: test).

Uses Ultralytics matching (IoU 0.5, confidence 0.25). For a single-class
detector the matrix is TP / FP / FN; true negatives are not defined.

Usage:
    python tools/evaluate_det_confusion.py
    python tools/evaluate_det_confusion.py --fold 1
    python tools/evaluate_det_confusion.py --weights path/to/best.pt \\
        --data yolo_dataset/folds/fold_1/data.yaml
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from ultralytics import YOLO

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from tools.evaluate_det_model_kfold import _resolve_weights


def _counts(matrix: np.ndarray) -> Tuple[int, int, int]:
    """TP, FP, FN from an Ultralytics detect matrix (pred × true, last index = background)."""
    nc = matrix.shape[0] - 1
    tp = int(np.trace(matrix[:nc, :nc]))
    fp = int(matrix[:nc, nc].sum())
    fn = int(matrix[nc, :nc].sum())
    return tp, fp, fn


def _plot(tp: int, fp: int, fn: int, title: str, out_path: str) -> None:
    # rows = true class, cols = predicted class (sklearn convention)
    cm = np.array([[tp, fn], [fp, 0]], dtype=int)
    fig, ax = plt.subplots(figsize=(5.2, 4.4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1], labels=["chimp", "background"])
    ax.set_yticks([0, 1], labels=["chimp", "background"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    thresh = cm.max() / 2 if cm.max() else 0
    texts = [["TP", "FN"], ["FP", "—"]]
    for i in range(2):
        for j in range(2):
            label = f"{texts[i][j]}\n{cm[i, j]}" if texts[i][j] != "—" else "n/a"
            ax.text(
                j, i, label, ha="center", va="center", fontsize=12,
                color="white" if cm[i, j] > thresh and texts[i][j] != "—" else "black",
            )
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"Saved {out_path}")


def evaluate(
    weights: str,
    data_yaml: str,
    split: str,
    imgsz: int,
    conf: float,
    iou: float,
    project: str,
    name: str,
):
    model = YOLO(weights)
    metrics = model.val(
        data=data_yaml,
        split=split,
        imgsz=imgsz,
        conf=conf,
        iou=iou,
        plots=True,
        project=project,
        name=name,
        exist_ok=True,
    )
    matrix = np.array(metrics.confusion_matrix.matrix, dtype=int)
    tp, fp, fn = _counts(matrix)
    return matrix, tp, fp, fn


def main() -> None:
    parser = argparse.ArgumentParser(description="Detection confusion matrix on the test split.")
    parser.add_argument("--weights", default=None, help="Single best.pt (overrides --fold / k-fold)")
    parser.add_argument("--data", default=None, help="data.yaml for --weights")
    parser.add_argument("--fold", type=int, default=None, help="Run one fold only (1-indexed)")
    parser.add_argument("--n_folds", type=int, default=5)
    parser.add_argument("--weights_dir", default="runs/detect/yolo_training")
    parser.add_argument("--data_dir", default="yolo_dataset/folds")
    parser.add_argument("--split", default="test")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence for TP/FP/FN matching")
    parser.add_argument("--iou", type=float, default=0.5, help="IoU for TP matching")
    parser.add_argument("--output_dir", default="runs/detect/confusion_test")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    jobs: List[Tuple[str, str, str]] = []
    if args.weights:
        if not args.data:
            parser.error("--data is required with --weights")
        jobs.append(("single", args.weights, args.data))
    else:
        folds = [args.fold] if args.fold is not None else list(range(1, args.n_folds + 1))
        for fold in folds:
            data_yaml = os.path.join(args.data_dir, f"fold_{fold}", "data.yaml")
            if not os.path.isfile(data_yaml):
                raise FileNotFoundError(data_yaml)
            jobs.append((f"fold_{fold}", _resolve_weights(args.weights_dir, fold), data_yaml))

    print(f"split={args.split}  conf={args.conf}  iou={args.iou}\n")
    summed: Optional[np.ndarray] = None

    for label, weights, data_yaml in jobs:
        print(f"--- {label} ---")
        print(f"weights: {weights}")
        matrix, tp, fp, fn = evaluate(
            weights, data_yaml, args.split, args.imgsz, args.conf, args.iou,
            args.output_dir, label,
        )
        p = tp / (tp + fp) if (tp + fp) else 0.0
        r = tp / (tp + fn) if (tp + fn) else 0.0
        print(f"TP={tp}  FP={fp}  FN={fn}  P={p:.4f}  R={r:.4f}")
        print(f"Ultralytics matrix (rows=pred, cols=true, last=background):\n{matrix}\n")
        _plot(tp, fp, fn, f"{label} {args.split}", os.path.join(args.output_dir, f"{label}_confusion.png"))
        summed = matrix if summed is None else summed + matrix

    if summed is not None and len(jobs) > 1:
        tp, fp, fn = _counts(summed)
        p = tp / (tp + fp) if (tp + fp) else 0.0
        r = tp / (tp + fn) if (tp + fn) else 0.0
        print("=" * 50)
        print(f"All folds pooled  TP={tp}  FP={fp}  FN={fn}  P={p:.4f}  R={r:.4f}")
        print(f"{summed}")
        _plot(tp, fp, fn, f"all folds {args.split}", os.path.join(args.output_dir, "all_folds_confusion.png"))


if __name__ == "__main__":
    main()
