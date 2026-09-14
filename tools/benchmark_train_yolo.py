#!/usr/bin/env python3
"""
Fine-tune YOLO nano models on the chimpanzee detection benchmark dataset.

Trains YOLOv8n, YOLOv10n, and YOLOv11n sequentially on the same prepared
dataset split, using identical hyperparameters throughout for a fair comparison.

Run benchmark_prepare_data.py first to generate the shared dataset, then
activate the nanochimp environment (see requirements.txt) before running
this script.

Usage:
    conda activate nanochimp
    python tools/benchmark_prepare_data.py --annotation_file ... --image_dir ... --kfold
    python tools/benchmark_train_yolo.py --data_dir yolo_benchmark_dataset/ --kfold
"""

from __future__ import annotations

import argparse
import json
import os

import torch
from ultralytics import YOLO


# YOLOv8/10/11 nano pretrained weights; downloaded automatically on first run.
MODELS: dict[str, str] = {
    "YOLOv8n":  "yolov8n.pt",   
    "YOLOv10n": "yolov10n.pt", 
    "YOLOv11n": "yolo11n.pt",  
}


def train_model(
    name: str,
    weights: str,
    data_yaml: str,
    args: argparse.Namespace,
    fold: int | None = None,
) -> dict | None:
    """Fine-tune a single YOLO model and return a summary dictionary.

    Args:
        name: Model identifier used for the output directory name.
        weights: Pretrained weights file (e.g. 'yolov8n.pt').
        data_yaml: Path to the Ultralytics data.yaml.
        args: Parsed CLI arguments.
        fold: Optional 1-indexed fold number, appended to the run name.

    Returns:
        Dictionary of summary metrics, or None if training failed.
    """
    run_name = f"{name}_fold_{fold}" if fold is not None else name
    print(f"\n{'='*60}\nTraining {run_name}\n{'='*60}")

    model = YOLO(weights)
    params = sum(p.numel() for p in model.model.parameters())

    results = model.train(
        data=data_yaml,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch_size,
        patience=args.patience,
        device=0 if torch.cuda.is_available() else "cpu",
        project=args.output_dir,
        name=run_name,
        exist_ok=True,
        save=True,
        plots=True,
        verbose=True,
        save_period=-1,  # only last.pt and best.pt; results.csv still logs every epoch
    )

    try:
        metrics = results.results_dict
        map50    = metrics.get("metrics/mAP50(B)", 0.0)
        map5095  = metrics.get("metrics/mAP50-95(B)", 0.0)
        infer_ms = results.speed.get("inference", 0.0)
    except Exception:
        map50    = getattr(getattr(results, "box", None), "map50", 0.0)
        map5095  = getattr(getattr(results, "box", None), "map", 0.0)
        infer_ms = 0.0

    return {
        "model":        name,
        "fold":         fold,
        "framework":    "Ultralytics",
        "params_M":     round(params / 1e6, 2),
        "mAP50":        round(float(map50), 4),
        "mAP50_95":     round(float(map5095), 4),
        "inference_ms": round(float(infer_ms), 2),
        "save_dir":     str(results.save_dir),
    }


def _eval_test(save_dir: str, data_yaml: str, imgsz: int) -> tuple[float, float] | None:
    """Evaluate best.pt on the yaml test split, if one exists."""
    test_txt = os.path.join(os.path.dirname(data_yaml), "test.txt")
    best = os.path.join(save_dir, "weights", "best.pt")
    if not os.path.isfile(test_txt) or not os.path.isfile(best):
        return None
    metrics = YOLO(best).val(data=data_yaml, split="test", imgsz=imgsz, plots=False)
    d = metrics.results_dict
    return float(d.get("metrics/mAP50(B)", 0.0)), float(d.get("metrics/mAP50-95(B)", 0.0))


def _fold_jobs(
    data_dir: str, kfold: bool, n_folds: int, fold: int | None
) -> list[tuple[int | None, str]]:
    """Return (fold_number, data.yaml) pairs. fold_number is None for the single split."""
    if not kfold:
        yaml_path = os.path.join(data_dir, "data.yaml")
        if not os.path.isfile(yaml_path):
            raise FileNotFoundError(
                f"{yaml_path} not found. Run benchmark_prepare_data.py first."
            )
        return [(None, yaml_path)]
    folds = [fold] if fold is not None else list(range(1, n_folds + 1))
    jobs = []
    for f in folds:
        yaml_path = os.path.join(data_dir, "folds", f"fold_{f}", "data.yaml")
        if not os.path.isfile(yaml_path):
            raise FileNotFoundError(
                f"{yaml_path} not found. Re-run benchmark_prepare_data.py with --kfold."
            )
        jobs.append((f, yaml_path))
    return jobs


def _print_mean_sd(summaries: list[dict]) -> None:
    """Print mean ± SD test/val mAP per model when multiple folds were run."""
    names = sorted({r["model"] for r in summaries if r.get("fold") is not None})
    if not names:
        return
    print("\nMean ± SD across folds")
    print(f"{'Model':<12} {'mAP50':<18} {'mAP50-95'}")
    print("-" * 50)
    for name in names:
        rows = [r for r in summaries if r["model"] == name]
        m50 = [r["mAP50"] for r in rows]
        m95 = [r["mAP50_95"] for r in rows]
        n = len(rows)
        sd50 = (sum((x - sum(m50) / n) ** 2 for x in m50) / (n - 1)) ** 0.5 if n > 1 else 0.0
        sd95 = (sum((x - sum(m95) / n) ** 2 for x in m95) / (n - 1)) ** 0.5 if n > 1 else 0.0
        print(
            f"{name:<12} {sum(m50)/n:.4f} ± {sd50:.4f}   "
            f"{sum(m95)/n:.4f} ± {sd95:.4f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fine-tune YOLO nano models on the chimpanzee benchmark dataset.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Run benchmark_prepare_data.py first (add --kfold for 5-fold splits).\n\n"
            "Example:\n"
            "  python tools/benchmark_train_yolo.py --data_dir yolo_benchmark_dataset/ --kfold"
        ),
    )
    parser.add_argument(
        "--data_dir", default="yolo_benchmark_dataset/",
        help="Directory produced by benchmark_prepare_data.py.",
    )
    parser.add_argument("--epochs",     type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--imgsz",      type=int, default=640)
    parser.add_argument(
        "--patience", type=int, default=50,
        help="Early-stopping patience in epochs. Set to 0 to disable.",
    )
    parser.add_argument(
        "--output_dir", default="benchmark_results/",
        help="Root directory for training outputs.",
    )
    parser.add_argument(
        "--kfold", action="store_true",
        help="Train on video-grouped folds under <data_dir>/folds/.",
    )
    parser.add_argument("--n_folds", type=int, default=5, help="Folds to run with --kfold.")
    parser.add_argument(
        "--fold", type=int, default=None,
        help="If --kfold, train only this fold (1-indexed).",
    )
    args = parser.parse_args()
    if args.fold is not None and not args.kfold:
        parser.error("--fold requires --kfold")

    jobs = _fold_jobs(args.data_dir, args.kfold, args.n_folds, args.fold)

    summaries = []
    for name, weights in MODELS.items():
        for fold, data_yaml in jobs:
            result = train_model(name, weights, data_yaml, args, fold=fold)
            if not result:
                continue
            test = _eval_test(result["save_dir"], data_yaml, args.imgsz)
            if test:
                result["mAP50"], result["mAP50_95"] = round(test[0], 4), round(test[1], 4)
                result["split"] = "test"
            else:
                result["split"] = "val"
            summaries.append(result)

    print(f"\n{'='*90}")
    print("YOLO BENCHMARK RESULTS")
    print(f"{'='*90}")
    print(
        f"{'Model':<12} {'Fold':<6} {'Split':<6} {'Params (M)':<12} "
        f"{'mAP50':<10} {'mAP50-95':<12} {'Speed (ms)'}"
    )
    print("-" * 90)
    for r in summaries:
        fold = r["fold"] if r["fold"] is not None else "-"
        print(
            f"{r['model']:<12} {fold:<6} {r['split']:<6} {r['params_M']:<12.2f} "
            f"{r['mAP50']:<10.4f} {r['mAP50_95']:<12.4f} {r['inference_ms']:.2f}"
        )
    print(f"{'='*90}")
    _print_mean_sd(summaries)

    os.makedirs(args.output_dir, exist_ok=True)
    results_path = os.path.join(args.output_dir, "yolo_benchmark_summary.json")
    with open(results_path, "w") as f:
        json.dump(summaries, f, indent=2)
    print(f"\nResults saved to: {results_path}")


if __name__ == "__main__":
    main()
