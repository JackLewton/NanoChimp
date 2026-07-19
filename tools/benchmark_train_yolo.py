#!/usr/bin/env python3
"""
Fine-tune YOLO nano models on the chimpanzee detection benchmark dataset.

Trains YOLOv8n, YOLOv10n, and YOLOv11n sequentially on the same prepared
dataset split, using identical hyperparameters throughout for a fair comparison.

Run prepare_benchmark_data.py first to generate the shared dataset, then
activate the nanochimp environment (see requirements.txt) before running
this script.

Usage:
    conda activate nanochimp
    python tools/prepare_benchmark_data.py --annotation_file ... --image_dir ...
    python tools/train_yolo_benchmark.py --data_dir yolo_benchmark_dataset/
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
) -> dict | None:
    """Fine-tune a single YOLO model and return a summary dictionary.

    Args:
        name: Model identifier used for the output directory name.
        weights: Pretrained weights file (e.g. 'yolov8n.pt').
        data_yaml: Path to the Ultralytics data.yaml.
        args: Parsed CLI arguments.

    Returns:
        Dictionary of summary metrics, or None if training failed.
    """
    print(f"\n{'='*60}\nTraining {name}\n{'='*60}")

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
        name=name,
        exist_ok=True,
        save=True,
        plots=True,
        verbose=True,
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
        "framework":    "Ultralytics",
        "params_M":     round(params / 1e6, 2),
        "mAP50":        round(float(map50), 4),
        "mAP50_95":     round(float(map5095), 4),
        "inference_ms": round(float(infer_ms), 2),
        "save_dir":     str(results.save_dir),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fine-tune YOLO nano models on the chimpanzee benchmark dataset.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Run prepare_benchmark_data.py first.\n\n"
            "Example:\n"
            "  python tools/train_yolo_benchmark.py \\\n"
            "      --data_dir yolo_benchmark_dataset/ \\\n"
            "      --epochs 200 --batch_size 96"
        ),
    )
    parser.add_argument(
        "--data_dir", default="yolo_benchmark_dataset/",
        help="Directory produced by prepare_benchmark_data.py.",
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
    args = parser.parse_args()

    data_yaml = os.path.join(args.data_dir, "data.yaml")
    if not os.path.isfile(data_yaml):
        raise FileNotFoundError(
            f"{data_yaml} not found. Run prepare_benchmark_data.py first."
        )

    summaries = []
    for name, weights in MODELS.items():
        result = train_model(name, weights, data_yaml, args)
        if result:
            summaries.append(result)

    # Print results table
    print(f"\n{'='*80}")
    print("YOLO BENCHMARK RESULTS")
    print(f"{'='*80}")
    print(f"{'Model':<12} {'Params (M)':<12} {'mAP50':<10} {'mAP50-95':<12} {'Speed (ms)'}")
    print("-" * 80)
    for r in summaries:
        print(
            f"{r['model']:<12} {r['params_M']:<12.2f} "
            f"{r['mAP50']:<10.4f} {r['mAP50_95']:<12.4f} {r['inference_ms']:.2f}"
        )
    print(f"{'='*80}")

    os.makedirs(args.output_dir, exist_ok=True)
    results_path = os.path.join(args.output_dir, "yolo_benchmark_summary.json")
    with open(results_path, "w") as f:
        json.dump(summaries, f, indent=2)
    print(f"\nResults saved to: {results_path}")


if __name__ == "__main__":
    main()
