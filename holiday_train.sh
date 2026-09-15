#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
eval "$(conda shell.bash hook)"
mkdir -p logs
exec > >(tee -a logs/holiday_train.log) 2>&1

echo "=== YOLO benchmark $(date) ==="
conda activate nanochimp
python tools/benchmark_train_yolo.py --data_dir yolo_benchmark_dataset/ \
    --kfold --output_dir benchmark_results_seed42 --epochs 250

echo "=== MMDet benchmark $(date) ==="
conda activate nanochimp-MMDet2
python tools/benchmark_train_mmdet.py --data_dir yolo_benchmark_dataset/ \
    --kfold --output_dir benchmark_results_seed42 --epochs 250

echo "=== YOLO11n seed 7 $(date) ==="
conda activate nanochimp
python configs/train.py --kfold --epochs 500 --batch_size 96 \
    --patience 0 --seed 7

echo "=== YOLO11n seed 2026 $(date) ==="
python configs/train.py --kfold --epochs 500 --batch_size 96 \
    --patience 0 --seed 2026

echo "=== DONE $(date) ==="
