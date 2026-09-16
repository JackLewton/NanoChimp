#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
eval "$(conda shell.bash hook)"
mkdir -p logs
exec > >(tee -a logs/holiday_train.log) 2>&1

# Uses existing video-grouped folds under yolo_benchmark_dataset/folds/

echo "=== YOLO 5-fold (YOLOv8n, YOLOv10n, YOLOv11n) $(date) ==="
conda activate nanochimp
python tools/benchmark_train_yolo.py --data_dir yolo_benchmark_dataset/ \
    --kfold --output_dir benchmark_results_seed42 --epochs 250 --patience 0

echo "=== MMDet 5-fold (Faster R-CNN, FCOS) $(date) ==="
conda activate nanochimp-MMDet2
python tools/benchmark_train_mmdet.py --data_dir yolo_benchmark_dataset/ \
    --kfold --output_dir benchmark_results_seed42 --epochs 250

echo "=== DONE $(date) ==="
