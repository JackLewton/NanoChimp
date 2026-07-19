#!/usr/bin/env python3
"""
Evaluate a YOLO detection model.

mAP50 and mAP50:95 are computed via the official Ultralytics validation engine.
Precision, recall, and F1 are computed with scikit-learn.
"""

import os
import sys
import json
import glob
import argparse
import numpy as np
from pathlib import Path
from collections import defaultdict
from typing import List, Dict, Optional, Tuple

import matplotlib.pyplot as plt
try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False

from tqdm import tqdm
from ultralytics import YOLO
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


def load_annotations(annotation_file: str) -> Dict:
    """Load COCO format annotations."""
    with open(annotation_file) as f:
        return json.load(f)


def load_split_images(split_file: str) -> Optional[set]:
    """Load image filenames from a YOLO split .txt file."""
    if not os.path.exists(split_file):
        return None
    with open(split_file) as f:
        return {os.path.basename(line.strip()) for line in f if line.strip()}


def calculate_iou(box1: List[float], box2: List[float]) -> float:
    """Compute IoU between two boxes in [x1, y1, x2, y2] format."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    if x2 <= x1 or y2 <= y1:
        return 0.0
    inter = (x2 - x1) * (y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    return inter / (area1 + area2 - inter)


def match_predictions_to_gt(
    predictions: List[Dict],
    ground_truth: List[Dict],
    iou_threshold: float = 0.5
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Greedy IoU matching of predictions to ground truth boxes.

    Returns binary y_true and continuous y_score arrays compatible with
    scikit-learn metric functions.
    """
    preds_sorted = sorted(predictions, key=lambda x: x['confidence'], reverse=True)
    gt_matched = [False] * len(ground_truth)

    y_true = []
    y_score = []

    for pred in preds_sorted:
        best_iou = 0.0
        best_gt_idx = -1
        for gt_idx, gt in enumerate(ground_truth):
            if gt_matched[gt_idx]:
                continue
            iou = calculate_iou(pred['bbox'], gt['bbox'])
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = gt_idx

        if best_iou >= iou_threshold and best_gt_idx >= 0:
            y_true.append(1)
            gt_matched[best_gt_idx] = True
        else:
            y_true.append(0)
        y_score.append(pred['confidence'])

    # Unmatched ground truth boxes are false negatives
    for matched in gt_matched:
        if not matched:
            y_true.append(1)
            y_score.append(0.0)

    return np.array(y_true), np.array(y_score)


def find_optimal_threshold(
    y_true: np.ndarray,
    y_score: np.ndarray,
    thresholds: Optional[List[float]] = None
) -> Tuple[float, Dict]:
    """Find the confidence threshold that maximises F1-score."""
    if thresholds is None:
        thresholds = np.arange(0.05, 0.95, 0.05).tolist()

    best_f1 = 0.0
    best_threshold = 0.5
    best_metrics: Dict = {'precision': 0.0, 'recall': 0.0, 'f1': 0.0}

    for t in thresholds:
        y_pred = (y_score >= t).astype(int)
        p, r, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average='binary', zero_division=0
        )
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = t
            best_metrics = {'precision': float(p), 'recall': float(r), 'f1': float(f1)}

    return best_threshold, best_metrics


def run_inference(model: YOLO, image_path: str, conf: float = 0.001) -> List[Dict]:
    """Run YOLO inference and return detections as a list of dicts."""
    results = model(image_path, conf=conf, verbose=False)[0]
    detections = []
    if results.boxes is not None:
        for box, score, cls in zip(
            results.boxes.xyxy.cpu().numpy(),
            results.boxes.conf.cpu().numpy(),
            results.boxes.cls.cpu().numpy()
        ):
            detections.append({
                'bbox': box.tolist(),
                'confidence': float(score),
                'class_id': int(cls)
            })
    return detections


def evaluate_model(
    model_path: str,
    annotation_file: str,
    image_dir: str,
    split: str = 'val',
    conf_threshold: Optional[float] = None,
    iou_threshold: float = 0.5,
    output_dir: str = 'evaluation_results'
) -> Dict:
    """Evaluate a single model and return a results dict."""
    print(f"\n{'='*60}")
    print(f"Evaluating: {model_path}")
    print(f"{'='*60}")

    model = YOLO(model_path)
    data = load_annotations(annotation_file)

    img_to_anns: Dict[int, List] = defaultdict(list)
    for ann in data['annotations']:
        img_to_anns[ann['image_id']].append(ann)

    image_info_map = {img['id']: img for img in data['images']}

    # filter by split file if available
    split_images = None
    if split:
        split_file = os.path.join('yolo_dataset', f'{split}.txt')
        split_images = load_split_images(split_file)
        if split_images:
            print(f"Split file: {split_file} ({len(split_images)} images)")

    images = (
        {k: v for k, v in image_info_map.items() if v['file_name'] in split_images}
        if split_images else image_info_map
    )
    print(f"Images to evaluate: {len(images)}")

    all_predictions: List[Dict] = []
    all_ground_truth: List[Dict] = []

    for img_id, img_info in tqdm(images.items(), desc="Running inference"):
        img_path = os.path.join(image_dir, img_info['file_name'])
        if not os.path.exists(img_path):
            continue

        gt_boxes = [
            {
                'bbox': [
                    ann['bbox'][0], ann['bbox'][1],
                    ann['bbox'][0] + ann['bbox'][2],
                    ann['bbox'][1] + ann['bbox'][3]
                ],
                'class_id': ann.get('category_id', 0)
            }
            for ann in img_to_anns.get(img_id, [])
        ]
        all_predictions.extend(run_inference(model, img_path))
        all_ground_truth.extend(gt_boxes)

    print(f"Predictions: {len(all_predictions)}  |  Ground truth boxes: {len(all_ground_truth)}")

    # match predictions to ground truth for sklearn metrics
    y_true, y_score = match_predictions_to_gt(all_predictions, all_ground_truth, iou_threshold)

    # find optimal operating threshold
    print("Finding optimal confidence threshold...")
    optimal_threshold, _ = find_optimal_threshold(y_true, y_score)
    print(f"Optimal threshold: {optimal_threshold:.3f}")

    eval_threshold = conf_threshold if conf_threshold is not None else optimal_threshold
    y_pred = (y_score >= eval_threshold).astype(int)

    p, r, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='binary', zero_division=0
    )

    # official mAP via Ultralytics validation engine
    map50, map50_95 = 0.0, 0.0
    data_yaml = os.path.join('yolo_dataset', 'data.yaml')
    if os.path.exists(data_yaml):
        print("Computing mAP via Ultralytics validation...")
        try:
            val = model.val(
                data=data_yaml, conf=0.001, iou=0.5, verbose=False,
                project=output_dir, name='temp_val', save=False, plots=False
            )
            if hasattr(val, 'box'):
                map50 = float(val.box.map50)
                map50_95 = float(val.box.map)
            print(f"mAP50: {map50:.4f}  |  mAP50:95: {map50_95:.4f}")
        except Exception as e:
            print(f"Warning: Ultralytics validation failed: {e}")
    else:
        print(f"Warning: {data_yaml} not found — mAP metrics unavailable.")

    # confusion matrix via scikit-learn
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    print(f"Precision: {p:.4f}  Recall: {r:.4f}  F1: {f1:.4f}")
    print(f"TP: {tp}  FP: {fp}  FN: {fn}")

    return {
        'model_path': model_path,
        'eval_threshold': float(eval_threshold),
        'optimal_threshold': float(optimal_threshold),
        'precision': float(p),
        'recall': float(r),
        'f1': float(f1),
        'map50': map50,
        'map50_95': map50_95,
        'tp': int(tp),
        'fp': int(fp),
        'fn': int(fn),
        'num_images': len(images),
        'num_ground_truth': len(all_ground_truth),
        'num_predictions': len(all_predictions),
    }


def plot_confusion_matrix(results: Dict, output_dir: str):
    """Plot and save a TP/FP/FN confusion matrix for a single-class detector."""
    tp, fp, fn = results['tp'], results['fp'], results['fn']
    # TN is undefined in object detection (background is not explicitly annotated)
    cm = np.array([[0, fp], [fn, tp]])
    labels = ['No Object', 'Object']
    model_name = Path(results['model_path']).stem

    fig, ax = plt.subplots(figsize=(6, 5))
    if HAS_SEABORN:
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                    xticklabels=labels, yticklabels=labels)
    else:
        ax.imshow(cm, cmap='Blues')
        for i in range(2):
            for j in range(2):
                ax.text(j, i, cm[i, j], ha='center', va='center',
                        fontsize=14, fontweight='bold')
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(labels)
        ax.set_yticklabels(labels)

    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_title(f'Confusion Matrix — {model_name}')
    plt.tight_layout()

    out_path = os.path.join(output_dir, f'confusion_matrix_{model_name}.png')
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved confusion matrix: {out_path}")


def print_results_table(results_list: List[Dict]):
    """Print a formatted summary table to stdout."""
    print("\n" + "=" * 90)
    print("MODEL EVALUATION RESULTS")
    print("=" * 90)
    header = (
        f"{'Model':<30} {'P':<10} {'R':<10} {'F1':<10} "
        f"{'mAP50':<10} {'mAP50:95':<12} {'TP':<6} {'FP':<6} {'FN':<6}"
    )
    print(header)
    print("-" * 90)
    for r in results_list:
        name = Path(r['model_path']).stem
        print(
            f"{name:<30} {r['precision']:<10.4f} {r['recall']:<10.4f} {r['f1']:<10.4f} "
            f"{r['map50']:<10.4f} {r['map50_95']:<12.4f} {r['tp']:<6} {r['fp']:<6} {r['fn']:<6}"
        )
    print("=" * 90)


def save_results(results_list: List[Dict], output_dir: str):
    """Save results to JSON and a plain-text summary."""
    json_path = os.path.join(output_dir, 'evaluation_results.json')
    with open(json_path, 'w') as f:
        json.dump(results_list, f, indent=2)
    print(f"JSON results saved to {json_path}")

    txt_path = os.path.join(output_dir, 'evaluation_results.txt')
    with open(txt_path, 'w') as f:
        for r in results_list:
            name = Path(r['model_path']).stem
            f.write(f"Model: {name}\n")
            f.write(f"  Path:               {r['model_path']}\n")
            f.write(f"  Optimal threshold:  {r['optimal_threshold']:.4f}\n")
            f.write(f"  Eval threshold:     {r['eval_threshold']:.4f}\n")
            f.write(f"  Precision:          {r['precision']:.4f}\n")
            f.write(f"  Recall:             {r['recall']:.4f}\n")
            f.write(f"  F1:                 {r['f1']:.4f}\n")
            f.write(f"  mAP50:              {r['map50']:.4f}\n")
            f.write(f"  mAP50:95:           {r['map50_95']:.4f}\n")
            f.write(f"  TP / FP / FN:       {r['tp']} / {r['fp']} / {r['fn']}\n")
            f.write(f"  Images evaluated:   {r['num_images']}\n")
            f.write(f"  Ground truth boxes: {r['num_ground_truth']}\n")
            f.write("\n")
    print(f"Text summary saved to {txt_path}")


def find_latest_model() -> Optional[str]:
    """Find the most recently modified best.pt in standard Ultralytics output directories."""
    for search_dir in [os.path.join('runs', 'detect', 'yolo_training'), 'yolo_training']:
        if os.path.exists(search_dir):
            pt_files = glob.glob(
                os.path.join(search_dir, '**/weights/best.pt'), recursive=True
            )
            if pt_files:
                return max(pt_files, key=os.path.getmtime)
    return None


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate YOLO detection models and report publication metrics.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python tools/evaluate_models.py --model runs/detect/yolo_training/bounding_box_model/weights/best.pt
  python tools/evaluate_models.py --models model1.pt model2.pt
  python tools/evaluate_models.py --model model1.pt --conf_threshold 0.5
        """
    )
    parser.add_argument('--model', type=str,
                        help='Path to a single model file (.pt)')
    parser.add_argument('--models', nargs='+',
                        help='Paths to multiple model files (.pt)')
    parser.add_argument('--annotation_file', type=str,
                        default='data/annotations/annotations.json',
                        help='COCO format annotation file (default: data/annotations/annotations.json)')
    parser.add_argument('--image_dir', type=str,
                        default='data/images/',
                        help='Image directory (default: data/images/)')
    parser.add_argument('--split', type=str, default='val',
                        choices=['train', 'val', 'test', 'all'],
                        help='Dataset split to evaluate on (default: val)')
    parser.add_argument('--conf_threshold', type=float, default=None,
                        help='Confidence threshold; omit to use the optimal threshold')
    parser.add_argument('--iou_threshold', type=float, default=0.5,
                        help='IoU threshold for matching predictions to ground truth (default: 0.5)')
    parser.add_argument('--output_dir', type=str, default='evaluation_results',
                        help='Output directory for results and plots (default: evaluation_results)')

    args = parser.parse_args()

    if args.models:
        model_paths = args.models
    elif args.model:
        model_paths = [args.model]
    else:
        latest = find_latest_model()
        if latest:
            print(f"No model specified, using latest: {latest}")
            model_paths = [latest]
        else:
            parser.error("No model found. Specify --model or --models.")

    os.makedirs(args.output_dir, exist_ok=True)

    results_list = []
    for model_path in model_paths:
        if not os.path.exists(model_path):
            print(f"Warning: model not found: {model_path}")
            continue
        results = evaluate_model(
            model_path=model_path,
            annotation_file=args.annotation_file,
            image_dir=args.image_dir,
            split=args.split if args.split != 'all' else None,
            conf_threshold=args.conf_threshold,
            iou_threshold=args.iou_threshold,
            output_dir=args.output_dir
        )
        results_list.append(results)
        plot_confusion_matrix(results, args.output_dir)

    if not results_list:
        print("No models were successfully evaluated.")
        return

    print_results_table(results_list)
    save_results(results_list, args.output_dir)
    print(f"\nEvaluation complete. Results saved to: {args.output_dir}")


if __name__ == '__main__':
    main()
