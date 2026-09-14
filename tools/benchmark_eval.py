#!/usr/bin/env python3
"""
Compare benchmark detection runs: training curves and a test-set table.

Reads Ultralytics results.csv and MMDetection *.log.json from
benchmark_results/, plots mean validation mAP across folds, then
re-evaluates each fold's best checkpoint on the held-out test split.

Table values are mean ± 95% CI (Student's t) across folds.

Usage (nanochimp env; MMDet rows need nanochimp-MMDet2):
    python tools/benchmark_eval.py
    python tools/benchmark_eval.py --runs_dir benchmark_results --data_dir yolo_benchmark_dataset
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import re
import time
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np


DISPLAY_NAMES = {
    "YOLOv8n": "YOLOv8n",
    "YOLOv10n": "YOLOv10n",
    "YOLOv11n": "YOLOv11n",
    "FasterRCNN-R50-FPN": "Faster R-CNN",
    "FCOS-R50-FPN": "FCOS",
}

COLORS = {
    "YOLOv8n": "#1f77b4",
    "YOLOv10n": "#ff7f0e",
    "YOLOv11n": "#2ca02c",
    "FasterRCNN-R50-FPN": "#d62728",
    "FCOS-R50-FPN": "#9467bd",
}

# Two-sided 95% t critical values by degrees of freedom.
_T95 = {
    1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571,
    6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228,
}


def _t_crit(df: int) -> float:
    if df <= 0:
        return float("nan")
    if df in _T95:
        return _T95[df]
    return 1.96


def mean_sd_ci(values: list[float]) -> tuple[float, float, float]:
    """Return mean, sample SD, and 95% CI half-width."""
    arr = np.asarray(values, dtype=float)
    arr = arr[~np.isnan(arr)]
    n = len(arr)
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    mean = float(arr.mean())
    if n == 1:
        return mean, 0.0, 0.0
    sd = float(arr.std(ddof=1))
    ci = _t_crit(n - 1) * sd / np.sqrt(n)
    return mean, sd, float(ci)


def _fmt(mean: float, ci: float, digits: int = 3) -> str:
    if np.isnan(mean):
        return "—"
    if ci == 0 or np.isnan(ci):
        return f"{mean:.{digits}f}"
    return f"{mean:.{digits}f} ± {ci:.{digits}f}"


def _f1(precision: float, recall: float) -> float:
    if precision + recall <= 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


# Discover runs

def discover_runs(runs_dir: str) -> dict[str, dict[int | None, str]]:
    """Map model name -> {fold: run_dir}. fold is None for a single split."""
    found: dict[str, dict[int | None, str]] = defaultdict(dict)
    fold_re = re.compile(r"^(?P<model>.+)_fold_(?P<fold>\d+)$")
    if not os.path.isdir(runs_dir):
        raise FileNotFoundError(f"{runs_dir} not found")

    for name in sorted(os.listdir(runs_dir)):
        if name.startswith("_") or name.startswith("."):
            continue
        path = os.path.join(runs_dir, name)
        if not os.path.isdir(path):
            continue
        match = fold_re.match(name)
        if match:
            found[match.group("model")][int(match.group("fold"))] = path
        elif name in DISPLAY_NAMES:
            found[name][None] = path
    return dict(found)


# Training curves from logs

def _read_yolo_csv(run_dir: str) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    csv_path = os.path.join(run_dir, "results.csv")
    if not os.path.isfile(csv_path):
        return None
    epochs, m50, m95 = [], [], []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            return None
        rename = {k: k.strip() for k in reader.fieldnames}
        for row in reader:
            row = {rename[k]: v.strip() for k, v in row.items() if k}
            try:
                epochs.append(int(float(row["epoch"])))
                m50.append(float(row["metrics/mAP50(B)"]))
                m95.append(float(row["metrics/mAP50-95(B)"]))
            except (KeyError, ValueError):
                continue
    if not epochs:
        return None
    return np.array(epochs), np.array(m50), np.array(m95)


def _read_mmdet_log(run_dir: str) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    logs = sorted(glob.glob(os.path.join(run_dir, "*.log.json")))
    if not logs:
        return None
    epochs, m50, m95 = [], [], []
    with open(logs[-1]) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            if entry.get("mode") != "val" or "bbox_mAP" not in entry:
                continue
            epochs.append(int(entry.get("epoch", len(epochs) + 1)))
            m50.append(float(entry.get("bbox_mAP_50", 0.0)))
            m95.append(float(entry.get("bbox_mAP", 0.0)))
    if not epochs:
        return None
    return np.array(epochs), np.array(m50), np.array(m95)


def load_curves(run_dir: str) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    return _read_yolo_csv(run_dir) or _read_mmdet_log(run_dir)


def mean_curve(
    fold_curves: list[tuple[np.ndarray, np.ndarray, np.ndarray]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Align folds on epoch and return mean ± SD for mAP50 and mAP50-95."""
    max_epoch = max(int(ep.max()) for ep, _, _ in fold_curves)
    grid = np.arange(1, max_epoch + 1)
    m50 = np.full((len(fold_curves), len(grid)), np.nan)
    m95 = np.full_like(m50, np.nan)
    for i, (ep, a, b) in enumerate(fold_curves):
        for e, x, y in zip(ep, a, b):
            if 1 <= e <= max_epoch:
                m50[i, e - 1] = x
                m95[i, e - 1] = y
    with np.errstate(all="ignore"):
        mu50 = np.nanmean(m50, axis=0)
        mu95 = np.nanmean(m95, axis=0)
        if len(fold_curves) > 1:
            sd50 = np.nanstd(m50, axis=0, ddof=1)
            sd95 = np.nanstd(m95, axis=0, ddof=1)
        else:
            sd50 = sd95 = np.zeros(len(grid))
    return grid, mu50, np.nan_to_num(sd50), mu95, np.nan_to_num(sd95)


def plot_curves(
    models: dict[str, dict[int | None, str]],
    out_prefix: str,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.4), sharex=True)
    plotted = 0
    for model, folds in models.items():
        series = []
        for run_dir in folds.values():
            curve = load_curves(run_dir)
            if curve is not None:
                series.append(curve)
        if not series:
            print(f"No training log for {model}, skipping curve.")
            continue
        grid, mu50, sd50, mu95, sd95 = mean_curve(series)
        colour = COLORS.get(model, None)
        label = DISPLAY_NAMES.get(model, model)
        for ax, mu, sd, ylabel in (
            (axes[0], mu50, sd50, "mAP@0.5"),
            (axes[1], mu95, sd95, "mAP@0.5:0.95"),
        ):
            ax.plot(grid, mu, color=colour, lw=2.0, label=label)
            if np.nanmax(sd) > 0:
                ax.fill_between(grid, mu - sd, mu + sd, color=colour, alpha=0.18, linewidth=0)
            ax.set_xlabel("Epoch")
            ax.set_ylabel(ylabel)
            ax.grid(True, linestyle="--", alpha=0.45)
            ax.set_xlim(left=1)
            ax.set_ylim(bottom=0)
        plotted += 1

    if plotted == 0:
        plt.close(fig)
        print("No training curves found.")
        return

    axes[0].legend(frameon=False, loc="lower right")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        path = f"{out_prefix}.{ext}"
        fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
        print(f"Wrote {path}")
    plt.close(fig)


# Test-set evaluation

def _fold_yaml(data_dir: str, fold: int | None) -> str:
    if fold is None:
        return os.path.join(data_dir, "data.yaml")
    return os.path.join(data_dir, "folds", f"fold_{fold}", "data.yaml")


def _fold_test_json(data_dir: str, fold: int | None) -> str:
    if fold is None:
        return os.path.join(data_dir, "test_split.json")
    return os.path.join(data_dir, "folds", f"fold_{fold}", "test_split.json")


def _yolo_best(run_dir: str) -> str | None:
    path = os.path.join(run_dir, "weights", "best.pt")
    return path if os.path.isfile(path) else None


def _mmdet_best(run_dir: str) -> str | None:
    best = glob.glob(os.path.join(run_dir, "best_bbox_mAP_epoch_*.pth"))
    if best:
        return best[0]
    epochs = glob.glob(os.path.join(run_dir, "epoch_*.pth"))
    if not epochs:
        return None

    def _num(p: str) -> int:
        m = re.search(r"epoch_(\d+)\.pth$", p)
        return int(m.group(1)) if m else -1

    return max(epochs, key=_num)


def eval_yolo(run_dir: str, data_yaml: str, imgsz: int, split: str) -> dict | None:
    weights = _yolo_best(run_dir)
    if not weights or not os.path.isfile(data_yaml):
        return None
    from ultralytics import YOLO

    model = YOLO(weights)
    params = sum(p.numel() for p in model.model.parameters()) / 1e6
    metrics = model.val(
        data=data_yaml,
        split=split,
        imgsz=imgsz,
        plots=False,
        verbose=False,
        project=os.path.abspath(os.path.dirname(run_dir)),
        name="_eval",
        exist_ok=True,
    )
    d = metrics.results_dict
    precision = float(d.get("metrics/precision(B)", 0.0))
    recall = float(d.get("metrics/recall(B)", 0.0))
    return {
        "params_M": round(params, 2),
        "mAP50": float(d.get("metrics/mAP50(B)", 0.0)),
        "mAP50_95": float(d.get("metrics/mAP50-95(B)", 0.0)),
        "precision": precision,
        "recall": recall,
        "f1": _f1(precision, recall),
        "inference_ms": float(metrics.speed.get("inference", 0.0)),
    }


def eval_mmdet(run_dir: str, data_dir: str, test_json: str, device: str) -> dict | None:
    try:
        import torch
        from mmcv import Config as MMCVConfig
        from mmcv.parallel import MMDataParallel
        from mmcv.runner import load_checkpoint
        from mmdet.apis import single_gpu_test
        from mmdet.datasets import build_dataloader, build_dataset
        from mmdet.models import build_detector
    except ImportError:
        return None

    ckpt = _mmdet_best(run_dir)
    configs = glob.glob(os.path.join(run_dir, "*_config.py"))
    if not ckpt or not configs or not os.path.isfile(test_json):
        return None

    cfg = MMCVConfig.fromfile(configs[0])
    cfg.data.test.ann_file = os.path.abspath(test_json)
    cfg.data.test.img_prefix = os.path.join(os.path.abspath(data_dir), "images") + "/"
    cfg.data.test.test_mode = True

    dataset = build_dataset(cfg.data.test)
    loader = build_dataloader(
        dataset, samples_per_gpu=1, workers_per_gpu=0, dist=False, shuffle=False,
    )
    model = build_detector(cfg.model, test_cfg=cfg.get("test_cfg"))
    load_checkpoint(model, ckpt, map_location="cpu")
    model.CLASSES = dataset.CLASSES
    params = sum(p.numel() for p in model.parameters()) / 1e6
    model = MMDataParallel(model, device_ids=[0] if device == "cuda" else [])
    outputs = single_gpu_test(model, loader, show=False)
    coco = dataset.evaluate(outputs, metric="bbox")

    times = []
    model.eval()
    with torch.no_grad():
        for i, data in enumerate(loader):
            if i >= 55:
                break
            if device == "cuda":
                torch.cuda.synchronize()
            start = time.perf_counter()
            model(return_loss=False, rescale=True, **data)
            if device == "cuda":
                torch.cuda.synchronize()
            if i >= 5:
                times.append((time.perf_counter() - start) * 1000)

    precision, recall = _pr_at_iou(dataset, outputs, iou_thr=0.5, score_thr=0.25)
    return {
        "params_M": round(params, 2),
        "mAP50": float(coco.get("bbox_mAP_50", 0.0)),
        "mAP50_95": float(coco.get("bbox_mAP", 0.0)),
        "precision": precision,
        "recall": recall,
        "f1": _f1(precision, recall),
        "inference_ms": float(np.mean(times)) if times else 0.0,
    }


def _pr_at_iou(dataset, outputs, iou_thr: float = 0.5, score_thr: float = 0.25) -> tuple[float, float]:
    """Greedy IoU matching for a single-class detector (TP/FP/FN → P, R)."""
    tp = fp = fn = 0
    for i, pred in enumerate(outputs):
        boxes = pred[0] if isinstance(pred, list) else pred
        boxes = np.asarray(boxes)
        if boxes.size == 0:
            boxes = np.zeros((0, 5))
        boxes = boxes[boxes[:, 4] >= score_thr]
        ann = dataset.get_ann_info(i)
        gt = np.asarray(ann.get("bboxes", []), dtype=float)
        if gt.size == 0:
            fp += len(boxes)
            continue
        matched = np.zeros(len(gt), dtype=bool)
        order = np.argsort(-boxes[:, 4]) if len(boxes) else []
        for j in order:
            x1, y1, x2, y2 = boxes[j, :4]
            ious = _iou_xyxy(x1, y1, x2, y2, gt)
            k = int(np.argmax(ious)) if len(ious) else -1
            if k >= 0 and ious[k] >= iou_thr and not matched[k]:
                matched[k] = True
                tp += 1
            else:
                fp += 1
        fn += int((~matched).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return precision, recall


def _iou_xyxy(x1, y1, x2, y2, gt: np.ndarray) -> np.ndarray:
    gx1, gy1, gx2, gy2 = gt[:, 0], gt[:, 1], gt[:, 2], gt[:, 3]
    ix1 = np.maximum(x1, gx1)
    iy1 = np.maximum(y1, gy1)
    ix2 = np.minimum(x2, gx2)
    iy2 = np.minimum(y2, gy2)
    inter = np.clip(ix2 - ix1, 0, None) * np.clip(iy2 - iy1, 0, None)
    union = (x2 - x1) * (y2 - y1) + (gx2 - gx1) * (gy2 - gy1) - inter
    return np.where(union > 0, inter / union, 0.0)


def evaluate_models(
    models: dict[str, dict[int | None, str]],
    data_dir: str,
    imgsz: int,
    split: str,
) -> list[dict]:
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    rows = []
    for model, folds in models.items():
        is_yolo = model.startswith("YOLO")
        for fold, run_dir in sorted(folds.items(), key=lambda kv: (kv[0] is None, kv[0] or 0)):
            print(f"Evaluating {model} fold={fold if fold is not None else '-'} ...")
            if is_yolo:
                metrics = eval_yolo(run_dir, _fold_yaml(data_dir, fold), imgsz, split)
            else:
                test_json = _fold_test_json(data_dir, fold)
                if not os.path.isfile(test_json):
                    test_json = os.path.join(
                        data_dir if fold is None else os.path.join(data_dir, "folds", f"fold_{fold}"),
                        "val_split.json",
                    )
                metrics = eval_mmdet(run_dir, data_dir, test_json, device)
            if metrics is None:
                print("  skipped (missing weights, data, or MMDet env)")
                continue
            metrics.update({"model": model, "fold": fold})
            rows.append(metrics)
            print(
                f"  mAP50={metrics['mAP50']:.4f}  mAP50-95={metrics['mAP50_95']:.4f}  "
                f"P={metrics['precision']:.4f}  R={metrics['recall']:.4f}"
            )
    return rows


def aggregate(rows: list[dict]) -> list[dict]:
    by_model: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_model[row["model"]].append(row)
    summary = []
    keys = ("mAP50", "mAP50_95", "precision", "recall", "f1", "inference_ms")
    for model, items in by_model.items():
        entry = {
            "model": model,
            "display": DISPLAY_NAMES.get(model, model),
            "params_M": items[0]["params_M"],
            "n": len(items),
        }
        for key in keys:
            mean, sd, ci = mean_sd_ci([item[key] for item in items])
            entry[key] = mean
            entry[f"{key}_sd"] = sd
            entry[f"{key}_ci"] = ci
        summary.append(entry)
    order = list(DISPLAY_NAMES)
    summary.sort(key=lambda r: order.index(r["model"]) if r["model"] in order else 99)
    return summary


def write_table(summary: list[dict], out_csv: str, out_md: str) -> None:
    headers = [
        "Model", "Size (M)", "mAP@0.5", "mAP@0.5:0.95",
        "Precision", "Recall", "F1", "Speed (ms)",
    ]
    lines = []
    for row in summary:
        lines.append([
            row["display"],
            f"{row['params_M']:.1f}",
            _fmt(row["mAP50"], row["mAP50_ci"]),
            _fmt(row["mAP50_95"], row["mAP50_95_ci"]),
            _fmt(row["precision"], row["precision_ci"]),
            _fmt(row["recall"], row["recall_ci"]),
            _fmt(row["f1"], row["f1_ci"]),
            _fmt(row["inference_ms"], row["inference_ms_ci"], digits=1),
        ])

    width = [max(len(h), *(len(line[i]) for line in lines)) if lines else len(h)
             for i, h in enumerate(headers)]
    def _row(vals: list[str]) -> str:
        return "  ".join(v.ljust(w) for v, w in zip(vals, width))

    print()
    print("Test-set comparison  (mean ± 95% CI across folds)")
    print(_row(headers))
    print("  ".join("-" * w for w in width))
    for line in lines:
        print(_row(line))

    os.makedirs(os.path.dirname(os.path.abspath(out_csv)) or ".", exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(lines)
        writer.writerow([])
        writer.writerow(["n_folds", summary[0]["n"] if summary else 0])
        writer.writerow(["interval", "95% CI (Student t)"])
    print(f"Wrote {out_csv}")

    md = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    for line in lines:
        md.append("| " + " | ".join(line) + " |")
    with open(out_md, "w") as f:
        f.write("\n".join(md) + "\n")
    print(f"Wrote {out_md}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot benchmark training curves and a test-set comparison table.",
    )
    parser.add_argument("--runs_dir", default="benchmark_results",
                        help="Directory written by the benchmark train scripts.")
    parser.add_argument("--data_dir", default="yolo_benchmark_dataset",
                        help="Shared dataset from benchmark_prepare_data.py.")
    parser.add_argument("--output_dir", default="benchmark_results",
                        help="Where to write the figure and table.")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--split", default="test",
                        help="Ultralytics split name (test if k-fold data.yaml has it).")
    parser.add_argument("--curves_only", action="store_true",
                        help="Only plot training curves; skip test-set evaluation.")
    args = parser.parse_args()

    models = discover_runs(args.runs_dir)
    if not models:
        raise SystemExit(f"No run folders found in {args.runs_dir}")
    print("Found:", ", ".join(f"{m} ({len(f)} run(s))" for m, f in models.items()))

    os.makedirs(args.output_dir, exist_ok=True)
    plot_curves(models, os.path.join(args.output_dir, "benchmark_training_curves"))

    if args.curves_only:
        return

    rows = evaluate_models(models, args.data_dir, args.imgsz, args.split)
    if not rows:
        raise SystemExit("No models could be evaluated. Train first, or use --curves_only.")
    summary = aggregate(rows)
    write_table(
        summary,
        os.path.join(args.output_dir, "benchmark_comparison.csv"),
        os.path.join(args.output_dir, "benchmark_comparison.md"),
    )
    with open(os.path.join(args.output_dir, "benchmark_eval.json"), "w") as f:
        json.dump({"folds": rows, "summary": summary}, f, indent=2)


if __name__ == "__main__":
    main()
