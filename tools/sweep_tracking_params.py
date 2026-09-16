#!/usr/bin/env python3
"""
Sweep tracker hyperparameters on the 5-minute MOT clip.

Runs one-at-a-time (OAT) ByteTrack changes around the current YAML baseline.
Not a full factorial — each row changes one thing so you can see which lever
moves MOTA / IDF1 / ID switches.

ByteTrack only. Does not write annotated video. Writes MOT .txt per config and a TSV table.

Usage (nanochimp env, on the GPU node):
    python tools/sweep_tracking_params.py \\
        --weights runs/detect/yolo_training/bounding_box_model_fold_4_seed42/weights/best.pt \\
        --video "temp/5 min track/Five_mins_NVR-0-Camera1-20250730091500-20250730092001.mp4" \\
        --gt "temp/5 min track/gt/gt.txt"

Cheap screen (first 500 frames only):
    python tools/sweep_tracking_params.py --max_frames 500 --suite quick
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from copy import deepcopy
from typing import Any, Dict, List, Optional

import yaml
from ultralytics import YOLO

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from tools.evaluate_tracking import compute_mot_metrics

COLUMNS = [
    "config",
    "tracker",
    "conf",
    "track_high_thresh",
    "track_low_thresh",
    "new_track_thresh",
    "match_thresh",
    "track_buffer",
    "fuse_score",
    "MOTA",
    "IDF1",
    "IDs",
    "Precision",
    "Recall",
    "FP",
    "FN",
    "FM",
    "fps",
    "seconds",
]


def _baseline_bytetrack() -> Dict[str, Any]:
    return {
        "tracker_type": "bytetrack",
        "track_high_thresh": 0.5,
        "track_low_thresh": 0.1,
        "new_track_thresh": 0.6,
        "track_buffer": 300,
        "match_thresh": 0.8,
        "fuse_score": False,
    }


def build_suite(suite: str) -> List[Dict[str, Any]]:
    """Named ByteTrack configs. Each dict is yaml keys plus 'conf' and 'name'."""
    runs: List[Dict[str, Any]] = []

    def add(name: str, yaml_cfg: Dict[str, Any], conf: float) -> None:
        row = deepcopy(yaml_cfg)
        row["name"] = name
        row["conf"] = conf
        runs.append(row)

    bt = _baseline_bytetrack()

    add("baseline_conf0.1", bt, 0.1)
    add("baseline_conf0.7", bt, 0.7)

    if suite == "quick":
        add("match_0.7", {**bt, "match_thresh": 0.7}, 0.1)
        add("high_0.3", {**bt, "track_high_thresh": 0.3}, 0.1)
        add("buffer_30", {**bt, "track_buffer": 30}, 0.1)
        return runs

    for v in (0.5, 0.6, 0.7, 0.9):
        add(f"match_{v}", {**bt, "match_thresh": v}, 0.1)

    for v in (0.3, 0.4, 0.6, 0.7):
        add(f"high_{v}", {**bt, "track_high_thresh": v}, 0.1)

    for v in (0.4, 0.5, 0.7):
        add(f"new_{v}", {**bt, "new_track_thresh": v}, 0.1)

    for v in (30, 75, 150):
        add(f"buffer_{v}", {**bt, "track_buffer": v}, 0.1)

    for v in (0.05, 0.2):
        add(f"low_{v}", {**bt, "track_low_thresh": v}, 0.1)

    add("fuse_true", {**bt, "fuse_score": True}, 0.1)
    return runs


def write_yaml(cfg: Dict[str, Any], path: str) -> None:
    dump = {k: v for k, v in cfg.items() if k not in ("name", "conf")}
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(dump, f, default_flow_style=False, sort_keys=False)


def run_track(
    model: YOLO,
    video: str,
    tracker_yaml: str,
    conf: float,
    out_txt: str,
    imgsz: int,
    max_frames: Optional[int],
) -> tuple[int, float]:
    """Write MOT txt; return (n_frames, elapsed_s). persist reset between configs."""
    model.predictor = None
    os.makedirs(os.path.dirname(os.path.abspath(out_txt)), exist_ok=True)
    t0 = time.perf_counter()
    n = 0
    with open(out_txt, "w", encoding="utf-8") as fh:
        for result in model.track(
            source=video,
            conf=conf,
            iou=0.5,
            imgsz=imgsz,
            stream=True,
            persist=True,
            tracker=tracker_yaml,
            verbose=False,
        ):
            n += 1
            if result.boxes is not None and result.boxes.id is not None:
                ids = result.boxes.id.int().cpu().tolist()
                boxes = result.boxes.xyxy.cpu().numpy()
                confs = result.boxes.conf.cpu().numpy()
                for box, track_id, score in zip(boxes, ids, confs):
                    x1, y1, x2, y2 = box
                    fh.write(
                        f"{n},{track_id},{x1:.2f},{y1:.2f},{x2 - x1:.2f},{y2 - y1:.2f},"
                        f"{score:.4f},-1,-1,-1\n"
                    )
            if max_frames and n >= max_frames:
                break
    elapsed = time.perf_counter() - t0
    return n, elapsed


def row_from(
    cfg: Dict[str, Any],
    metrics: Optional[Dict[str, float]],
    n_frames: int,
    elapsed: float,
) -> Dict[str, Any]:
    fps = (n_frames / elapsed) if elapsed > 0 else 0.0
    out: Dict[str, Any] = {
        "config": cfg["name"],
        "tracker": cfg.get("tracker_type", ""),
        "conf": cfg["conf"],
        "track_high_thresh": cfg.get("track_high_thresh", ""),
        "track_low_thresh": cfg.get("track_low_thresh", ""),
        "new_track_thresh": cfg.get("new_track_thresh", ""),
        "match_thresh": cfg.get("match_thresh", ""),
        "track_buffer": cfg.get("track_buffer", ""),
        "fuse_score": cfg.get("fuse_score", ""),
        "MOTA": "",
        "IDF1": "",
        "IDs": "",
        "Precision": "",
        "Recall": "",
        "FP": "",
        "FN": "",
        "FM": "",
        "fps": f"{fps:.2f}",
        "seconds": f"{elapsed:.1f}",
    }
    if metrics:
        out["MOTA"] = f"{metrics['mota']:.2f}"
        out["IDF1"] = f"{metrics['idf1']:.2f}"
        out["IDs"] = metrics["ids"]
        out["Precision"] = f"{metrics['precision']:.2f}"
        out["Recall"] = f"{metrics['recall']:.2f}"
        out["FP"] = metrics["fp"]
        out["FN"] = metrics["fn"]
        out["FM"] = metrics["fm"]
    return out


def print_table(rows: List[Dict[str, Any]]) -> None:
    show = [
        "config",
        "tracker",
        "conf",
        "match_thresh",
        "track_high_thresh",
        "track_buffer",
        "MOTA",
        "IDF1",
        "IDs",
        "fps",
    ]
    widths = {k: max(len(k), *(len(str(r[k])) for r in rows)) for k in show}
    header = "\t".join(k.ljust(widths[k]) for k in show)
    print("\n" + header)
    print("-" * len(header.expandtabs()))
    for r in rows:
        print("\t".join(str(r[k]).ljust(widths[k]) for k in show))


def write_tsv(path: str, rows: List[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="One-at-a-time tracker hyperparameter sweep on the MOT clip.",
    )
    parser.add_argument("--weights", required=True, help="YOLO best.pt (one fold is enough)")
    parser.add_argument(
        "--video",
        default="temp/5 min track/Five_mins_NVR-0-Camera1-20250730091500-20250730092001.mp4",
    )
    parser.add_argument("--gt", default="temp/5 min track/gt/gt.txt")
    parser.add_argument(
        "--output_dir",
        default="infer_output/tracking_sweeps",
        help="MOT txts + results.tsv",
    )
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument(
        "--max_frames",
        type=int,
        default=None,
        help="Score only the first N frames (cheap screen; also clips GT)",
    )
    parser.add_argument(
        "--suite",
        choices=("full", "quick"),
        default="full",
        help="full = OAT around ByteTrack baseline (~19 runs); quick = 5 runs",
    )
    args = parser.parse_args()

    if not os.path.isfile(args.weights):
        raise FileNotFoundError(args.weights)
    if not os.path.isfile(args.video):
        raise FileNotFoundError(args.video)
    if not os.path.isfile(args.gt):
        raise FileNotFoundError(args.gt)

    configs = build_suite(args.suite)
    print(f"{len(configs)} configs  suite={args.suite}  max_frames={args.max_frames or 'all'}")
    print("ByteTrack on this clip is ~18 min/run. Use --max_frames 500 first.")

    model = YOLO(args.weights)
    rows: List[Dict[str, Any]] = []
    tsv_path = os.path.join(args.output_dir, "results.tsv")

    for i, cfg in enumerate(configs, start=1):
        yaml_path = os.path.join(args.output_dir, "yamls", f"{cfg['name']}.yaml")
        os.makedirs(os.path.dirname(yaml_path), exist_ok=True)
        write_yaml(cfg, yaml_path)
        mot_path = os.path.join(args.output_dir, f"{cfg['name']}.txt")
        print(f"\n[{i}/{len(configs)}] {cfg['name']}")
        n, elapsed = run_track(
            model,
            args.video,
            yaml_path,
            cfg["conf"],
            mot_path,
            args.imgsz,
            args.max_frames,
        )
        metrics, _ = compute_mot_metrics(args.gt, mot_path, max_frame=args.max_frames)
        row = row_from(cfg, metrics, n, elapsed)
        rows.append(row)
        write_tsv(tsv_path, rows)
        if metrics:
            print(
                f"  MOTA={metrics['mota']:.2f}  IDF1={metrics['idf1']:.2f}  "
                f"IDs={metrics['ids']}  fps={n / elapsed:.2f}"
            )
        else:
            print("  MOT eval failed")

    print_table(rows)
    print(f"\nTable written to {tsv_path}")


if __name__ == "__main__":
    main()
