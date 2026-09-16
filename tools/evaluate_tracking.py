#!/usr/bin/env python3
"""
Evaluate Multi-Object Tracking (MOT) performance using standard MOTChallenge metrics.

Computes standard tracking metrics (MOTA, IDF1, MOTP, Mostly Tracked, Mostly Lost,
ID Switches, Fragmentations, etc.) using the standard motmetrics library.
"""

from __future__ import annotations

import argparse
import os
from typing import Optional

import motmetrics as mm


def compute_mot_metrics(
    gt_file: str,
    ts_file: str,
    max_frame: Optional[int] = None,
):
    """Return (metrics dict, motmetrics summary) or (None, None) on failure."""
    if not os.path.exists(gt_file) or not os.path.exists(ts_file):
        return None, None

    gt = mm.io.loadtxt(gt_file, fmt="mot15-2D")
    ts = mm.io.loadtxt(ts_file, fmt="mot15-2D")
    acc = mm.MOTAccumulator(auto_id=True)

    gt_max = int(gt.index.get_level_values("FrameId").max())
    ts_max = int(ts.index.get_level_values("FrameId").max()) if len(ts) else 0
    last = min(max_frame, gt_max) if max_frame else max(gt_max, ts_max)

    for frame_id in range(1, last + 1):
        gt_frame = gt[gt.index.get_level_values("FrameId") == frame_id]
        ts_frame = ts[ts.index.get_level_values("FrameId") == frame_id]
        distances = mm.distances.iou_matrix(
            gt_frame[["X", "Y", "Width", "Height"]].values,
            ts_frame[["X", "Y", "Width", "Height"]].values,
            max_iou=0.5,
        )
        acc.update(
            gt_frame.index.get_level_values("Id").values,
            ts_frame.index.get_level_values("Id").values,
            distances,
        )

    mh = mm.metrics.create()
    summary = mh.compute(acc, metrics=mm.metrics.motchallenge_metrics, name="overall")
    row = summary.loc["overall"]
    metrics = {
        "mota": float(row["mota"]) * 100.0,
        "idf1": float(row["idf1"]) * 100.0,
        "ids": int(row["num_switches"]),
        "precision": float(row["precision"]) * 100.0,
        "recall": float(row["recall"]) * 100.0,
        "fp": int(row["num_false_positives"]),
        "fn": int(row["num_misses"]),
        "fm": int(row["num_fragmentations"]),
        "motp": float(row["motp"]),
        "frames": last,
    }
    return metrics, summary


def evaluate_mot(gt_file: str, ts_file: str):
    """
    Load MOT ground truth and tracker results and compute MOTChallenge metrics.
    """
    if not os.path.exists(gt_file):
        print(f"Error: Ground truth file not found: {gt_file}")
        return

    if not os.path.exists(ts_file):
        print(f"Error: Tracking results file not found: {ts_file}")
        return

    print(f"\n{'='*60}")
    print("Evaluating Multi-Object Tracking (MOT) Performance")
    print(f"{'='*60}")
    print(f"Ground Truth:      {gt_file}")
    print(f"Tracking Results:  {ts_file}")
    print("-" * 60)

    try:
        metrics, summary = compute_mot_metrics(gt_file, ts_file)
    except Exception as e:
        print(f"Error loading MOT files: {e}")
        print("Please ensure your files are in the standard MOT15 format.")
        return

    if metrics is None or summary is None:
        print("Error: could not compute MOT metrics.")
        return

    print(f"Processing {metrics['frames']} frames...")
    print("Computing MOTChallenge metrics...")

    mh = mm.metrics.create()
    str_summary = mm.io.render_summary(
        summary,
        formatters=mh.formatters,
        namemap=mm.io.motchallenge_metric_names,
    )

    print("\n" + "=" * 90)
    print("MOTCHALLENGE EVALUATION SUMMARY")
    print("=" * 90)
    print(str_summary)
    print("=" * 90 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate Multi-Object Tracking (MOT) results using MOTChallenge metrics.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python tools/evaluate_tracking.py --gt data/gt.txt --ts infer_output/tracking_results.txt
        """,
    )
    parser.add_argument(
        "--gt",
        type=str,
        required=True,
        help="Path to the ground truth file (MOT15-2D format)",
    )
    parser.add_argument(
        "--ts",
        type=str,
        required=True,
        help="Path to the tracking results file (MOT15-2D format)",
    )

    args = parser.parse_args()
    evaluate_mot(args.gt, args.ts)


if __name__ == "__main__":
    main()
