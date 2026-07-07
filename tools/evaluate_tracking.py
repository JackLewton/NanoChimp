#!/usr/bin/env python3
"""
Evaluate Multi-Object Tracking (MOT) performance using standard MOTChallenge metrics.

Computes standard tracking metrics (MOTA, IDF1, MOTP, Mostly Tracked, Mostly Lost, 
ID Switches, Fragmentations, etc.) using the standard motmetrics library.
"""

import os
import sys
import argparse
import numpy as np
import motmetrics as mm


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
        # Load ground truth and tracker results in MOT15-2D format
        gt = mm.io.loadtxt(gt_file, fmt='mot15-2D')
        ts = mm.io.loadtxt(ts_file, fmt='mot15-2D')
    except Exception as e:
        print(f"Error loading MOT files: {e}")
        print("Please ensure your files are in the standard MOT15 format.")
        return

    # Create an accumulator that will be updated during each frame
    acc = mm.MOTAccumulator(auto_id=True)

    # Determine maximum frame number across both files
    try:
        max_frame = int(max(
            gt.index.get_level_values('FrameId').max(),
            ts.index.get_level_values('FrameId').max()
        ))
    except Exception as e:
        print(f"Error determining frame range: {e}")
        return

    print(f"Processing {max_frame} frames...")

    for frame_id in range(1, max_frame + 1):
        gt_frame = gt[gt.index.get_level_values('FrameId') == frame_id]
        ts_frame = ts[ts.index.get_level_values('FrameId') == frame_id]

        gt_ids = gt_frame.index.get_level_values('Id').values
        ts_ids = ts_frame.index.get_level_values('Id').values

        gt_boxes = gt_frame[['X', 'Y', 'Width', 'Height']].values
        ts_boxes = ts_frame[['X', 'Y', 'Width', 'Height']].values

        # Compute IoU distance matrix (distances are 1 - IoU, thresholded at 0.5)
        distances = mm.distances.iou_matrix(gt_boxes, ts_boxes, max_iou=0.5)
        acc.update(gt_ids, ts_ids, distances)

    print("Computing MOTChallenge metrics...")
    mh = mm.metrics.create()
    summary = mh.compute(acc, metrics=mm.metrics.motchallenge_metrics, name='overall')
    
    str_summary = mm.io.render_summary(
        summary,
        formatters=mh.formatters,
        namemap=mm.io.motchallenge_metric_names
    )
    
    print("\n" + "=" * 90)
    print("MOTCHALLENGE EVALUATION SUMMARY")
    print("=" * 90)
    print(str_summary)
    print("=" * 90 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate Multi-Object Tracking (MOT) results using MOTChallenge metrics.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python tools/evaluate_tracking.py --gt data/gt.txt --ts infer_output/tracking_results.txt
        """
    )
    parser.add_argument('--gt', type=str, required=True,
                        help='Path to the ground truth file (MOT15-2D format)')
    parser.add_argument('--ts', type=str, required=True,
                        help='Path to the tracking results file (MOT15-2D format)')
    
    args = parser.parse_args()
    evaluate_mot(args.gt, args.ts)


if __name__ == '__main__':
    main()
