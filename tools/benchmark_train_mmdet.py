#!/usr/bin/env python3
"""
Fine-tune MMDetection models on the chimpanzee detection benchmark dataset.

Trains Faster R-CNN R50-FPN and FCOS R50-FPN sequentially on the same
prepared dataset split as train_yolo_benchmark.py for a fair comparison.

Requires the nanochimp-MMDet2 conda environment:

    conda create -n nanochimp-MMDet2 python=3.9 -y
    conda activate nanochimp-MMDet2
    conda install pytorch==1.10.2 torchvision==0.11.3 cudatoolkit=11.3 -c pytorch -y
    pip install "numpy<2" "opencv-python<4.12"
    pip install "mmcv-full==1.6.0" -f https://download.openmmlab.com/mmcv/dist/cu113/torch1.10.0/index.html
    pip install "mmdet==2.25.0" tqdm pyyaml pillow

Run prepare_benchmark_data.py (nanochimp env) before running this script.

Usage:
    conda activate nanochimp-MMDet2
    python tools/train_mmdet_benchmark.py --data_dir yolo_benchmark_dataset/
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import time
import traceback

import torch
from mmcv import Config as MMCVConfig
from mmdet.apis import train_detector
from mmdet.datasets import build_dataloader, build_dataset
from mmdet.models import build_detector


MODELS = ["FasterRCNN-R50-FPN", "FCOS-R50-FPN"]

# SGD learning rate follows the linear scaling rule: lr = 0.02 * (batch / 16).
_BASE_LR    = 0.02
_BASE_BATCH = 16



# MMDetection configuration builders
# standard to put config in string, even if ugly

def _faster_rcnn_head(data_root: str, train_json: str, val_json: str,
                      imgsz: int, batch: int, epochs: int) -> str:
    return """
# Faster R-CNN R50-FPN — fine-tuned on ChimpTZ-26
model = dict(
    type='FasterRCNN',
    backbone=dict(
        type='ResNet', depth=50, num_stages=4, out_indices=(0, 1, 2, 3),
        frozen_stages=1, norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True, style='pytorch',
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50')),
    neck=dict(type='FPN', in_channels=[256, 512, 1024, 2048], out_channels=256, num_outs=5),
    rpn_head=dict(
        type='RPNHead', in_channels=256, feat_channels=256,
        anchor_generator=dict(
            type='AnchorGenerator', scales=[8], ratios=[0.5, 1.0, 2.0],
            strides=[4, 8, 16, 32, 64]),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[.0, .0, .0, .0], target_stds=[1., 1., 1., 1.]),
        loss_cls=dict(type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
        loss_bbox=dict(type='L1Loss', loss_weight=1.0)),
    roi_head=dict(
        type='StandardRoIHead',
        bbox_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=0),
            out_channels=256, featmap_strides=[4, 8, 16, 32]),
        bbox_head=dict(
            type='Shared2FCBBoxHead', in_channels=256, fc_out_channels=1024,
            roi_feat_size=7, num_classes=1,
            bbox_coder=dict(
                type='DeltaXYWHBBoxCoder',
                target_means=[0., 0., 0., 0.], target_stds=[0.1, 0.1, 0.2, 0.2]),
            reg_class_agnostic=False,
            loss_cls=dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
            loss_bbox=dict(type='L1Loss', loss_weight=1.0))),
    train_cfg=dict(
        rpn=dict(
            assigner=dict(
                type='MaxIoUAssigner', pos_iou_thr=0.7, neg_iou_thr=0.3,
                min_pos_iou=0.3, match_low_quality=True, ignore_iof_thr=-1),
            sampler=dict(
                type='RandomSampler', num=256, pos_fraction=0.5,
                neg_pos_ub=-1, add_gt_as_proposals=False),
            allowed_border=-1, pos_weight=-1, debug=False),
        rpn_proposal=dict(
            nms_pre=2000, max_per_img=1000,
            nms=dict(type='nms', iou_threshold=0.7), min_bbox_size=0),
        rcnn=dict(
            assigner=dict(
                type='MaxIoUAssigner', pos_iou_thr=0.5, neg_iou_thr=0.5,
                min_pos_iou=0.5, match_low_quality=False, ignore_iof_thr=-1),
            sampler=dict(
                type='RandomSampler', num=512, pos_fraction=0.25,
                neg_pos_ub=-1, add_gt_as_proposals=True),
            pos_weight=-1, debug=False)),
    test_cfg=dict(
        rpn=dict(
            nms_pre=1000, max_per_img=1000,
            nms=dict(type='nms', iou_threshold=0.7), min_bbox_size=0),
        rcnn=dict(
            score_thr=0.05, nms=dict(type='nms', iou_threshold=0.5), max_per_img=100)))
""" + _common_config(data_root, train_json, val_json, imgsz, batch, epochs)


def _fcos_head(data_root: str, train_json: str, val_json: str,
               imgsz: int, batch: int, epochs: int) -> str:
    return """
# FCOS R50-FPN — fine-tuned on ChimpTZ-26
model = dict(
    type='FCOS',
    backbone=dict(
        type='ResNet', depth=50, num_stages=4, out_indices=(0, 1, 2, 3),
        frozen_stages=1, norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True, style='pytorch',
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50')),
    neck=dict(
        type='FPN', in_channels=[256, 512, 1024, 2048], out_channels=256,
        start_level=1, add_extra_convs='on_output', num_outs=5),
    bbox_head=dict(
        type='FCOSHead', num_classes=1, in_channels=256, stacked_convs=4,
        feat_channels=256, strides=[8, 16, 32, 64, 128],
        loss_cls=dict(
            type='FocalLoss', use_sigmoid=True, gamma=2.0, alpha=0.25, loss_weight=1.0),
        loss_bbox=dict(type='IoULoss', loss_weight=1.0),
        loss_centerness=dict(type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0)),
    train_cfg=dict(),
    test_cfg=dict(
        nms_pre=1000, min_bbox_size=0, score_thr=0.05,
        nms=dict(type='nms', iou_threshold=0.5), max_per_img=100))
""" + _common_config(data_root, train_json, val_json, imgsz, batch, epochs)


def _common_config(data_root: str, train_json: str, val_json: str,
                   imgsz: int, batch: int, epochs: int) -> str:
    """Shared dataset, optimiser, and runtime settings for both models."""
    lr     = _BASE_LR * batch / _BASE_BATCH
    step1  = int(epochs * 0.67)
    step2  = int(epochs * 0.89)
    images = os.path.join(data_root, "images")
    return f"""
dataset_type = 'CocoDataset'
data_root = '{data_root}/'
img_norm_cfg = dict(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', img_scale=({imgsz}, {imgsz}), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='MultiScaleFlipAug', img_scale=({imgsz}, {imgsz}), flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]

data = dict(
    samples_per_gpu={batch},
    workers_per_gpu=2,
    train=dict(type=dataset_type,
        ann_file='{train_json}', img_prefix='{images}/',
        classes=('chimp',), pipeline=train_pipeline),
    val=dict(type=dataset_type,
        ann_file='{val_json}', img_prefix='{images}/',
        classes=('chimp',), pipeline=test_pipeline),
    test=dict(type=dataset_type,
        ann_file='{val_json}', img_prefix='{images}/',
        classes=('chimp',), pipeline=test_pipeline))

evaluation = dict(interval=1, metric='bbox', save_best='auto')

optimizer = dict(type='SGD', lr={lr}, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
lr_config = dict(
    policy='step', warmup='linear', warmup_iters=500,
    warmup_ratio=0.001, step=[{step1}, {step2}])

runner = dict(type='EpochBasedRunner', max_epochs={epochs})
# save_best='auto' above (bbox_mAP is the primary CocoDataset metric) writes
# best_bbox_mAP_epoch_<N>.pth. Cap regular per-epoch checkpoints so long runs
# don't fill the disk; the "best" checkpoint is kept independently of this limit.
checkpoint_config = dict(interval=max(1, {epochs} // 10), max_keep_ckpts=3)
log_config = dict(interval=50, hooks=[dict(type='TextLoggerHook')])

seed = 42
deterministic = False
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
"""


def build_config(
    model_name: str,
    data_root: str,
    train_json: str,
    val_json: str,
    imgsz: int,
    batch: int,
    epochs: int,
    config_dir: str,
) -> str:
    """Generate and write an MMDetection config file, returning its path.

    Args:
        model_name: One of 'FasterRCNN-R50-FPN' or 'FCOS-R50-FPN'.
        data_root: Absolute path to the benchmark dataset directory.
        train_json: Absolute path to train_split.json.
        val_json: Absolute path to val_split.json.
        imgsz: Input image size (square).
        batch: Samples per GPU.
        epochs: Total training epochs.
        config_dir: Directory to write the config file.

    Returns:
        Path to the written config file.
    """
    if "FasterRCNN" in model_name:
        content = _faster_rcnn_head(data_root, train_json, val_json, imgsz, batch, epochs)
    elif "FCOS" in model_name:
        content = _fcos_head(data_root, train_json, val_json, imgsz, batch, epochs)
    else:
        raise ValueError(f"Unknown model: {model_name}")

    config_path = os.path.join(
        config_dir, f"{model_name.lower().replace('-', '_')}_config.py"
    )
    with open(config_path, "w") as f:
        f.write(content)
    return config_path


# Training

def _find_best_checkpoint(work_dir: str) -> str | None:
    """Return the best checkpoint, preferring the one MMDetection's eval hook saved.

    Falls back to the highest-numbered `epoch_*.pth` if no `best_bbox_mAP_epoch_*.pth`
    exists (e.g. training was interrupted before any validation ran). Epoch numbers are
    compared numerically — a plain `sorted()` on filenames is a lexicographic string
    sort, which incorrectly ranks e.g. "epoch_99.pth" above "epoch_100.pth".
    """
    best_ckpts = glob.glob(os.path.join(work_dir, "best_bbox_mAP_epoch_*.pth"))
    if best_ckpts:
        return best_ckpts[0]

    epoch_ckpts = glob.glob(os.path.join(work_dir, "epoch_*.pth"))
    if not epoch_ckpts:
        return None

    def _epoch_num(path: str) -> int:
        match = re.search(r"epoch_(\d+)\.pth$", path)
        return int(match.group(1)) if match else -1

    return max(epoch_ckpts, key=_epoch_num)


def _parse_best_val_metrics(work_dir: str) -> dict:
    """Parse MMDetection's *.log.json for the best validation bbox_mAP / bbox_mAP_50.

    MMDetection logs one JSON line per epoch with `"mode": "val"` when `evaluation`
    is configured. `bbox_mAP` is the COCO-style mAP averaged over IoU 0.5:0.95
    (matches YOLO's mAP50-95); `bbox_mAP_50` is mAP at IoU 0.5 (matches YOLO's mAP50).
    Returns the entry with the highest bbox_mAP, i.e. the epoch save_best='auto' picked.
    """
    log_files = sorted(glob.glob(os.path.join(work_dir, "*.log.json")))
    if not log_files:
        return {}

    best: dict = {}
    with open(log_files[-1]) as f:
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
            if not best or entry["bbox_mAP"] > best["bbox_mAP"]:
                best = entry
    return best


def _measure_inference_ms(
    model, cfg, device: str, num_images: int = 50, warmup: int = 5
) -> float:
    """Measure average per-image inference latency (ms) on the validation set."""
    try:
        val_dataset = build_dataset(cfg.data.val)
        loader = build_dataloader(
            val_dataset, samples_per_gpu=1, workers_per_gpu=0,
            dist=False, shuffle=False,
        )
    except Exception as e:
        print(f"Warning: could not build val dataloader for timing: {e}")
        return 0.0

    from mmcv.parallel import MMDataParallel
    if device == "cuda":
        model_wrapped = MMDataParallel(model, device_ids=[0])
    else:
        model_wrapped = MMDataParallel(model, device_ids=[])

    model_wrapped.eval()
    times = []
    with torch.no_grad():
        for i, data in enumerate(loader):
            if i >= warmup + num_images:
                break
            if device == "cuda":
                torch.cuda.synchronize()
            start = time.perf_counter()
            model_wrapped(return_loss=False, rescale=True, **data)
            if device == "cuda":
                torch.cuda.synchronize()
            elapsed_ms = (time.perf_counter() - start) * 1000
            if i >= warmup:
                times.append(elapsed_ms)

    return sum(times) / len(times) if times else 0.0


def train_model(model_name: str, data_dir: str, args: argparse.Namespace) -> dict | None:
    """Fine-tune a single MMDetection model and return a summary dictionary.

    Args:
        model_name: One of MODELS.
        data_dir: Directory produced by prepare_benchmark_data.py.
        args: Parsed CLI arguments.

    Returns:
        Summary dict with checkpoint path, or None on failure.
    """
    print(f"\n{'='*60}\nTraining {model_name}\n{'='*60}")

    data_root  = os.path.abspath(data_dir)
    train_json = os.path.join(data_root, "train_split.json")
    val_json   = os.path.join(data_root, "val_split.json")

    config_path = build_config(
        model_name=model_name,
        data_root=data_root,
        train_json=train_json,
        val_json=val_json,
        imgsz=args.imgsz,
        batch=args.batch_size,
        epochs=args.epochs,
        config_dir=data_root,
    )

    cfg          = MMCVConfig.fromfile(config_path)
    work_dir     = os.path.join(args.output_dir, model_name)
    cfg.work_dir = work_dir
    cfg.seed     = 42
    os.makedirs(work_dir, exist_ok=True)

    cfg.gpu_ids = [0]
    cfg.device  = "cuda" if torch.cuda.is_available() else "cpu"
    if not torch.cuda.is_available():
        print("Note: CUDA unavailable, training on CPU.")

    datasets = [build_dataset(cfg.data.train)]
    model    = build_detector(cfg.model, train_cfg=cfg.get("train_cfg"), test_cfg=cfg.get("test_cfg"))
    model.CLASSES = ("chimp",)
    if not torch.cuda.is_available():
        model = model.cpu()

    params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {params / 1e6:.2f}M")

    meta: dict = {}
    try:
        meta["config"] = cfg.pretty_text
    except TypeError:
        # yapf >= 0.40 removed the 'verify' argument used internally by older mmcv
        meta["config"] = cfg.text

    train_detector(model, datasets, cfg, distributed=False, validate=True, timestamp=None, meta=meta)

    best_ckpt = _find_best_checkpoint(work_dir)
    if best_ckpt:
        print(f"Best checkpoint: {best_ckpt}")
    else:
        print("Warning: no checkpoint found.")

    best_metrics = _parse_best_val_metrics(work_dir)
    map50    = best_metrics.get("bbox_mAP_50", 0.0)
    map50_95 = best_metrics.get("bbox_mAP", 0.0)
    if not best_metrics:
        print(f"Warning: could not parse validation mAP from logs in {work_dir}.")

    print("Measuring inference speed...")
    infer_ms = _measure_inference_ms(model, cfg, device=cfg.device)

    return {
        "model":        model_name,
        "framework":    "MMDetection",
        "params_M":     round(params / 1e6, 2),
        "mAP50":        round(float(map50), 4),
        "mAP50_95":     round(float(map50_95), 4),
        "inference_ms": round(float(infer_ms), 2),
        "checkpoint":   best_ckpt,
        "save_dir":     work_dir,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fine-tune MMDetection models on the chimpanzee benchmark dataset.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Run prepare_benchmark_data.py (nanochimp env) first, then:\n\n"
            "  conda activate nanochimp-MMDet2\n"
            "  python tools/train_mmdet_benchmark.py \\\n"
            "      --data_dir yolo_benchmark_dataset/ --epochs 200"
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
        "--output_dir", default="benchmark_results/",
        help="Root directory for training outputs.",
    )
    args = parser.parse_args()

    for required in ("train_split.json", "val_split.json"):
        path = os.path.join(args.data_dir, required)
        if not os.path.isfile(path):
            raise FileNotFoundError(
                f"{path} not found. Run prepare_benchmark_data.py first."
            )

    summaries = []
    for name in MODELS:
        try:
            result = train_model(name, args.data_dir, args)
            if result:
                summaries.append(result)
        except Exception as e:
            print(f"Training failed for {name}: {e}")
            traceback.print_exc()

    print(f"\n{'='*80}")
    print("MMDETECTION BENCHMARK RESULTS")
    print(f"{'='*80}")
    print(f"{'Model':<20} {'Params (M)':<12} {'mAP50':<10} {'mAP50-95':<12} {'Speed (ms)'}")
    print("-" * 80)
    for r in summaries:
        print(
            f"{r['model']:<20} {r['params_M']:<12.2f} "
            f"{r['mAP50']:<10.4f} {r['mAP50_95']:<12.4f} {r['inference_ms']:.2f}"
        )
    print(f"{'='*80}")

    os.makedirs(args.output_dir, exist_ok=True)
    results_path = os.path.join(args.output_dir, "mmdet_benchmark_summary.json")
    with open(results_path, "w") as f:
        json.dump(summaries, f, indent=2)
    print(f"\nResults saved to: {results_path}")


if __name__ == "__main__":
    main()
