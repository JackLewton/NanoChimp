#!/usr/bin/env python3
"""
Train a ResNet-based Re-ID model for chimp identification.

Uses triplet loss with hard triplet mining (hardest positive and hardest negative). 
Handles class imbalance (chimp ID) using a PK-sampler (instead of random), garunteeing balance in every batch.
Video-level balance is also included to sample images from different video stems.
"""

import os
import sys
import argparse
import datetime
import math
import random
import json
from pathlib import Path
from collections import defaultdict
from typing import Tuple, List, Dict, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm
import matplotlib.pyplot as plt

# Add project root to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from lib.reid_model import ReIDNet
from tools.split_dataset import _extract_video_id

try:
    from tools.split_reid_dataset import create_reid_splits
except ImportError:
    create_reid_splits = None

# Central configuration dictionary
CONFIG = {
    # Paths & Directories
    "data_dir": "data/reid",
    "split_dir": "splits/reid_v1",
    "model_output_path": "reid_training/reid_model.pt",

    # Dataset & Splitting
    "auto_split": True,
    "val_ratio": 0.1,
    "test_ratio": 0.1,
    "seed": 42,
    "overwrite_splits": False,
    "ensure_min_per_identity": True,
    "min_val_videos": 1,
    "min_test_videos": 1,

    # Core Training Hyperparameters
    "epochs": 30,
    "batch_size": 16,
    "learning_rate": 0.0001,
    "embedding_dim": 128,  # Dimension of the output feature vector
    "margin": 0.3,         # Triplet loss margin
    "image_size": 224,     # Input resolution for the ResNet backbone

    # Balanced Sampler Settings
    "balanced_sampling": True,  # Prevent frame-heavy videos from dominating
    "samples_per_id": 4,        # K in PK-sampling (samples per identity per batch)
    "steps_per_epoch": None,    # Auto-calculated if None
    "balanced_val": True,       # Use balanced sampling for validation
    "val_steps": None,          # Auto-calculated if None

    # Validation Retrieval Settings
    "track_retrieval": True,    # Track Rank-k and mAP metrics during training
    "eval_every": 1,            # Compute retrieval metrics every N epochs
    "gallery_k": 5,             # Gallery images per identity to average
}


# manifest dataset (train/val from split .txt files)

def _read_manifest(path: Path) -> List[Path]:
    if not path.is_file():
        raise FileNotFoundError(f"Manifest not found: {path}")
    lines = [ln.strip() for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    return [Path(ln) for ln in lines]


def _identity_from_path(p: Path) -> str:
    return p.parent.name


class ReidPathDataset(Dataset):
    def __init__(self, paths: List[Path], label_to_idx: Dict[str, int], transform=None):
        self.paths = paths
        self.label_to_idx = label_to_idx
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        p = self.paths[idx]
        img = Image.open(p).convert("RGB")
        if self.transform:
            img = self.transform(img)
        label = _identity_from_path(p)
        return img, self.label_to_idx[label]


class BalancedIdentityVideoBatchSampler:
    """
    PK-style balanced sampling:
    - pick P identities uniformly each batch
    - for each identity pick K samples, preferring distinct video_ids

    This maximizes use of the dataset over time without allowing frame-heavy
    identities/videos to dominate batches.
    """

    def __init__(
        self,
        paths: List[Path],
        label_to_idx: Dict[str, int],
        batch_size: int,
        samples_per_id: int,
        steps_per_epoch: int,
        seed: int,
    ):
        if samples_per_id <= 0:
            raise ValueError("samples_per_id must be > 0")
        if batch_size % samples_per_id != 0:
            raise ValueError(f"batch_size ({batch_size}) must be divisible by samples_per_id ({samples_per_id})")

        self.paths = paths
        self.label_to_idx = label_to_idx
        self.batch_size = int(batch_size)
        self.samples_per_id = int(samples_per_id)
        self.identities_per_batch = self.batch_size // self.samples_per_id
        self.steps_per_epoch = int(steps_per_epoch)
        self.seed = int(seed)

        # id_idx -> video_id -> [dataset indices]
        self.id_video_to_indices: Dict[int, Dict[str, List[int]]] = defaultdict(lambda: defaultdict(list))
        for i, p in enumerate(paths):
            ident = _identity_from_path(p)
            if ident not in label_to_idx:
                continue
            id_idx = label_to_idx[ident]
            video_id = _extract_video_id(p.name)
            self.id_video_to_indices[id_idx][video_id].append(i)

        self.identity_ids = sorted(self.id_video_to_indices.keys())
        if len(self.identity_ids) < 2:
            raise ValueError("Need at least 2 identities for ReID training batches.")

        for id_idx in self.identity_ids:
            for vid in self.id_video_to_indices[id_idx]:
                self.id_video_to_indices[id_idx][vid].sort()

    def __len__(self) -> int:
        return self.steps_per_epoch

    def __iter__(self):
        rng = random.Random(self.seed)
        for _ in range(self.steps_per_epoch):
            step_rng = random.Random(rng.randint(0, 2**31 - 1))

            if len(self.identity_ids) >= self.identities_per_batch:
                chosen_ids = step_rng.sample(self.identity_ids, self.identities_per_batch)
            else:
                chosen_ids = [step_rng.choice(self.identity_ids) for _ in range(self.identities_per_batch)]

            batch: List[int] = []
            for id_idx in chosen_ids:
                video_map = self.id_video_to_indices[id_idx]
                video_ids = list(video_map.keys())
                if not video_ids:
                    continue

                # Prefer distinct videos if possible
                if len(video_ids) >= self.samples_per_id:
                    chosen_videos = step_rng.sample(video_ids, self.samples_per_id)
                else:
                    chosen_videos = [step_rng.choice(video_ids) for _ in range(self.samples_per_id)]

                for vid in chosen_videos:
                    indices = video_map[vid]
                    if indices:
                        batch.append(step_rng.choice(indices))
                    else:
                        all_indices = [ix for lst in video_map.values() for ix in lst]
                        batch.append(step_rng.choice(all_indices))

            # Pad/truncate to exact batch size
            if len(batch) < self.batch_size and batch:
                batch.extend(batch[: (self.batch_size - len(batch))])
            batch = batch[: self.batch_size]

            yield batch


def report_sampling(paths: List[Path], batch_sampler: BalancedIdentityVideoBatchSampler, max_videos_print: int = 8) -> None:
    """
    Summarize what the sampler will "show" the model over one epoch (i.e., over steps_per_epoch).
    Prints per-identity sampled frame counts and unique video counts.
    """
    frames_per_identity: Dict[str, int] = defaultdict(int)
    videos_per_identity: Dict[str, set[str]] = defaultdict(set)
    frames_per_identity_video: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))

    total = 0
    for batch in batch_sampler:
        for idx in batch:
            p = paths[idx]
            ident = _identity_from_path(p)
            vid = _extract_video_id(p.name)
            frames_per_identity[ident] += 1
            videos_per_identity[ident].add(vid)
            frames_per_identity_video[ident][vid] += 1
            total += 1

    print("\n=== Balanced sampler report (1 epoch) ===")
    print(f"Steps: {len(batch_sampler)} | Batch size: {batch_sampler.batch_size} | Total sampled frames: {total}")
    print(f"P (identities/batch): {batch_sampler.identities_per_batch} | K (samples/identity): {batch_sampler.samples_per_id}")
    print("")
    print("Per-identity sampled (frames | unique videos):")
    for ident in sorted(frames_per_identity.keys()):
        f = frames_per_identity[ident]
        v = len(videos_per_identity[ident])
        share = (f / total) if total else 0.0
        print(f"- {ident}: {f} | {v}  ({share:.1%} of sampled frames)")

    print("\nTop videos per identity (video_id: frames_sampled_in_epoch):")
    for ident in sorted(frames_per_identity_video.keys()):
        items = list(frames_per_identity_video[ident].items())
        items.sort(key=lambda kv: kv[1], reverse=True)
        top = items[:max_videos_print]
        top_str = ", ".join([f"{vid}:{cnt}" for vid, cnt in top])
        print(f"- {ident}: {top_str}")


class _EvalPathDataset(Dataset):
    def __init__(self, paths: List[Path], transform):
        self.paths = paths
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        p = self.paths[idx]
        img = Image.open(p).convert("RGB")
        img = self.transform(img)
        label = _identity_from_path(p)
        return img, label


def _get_embeddings_labels(model: nn.Module, paths: List[Path], transform, device, batch_size: int = 64) -> Tuple[np.ndarray, List[str]]:
    ds = _EvalPathDataset(paths, transform=transform)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=4)
    model.eval()
    embs = []
    labels: List[str] = []
    with torch.no_grad():
        for imgs, labs in loader:
            imgs = imgs.to(device)
            e = model(imgs).detach().cpu().numpy()
            embs.append(e)
            labels.extend(list(labs))
    return np.vstack(embs) if embs else np.zeros((0, 1), dtype=np.float32), labels


def _calc_micro_macro(query_emb: np.ndarray, query_labels: List[str], gallery_emb: np.ndarray, gallery_labels: List[str]) -> Dict:
    if len(query_labels) == 0 or len(gallery_labels) == 0:
        return {"micro": {"rank1": 0.0, "mAP": 0.0}, "macro": {"rank1": 0.0, "mAP": 0.0}}

    # cosine similarity
    q = query_emb / np.linalg.norm(query_emb, axis=1, keepdims=True)
    g = gallery_emb / np.linalg.norm(gallery_emb, axis=1, keepdims=True)
    sim = np.dot(q, g.T)
    indices = np.argsort(-sim, axis=1)
    g_labels = np.array(gallery_labels)
    q_labels = np.array(query_labels)

    matches = (g_labels[indices] == q_labels[:, None])
    correct_rank1 = matches[:, 0].astype(np.float32)

    aps = np.zeros((len(query_labels),), dtype=np.float32)
    for i in range(len(query_labels)):
        correct_idx = np.where(matches[i])[0]
        if len(correct_idx) == 0:
            aps[i] = 0.0
            continue
        precisions = [(k_idx + 1) / (k + 1) for k_idx, k in enumerate(correct_idx)]
        aps[i] = float(np.mean(precisions)) if precisions else 0.0

    micro = {"rank1": float(np.mean(correct_rank1)), "mAP": float(np.mean(aps))}

    per_id = {}
    for ident in sorted(set(query_labels)):
        idxs = np.where(q_labels == ident)[0]
        if len(idxs) == 0:
            continue
        per_id[ident] = {"rank1": float(np.mean(correct_rank1[idxs])), "mAP": float(np.mean(aps[idxs]))}
    macro = {"rank1": float(np.mean([v["rank1"] for v in per_id.values()])), "mAP": float(np.mean([v["mAP"] for v in per_id.values()]))} if per_id else {"rank1": 0.0, "mAP": 0.0}
    return {"micro": micro, "macro": macro, "per_identity": per_id}


def _build_gallery_prototypes(
    model: nn.Module,
    gallery_paths_by_id: Dict[str, List[Path]],
    transform,
    device,
) -> Tuple[np.ndarray, List[str]]:
    """
    Build one gallery embedding per identity by averaging embeddings of multiple gallery images.
    Returns (gallery_embeddings, gallery_identity_labels).
    """
    embs = []
    labels = []
    for ident in sorted(gallery_paths_by_id.keys()):
        paths = gallery_paths_by_id[ident]
        if not paths:
            continue
        e, _ = _get_embeddings_labels(model, paths, transform, device)
        if e.shape[0] == 0:
            continue
        avg = np.mean(e, axis=0, keepdims=True)
        embs.append(avg)
        labels.append(ident)
    if not embs:
        return np.zeros((0, 1), dtype=np.float32), []
    return np.vstack(embs), labels


def compute_val_retrieval_metrics(model: nn.Module, val_paths: List[Path], transform, device, gallery_k: int = 5) -> Dict:
    """
    Deterministic query/gallery split within val:
    - for each identity, first K sorted images are gallery (averaged), rest are queries
    """
    by_id: Dict[str, List[Path]] = defaultdict(list)
    for p in val_paths:
        by_id[_identity_from_path(p)].append(p)
    gallery_by_id: Dict[str, List[Path]] = {}
    query = []
    for ident, paths in by_id.items():
        paths = sorted(paths)
        if len(paths) < 2:
            continue
        k = max(1, min(int(gallery_k), len(paths) - 1))
        gallery_by_id[ident] = paths[:k]
        query.extend(paths[k:])
    if not gallery_by_id or not query:
        return {"micro": {"rank1": 0.0, "mAP": 0.0}, "macro": {"rank1": 0.0, "mAP": 0.0}}
    qe, ql = _get_embeddings_labels(model, query, transform, device)
    ge, gl = _build_gallery_prototypes(model, gallery_by_id, transform, device)
    return _calc_micro_macro(qe, ql, ge, gl)


# triplet loss with hard mining

def pairwise_distances(embeddings, squared=False):
    """
    Compute the 2D matrix of distances between all pairs of embeddings.
    """
    dot_product = torch.matmul(embeddings, embeddings.t())
    square_norm = torch.diag(dot_product)
    distances = square_norm.unsqueeze(1) - 2.0 * dot_product + square_norm.unsqueeze(0)
    distances = torch.clamp(distances, min=0)

    if not squared:
        mask = (distances == 0.0).float()
        distances = distances + mask * 1e-16
        distances = torch.sqrt(distances)
        distances = distances * (1.0 - mask)
        
    return distances


def get_hard_triplets(embeddings, labels):
    """
    For each anchor in the batch, find the hardest positive and hardest negative.
    """
    distances = pairwise_distances(embeddings)
    
    mask_anchor_positive = (labels.unsqueeze(1) == labels.unsqueeze(0)).bool()
    mask_anchor_negative = ~mask_anchor_positive

    # For each anchor, find the hardest positive (the one with the largest distance)
    anchor_positive_dist = distances * mask_anchor_positive.float()
    hardest_positive_dist, _ = torch.max(anchor_positive_dist, dim=1, keepdim=True)

    # For each anchor, find the hardest negative (the one with the smallest distance)
    # Add a large value to positive pairs to exclude them from the min search
    max_dist = torch.max(distances).item()
    anchor_negative_dist = distances + max_dist * (1.0 - mask_anchor_negative.float())
    hardest_negative_dist, _ = torch.min(anchor_negative_dist, dim=1, keepdim=True)
    
    return hardest_positive_dist, hardest_negative_dist


# main training loop

def train(config):
    """
    The main function to train the Re-ID model.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    torch.manual_seed(int(config.get("seed", 42)))
    np.random.seed(int(config.get("seed", 42)))

    # data preparation
    data_transforms = transforms.Compose([
        transforms.Resize((config["image_size"], config["image_size"])),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(20),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.15),
        transforms.RandomPerspective(distortion_scale=0.2, p=0.5), 
        transforms.RandomApply([
            transforms.GaussianBlur(kernel_size=3)
        ], p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.6, scale=(0.02, 0.25), ratio=(0.3, 3.3), value=0, inplace=False) 
    ])
    
    val_transforms = transforms.Compose([
        transforms.Resize((config["image_size"], config["image_size"])),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # manifest-based split (video-disjoint)
    split_dir = config.get("split_dir")
    if not split_dir:
        raise RuntimeError("split_dir is required for video-level splitting.")

    split_dir_path = Path(split_dir)
    train_list = split_dir_path / "train.txt"
    val_list = split_dir_path / "val.txt"
    test_list = split_dir_path / "test.txt"

    if config.get("auto_split", False):
        if create_reid_splits is None:
            raise RuntimeError(
                "auto_split requested but tools.split_reid_dataset could not be imported. "
                "Run from the repo root or fix PYTHONPATH."
            )

        if any(p.exists() for p in (train_list, val_list, test_list)):
            if config.get("overwrite_splits", False):
                for p in (train_list, val_list, test_list):
                    if p.exists():
                        p.unlink()
            else:
                print(f"Split manifests already exist in '{split_dir_path}'. Using them (set --overwrite-splits to regenerate).")

        if not (train_list.exists() and val_list.exists() and test_list.exists()):
            create_reid_splits(
                data_dir=config["data_dir"],
                output_dir=str(split_dir_path),
                val_ratio=float(config.get("val_ratio", 0.1)),
                test_ratio=float(config.get("test_ratio", 0.1)),
                seed=int(config.get("seed", 42)),
                ensure_min_per_identity=bool(config.get("ensure_min_per_identity", True)),
                min_val_videos=int(config.get("min_val_videos", 1)),
                min_test_videos=int(config.get("min_test_videos", 1)),
            )

    if not train_list.exists():
        raise RuntimeError(f"Missing train manifest: {train_list} (run with --auto-split or run tools/split_reid_dataset.py)")
    if not val_list.exists():
        raise RuntimeError(f"Missing val manifest: {val_list} (run with --auto-split or run tools/split_reid_dataset.py)")

    train_paths = _read_manifest(train_list)
    val_paths = _read_manifest(val_list)
    if not train_paths:
        raise RuntimeError(f"train.txt is empty: {train_list}")
    if not val_paths:
        print("Warning: val.txt is empty; validation will be skipped.")

    # Build label map from train (+val) for stability
    label_names = sorted({_identity_from_path(p) for p in (train_paths + val_paths)})
    label_to_idx = {n: i for i, n in enumerate(label_names)}
    print(f"Found {len(label_names)} classes for training: {', '.join(label_names)}")

    train_dataset = ReidPathDataset(train_paths, label_to_idx, transform=data_transforms)
    val_dataset = ReidPathDataset(val_paths, label_to_idx, transform=val_transforms) if val_paths else None

    batch_size = int(config["batch_size"])
    balanced_sampling = bool(config.get("balanced_sampling", True))
    samples_per_id = int(config.get("samples_per_id", 4))
    steps_per_epoch = config.get("steps_per_epoch", None)
    if steps_per_epoch is None:
        steps_per_epoch = int(math.ceil(len(train_dataset) / max(1, batch_size)))
    else:
        steps_per_epoch = int(steps_per_epoch)

    if balanced_sampling:
        train_batch_sampler = BalancedIdentityVideoBatchSampler(
            paths=train_paths,
            label_to_idx=label_to_idx,
            batch_size=batch_size,
            samples_per_id=samples_per_id,
            steps_per_epoch=steps_per_epoch,
            seed=int(config.get("seed", 42)),
        )
        train_loader = DataLoader(train_dataset, batch_sampler=train_batch_sampler, num_workers=4)
        print(f"Training sampler: balanced (P={batch_size // samples_per_id}, K={samples_per_id}), steps/epoch={steps_per_epoch}")
    else:
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
        print("Training sampler: shuffle")

    val_loader = None
    if val_dataset:
        balanced_val = bool(config.get("balanced_val", True))
        if balanced_val and balanced_sampling:
            val_steps = config.get("val_steps", None)
            if val_steps is None:
                val_steps = int(math.ceil(len(val_dataset) / max(1, batch_size)))
            else:
                val_steps = int(val_steps)
            val_batch_sampler = BalancedIdentityVideoBatchSampler(
                paths=val_paths,
                label_to_idx=label_to_idx,
                batch_size=batch_size,
                samples_per_id=samples_per_id,
                steps_per_epoch=val_steps,
                seed=int(config.get("seed", 42)) + 999,
            )
            val_loader = DataLoader(val_dataset, batch_sampler=val_batch_sampler, num_workers=4)
            print(f"Validation sampler: balanced, steps={val_steps}")
        else:
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
            print("Validation sampler: sequential")

    print(f"\nDataset split (manifest): {len(train_dataset)} training samples, {len(val_dataset) if val_dataset else 0} validation samples.")

    if config.get("sampler_report_only", False):
        if not balanced_sampling:
            raise RuntimeError("sampler_report_only requires balanced_sampling to be enabled.")
        report_sampling(train_paths, train_batch_sampler, max_videos_print=int(config.get("report_max_videos", 8)))
        if val_dataset and val_loader and isinstance(val_loader.batch_sampler, BalancedIdentityVideoBatchSampler):
            report_sampling(val_paths, val_loader.batch_sampler, max_videos_print=int(config.get("report_max_videos", 8)))
        return

    # model, loss, and optimizer
    model = ReIDNet(config["embedding_dim"]).to(device)
    
    triplet_loss_fn = nn.TripletMarginLoss(margin=config["margin"], p=2)
    optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"])
    
    best_val_loss = float('inf')
    best_model_state = None
    best_map_micro = -1.0
    best_map_micro_state = None
    best_map_micro_epoch = None
    best_map_macro = -1.0
    best_map_macro_state = None
    best_map_macro_epoch = None
    history = {
        "train_loss": [],
        "val_loss": [],
        "val_rank1_micro": [],
        "val_map_micro": [],
        "val_rank1_macro": [],
        "val_map_macro": [],
    }

    # training loop
    print("\nStarting training...")
    for epoch in range(config["epochs"]):
        model.train()
        running_loss = 0.0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['epochs']} [Train]", leave=False)
        for images, labels in progress_bar:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            embeddings = model(images)
            hardest_pos, hardest_neg = get_hard_triplets(embeddings, labels)
            loss = nn.functional.relu(hardest_pos - hardest_neg + config["margin"]).mean()

            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            progress_bar.set_postfix(loss=f"{loss.item():.4f}")

        epoch_loss = running_loss / len(train_loader)
        history["train_loss"].append(float(epoch_loss))
        
        # validation phase
        model.eval()
        if val_loader is not None:
            val_loss = 0.0
            val_progress_bar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{config['epochs']} [Val]", leave=False)
            with torch.no_grad():
                for images, labels in val_progress_bar:
                    images, labels = images.to(device), labels.to(device)
                    embeddings = model(images)
                    
                    hardest_pos, hardest_neg = get_hard_triplets(embeddings, labels)
                    loss = nn.functional.relu(hardest_pos - hardest_neg + config["margin"]).mean()
                    val_loss += loss.item()
            
            epoch_val_loss = val_loss / len(val_loader)
            history["val_loss"].append(float(epoch_val_loss))
            print(f"Epoch {epoch+1}/{config['epochs']} - Train Loss: {epoch_loss:.4f}, Val Loss: {epoch_val_loss:.4f}")
        else:
            epoch_val_loss = epoch_loss
            history["val_loss"].append(float(epoch_val_loss))
            print(f"Epoch {epoch+1}/{config['epochs']} - Train Loss: {epoch_loss:.4f}")

        # retrieval metrics on val split
        track_retrieval = bool(config.get("track_retrieval", True))
        eval_every = int(config.get("eval_every", 1))
        if track_retrieval and val_paths and (eval_every <= 1 or ((epoch + 1) % eval_every == 0)):
            metrics = compute_val_retrieval_metrics(
                model,
                val_paths,
                val_transforms,
                device,
                gallery_k=int(config.get("gallery_k", 5)),
            )
            history["val_rank1_micro"].append(metrics["micro"]["rank1"])
            history["val_map_micro"].append(metrics["micro"]["mAP"])
            history["val_rank1_macro"].append(metrics["macro"]["rank1"])
            history["val_map_macro"].append(metrics["macro"]["mAP"])
            print(
                f"  Val retrieval: "
                f"R1 micro={metrics['micro']['rank1']:.2%}, mAP micro={metrics['micro']['mAP']:.2%} | "
                f"R1 macro={metrics['macro']['rank1']:.2%}, mAP macro={metrics['macro']['mAP']:.2%}"
            )

            # Track best-by-retrieval checkpoints
            if metrics["micro"]["mAP"] > best_map_micro:
                best_map_micro = float(metrics["micro"]["mAP"])
                best_map_micro_state = {k: v.detach().cpu() if hasattr(v, "detach") else v for k, v in model.state_dict().items()}
                best_map_micro_epoch = epoch + 1
            if metrics["macro"]["mAP"] > best_map_macro:
                best_map_macro = float(metrics["macro"]["mAP"])
                best_map_macro_state = {k: v.detach().cpu() if hasattr(v, "detach") else v for k, v in model.state_dict().items()}
                best_map_macro_epoch = epoch + 1
        else:
            history["val_rank1_micro"].append(None)
            history["val_map_micro"].append(None)
            history["val_rank1_macro"].append(None)
            history["val_map_macro"].append(None)

        # Check if this is the best model so far
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            best_model_state = model.state_dict()
            print(f"  -> New best model found with validation loss: {best_val_loss:.4f}")

    # save the best model
    print("\nTraining complete.")
    
    output_dir = os.path.dirname(config["model_output_path"])
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        output_dir = os.path.dirname(config["model_output_path"]) or "."
        if best_map_micro_state is not None:
            path = os.path.join(output_dir, "reid_model_best_map_micro.pt")
            torch.save(best_map_micro_state, path)
            print(f"Best-by-mAP(micro) saved to: {path} (epoch {best_map_micro_epoch}, mAP={best_map_micro:.2%})")
            torch.save(best_map_micro_state, config["model_output_path"])
            print(f"✓ PRIMARY MODEL saved to: {config['model_output_path']} (best mAP micro, epoch {best_map_micro_epoch})")
            stable_best = os.path.join(output_dir, "reid_model_best.pt")
            torch.save(best_map_micro_state, stable_best)
            print(f"  (also copied to: {stable_best})")
        elif best_model_state:
            torch.save(best_model_state, config["model_output_path"])
            print(f"Best model (by val loss) saved to: {config['model_output_path']}")
        else:
            print("Warning: No best model was found. Saving the final model instead.")
            torch.save(model.state_dict(), config["model_output_path"])
            print(f"Final model saved to: {config['model_output_path']}")
        
        if best_map_macro_state is not None:
            path = os.path.join(output_dir, "reid_model_best_map_macro.pt")
            torch.save(best_map_macro_state, path)
            print(f"Best-by-mAP(macro) saved to: {path} (epoch {best_map_macro_epoch}, mAP={best_map_macro:.2%})")
        
        if best_model_state and best_map_micro_state is not None:
            loss_path = os.path.join(output_dir, "reid_model_best_loss.pt")
            torch.save(best_model_state, loss_path)
            print(f"Best-by-loss saved (for reference): {loss_path}")
    except Exception as e:
        print(f"Warning: failed to save best-by-mAP checkpoints: {e}")

    # save training curves
    try:
        output_dir = os.path.dirname(config["model_output_path"]) or "."
        run_json = os.path.join(output_dir, "training_metrics.json")
        with open(run_json, "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)

        epochs = list(range(1, len(history["train_loss"]) + 1))
        plt.figure(figsize=(10, 5))

        ax1 = plt.gca()
        ax1.plot(epochs, history["train_loss"], label="train loss")
        ax1.plot(epochs, history["val_loss"], label="val loss")
        ax1.set_xlabel("epoch")
        ax1.set_ylabel("loss")
        ax1.grid(True, alpha=0.3)

        ax2 = ax1.twinx()
        def _mask_none(vals):
            xs = []
            ys = []
            for e, v in zip(epochs, vals):
                if v is None:
                    continue
                xs.append(e)
                ys.append(v)
            return xs, ys

        x, y = _mask_none(history["val_map_micro"])
        if x:
            ax2.plot(x, y, label="val mAP micro", linestyle="--")
        x, y = _mask_none(history["val_map_macro"])
        if x:
            ax2.plot(x, y, label="val mAP macro", linestyle="--")
        ax2.set_ylabel("mAP")

        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc="best")

        out_png = os.path.join(output_dir, "results.png")
        plt.tight_layout()
        plt.savefig(out_png, dpi=150)
        plt.close()
        print(f"Saved training curves: {out_png}")
        print(f"Saved training metrics JSON: {run_json}")
    except Exception as e:
        print(f"Warning: failed to write training curves: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a Re-ID model for chimp identification.")
    parser.add_argument('--data-dir', type=str, default=CONFIG["data_dir"], help="Directory with curated Re-ID data.")
    parser.add_argument('--epochs', type=int, default=CONFIG["epochs"], help="Number of training epochs.")
    parser.add_argument('--batch-size', type=int, default=CONFIG["batch_size"], help="Training batch size.")
    parser.add_argument('--lr', type=float, default=CONFIG["learning_rate"], help="Learning rate.")
    parser.add_argument('--margin', type=float, default=CONFIG["margin"], help="Margin for the triplet loss.")
    parser.add_argument('--output-path', type=str, default=CONFIG["model_output_path"], help="Path to save the trained model.")
    parser.add_argument('--split-dir', type=str, default=CONFIG["split_dir"], help="Directory containing/writing train.txt and val.txt manifests.")
    parser.add_argument('--overwrite-splits', action='store_true', help="Overwrite existing manifests when using --auto-split.")
    parser.add_argument('--seed', type=int, default=CONFIG["seed"], help="Seed for stable split hashing.")
    args = parser.parse_args()
    
    # Update config with command line arguments
    CONFIG["data_dir"] = args.data_dir
    CONFIG["epochs"] = args.epochs
    CONFIG["batch_size"] = args.batch_size
    CONFIG["learning_rate"] = args.lr
    CONFIG["margin"] = args.margin
    CONFIG["split_dir"] = args.split_dir
    CONFIG["overwrite_splits"] = args.overwrite_splits
    CONFIG["seed"] = args.seed
    
    # Handle model output path for versioning
    output_path = args.output_path
    if output_path == 'reid_training/reid_model.pt':
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        name, ext = os.path.splitext(os.path.basename(output_path))
        versioned_filename = f"{name}_{timestamp}{ext}"
        CONFIG["model_output_path"] = os.path.join("reid_training", versioned_filename)
    else:
        CONFIG["model_output_path"] = output_path
    
    train(CONFIG)
