#!/usr/bin/env python3
"""
Evaluate a trained Re-ID model and report metrics suitable for publication.

Computes micro (query-weighted) and macro (identity-weighted) Rank-1, Rank-3,
and mAP metrics using scikit-learn.
"""

import os
import sys
import glob
import argparse
import numpy as np
from pathlib import Path
from collections import defaultdict
from typing import List, Dict, Optional, Tuple

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import average_precision_score

# Add project root to path to allow imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from lib.reid_model import ReIDNet, infer_reid_head_from_state_dict
from configs.inference import find_latest_reid_model


class ReidTestDataset(Dataset):
    """Custom dataset to handle Re-ID evaluation images."""
    def __init__(self, image_paths: List[Path], transform=None):
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img = Image.open(img_path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        # Identity name is the parent folder name
        label = img_path.parent.name
        return img, label


def collect_images(test_dir: Optional[str] = None, test_list: Optional[str] = None) -> List[Path]:
    """Collect image paths from a manifest file or directory."""
    if test_list:
        if not os.path.exists(test_list):
            raise FileNotFoundError(f"Test list manifest not found: {test_list}")
        with open(test_list) as f:
            return [Path(line.strip()) for line in f if line.strip()]

    if not test_dir:
        raise ValueError("Either test_dir or test_list must be provided.")

    test_path = Path(test_dir)
    if not test_path.is_dir():
        raise FileNotFoundError(f"Test directory not found: {test_dir}")

    extensions = ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']
    image_paths = []
    for ext in extensions:
        image_paths.extend(test_path.glob(f"**/{ext}"))
    return image_paths


def get_embeddings(model: torch.nn.Module, dataloader: DataLoader, device: torch.device) -> Tuple[np.ndarray, List[str]]:
    """Extract embeddings and labels for all images in a dataloader."""
    model.eval()
    all_embeddings = []
    all_labels = []

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            embeddings = model(images)
            all_embeddings.append(embeddings.cpu().numpy())
            all_labels.extend(labels)

    return np.vstack(all_embeddings), all_labels


def calculate_metrics(
    query_embeddings: np.ndarray,
    query_labels: List[str],
    gallery_embeddings: np.ndarray,
    gallery_labels: List[str]
) -> Dict:
    """
    Calculate micro (per-query) and macro (per-identity) Rank-k and mAP metrics.
    Uses standard scikit-learn functions for similarity and AP computation.
    """
    num_queries = len(query_labels)
    if num_queries == 0 or len(gallery_labels) == 0:
        return {
            "micro": {"rank1": 0.0, "rank3": 0.0, "mAP": 0.0, "queries": 0},
            "macro": {"rank1": 0.0, "rank3": 0.0, "mAP": 0.0, "identities": 0},
            "per_identity": {}
        }

    # Compute cosine similarity matrix
    similarity_matrix = cosine_similarity(query_embeddings, gallery_embeddings)

    # Sort gallery indices by similarity in descending order
    indices = np.argsort(-similarity_matrix, axis=1)
    gallery_labels_arr = np.array(gallery_labels)
    query_labels_arr = np.array(query_labels)

    # Compute Rank-1 and Rank-3 correctness
    matches = (gallery_labels_arr[indices] == query_labels_arr[:, np.newaxis])
    correct_rank1 = matches[:, 0].astype(float)
    correct_rank3 = np.any(matches[:, :3], axis=1).astype(float)

    # Compute Average Precision (AP) for each query using scikit-learn
    aps = np.zeros(num_queries)
    for i in range(num_queries):
        query_label = query_labels[i]
        y_true = (gallery_labels_arr == query_label).astype(int)
        y_score = similarity_matrix[i]
        if np.sum(y_true) > 0:
            aps[i] = average_precision_score(y_true, y_score)

    # Micro metrics (query-weighted)
    rank1_micro = float(np.mean(correct_rank1))
    rank3_micro = float(np.mean(correct_rank3))
    map_micro = float(np.mean(aps))

    # Per-identity and Macro metrics (identity-weighted)
    per_identity = {}
    identities = sorted(set(query_labels))
    for ident in identities:
        idxs = np.where(query_labels_arr == ident)[0]
        if len(idxs) == 0:
            continue
        per_identity[ident] = {
            "queries": int(len(idxs)),
            "rank1": float(np.mean(correct_rank1[idxs])),
            "rank3": float(np.mean(correct_rank3[idxs])),
            "mAP": float(np.mean(aps[idxs])),
        }

    if per_identity:
        rank1_macro = float(np.mean([v["rank1"] for v in per_identity.values()]))
        rank3_macro = float(np.mean([v["rank3"] for v in per_identity.values()]))
        map_macro = float(np.mean([v["mAP"] for v in per_identity.values()]))
    else:
        rank1_macro = rank3_macro = map_macro = 0.0

    return {
        "micro": {"rank1": rank1_micro, "rank3": rank3_micro, "mAP": map_micro, "queries": int(num_queries)},
        "macro": {"rank1": rank1_macro, "rank3": rank3_macro, "mAP": map_macro, "identities": int(len(per_identity))},
        "per_identity": per_identity,
    }


def evaluate_single_model(
    model_path: str,
    query_loader: DataLoader,
    gallery_loader: DataLoader,
    device: torch.device
) -> Optional[Dict]:
    """Run evaluation for a single Re-ID model and return metrics."""
    try:
        state = torch.load(model_path, map_location=device)
        head = infer_reid_head_from_state_dict(state)
        model = ReIDNet(embedding_dim=128, head=head)
        model.load_state_dict(state)
        model.to(device)
    except Exception as e:
        print(f"Error loading model {model_path}: {e}")
        return None

    query_embeddings, query_labels = get_embeddings(model, query_loader, device)
    gallery_embeddings, gallery_labels = get_embeddings(model, gallery_loader, device)

    # Average gallery embeddings per identity to form one prototype per identity (stabilises metrics)
    by_id = {}
    for emb, lab in zip(gallery_embeddings, gallery_labels):
        by_id.setdefault(lab, []).append(emb)

    if by_id:
        proto_embs = []
        proto_labels = []
        for lab in sorted(by_id.keys()):
            proto_embs.append(np.mean(np.stack(by_id[lab], axis=0), axis=0))
            proto_labels.append(lab)
        gallery_embeddings = np.stack(proto_embs, axis=0)
        gallery_labels = proto_labels

    metrics = calculate_metrics(query_embeddings, query_labels, gallery_embeddings, gallery_labels)

    return {
        "model_name": Path(model_path).name,
        "micro": metrics["micro"],
        "macro": metrics["macro"],
        "per_identity": metrics["per_identity"],
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate and compare trained Re-ID models.")
    parser.add_argument('--model_path', type=str, default=None,
                        help="Path to a specific Re-ID model file (.pt). If omitted, auto-selects from reid_training/.")
    parser.add_argument('--test_dir', type=str, default='data/reid_test',
                        help="Directory containing the hold-out test set (fallback if no manifest is used).")
    parser.add_argument('--test_list', type=str, default=None,
                        help="Path to a test manifest (.txt) listing absolute image paths (overrides --test_dir).")
    parser.add_argument('--gallery_k', type=int, default=5,
                        help="Number of gallery images per identity to use (averaged into one prototype per identity).")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Prefer manifest-based split if available
    test_list = args.test_list
    if test_list is None:
        default_manifest = Path("splits/reid_v1/test.txt")
        if default_manifest.is_file():
            test_list = str(default_manifest)

    try:
        all_images = collect_images(test_dir=args.test_dir, test_list=test_list)
    except Exception as e:
        print(f"Error: {e}")
        print("Please ensure your test dataset or split manifest is prepared.")
        return

    if not all_images:
        print("Error: No test images found.")
        return

    # Create gallery (K images per ID) and query (the rest) sets
    gallery_paths = []
    query_paths = []
    identities = {p.parent.name for p in all_images}

    for identity in identities:
        identity_images = sorted([p for p in all_images if p.parent.name == identity])
        if len(identity_images) < 2:
            continue
        k = max(1, min(args.gallery_k, len(identity_images) - 1))
        gallery_paths.extend(identity_images[:k])
        query_paths.extend(identity_images[k:])

    print(f"\nTest set split: {len(set([p.parent.name for p in gallery_paths]))} gallery identities "
          f"({len(gallery_paths)} images, K={args.gallery_k}), {len(query_paths)} query images.")

    eval_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    gallery_dataset = ReidTestDataset(gallery_paths, transform=eval_transforms)
    query_dataset = ReidTestDataset(query_paths, transform=eval_transforms)

    gallery_loader = DataLoader(gallery_dataset, batch_size=32, shuffle=False)
    query_loader = DataLoader(query_dataset, batch_size=32, shuffle=False)

    # Find models to evaluate
    if args.model_path:
        model_files = [args.model_path]
        print(f"Evaluating specified model: {args.model_path}")
    else:
        best_micro = Path("reid_training/reid_model_best_map_micro.pt")
        best = Path("reid_training/reid_model_best.pt")
        if best_micro.is_file():
            model_files = [str(best_micro)]
            print(f"Evaluating best-by-mAP(micro): {best_micro}")
        elif best.is_file():
            model_files = [str(best)]
            print(f"Evaluating stable best model: {best}")
        else:
            latest = find_latest_reid_model("reid_training")
            if latest:
                model_files = [latest]
                print(f"Evaluating latest Re-ID model in reid_training/: {latest}")
            else:
                print("Searching for all Re-ID models in the project directory...")
                model_files = glob.glob("reid_model_*.pt")
                if not model_files:
                    print("No models found. Exiting.")
                    return
                print(f"Found {len(model_files)} models to evaluate.")

    results = []
    for path in tqdm(model_files, desc="Evaluating models"):
        result = evaluate_single_model(path, query_loader, gallery_loader, device)
        if result:
            results.append(result)

    if not results:
        print("No models were successfully evaluated.")
        return

    results.sort(key=lambda x: x["micro"]["rank1"], reverse=True)

    print("\n" + "=" * 90)
    print("RE-ID BENCHMARK RESULTS")
    print("=" * 90)
    print(f"{'Model':<35} | {'R1 micro':>9} | {'R1 macro':>9} | {'mAP micro':>9} | {'mAP macro':>9}")
    print("-" * 90)
    for res in results:
        print(
            f"{res['model_name']:<35} | "
            f"{res['micro']['rank1']:>8.2%} | {res['macro']['rank1']:>8.2%} | "
            f"{res['micro']['mAP']:>8.2%} | {res['macro']['mAP']:>8.2%}"
        )
    print("=" * 90)

    # Per-identity breakdown for the best model
    best_res = results[0]
    per_id = best_res.get("per_identity", {})
    if per_id:
        print(f"\nPer-identity metrics for best model: {best_res['model_name']}")
        print(f"{'Identity':<20} | {'Queries':>7} | {'Rank-1':>9} | {'mAP':>9}")
        print("-" * 55)
        for ident in sorted(per_id.keys()):
            v = per_id[ident]
            print(f"{ident:<20} | {v['queries']:>7d} | {v['rank1']:>8.2%} | {v['mAP']:>8.2%}")
        print("-" * 55)
        print(f"Micro (query-weighted): Rank-1={best_res['micro']['rank1']:.2%}, mAP={best_res['micro']['mAP']:.2%}")
        print(f"Macro (identity-weighted): Rank-1={best_res['macro']['rank1']:.2%}, mAP={best_res['macro']['mAP']:.2%}")
        print()


if __name__ == "__main__":
    main()
