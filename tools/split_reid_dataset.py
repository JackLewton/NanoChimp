#!/usr/bin/env python3
"""
This script creates split manifests for the Re-ID dataset.
It creates leak-resistant train/val/test splits.
Ensuring no leakage across splits from the same underlying video/clip source unit. 
"""

import argparse
import hashlib
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

# Allow imports from project root when running as a script
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from tools.split_dataset import _extract_video_id  # reuse existing robust parsing


IMAGE_SUFFIXES: Tuple[str, ...] = (".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp")


@dataclass(frozen=True)
class SplitPaths:
	train_list: Path
	val_list: Path
	test_list: Path


def _is_image(p: Path, suffixes: Sequence[str]) -> bool:
	return p.is_file() and p.suffix.lower() in suffixes


def _iter_identity_dirs(data_dir: Path) -> List[Path]:
	return sorted([p for p in data_dir.iterdir() if p.is_dir() and not p.name.startswith("_")], key=lambda p: p.name)


def _stable_bucket(key: str, seed: int) -> float:
	"""
	Return a stable pseudo-random number in [0, 1) from key+seed.
	Uses md5 for stability across Python versions/platforms.
	"""
	msg = f"{seed}:{key}".encode("utf-8")
	digest = hashlib.md5(msg).hexdigest()  # noqa: S324 (non-crypto use; stability only)
	return int(digest[:12], 16) / float(16**12)


def _assign_split(bucket: float, val_ratio: float, test_ratio: float) -> str:
	if bucket < test_ratio:
		return "test"
	if bucket < test_ratio + val_ratio:
		return "val"
	return "train"


def _group_images_by_video(identity_dir: Path, image_suffixes: Sequence[str]) -> Dict[str, List[Path]]:
	video_to_paths: Dict[str, List[Path]] = {}
	for p in identity_dir.iterdir():
		if not _is_image(p, image_suffixes):
			continue
		video_id = _extract_video_id(p.name)
		video_to_paths.setdefault(video_id, []).append(p)
	for vid in video_to_paths:
		video_to_paths[vid].sort()
	return video_to_paths


def _ensure_min_per_identity(
	identity: str,
	video_ids: List[str],
	assignments: Dict[str, str],
	min_val_videos: int,
	min_test_videos: int,
	seed: int,
) -> None:
	"""
	Ensure each identity has at least some videos in val/test if possible.
	Deterministic: picks videos with smallest stable buckets for the needed split.
	"""
	if not video_ids:
		return

	def bucket_for(vid: str) -> float:
		return _stable_bucket(f"{identity}:{vid}", seed=seed)

	def count(split: str) -> int:
		return sum(1 for v in video_ids if assignments.get(v) == split)

	# Prefer to keep existing assignments; only reassign if missing.
	if min_test_videos > 0 and count("test") < min_test_videos and len(video_ids) >= (1 + min_test_videos):
		candidates = [v for v in video_ids if assignments.get(v) != "test"]
		candidates.sort(key=bucket_for)
		for v in candidates[: (min_test_videos - count("test"))]:
			assignments[v] = "test"

	if min_val_videos > 0 and count("val") < min_val_videos and len(video_ids) >= (2 + min_val_videos):
		candidates = [v for v in video_ids if assignments.get(v) not in ("val", "test")]
		candidates.sort(key=bucket_for)
		for v in candidates[: (min_val_videos - count("val"))]:
			assignments[v] = "val"


def create_reid_splits(
	data_dir: str,
	output_dir: str,
	val_ratio: float,
	test_ratio: float,
	seed: int,
	ensure_min_per_identity: bool,
	min_val_videos: int,
	min_test_videos: int,
	image_suffixes: Sequence[str] = IMAGE_SUFFIXES,
) -> Tuple[SplitPaths, Dict[str, Dict[str, int]]]:
	data_path = Path(data_dir)
	if not data_path.is_dir():
		raise FileNotFoundError(f"ReID data_dir not found: {data_dir}")

	out_path = Path(output_dir)
	out_path.mkdir(parents=True, exist_ok=True)

	splits = SplitPaths(
		train_list=out_path / "train.txt",
		val_list=out_path / "val.txt",
		test_list=out_path / "test.txt",
	)

	train_paths: List[str] = []
	val_paths: List[str] = []
	test_paths: List[str] = []

	stats: Dict[str, Dict[str, int]] = {}

	identity_dirs = _iter_identity_dirs(data_path)
	if not identity_dirs:
		raise RuntimeError(f"No identity subfolders found in: {data_dir}")

	for ident_dir in identity_dirs:
		identity = ident_dir.name
		video_to_paths = _group_images_by_video(ident_dir, image_suffixes=image_suffixes)
		video_ids = sorted(video_to_paths.keys())

		assignments: Dict[str, str] = {}
		for vid in video_ids:
			b = _stable_bucket(f"{identity}:{vid}", seed=seed)
			assignments[vid] = _assign_split(b, val_ratio=val_ratio, test_ratio=test_ratio)

		if ensure_min_per_identity:
			_ensure_min_per_identity(
				identity=identity,
				video_ids=video_ids,
				assignments=assignments,
				min_val_videos=min_val_videos,
				min_test_videos=min_test_videos,
				seed=seed,
			)

		id_train = 0
		id_val = 0
		id_test = 0
		for vid in video_ids:
			paths = video_to_paths[vid]
			if assignments[vid] == "test":
				test_paths.extend([str(p.resolve()) for p in paths])
				id_test += len(paths)
			elif assignments[vid] == "val":
				val_paths.extend([str(p.resolve()) for p in paths])
				id_val += len(paths)
			else:
				train_paths.extend([str(p.resolve()) for p in paths])
				id_train += len(paths)

		stats[identity] = {"train": id_train, "val": id_val, "test": id_test, "videos": len(video_ids)}

	# Sort for reproducibility
	train_paths.sort()
	val_paths.sort()
	test_paths.sort()

	splits.train_list.write_text("\n".join(train_paths) + ("\n" if train_paths else ""), encoding="utf-8")
	splits.val_list.write_text("\n".join(val_paths) + ("\n" if val_paths else ""), encoding="utf-8")
	splits.test_list.write_text("\n".join(test_paths) + ("\n" if test_paths else ""), encoding="utf-8")

	return splits, stats


def main() -> None:
	parser = argparse.ArgumentParser(description="Create stable train/val/test split manifests for ReID (video-disjoint).")
	parser.add_argument("--data_dir", type=str, default="data/reid", help="Root ReID directory (one folder per identity).")
	parser.add_argument(
		"--output_dir",
		type=str,
		default="splits/reid_v1",
		help="Directory to write train.txt/val.txt/test.txt (manifests).",
	)
	parser.add_argument("--val_ratio", type=float, default=0.1, help="Fraction of videos assigned to val.")
	parser.add_argument("--test_ratio", type=float, default=0.1, help="Fraction of videos assigned to test.")
	parser.add_argument("--seed", type=int, default=42, help="Seed for stable hashing.")
	parser.add_argument(
		"--ensure_min_per_identity",
		action="store_true",
		help="Ensure each identity has at least some videos in val/test when possible (may slightly alter pure hash assignment).",
	)
	parser.add_argument("--min_val_videos", type=int, default=1, help="Min distinct videos per identity in val (if ensured).")
	parser.add_argument("--min_test_videos", type=int, default=1, help="Min distinct videos per identity in test (if ensured).")
	parser.add_argument("--overwrite", action="store_true", help="Allow overwriting existing manifest files.")
	args = parser.parse_args()

	out = Path(args.output_dir)
	if out.exists() and not args.overwrite:
		# If any split exists, treat as existing and fail fast.
		existing = [p for p in (out / "train.txt", out / "val.txt", out / "test.txt") if p.exists()]
		if existing:
			raise SystemExit(
				f"Refusing to overwrite existing split manifests in '{args.output_dir}'. "
				f"Pass --overwrite or use a new --output_dir (e.g. splits/reid_v2)."
			)

	splits, stats = create_reid_splits(
		data_dir=args.data_dir,
		output_dir=args.output_dir,
		val_ratio=args.val_ratio,
		test_ratio=args.test_ratio,
		seed=args.seed,
		ensure_min_per_identity=args.ensure_min_per_identity,
		min_val_videos=args.min_val_videos,
		min_test_videos=args.min_test_videos,
	)

	total_train = sum(v["train"] for v in stats.values())
	total_val = sum(v["val"] for v in stats.values())
	total_test = sum(v["test"] for v in stats.values())
	print(f"Wrote manifests:\n- {splits.train_list}\n- {splits.val_list}\n- {splits.test_list}")
	print(f"Totals: train={total_train} val={total_val} test={total_test}")
	print("\nPer-identity (images):")
	for identity, s in stats.items():
		print(f"- {identity}: train={s['train']} val={s['val']} test={s['test']} (videos={s['videos']})")


if __name__ == "__main__":
	main()


