#!/usr/bin/env python3
"""
Create rain/val splits for a YOLO images folder based on video stem.

Splits images by source video so that frames from the same video are not
distributed across both train and val sets. Writes train.txt and val.txt
listing absolute image paths for use with Ultralytics YOLO training.
"""

from __future__ import annotations

import os
import random
import argparse
import re
from typing import Tuple, Sequence, Dict, List, Optional
from collections import defaultdict


def _list_images(images_dir: str, suffixes: Sequence[str]) -> list:
	files = []
	for name in os.listdir(images_dir):
		lower = name.lower()
		if any(lower.endswith(suf) for suf in suffixes):
			files.append(name)
	return files


def _extract_video_id(filename: str) -> str:
	"""Extract video ID from image filename.
	
	Common patterns:
	- video_name_frame_000001.jpg -> video_name
	- NVR-0-Window 1-0-20251215171925-0_frame_000001.jpg -> NVR-0-Window 1-0-20251215171925-0
	- video_name_track123_frame456.jpg -> video_name
	- frame_000001.jpg -> frame (fallback)
	
	Args:
		filename: Image filename (without path)
		
	Returns:
		Video ID string
	"""
	# Remove file extension
	name_without_ext = os.path.splitext(filename)[0]
	
	# Pattern 1: Look for _frame_ or _frame followed by numbers (may have more content after)
	# Handles both "_frame_000058" and "_frame000058" patterns
	match = re.search(r'^(.+?)_frame[_0-9]+\d', name_without_ext, re.IGNORECASE)
	if match:
		return match.group(1)
	
	# Pattern 2: Look for _track followed by numbers and then _frame
	match = re.search(r'^(.+?)_track\d+_frame', name_without_ext, re.IGNORECASE)
	if match:
		return match.group(1)
	
	# Pattern 3: Try to find common prefix before last underscore followed by numbers
	# This handles cases like "video_001.jpg" or "video_001_frame.jpg"
	parts = name_without_ext.split('_')
	if len(parts) > 1:
		# Check if last part is numeric (frame number)
		if parts[-1].isdigit() or (len(parts) > 2 and parts[-2].isdigit()):
			# Return everything except the numeric suffix
			non_numeric_parts = []
			for part in parts:
				if part.isdigit():
					break
				non_numeric_parts.append(part)
			if non_numeric_parts:
				return '_'.join(non_numeric_parts)
	
	# Fallback: return filename without extension as video ID
	# This ensures each unique filename pattern gets its own video ID
	return name_without_ext


def create_train_val_split(
	images_dir: str,
	val_ratio: float = 0.2,
	seed: int = 42,
	output_dir: Optional[str] = None,
	image_suffixes: Sequence[str] = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff'),
) -> Tuple[str, str, int, int]:
	"""Create train/val split lists for Ultralytics from a YOLO images folder.
	
	This function ensures that images from the same video (same video ID) are not
	split across train and validation sets. Videos are split as whole units.

	Writes two text files listing absolute image paths: ``train.txt`` and ``val.txt``.

	Args:
		images_dir: Directory containing images
		val_ratio: Fraction of videos (not images) for validation split
		seed: RNG seed for reproducibility
		output_dir: Directory to write the list files (defaults to parent of images_dir)
		image_suffixes: Allowed image extensions

	Returns:
		(train_list_path, val_list_path, train_count, val_count)
	"""
	if not os.path.isdir(images_dir):
		raise FileNotFoundError(f"Images directory not found: {images_dir}")

	output_dir = output_dir or os.path.dirname(images_dir)
	os.makedirs(output_dir, exist_ok=True)

	image_names = _list_images(images_dir, image_suffixes)
	image_paths = [os.path.abspath(os.path.join(images_dir, n)) for n in image_names]

	# Group images by video ID
	video_to_images: Dict[str, List[str]] = defaultdict(list)
	for img_path in image_paths:
		img_filename = os.path.basename(img_path)
		video_id = _extract_video_id(img_filename)
		video_to_images[video_id].append(img_path)
	
	# Sort images within each video for reproducibility
	for video_id in video_to_images:
		video_to_images[video_id].sort()
	
	# Get list of video IDs and shuffle them
	video_ids = list(video_to_images.keys())
	video_ids.sort()  # Sort for reproducibility before shuffling
	random.Random(seed).shuffle(video_ids)
	
	# Split videos (not individual images)
	n_videos = len(video_ids)
	n_val_videos = max(1, int(n_videos * val_ratio)) if n_videos > 0 else 0
	val_video_ids = set(video_ids[:n_val_videos])
	train_video_ids = set(video_ids[n_val_videos:])
	
	# Collect all images from train and val videos
	train_paths = []
	val_paths = []
	
	for video_id in video_ids:
		images = video_to_images[video_id]
		if video_id in val_video_ids:
			val_paths.extend(images)
		else:
			train_paths.extend(images)
	
	# Sort paths for reproducibility
	train_paths.sort()
	val_paths.sort()

	train_list = os.path.join(output_dir, 'train.txt')
	val_list = os.path.join(output_dir, 'val.txt')

	with open(train_list, 'w') as f:
		for p in train_paths:
			f.write(p + '\n')

	with open(val_list, 'w') as f:
		for p in val_paths:
			f.write(p + '\n')

	print(f"Split {n_videos} videos ({len(image_paths)} images) -> train: {len(train_video_ids)} videos ({len(train_paths)} images), val: {len(val_video_ids)} videos ({len(val_paths)} images)")
	print(f"Wrote {train_list} and {val_list}")

	return train_list, val_list, len(train_paths), len(val_paths)


def main() -> None:
	parser = argparse.ArgumentParser(description='Create train/val splits for a YOLO images folder')
	parser.add_argument('--images_dir', required=True, help='Path to YOLO images directory (e.g., yolo_dataset/images)')
	parser.add_argument('--val_ratio', type=float, default=0.2, help='Validation ratio, default 0.2')
	parser.add_argument('--seed', type=int, default=42, help='Random seed, default 42')
	parser.add_argument('--output_dir', default=None, help='Output directory for list files (default: parent of images_dir)')
	args = parser.parse_args()

	create_train_val_split(
		images_dir=args.images_dir,
		val_ratio=args.val_ratio,
		seed=args.seed,
		output_dir=args.output_dir,
	)


if __name__ == '__main__':
	main()


