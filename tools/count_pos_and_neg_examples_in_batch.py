#!/usr/bin/env python3
"""
Analyse a COCO format annotation file to count positive and negative samples.

Positive samples are images with at least one annotation. Negative (background)
samples are images tagged as 'background' with no annotations.
"""

import json
import argparse
import os
from collections import defaultdict
from pathlib import Path

def analyze_annotations(annotation_file, name):
    """
    Analyzes a COCO JSON annotation file to count positive and negative samples.
    Negative samples are identified by the "background" tag.
    """
    try:
        with open(annotation_file, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: Annotation file '{annotation_file}' not found.")
        return

    total_annotations = len(data.get('annotations', []))
    total_images = 0
    positive_samples = 0
    negative_samples = 0

    image_list = data.get('images', [])
    if not image_list:
        print("Warning: No images found in the annotation file.")
        return
        
    for img_info in image_list:
        total_images += 1
        tags = img_info.get('tags', [])
        if "background" in tags:
            negative_samples += 1
        else:
            positive_samples += 1

    print(f"\n--- {name} Analysis ---")
    print(f"Total images: {total_images}")
    print(f"  - Positive samples (with annotations): {positive_samples}")
    print(f"  - Negative samples (tagged as 'background'): {negative_samples}")
    print(f"Total chimp instances (annotations): {total_annotations}")

    if positive_samples > 0:
        avg_chimps_per_image = total_annotations / positive_samples
        print(f"Average chimps per positive image: {avg_chimps_per_image:.2f}")

def main():
    parser = argparse.ArgumentParser(description='Analyze batch annotations to count positive and negative samples based on tags.')
    parser.add_argument('--annotation_file', type=str, required=True, help='Path to the COCO JSON annotation file to analyze.')
    
    args = parser.parse_args()
    
    # Extract a descriptive name from the file path
    file_name = Path(args.annotation_file).stem
    analyze_annotations(args.annotation_file, f"Analysis of: {file_name}")

if __name__ == "__main__":
    main() 