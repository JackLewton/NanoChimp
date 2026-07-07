#!/usr/bin/env python3
"""
Visually verify COCO format annotations by drawing bounding boxes on images.

Draws bounding boxes on annotated images and labels background images,
saving results to an output directory for manual inspection. 

I found this script helpful to just check that the annotations are correct before training.
"""

import json
import os
import cv2
import numpy as np
from pathlib import Path
import argparse
import random
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def load_annotations(json_path):
    """
    Load COCO format annotations from JSON file.
    """
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data

def draw_bbox_on_image(image_path, annotations, output_path, image_id):
    """
    Draw bounding boxes on an image and save it.
    
    Args:
        image_path: Path to the image file
        annotations: List of annotations for this image
        output_path: Path to save the annotated image
        image_id: ID of the image in the COCO dataset
    """
    # Load image
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"Warning: Could not load image {image_path}")
        return False
    
    # Convert BGR to RGB for matplotlib
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Create figure and axis
    fig, ax = plt.subplots(1, figsize=(12, 8))
    ax.imshow(image_rgb)
    
    # Draw bounding boxes
    colors = ['red', 'blue', 'green', 'yellow', 'purple', 'orange', 'pink', 'brown']
    color_idx = 0
    
    for ann in annotations:
        if ann['image_id'] == image_id:
            bbox = ann['bbox']  # [x, y, width, height]
            category_id = ann.get('category_id', 1)
            
            # Create rectangle patch
            rect = patches.Rectangle(
                (bbox[0], bbox[1]), bbox[2], bbox[3],
                linewidth=2, edgecolor=colors[color_idx % len(colors)],
                facecolor='none'
            )
            ax.add_patch(rect)
            
            # Add label
            label = f"Chimp (ID: {category_id})"
            ax.text(bbox[0], bbox[1] - 10, label,
                   bbox=dict(boxstyle="round,pad=0.3", facecolor=colors[color_idx % len(colors)], alpha=0.7),
                   fontsize=10, color='white', weight='bold')
            
            color_idx += 1
    
    # Remove axes
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(f"Image: {image_path.name} (ID: {image_id})", fontsize=14, weight='bold')
    
    # Save the annotated image
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return True

def draw_negative_label_on_image(image_path, output_path):
    """
    Saves a copy of an image with a "Negative Sample" label.
    """
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"Warning: Could not load image {image_path}")
        return False
    
    # Convert BGR to RGB for matplotlib
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Create figure and axis
    fig, ax = plt.subplots(1, figsize=(12, 8))
    ax.imshow(image_rgb)
    
    # Add a prominent text label
    ax.text(0.5, 0.5, 'Background\n(Negative Sample)',
            horizontalalignment='center',
            verticalalignment='center',
            transform=ax.transAxes,
            fontsize=30,
            color='white',
            weight='bold',
            bbox=dict(boxstyle="round,pad=0.5", facecolor='black', alpha=0.7))
    
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(f"Image: {image_path.name}", fontsize=14, weight='bold')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return True


def check_annotations_for_batch(image_dir, annotation_file, output_dir, sample=False):
    """
    Check annotations for a specific batch by drawing bounding boxes on images.
    
    Args:
        image_dir: Directory containing images for this batch
        annotation_file: Path to the annotation JSON file
        output_dir: Directory to save annotated images
        sample: Whether to process a 10% sample of the images
    """
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load annotations
    if not os.path.exists(annotation_file):
        print(f"Error: Annotation file {annotation_file} not found!")
        return
    
    data = load_annotations(annotation_file)
    
    # Create image_id to annotations mapping
    image_annotations = {}
    for ann in data.get('annotations', []):
        image_id = ann['image_id']
        if image_id not in image_annotations:
            image_annotations[image_id] = []
        image_annotations[image_id].append(ann)
    
    # Create image_id to image_info mapping
    image_info_map = {img['id']: img for img in data.get('images', [])}
    
    # Find negative samples (images with no annotations)
    annotated_image_ids = {ann['image_id'] for ann in data.get('annotations', [])}

    print(f"Processing: {annotation_file}")
    print(f"Found {len(image_info_map)} total images in JSON.")
    print(f"Found {len(data.get('annotations', []))} total annotations.")
    positive_count = len(annotated_image_ids)
    negative_count = len(image_info_map) - positive_count
    print(f"Positive samples (with annotations): {positive_count}")
    print(f"Negative samples (no annotations): {negative_count}")

    processed_pos_count = 0
    processed_neg_count = 0
    neg_samples_to_visualize = 5  # Max number of negative samples to save

    # Handle sampling
    images_to_process = image_info_map
    if sample:
        print("\nProcessing a 10% random sample of images...")
        num_samples = max(1, int(len(image_info_map) * 0.1))
        
        # Ensure we don't try to sample more images than exist
        if num_samples > len(image_info_map):
            num_samples = len(image_info_map)
            
        random_image_ids = random.sample(list(image_info_map.keys()), num_samples)
        images_to_process = {img_id: image_info_map[img_id] for img_id in random_image_ids}
        print(f"Sample size: {len(images_to_process)} images")

    # Process all images from the JSON file
    for image_id, img_info in images_to_process.items():
        try:
            image_filename = img_info['file_name']
            image_path = Path(image_dir) / image_filename

            if not image_path.exists():
                print(f"✗ Image file not found, skipping: {image_path}")
                continue

            # Check if it's a positive or negative sample
            if image_id in annotated_image_ids:
                # It's a positive sample, draw bounding boxes
                output_filename = f"annotated_{image_filename}"
                output_path = output_dir / output_filename
                annotations = image_annotations[image_id]
                
                if draw_bbox_on_image(image_path, annotations, output_path, image_id):
                    processed_pos_count += 1
                    print(f"✓ Processed Positive: {image_filename} ({len(annotations)} annotations)")
                else:
                    print(f"✗ Failed to draw boxes on: {image_filename}")

            else:
                # It's a negative sample, visualize a few of them
                if processed_neg_count < neg_samples_to_visualize:
                    output_filename = f"negative_{image_filename}"
                    output_path = output_dir / output_filename
                    
                    if draw_negative_label_on_image(image_path, output_path):
                        processed_neg_count += 1
                        print(f"✓ Visualized Negative: {image_filename}")
                    else:
                        print(f"✗ Failed to visualize negative sample: {image_filename}")
        
        except Exception as e:
            print(f"An unexpected error occurred processing image ID {image_id} ({img_info.get('file_name', 'N/A')}): {e}")

    print(f"\nCompleted! Processed {processed_pos_count} positive images and visualized {processed_neg_count} out of {negative_count} negative images.")
    print(f"Annotated images saved to: {output_dir}")

def check_single_batch(image_dir, annotation_file, output_dir=None, sample=False):
    """
    Check annotations for a single batch.
    
    Args:
        image_dir: Path to image directory
        annotation_file: Path to annotation JSON file
        output_dir: Output directory (optional)
        sample: Whether to process a 10% sample of the images
    """
    if output_dir is None:
        output_dir = Path("annotation_checks")
    
    check_annotations_for_batch(image_dir, annotation_file, output_dir, sample)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Check annotations by drawing bounding boxes on images")
    parser.add_argument("--image_dir", help="Path to image directory", required=True)
    parser.add_argument("--annotation_file", help="Path to annotation JSON file", required=True)
    parser.add_argument("--output_dir", help="Output directory for annotated images")
    parser.add_argument("--sample", action="store_true", help="Process a random 10% sample of the images")
    
    args = parser.parse_args()
    
    print(f"Processing single batch...")
    check_single_batch(args.image_dir, args.annotation_file, args.output_dir, args.sample) 