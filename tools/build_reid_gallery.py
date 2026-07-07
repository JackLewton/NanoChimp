#!/usr/bin/env python3
"""
Build a Re-ID feature gallery from a directory of identity images.

For each identity subfolder under data_dir, extracts and averages feature
embeddings from all images to produce a single prototype embedding per identity.
The resulting gallery is saved as a JSON file for use during inference.
"""

import sys
import os
import argparse
import json
import numpy as np
from PIL import Image
from tqdm import tqdm
import glob

# Add the parent directory to the Python path to allow imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from tools.feature_extractor import FeatureExtractor

def find_latest_reid_model(directory="."):
    """Finds the most recent Re-ID model in a directory based on timestamp."""
    pattern = os.path.join(directory, "reid_model_*.pt")
    model_files = glob.glob(pattern)
    
    latest_model = None
    
    # Find the most recently modified file among the matches
    if model_files:
        latest_model = max(model_files, key=os.path.getmtime)

    if latest_model:
        print(f"Found latest Re-ID model: {os.path.basename(latest_model)}")
        return latest_model

    # Fallback to the default name if no versioned models are found
    default_path = os.path.join(directory, 'reid_model.pt')
    if os.path.exists(default_path):
        print("No versioned Re-ID models found. Falling back to 'reid_model.pt'.")
        return default_path
        
    return None # No model found

def build_gallery(data_dir, model_path, output_path):
    """
    Builds a feature gallery from a directory of curated chimp images.
    """
    if not os.path.isdir(data_dir):
        print(f"Error: Data directory '{data_dir}' not found.")
        return

    print(f"Loading Re-ID model from: {model_path}")
    try:
        feature_extractor = FeatureExtractor(model_path)
    except Exception as e:
        print(f"Error loading Re-ID model: {e}")
        return

    gallery = {}
    identity_folders = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d)) and not d.startswith('_')]
    
    if not identity_folders:
        print(f"No identity subdirectories found in '{data_dir}'.")
        print("Please ensure the directory contains one folder per chimp, named after them.")
        return

    print(f"Building gallery from {len(identity_folders)} identities...")
    for identity_name in tqdm(identity_folders, desc="Processing Identities"):
        identity_dir = os.path.join(data_dir, identity_name)
        image_files = [f for f in os.listdir(identity_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        if not image_files:
            print(f"Warning: No images found for '{identity_name}'. Skipping.")
            continue
            
        identity_features = []
        for img_file in image_files:
            img_path = os.path.join(identity_dir, img_file)
            try:
                with Image.open(img_path) as img:
                    features = feature_extractor.extract_features(img)
                    identity_features.append(features)
            except Exception as e:
                print(f"Warning: Could not process image '{img_path}'. Error: {e}")
        
        if identity_features:
            avg_features = np.mean(identity_features, axis=0)
            gallery[identity_name] = avg_features.tolist()

    # Save the gallery to a JSON file
    try:
        with open(output_path, 'w') as f:
            json.dump(gallery, f, indent=4)
        print(f"\nGallery built successfully with {len(gallery)} identities.")
        print(f"Saved to: {output_path}")
    except Exception as e:
        print(f"\nError saving gallery file: {e}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Build a Re-ID feature gallery for known chimpanzee identities.")
    parser.add_argument('--data_dir', type=str, default='data/reid',
                        help="Directory containing curated images for each identity (e.g., 'data/reid').")
    parser.add_argument(
        '--reid_model_path',
        type=str,
        default=None,
        help="Path to the trained Re-ID model (.pt). If omitted, will auto-select the newest model in reid_training/.",
    )
    parser.add_argument('--output_path', type=str, default='reid_gallery.json',
                        help="Path to save the output gallery JSON file.")
    
    args = parser.parse_args()
    
    model_path = args.reid_model_path
    if not model_path:
        # Prefer the stable "best" model (query-weighted) if present.
        best_path = os.path.join("reid_training", "reid_model_best.pt")
        best_micro_path = os.path.join("reid_training", "reid_model_best_map_micro.pt")
        if os.path.exists(best_micro_path):
            model_path = best_micro_path
        elif os.path.exists(best_path):
            model_path = best_path
        else:
            # Default behavior: newest model in reid_training/
            latest_model = find_latest_reid_model("reid_training")
            if latest_model:
                model_path = latest_model
            else:
                # Fallback to previous behavior (project root)
                latest_model = find_latest_reid_model(".")
                if latest_model:
                    model_path = latest_model
                else:
                    print("Error: No Re-ID model found. Please train a model first.")
                    sys.exit(1)
            
    build_gallery(args.data_dir, model_path, args.output_path)
