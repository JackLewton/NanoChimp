"""
The below script can be used to create the cropped ID images required to train the Re-ID model using your trained detection model.
"""

import os
import argparse
import json
import cv2
from tqdm import tqdm
import imagehash
from PIL import Image
import sys
import glob
import numpy as np

# Add project root to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import necessary components from inference script and other tools
from ultralytics import YOLO
from tools.feature_extractor import FeatureExtractor
from numpy.linalg import norm

# --- Helper Functions (copied from configs/inference.py) ---

def find_latest_model():
    """Find the most recent trained YOLO model in runs/detect/yolo_training, yolo_training, or current directory."""
    search_dirs = [os.path.join('runs', 'detect', 'yolo_training'), 'yolo_training']
    for yolo_training_dir in search_dirs:
        if os.path.exists(yolo_training_dir):
            pt_files = glob.glob(os.path.join(yolo_training_dir, '**/weights/best.pt'), recursive=True)
            if pt_files:
                return max(pt_files, key=os.path.getmtime)
            
    # Check current directory for any trained model
    local_models = glob.glob('*_bounding_box_model_*.pt') + glob.glob('yolov11_bounding_box_model.pt')
    if local_models:
        return max(local_models, key=os.path.getmtime)
        
    return 'yolo11n.pt'

def find_latest_reid_model(directory="."):
    """Finds the most recent Re-ID model."""
    pattern = os.path.join(directory, "reid_model_*.pt")
    model_files = glob.glob(pattern)
    if model_files:
        latest_model = max(model_files, key=os.path.getmtime)
        print(f"Found latest Re-ID model: {os.path.basename(latest_model)}")
        return latest_model
    if os.path.exists('reid_model.pt'):
        return 'reid_model.pt'
    return None

def load_gallery(file_path):
    """Loads a Re-ID gallery from a JSON file."""
    if not os.path.exists(file_path):
        return None
    try:
        with open(file_path, 'r') as f:
            gallery_from_file = json.load(f)
        return {name: np.array(features) for name, features in gallery_from_file.items()}
    except Exception:
        return None

def find_best_match(embedding, gallery, threshold=0.6):
    """Finds the best match for an embedding in the gallery using cosine similarity."""
    if gallery is None: return "Unknown", 0.0
    max_similarity = -1
    best_match_name = "Unknown"
    query_embedding = embedding / norm(embedding)
    for name, gallery_embedding in gallery.items():
        similarity = np.dot(query_embedding, gallery_embedding / norm(gallery_embedding))
        if similarity > max_similarity:
            max_similarity = similarity
            best_match_name = name
    return (best_match_name, max_similarity) if max_similarity >= threshold else ("Unknown", max_similarity)

def are_images_similar(image1, image2, hash_size=8, similarity_cutoff=5):
    """
    Compares two images using perceptual hashing to see if they are too similar.
    Returns True if the difference is less than the cutoff.
    """
    hash1 = imagehash.phash(Image.fromarray(cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)), hash_size=hash_size)
    hash2 = imagehash.phash(Image.fromarray(cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)), hash_size=hash_size)
    
    difference = hash1 - hash2
    return difference < similarity_cutoff

def run_inference_on_image_and_extract_crops(image_path, yolo_model, feature_extractor, reid_gallery, output_dir, last_saved_crop, args):
    """
    Runs detection and Re-ID on a single image and extracts crops.
    """
    print(f"\nProcessing image: {os.path.basename(image_path)}")

    extracted_count = 0
    discarded_count = 0

    results = yolo_model(image_path, conf=args.conf_threshold)
    result = results[0]
    frame_bgr = result.orig_img

    if result.boxes is not None:
        boxes = result.boxes.xyxy.cpu().numpy()

        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = map(int, box)
            crop = frame_bgr[y1:y2, x1:x2]
            if crop.size == 0: continue

            # --- Re-ID Logic ---
            features = feature_extractor.extract_features(Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)))
            name, similarity = find_best_match(features, reid_gallery, args.reid_threshold)

            # --- De-duplication and Saving Logic ---
            identity_for_saving = name
            if identity_for_saving in last_saved_crop:
                if are_images_similar(crop, last_saved_crop[identity_for_saving]):
                    discarded_count += 1
                    continue
            
            identity_dir = os.path.join(output_dir, identity_for_saving)
            os.makedirs(identity_dir, exist_ok=True)
            
            image_name = os.path.splitext(os.path.basename(image_path))[0]
            filename = f"{image_name}_box{i}_{identity_for_saving}.jpg"
            output_path = os.path.join(identity_dir, filename)
            
            cv2.imwrite(output_path, crop)
            last_saved_crop[identity_for_saving] = crop
            extracted_count += 1

    print(f"  - Extracted: {extracted_count} new crops")
    print(f"  - Discarded: {discarded_count} similar crops")
    return extracted_count, discarded_count

def run_inference_and_extract_crops(video_path, yolo_model, feature_extractor, reid_gallery, output_dir, last_saved_crop, args):
    """
    Runs the full tracking and Re-ID pipeline on a video and extracts crops.
    """
    print(f"\nProcessing video: {os.path.basename(video_path)}")

    track_identities = {}
    identity_assignments = {}
    track_feature_history = {}
    extracted_count = 0
    discarded_count = 0

    frame_idx = 0
    for result in yolo_model.track(
        source=video_path,
        conf=args.conf_threshold,
        stream=True,
        persist=True,
        tracker=args.tracker,
        vid_stride=args.vid_stride
    ):
        frame_bgr = result.orig_img
        
        if result.boxes.id is not None:
            track_ids = result.boxes.id.int().cpu().tolist()
            boxes = result.boxes.xyxy.cpu().numpy()

            for box, track_id in zip(boxes, track_ids):
                x1, y1, x2, y2 = map(int, box)
                crop = frame_bgr[y1:y2, x1:x2]
                if crop.size == 0: continue

                # --- Re-ID Logic ---
                features = feature_extractor.extract_features(Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)))
                if track_id not in track_feature_history:
                    track_feature_history[track_id] = []
                track_feature_history[track_id].append(features)
                avg_features = np.mean(track_feature_history[track_id], axis=0)
                
                name, similarity = find_best_match(avg_features, reid_gallery, args.reid_threshold)

                if name != "Unknown":
                    current_holder_track_id, current_holder_sim = identity_assignments.get(name, (None, -1.0))
                    if similarity > current_holder_sim:
                        if current_holder_track_id is not None and current_holder_track_id in track_identities:
                            del track_identities[current_holder_track_id]
                        identity_assignments[name] = (track_id, similarity)
                        track_identities[track_id] = (name, similarity)
                
                # --- De-duplication and Saving Logic ---
                identity_for_saving = track_identities.get(track_id, ("Unknown", 0.0))[0]
                if identity_for_saving in last_saved_crop:
                    if are_images_similar(crop, last_saved_crop[identity_for_saving]):
                        discarded_count += 1
                        continue
                
                identity_dir = os.path.join(output_dir, identity_for_saving)
                os.makedirs(identity_dir, exist_ok=True)
                
                video_name = os.path.splitext(os.path.basename(video_path))[0]
                filename = f"{video_name}_track{track_id}_frame{frame_idx}.jpg"
                output_path = os.path.join(identity_dir, filename)
                
                cv2.imwrite(output_path, crop)
                last_saved_crop[identity_for_saving] = crop
                extracted_count += 1

        frame_idx += args.vid_stride

    print(f"  - Extracted: {extracted_count} new crops")
    print(f"  - Discarded: {discarded_count} similar crops")
    return extracted_count, discarded_count

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run inference on videos and images to extract unique, pre-sorted crops for labeling.")
    parser.add_argument('--input_dir', type=str, default='temp/tools_input_extract_crops_for_labelling',
                        help="Directory containing the raw input videos and images.")
    parser.add_argument('--output_dir', type=str, default='temp/tools_output_extract_crops_for_labelling',
                        help="Directory to save the extracted and sorted crops.")
    
    # --- Model and Threshold Arguments ---
    parser.add_argument('--model_path', help='Path to a specific YOLO model file (optional).')
    parser.add_argument('--reid_model_path', help='Path to a specific Re-ID model file (optional).')
    parser.add_argument('--reid_gallery_path', type=str, default='reid_gallery.json', help='Path to the pre-built Re-ID gallery file.')
    parser.add_argument('--conf_threshold', type=float, default=0.3, help='YOLO detection confidence threshold.')
    parser.add_argument('--reid_threshold', type=float, default=0.6, help='Re-ID similarity threshold for a match.')
    parser.add_argument('--tracker', type=str, default='bytetrack.yaml', help='Tracker configuration file.')
    parser.add_argument('--vid_stride', type=int, default=1, help='Process every Nth frame.')
    parser.add_argument(
        '--phash_threshold', 
        type=int, 
        default=5, 
        help='Perceptual hash difference threshold. Lower is stricter (fewer, more unique images).'
    )
    parser.add_argument('--min_width', type=int, default=50, help='Minimum width of a crop to be saved.')
    parser.add_argument('--min_height', type=int, default=50, help='Minimum height of a crop to be saved.')

    args = parser.parse_args()
    
    # --- Load Models and Gallery ---
    print("--- Loading Models and Gallery ---")
    yolo_model_path = args.model_path or find_latest_model()
    reid_model_path = args.reid_model_path or find_latest_reid_model()
    reid_gallery = load_gallery(args.reid_gallery_path)

    if not yolo_model_path or not os.path.exists(yolo_model_path):
        print("Error: YOLO model not found. Please train a model first.")
        sys.exit(1)
    if not reid_model_path or not os.path.exists(reid_model_path):
        print("Error: Re-ID model not found. Please train a model and build a gallery first.")
        sys.exit(1)
    if not reid_gallery:
        print("Error: Re-ID gallery not found. Please build the gallery first.")
        sys.exit(1)
        
    yolo_model = YOLO(yolo_model_path)
    feature_extractor = FeatureExtractor(reid_model_path)
    print("All models loaded successfully.\n")

    # --- Find and Process Media ---
    if not os.path.isdir(args.input_dir):
        print(f"Error: Input directory not found at '{args.input_dir}'")
    else:
        video_extensions = ('.mp4', '.avi', '.mov', '.mkv')
        image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')
        media_files = []
        for root, _, files in os.walk(args.input_dir):
            for f in files:
                if f.lower().endswith(video_extensions + image_extensions):
                    media_files.append(os.path.join(root, f))
        
        if not media_files:
            print(f"No videos or images found in '{args.input_dir}'.")
        else:
            last_saved_crop = {}
            total_extracted = 0
            total_discarded = 0
            
            for media_path in media_files:
                if media_path.lower().endswith(video_extensions):
                    ext, disc = run_inference_and_extract_crops(media_path, yolo_model, feature_extractor, reid_gallery, args.output_dir, last_saved_crop, args)
                else:
                    ext, disc = run_inference_on_image_and_extract_crops(media_path, yolo_model, feature_extractor, reid_gallery, args.output_dir, last_saved_crop, args)
                
                total_extracted += ext
                total_discarded += disc
            
            print("\n--- Summary ---")
            print(f"Processed {len(media_files)} file(s).")
            print(f"Total new crops extracted: {total_extracted}")
            print(f"Total similar crops discarded: {total_discarded}")
            print(f"New data is ready for review in: '{args.output_dir}'")
