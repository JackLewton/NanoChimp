import os
import sys
import argparse
import glob
import json
from datetime import datetime
from typing import Optional
import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from ultralytics import YOLO
from numpy.linalg import norm

# Add parent directory to path to allow imports from 'tools'
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from tools.feature_extractor import FeatureExtractor


def load_gallery(file_path):
    """Loads a Re-ID gallery from a JSON file."""
    if not os.path.exists(file_path):
        print(f"Warning: Re-ID gallery not found at '{file_path}'.")
        return None
    try:
        with open(file_path, 'r') as f:
            gallery_from_file = json.load(f)
        # Convert feature lists back to numpy arrays
        gallery = {name: np.array(features) for name, features in gallery_from_file.items()}
        return gallery
    except Exception as e:
        print(f"Error loading gallery from file: {e}")
        return None


def find_latest_reid_model(directory="reid_training"):
    """Finds the most recent Re-ID model in a directory based on timestamp."""
    if not os.path.exists(directory):
        directory = "."
    pattern = os.path.join(directory, "reid_model_*.pt")
    model_files = glob.glob(pattern)
    
    if model_files:
        return max(model_files, key=os.path.getmtime)

    # Fallback to the default name if no versioned models are found
    default_path = os.path.join(directory, 'reid_model.pt')
    if os.path.exists(default_path):
        return default_path
        
    return None


def find_top_matches(embedding, gallery, top_n=5):
    """Finds the top N matches for an embedding in the gallery using cosine similarity."""
    if gallery is None or len(gallery) == 0:
        return []

    matches = []
    embedding_norm = norm(embedding)
    if embedding_norm == 0:
        return []
    query_embedding = embedding / embedding_norm

    for name, gallery_embedding in gallery.items():
        gallery_embedding_norm = norm(gallery_embedding)
        if gallery_embedding_norm == 0:
            continue
        gallery_embedding = gallery_embedding / gallery_embedding_norm
        
        # Cosine similarity
        similarity = np.dot(query_embedding, gallery_embedding)
        matches.append((name, similarity))

    # Sort matches by similarity in descending order and get the top N
    matches.sort(key=lambda x: x[1], reverse=True)
    return matches[:top_n]


class ColorGenerator:
    """Generates consistent, unique colors for each identity."""
    def __init__(self):
        self.colors = {}
        # Colorblind-friendly palette (BGR format for OpenCV)
        self.palette = [
            (0, 114, 178),      # Deep blue
            (230, 159, 0),      # Orange
            (0, 158, 115),      # Teal
            (213, 94, 0),       # Vermillion
            (86, 180, 233),     # Sky blue
            (204, 121, 167),    # Reddish purple
            (0, 142, 204),      # Cyan-blue
            (255, 157, 167),    # Light pink
            (156, 117, 95),     # Brown
            (186, 176, 172),    # Tan
            (140, 140, 140),    # Grey
            (240, 228, 66),     # Muted yellow
        ]

    def get_color(self, identity):
        if identity not in self.colors:
            idx = len(self.colors) % len(self.palette)
            self.colors[identity] = self.palette[idx]
        return self.colors[identity]


def load_yolo_model(model_path):
    """Load the trained YOLO model"""
    return YOLO(model_path)


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


def predict_bbox_yolo(model, image_path, conf_threshold=0.5, nms_iou_threshold=0.5, imgsz=640):
    """Predict bounding boxes for a single image using YOLO"""
    print(f"--> Running prediction with conf_threshold={conf_threshold}, nms_iou_threshold={nms_iou_threshold}, imgsz={imgsz}")
    results = model(image_path, conf=conf_threshold, iou=nms_iou_threshold, imgsz=imgsz)
    result = results[0]
    
    original_image = Image.open(image_path).convert('RGB')
    predictions = []
    if result.boxes is not None:
        boxes = result.boxes.xyxy.cpu().numpy()
        confidences = result.boxes.conf.cpu().numpy()
        class_ids = result.boxes.cls.cpu().numpy()
        
        for box, conf, class_id in zip(boxes, confidences, class_ids):
            x1, y1, x2, y2 = box
            predictions.append({
                'bbox': (x1, y1, x2-x1, y2-y1),  # Convert to (x, y, width, height)
                'confidence': conf,
                'class_id': int(class_id)
            })
    
    return predictions, original_image


def visualize_prediction_yolo(image_path, predictions, original_image, save_path=None):
    """Visualize YOLO predictions on the original image"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].imshow(original_image)
    
    for pred in predictions:
        bbox = pred['bbox']
        conf = pred['confidence']
        class_id = pred['class_id']
        
        x, y, w, h = bbox
        rect = patches.Rectangle((x, y), w, h, linewidth=2, edgecolor='red', facecolor='none')
        axes[0].add_patch(rect)
        axes[0].text(x, y-5, f'Class {class_id}: {conf:.2f}', color='red', fontsize=8)
    
    axes[0].set_title('YOLO Predictions')
    axes[0].axis('off')
    
    if predictions:
        best_pred = max(predictions, key=lambda x: x['confidence'])
        bbox = best_pred['bbox']
        x, y, w, h = bbox
        
        x1, y1 = max(0, int(x)), max(0, int(y))
        x2, y2 = min(original_image.width, int(x + w)), min(original_image.height, int(y + h))
        
        if x2 > x1 and y2 > y1:
            cropped = original_image.crop((x1, y1, x2, y2))
            axes[1].imshow(cropped)
            axes[1].set_title(f'Best Detection (conf: {best_pred["confidence"]:.2f})')
        else:
            axes[1].text(0.5, 0.5, 'No valid detection', ha='center', va='center', transform=axes[1].transAxes)
            axes[1].set_title('No Valid Detection')
    else:
        axes[1].text(0.5, 0.5, 'No detections', ha='center', va='center', transform=axes[1].transAxes)
        axes[1].set_title('No Detections')
    
    axes[1].axis('off')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def run_video_inference(model, video_path: str, output_path: str, conf_threshold: float = 0.5, vid_stride: int = 1, nms_iou_threshold: float = 0.5, imgsz: int = 640):
    """Run detection on a video and save an annotated video."""
    import cv2

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if output_path:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    for result in model.predict(source=video_path, conf=conf_threshold, iou=nms_iou_threshold, imgsz=imgsz, stream=True, vid_stride=vid_stride):
        frame_bgr = result.plot()
        writer.write(frame_bgr)

    writer.release()
    cap.release()


# core engine, need custom model.track()
def run_video_tracking(
    model,
    video_path: str,
    output_path: str,
    conf_threshold: float = 0.5,
    tracker: str = 'bytetrack.yaml',
    vid_stride: int = 1,
    feature_extractor: Optional[FeatureExtractor] = None,
    reid_gallery: Optional[dict] = None,
    color_generator: Optional[ColorGenerator] = None,
    reid_threshold: float = 0.6,
    nms_iou_threshold: float = 0.5,
    show_top_matches: bool = False,
    tracking_output_path: Optional[str] = None,
    imgsz: int = 640,
):
    """Run detection + multi-object tracking on a video and save annotated video with IDs."""
    import cv2
    track_features = {}  # Dictionary to store features for each track ID
    track_identities = {} # Dictionary to store the identified name for each track ID
    track_feature_history = {} # Stores raw feature vectors for averaging
    identity_assignments = {} # Maps a known identity name to its canonical track_id and best similarity

    if tracking_output_path:
        os.makedirs(os.path.dirname(os.path.abspath(tracking_output_path)), exist_ok=True)
        tracking_output_file = open(tracking_output_path, 'w')
    else:
        tracking_output_file = None

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if output_path:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    frame_idx = 0
    for result in model.track(
        source=video_path,
        conf=conf_threshold,
        iou=nms_iou_threshold,
        imgsz=imgsz,
        stream=True,
        persist=True,
        tracker=tracker,
        vid_stride=vid_stride
    ):
        frame_bgr = result.orig_img
        
        if tracking_output_file and result.boxes.id is not None:
            track_ids = result.boxes.id.int().cpu().tolist()
            boxes = result.boxes.xyxy.cpu().numpy()
            confs = result.boxes.conf.cpu().numpy()
            
            for box, track_id, conf in zip(boxes, track_ids, confs):
                x1, y1, x2, y2 = box
                w, h = x2 - x1, y2 - y1
                tracking_output_file.write(f'{frame_idx + 1},{track_id},{x1},{y1},{w},{h},{conf},-1,-1,-1\n')

        # re-id and identity matching
        if feature_extractor and reid_gallery and result.boxes.id is not None:
            track_ids = result.boxes.id.int().cpu().tolist()
            boxes = result.boxes.xyxy.cpu().numpy()
            
            current_track_ids = set(track_ids)
            
            for box, track_id in zip(boxes, track_ids):
                x1, y1, x2, y2 = map(int, box)
                crop = frame_bgr[y1:y2, x1:x2]
                if crop.size == 0: continue

                # Extract features and average them over time for stability
                features = feature_extractor.extract_features(Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)))
                if track_id not in track_feature_history:
                    track_feature_history[track_id] = []
                track_feature_history[track_id].append(features)
                avg_features = np.mean(track_feature_history[track_id], axis=0)
                
                # Save latest features for export
                track_features[track_id] = avg_features.tolist()
                
                # Find the top 5 matches for the track's averaged features
                top_matches = find_top_matches(avg_features, reid_gallery, top_n=5)
                
                # identity persistence logic
                if top_matches:
                    best_name, best_sim = top_matches[0]
                    
                    if best_name != "Unknown" and best_sim >= reid_threshold:
                        current_holder_track_id, current_holder_sim = identity_assignments.get(best_name, (None, -1.0))

                        if current_holder_track_id is None:
                            # First time this identity is confidently recognized.
                            # This track_id becomes the canonical ID for this identity.
                            identity_assignments[best_name] = (track_id, best_sim)
                            track_identities[track_id] = top_matches
                        
                        elif track_id == current_holder_track_id:
                             # The holder is still visible, update similarity score
                            if best_sim > current_holder_sim:
                                identity_assignments[best_name] = (track_id, best_sim)
                            track_identities[track_id] = top_matches

                        elif current_holder_track_id not in current_track_ids:
                            # Original holder of this ID is gone. New track takes over.
                            print(f"Identity '{best_name}' re-acquired by new track {track_id} (previous: {current_holder_track_id}).")
                            track_feature_history[track_id].extend(track_feature_history.get(current_holder_track_id, []))
                            
                            # Clean up old track data
                            track_feature_history.pop(current_holder_track_id, None)
                            track_identities.pop(current_holder_track_id, None)
                            
                            identity_assignments[best_name] = (track_id, best_sim)
                            track_identities[track_id] = top_matches
                        
                        elif best_sim > current_holder_sim:
                            # Conflict: two tracks claim same ID. Higher similarity wins.
                            print(f"Conflict for '{best_name}'. Track {track_id} ({best_sim:.2f}) wins over {current_holder_track_id} ({current_holder_sim:.2f}).")
                            
                            if current_holder_track_id in track_identities:
                                track_identities[current_holder_track_id] = [m for m in track_identities[current_holder_track_id] if m[0] != best_name]
                                if not track_identities[current_holder_track_id]:
                                    track_identities[current_holder_track_id] = [("Unknown", 0.0)]
                            
                            identity_assignments[best_name] = (track_id, best_sim)
                            track_identities[track_id] = top_matches
                        
                        else:
                            # Weaker match for already-assigned identity. Mark as Unknown.
                            track_identities[track_id] = [("Unknown", 0.0)]
                    else:
                        track_identities[track_id] = top_matches if top_matches else [("Unknown", 0.0)]
                else:
                    track_identities[track_id] = [("Unknown", 0.0)]

        # custom drawing
        annotated_frame = frame_bgr.copy()
        if result.boxes.id is not None:
            track_ids = result.boxes.id.int().cpu().tolist()
            boxes = result.boxes.xyxy.cpu().numpy()
            
            for box, track_id in zip(boxes, track_ids):
                x1, y1, x2, y2 = map(int, box)
                
                # get identity info
                top_matches = track_identities.get(track_id)
                best_match_name, best_match_sim = ("Unknown", 0.0)
                display_track_id = track_id

                if top_matches:
                    best_match_name, best_match_sim = top_matches[0]
                
                if best_match_name != "Unknown":
                    canonical_track_id, _ = identity_assignments.get(best_match_name, (track_id, 0))
                    display_track_id = canonical_track_id

                # Get a unique color for the identity
                color = (128, 128, 128) # Default grey for "Unknown"
                if color_generator and best_match_name != "Unknown" and best_match_sim >= reid_threshold:
                    color = color_generator.get_color(best_match_name)

                # Draw bounding box
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 3)
                
                if show_top_matches:
                    # create multi-line label for top 5 matches
                    label_lines = []
                    if top_matches:
                        primary_name = best_match_name if best_match_sim >= reid_threshold else "Unknown"
                        label_lines.append(f"{primary_name} ({best_match_sim:.2f}) ID: {display_track_id}")

                        for name, sim in top_matches[1:]:
                            if sim > 0.0:
                                label_lines.append(f"  {name} ({sim:.2f})")
                    else:
                        label_lines.append(f"Unknown ID: {display_track_id}")

                    font = cv2.FONT_HERSHEY_SIMPLEX
                    font_scale = 0.96
                    thickness = 3
                    
                    for i, line in enumerate(label_lines):
                        (text_width, text_height), baseline = cv2.getTextSize(line, font, font_scale, thickness)
                        label_y = y1 - (len(label_lines) - i) * (text_height + baseline + 5)
                        
                        bg_y1 = max(0, label_y - text_height - 5)
                        bg_y2 = max(0, label_y + baseline)
                        
                        cv2.rectangle(annotated_frame, (x1, bg_y1), (x1 + text_width, bg_y2), color, -1)
                        cv2.putText(annotated_frame, line, (x1, label_y), font, font_scale, (255, 255, 255), thickness)
                else:
                    # create a single label for the best match
                    primary_name = best_match_name if best_match_sim >= reid_threshold else "Unknown"
                    label = f"{primary_name} ({best_match_sim:.2f}) ID: {display_track_id}"
                    
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    font_scale = 0.96
                    thickness = 3
                    (text_width, text_height), baseline = cv2.getTextSize(label, font, font_scale, thickness)
                    
                    cv2.rectangle(annotated_frame, (x1, y1 - text_height - baseline - 5), (x1 + text_width, y1), color, -1)
                    cv2.putText(annotated_frame, label, (x1, y1 - baseline - 2), font, font_scale, (255, 255, 255), thickness)

        writer.write(annotated_frame)
        frame_idx += vid_stride

    writer.release()
    cap.release()

    if tracking_output_file:
        tracking_output_file.close()

    # Save features to a JSON file
    if feature_extractor and track_features:
        features_output_path = os.path.splitext(output_path)[0] + '_features.json'
        with open(features_output_path, 'w') as f:
            json.dump(track_features, f, indent=4)
        print(f"Saved Re-ID features to: {features_output_path}")


def export_cvat_xml_zip(
    tracking_txt_path: str,
    label_name: str = "chimp",
    keyframe_interval: int = 1,
    total_frames: int | None = None,
) -> str:
    """Convert a MOT tracking .txt to a CVAT for video 1.1 XML zip with interpolation keyframes.

    CVAT XML natively supports keyframe interpolation: only keyframe boxes are emitted,
    and CVAT linearly interpolates bounding boxes for intermediate frames.
    """
    import zipfile
    from xml.etree.ElementTree import Element, SubElement, tostring
    from collections import defaultdict

    rows_by_track: dict[int, list[tuple]] = defaultdict(list)
    max_frame_0based = 0
    with open(tracking_txt_path, 'r') as f:
        for line in f:
            parts = line.strip().split(',')
            if len(parts) < 7:
                continue
            frame_1based = int(parts[0])
            frame_0based = frame_1based - 1
            track_id = int(parts[1])
            x, y, w, h = float(parts[2]), float(parts[3]), float(parts[4]), float(parts[5])
            rows_by_track[track_id].append((frame_0based, x, y, w, h))
            max_frame_0based = max(max_frame_0based, frame_0based)

    if total_frames is not None:
        video_last_frame = total_frames - 1
    else:
        video_last_frame = max_frame_0based

    root = Element('annotations')
    SubElement(root, 'version').text = '1.1'

    for track_id in sorted(rows_by_track):
        rows = sorted(rows_by_track[track_id], key=lambda r: r[0])
        track_el = SubElement(root, 'track', id=str(track_id), label=label_name)

        last_idx = len(rows) - 1
        for i, (frame_id, x, y, w, h) in enumerate(rows):
            is_keyframe = (i == 0 or i == last_idx or i % keyframe_interval == 0)
            if not is_keyframe:
                continue
            SubElement(track_el, 'box', **{
                'frame': str(frame_id),
                'xtl': f'{x:.2f}',
                'ytl': f'{y:.2f}',
                'xbr': f'{x + w:.2f}',
                'ybr': f'{y + h:.2f}',
                'outside': '0',
                'occluded': '0',
                'keyframe': '1',
            })

        last_frame = rows[-1][0]
        if last_frame < video_last_frame:
            SubElement(track_el, 'box', **{
                'frame': str(last_frame + 1),
                'xtl': '0', 'ytl': '0', 'xbr': '0', 'ybr': '0',
                'outside': '1',
                'occluded': '0',
                'keyframe': '1',
            })

    xml_bytes = b'<?xml version="1.0" encoding="utf-8"?>\n' + tostring(root, encoding='unicode').encode('utf-8')

    cvat_zip_path = tracking_txt_path.replace('.txt', '_cvat.zip')
    with zipfile.ZipFile(cvat_zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
        zf.writestr('annotations.xml', xml_bytes)

    return cvat_zip_path


def process_video(model, video_path, args, feature_extractor=None, reid_gallery=None, color_generator=None):
    video_stem = os.path.splitext(os.path.basename(video_path))[0]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_video_path = os.path.join(args.output_dir, f"output_{video_stem}_{timestamp}.mp4")

    tracking_output_path = None
    if args.use_tracking and args.output_dir:
        video_stem = os.path.splitext(os.path.basename(video_path))[0]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        tracking_output_path = os.path.join(args.output_dir, f"tracking_{video_stem}_{timestamp}.txt")

    print(f"--> Running video processing with conf_threshold={args.conf_threshold}, nms_iou_threshold={args.nms_iou_threshold}, imgsz={args.imgsz}")
    print(f"Running {'tracking' if args.use_tracking else 'detection'} on video: {video_path}")
    if args.use_tracking:
        run_video_tracking(
            model,
            video_path=video_path,
            output_path=output_video_path,
            conf_threshold=args.conf_threshold,
            tracker=args.tracker,
            vid_stride=args.vid_stride,
            feature_extractor=feature_extractor,
            reid_gallery=reid_gallery,
            color_generator=color_generator,
            reid_threshold=args.reid_threshold,
            nms_iou_threshold=args.nms_iou_threshold,
            show_top_matches=args.show_top_matches,
            tracking_output_path=tracking_output_path,
            imgsz=args.imgsz,
        )
    else:
        run_video_inference(
            model,
            video_path=video_path,
            output_path=output_video_path,
            conf_threshold=args.conf_threshold,
            vid_stride=args.vid_stride,
            nms_iou_threshold=args.nms_iou_threshold,
            imgsz=args.imgsz,
        )
    print(f"Video processing completed! Saved to {output_video_path}")
    if tracking_output_path:
        print(f"Tracking results saved to {tracking_output_path}")
        if getattr(args, 'export_cvat_mot', False):
            interval = getattr(args, 'cvat_keyframe_interval', 15)
            import cv2 as _cv2
            _cap = _cv2.VideoCapture(video_path)
            _total = int(_cap.get(_cv2.CAP_PROP_FRAME_COUNT))
            _cap.release()
            cvat_zip = export_cvat_xml_zip(tracking_output_path, keyframe_interval=interval, total_frames=_total)
            print(f"CVAT XML zip saved to {cvat_zip} (keyframe interval: {interval})")


def batch_inference(model, image_dir, output_dir='infer_output', conf_threshold=0.5, nms_iou_threshold=0.5, imgsz=640):
    """Run inference on all images in a directory"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all image files
    image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
    image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(image_extensions)]
    
    print(f"Found {len(image_files)} images in {image_dir}")
    
    for i, img_file in enumerate(image_files):
        print(f"Processing {i+1}/{len(image_files)}: {img_file}")
        
        img_path = os.path.join(image_dir, img_file)
        predictions, original_image = predict_bbox_yolo(model, img_path, conf_threshold, nms_iou_threshold, imgsz)
        
        # Save visualization
        output_path = os.path.join(output_dir, f"output_{img_file}")
        visualize_prediction_yolo(img_path, predictions, original_image, output_path)
        
        # Print predictions
        if predictions:
            print(f"  Found {len(predictions)} objects:")
            for j, pred in enumerate(predictions):
                bbox = pred['bbox']
                print(f"    Object {j+1}: bbox={bbox}, conf={pred['confidence']:.3f}, class={pred['class_id']}")
        else:
            print("  No objects detected")


def main():
    parser = argparse.ArgumentParser(description='Run YOLO inference and tracking on images or videos')
    parser.add_argument('--input_dir', default='infer_input', help='Input directory containing images or videos (default: infer_input)')
    parser.add_argument('--output_dir', default='infer_output', help='Output directory for results (default: infer_output)')
    
    # Detection thresholds
    parser.add_argument('--conf_threshold', type=float, default=0.6, help='Confidence threshold for object detection (default: 0.6)')
    parser.add_argument('--nms_iou_threshold', type=float, default=0.5, help='IoU threshold for Non-Maximum Suppression (default: 0.5)')
    parser.add_argument('--imgsz', type=int, default=640, help='Inference image size (longer side).')

    # Model
    parser.add_argument('--model_path', help='Path to a specific YOLO model file (optional, overrides auto-selection)')
    
    # Video and tracking options
    parser.add_argument('--use_tracking', action='store_true', help='Enable multi-object tracking')
    parser.add_argument('--vid_stride', type=int, default=1, help='Process every Nth frame for tracking (default: 1)')
    parser.add_argument('--tracker', type=str, default='bytetrack.yaml', help='Tracker configuration: bytetrack.yaml or botsort.yaml')

    # Re-ID
    parser.add_argument('--use_reid', action='store_true', help='Enable Re-ID to assign persistent identities.')
    parser.add_argument('--reid_model_path', type=str, default=None, help='Path to the trained Re-ID model. If not specified, will auto-detect latest model in reid_training/')
    parser.add_argument('--reid_gallery_path', type=str, default='reid_gallery.json', help='Path to the Re-ID gallery file.')
    parser.add_argument('--reid_threshold', type=float, default=0.5, help='Cosine similarity threshold for a Re-ID match (default: 0.5).')
    parser.add_argument('--show_top_matches', action='store_true', help='Show top N matches for each track.')
    parser.add_argument('--export_cvat_mot', action='store_true', help='Also export a CVAT-compatible MOT .zip alongside the tracking .txt')
    parser.add_argument('--cvat_keyframe_interval', type=int, default=15, help='Keep every Nth frame per track in CVAT export (default: 15).')

    args = parser.parse_args()
    
    # argument validation
    if args.use_reid and not args.use_tracking:
        print("Error: --use_reid requires --use_tracking to be enabled.")
        return

    # load models
    model_path = args.model_path or find_latest_model()
    
    if model_path is None:
        print('No trained YOLO model found and no default specified. Please train a model first.')
        return
    
    print(f"Loading YOLO model from: {model_path}")
    try:
        model = load_yolo_model(model_path)
        print(f'Model loaded successfully from {model_path}')
    except Exception as e:
        print(f'Error loading model from {model_path}: {e}')
        return
    
    # Initialize Feature Extractor if Re-ID is enabled
    feature_extractor = None
    reid_gallery = None
    color_generator = None

    if args.use_reid:
        reid_model_path = args.reid_model_path
        
        # If no path specified, try to find the latest model
        if not reid_model_path:
            reid_model_path = find_latest_reid_model('reid_training')
        
        if not reid_model_path or not os.path.exists(reid_model_path):
            print(f"Re-ID model not found. Searched for: '{reid_model_path}'. Disabling Re-ID.")
            print(f"Please specify the correct path with --reid_model_path or ensure the model exists in reid_training/")
            args.use_reid = False
        else:
            try:
                print(f"Loading Re-ID model from: {reid_model_path}")
                feature_extractor = FeatureExtractor(reid_model_path)
                
                # Load the pre-built gallery
                reid_gallery = load_gallery(args.reid_gallery_path)
                
                if reid_gallery:
                    print(f"Loaded Re-ID gallery with {len(reid_gallery)} known identities.")
                    color_generator = ColorGenerator()
                else:
                    print("Disabling Re-ID due to gallery loading failure.")
                    args.use_reid = False

            except Exception as e:
                print(f"Error initializing Re-ID: {e}. Disabling Re-ID.")
                args.use_reid = False

    # Check if input directory exists
    if not os.path.exists(args.input_dir):
        print(f'Input directory {args.input_dir} not found.')
        print('Please create the directory and add an image or a video to it.')
        return

    # Auto-detect content in input directory
    video_extensions = ('.mp4', '.avi', '.mov', '.mkv', '.m4v', '.webm')
    image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')

    entries = [f for f in os.listdir(args.input_dir) if not f.startswith('.')]
    video_files = [f for f in entries if f.lower().endswith(video_extensions)]
    image_files = [f for f in entries if f.lower().endswith(image_extensions)]
    
    # Process all found videos
    if video_files:
        video_files.sort()
        print(f"Found {len(video_files)} video(s) in '{args.input_dir}'. Processing them now.")
        for video_file in video_files:
            video_path = os.path.join(args.input_dir, video_file)
            process_video(model, video_path, args, feature_extractor, reid_gallery, color_generator)

        # Process all found images
        if image_files:
            image_files.sort()
            print(f"Found {len(image_files)} image(s) in '{args.input_dir}'.")
            if len(image_files) == 1:
                # Single image: run one-off inference and save visualization
                img_name = image_files[0]
                img_path = os.path.join(args.input_dir, img_name)
                print(f"Running inference on single image: {img_name}")
                predictions, original_image = predict_bbox_yolo(model, img_path, args.conf_threshold, args.nms_iou_threshold, args.imgsz)
                os.makedirs(args.output_dir, exist_ok=True)
                output_path = os.path.join(args.output_dir, f"output_{img_name}")
                visualize_prediction_yolo(img_path, predictions, original_image, output_path)
                if predictions:
                    print(f"Found {len(predictions)} objects. Result saved to {output_path}")
                else:
                    print(f"No objects detected. Result saved to {output_path}")
            else:
                # Multiple images: run batch inference
                print(f"Running batch inference on images in {args.input_dir}...")
                batch_inference(model, args.input_dir, args.output_dir, args.conf_threshold, args.nms_iou_threshold, args.imgsz)
                print(f"Image inference completed! Results saved to {args.output_dir}")

        if not video_files and not image_files:
            print(f"No images or videos found in {args.input_dir}. Please add files and retry.")
    
    print(f"Inference completed!")


if __name__ == '__main__':
    main()
