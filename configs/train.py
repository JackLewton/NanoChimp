import json
import os
import sys
import shutil
import argparse
import torch
import numpy as np
import yaml
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from ultralytics import YOLO

# Add parent directory to path to allow imports from 'tools'
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import nighttime augmentation and split dataset tools
from tools.nighttime_augmentation import RandomNighttimeAugmentation
from tools.split_dataset_with_test import create_train_val_test_split


def convert_coco_to_yolo_format(annotation_file, image_dir, output_dir, single_class: bool = False, use_nighttime_aug=False, nighttime_aug_prob=0.5):
    """
    Convert COCO format annotations to YOLO format
    
    Args:
        annotation_file: Path to COCO annotation file
        image_dir: Directory containing images
        output_dir: Output directory for YOLO format
        single_class: Whether to use single class (merge occluded/non-occluded)
        use_nighttime_aug: Whether to apply nighttime augmentation (default: False)
        nighttime_aug_prob: Probability of applying nighttime augmentation (default: 0.5 for 50%)

    A custom function is used as originally I had two classes (occluded/non-occluded)
    to help with evalutation, plus I needed to augment the images for the nighttime option.
    """
    with open(annotation_file, 'r') as f:
        data = json.load(f)
    
    # Create output directories
    images_dir = os.path.join(output_dir, 'images')
    labels_dir = os.path.join(output_dir, 'labels')
    
    # Clear existing directories to ensure clean dataset
    if os.path.exists(images_dir):
        shutil.rmtree(images_dir)
    if os.path.exists(labels_dir):
        shutil.rmtree(labels_dir)
    
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)
    
    # Create image_id to annotations mapping
    img_to_anns = {}
    for ann in data['annotations']:
        img_id = ann['image_id']
        if img_id not in img_to_anns:
            img_to_anns[img_id] = []
        img_to_anns[img_id].append(ann)
    
    # Initialize nighttime augmentation if enabled
    nighttime_aug = None
    if use_nighttime_aug:
        nighttime_aug = RandomNighttimeAugmentation(
            p=nighttime_aug_prob,
            vignette_range=(0.2, 0.8),
            brightness_limit=(-0.1, 0.1),
            contrast_limit=(0.3, 0.5),
            gamma_limit=(50, 70),
            noise_std_limit=(5, 15),
            blur_limit=(3, 5),
            apply_blur=True,
            shadow_strength=(0.3, 0.6),
            shadow_falloff=0.5
        )
        print(f"Nighttime augmentation enabled: {nighttime_aug_prob*100:.0f}% of images will be augmented")
    
    # Process each image
    images_processed = 0
    images_with_anns = 0
    nighttime_count = 0
    
    print(f"Processing {len(data['images'])} images...")
    for img_info in tqdm(data['images'], desc="Converting to YOLO format"):
        img_id = img_info['id']
        
        # Load image if it exists
        src_img_path = os.path.join(image_dir, img_info['file_name'])
        if not os.path.isfile(src_img_path):
            continue

        # Load image
        image = Image.open(src_img_path).convert('RGB')
        
        # Apply nighttime augmentation randomly (50% probability)
        # RandomNighttimeAugmentation handles the probability internally
        if nighttime_aug is not None:
            # Store original to check if augmentation was applied
            original_array = np.array(image)
            image = nighttime_aug(image)
            augmented_array = np.array(image)
            # Check if images are different (augmentation was applied)
            if not np.array_equal(original_array, augmented_array):
                nighttime_count += 1
        
        # Save image (either original or augmented)
        dst_img_path = os.path.join(images_dir, img_info['file_name'])
        image.save(dst_img_path, quality=95)
        
        # Create YOLO annotation file
        label_filename = os.path.splitext(img_info['file_name'])[0] + '.txt'
        label_path = os.path.join(labels_dir, label_filename)
        
        annotations = img_to_anns.get(img_id)
        
        if annotations:
            images_with_anns += 1
            orig_width, orig_height = img_info['width'], img_info['height']
            
            with open(label_path, 'w') as f:
                for ann in annotations:
                    bbox = ann['bbox']  # [x, y, width, height]
                    x, y, w, h = bbox
                    
                    # Convert to center coordinates and normalize
                    x_center = (x + w/2) / orig_width
                    y_center = (y + h/2) / orig_height
                    w_norm = w / orig_width
                    h_norm = h / orig_height
                    
                    # Check if the chimp is occluded
                    is_occluded = ann.get('attributes', {}).get('occluded', False)

                    # Class ID mapping
                    # - single_class=True: map all chimp instances to class 0
                    # - single_class=False: 0=chimp, 1=occluded_chimp
                    if single_class:
                        class_id = 0
                    else:
                        class_id = 1 if is_occluded else 0
                    
                    # Write YOLO format: class_id x_center y_center width height
                    f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}\n")
        else:
            # This is a background image (no annotations), create an empty file
            open(label_path, 'w').close()
            
        images_processed += 1

    aug_info = f" ({nighttime_count} nighttime augmented)" if use_nighttime_aug else ""
    print(f"Converted {images_processed} total images ({images_with_anns} with annotations) to YOLO format in {output_dir} (single_class={single_class}){aug_info}")


def create_yolo_config(output_dir, num_classes=2, single_class: bool = False):
    """
    Create YOLO configuration file
    """
    # If split files exist, point Ultralytics to them; else default to folder
    output_dir_abs = os.path.abspath(output_dir)
    train_list = os.path.join(output_dir_abs, 'train.txt')
    val_list = os.path.join(output_dir_abs, 'val.txt')
    test_list = os.path.join(output_dir_abs, 'test.txt')

    # When using text list files, reference them relative to `path`
    train_ref = os.path.basename(train_list) if os.path.isfile(train_list) else 'images'
    val_ref = os.path.basename(val_list) if os.path.isfile(val_list) else 'images'
    test_ref = os.path.basename(test_list) if os.path.isfile(test_list) else None

    names = {0: 'chimp'} if single_class or num_classes == 1 else {0: 'chimp', 1: 'occluded_chimp'}
    nc_value = 1 if single_class else num_classes

    config = {
        # Use absolute path so entries like 'images/...' resolve correctly
        'path': output_dir_abs,
        'train': train_ref,
        'val': val_ref,
        'names': names,
        'nc': nc_value
    }
    if test_ref:
        config['test'] = test_ref
    
    config_path = os.path.join(output_dir_abs, 'data.yaml')
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    print(f"Created YOLO config at {config_path}")
    return config_path


def train_yolo_model(
    data_yaml_path,
    model_name='yolo11n.pt',
    epochs=100,
    imgsz=640,
    batch_size=16,
    resume_from=None,
    aug_level='default',
    patience: int = 10,
):
    """
    Train YOLO model using ultralytics
    
    Args:
        data_yaml_path: Path to YOLO data config file
        model_name: YOLO model file name (e.g., 'yolo11n.pt')
        epochs: Number of training epochs
        imgsz: Input image size
        batch_size: Batch size for training
        resume_from: Path to checkpoint to resume from
        aug_level: Augmentation level - 'default', 'moderate', or 'aggressive'
        patience: Early stopping patience
    """
    model_source = resume_from if resume_from else model_name

    try:
        print(f"Loading model from: {model_source}")
        model = YOLO(model_source)
        print(f"Successfully loaded {model_source}")
    except Exception as e:
        print(f"Error: Failed to load model from '{model_source}'.")
        print(f"Details: {e}")
        print("\nPlease ensure that 'ultralytics' is installed and up-to-date (`pip install -U ultralytics`)")
        print("and that the model name is correct.")
        raise Exception("Could not load the YOLO model.")

    # Determine device
    if torch.cuda.is_available() and torch.cuda.device_count() > 0:
        device = 0  # Use GPU device 0 explicitly
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = 'cpu'
        print("Using CPU for training")
    
    # Augmentation parameters based on level
    aug_params = {}
    
    if aug_level == 'moderate':
        print("\nUsing MODERATE augmentation to handle domain shift between cameras...")
        # Moderate augmentations - balanced approach to help with domain shift
        aug_params = {
            # Color augmentations (moderate increases)
            'hsv_h': 0.018,     # Hue: slight increase from default 0.015
            'hsv_s': 0.75,      # Saturation: slight increase from default 0.7
            'hsv_v': 0.45,      # Value (brightness): slight increase from default 0.4
            
            # Geometric augmentations (moderate values)
            'degrees': 5.0,     # Rotation: moderate rotation for camera angle variation
            'translate': 0.12,  # Translation: slight increase from default 0.1
            'scale': 0.55,      # Scaling: slight increase from default 0.5
            'shear': 2.0,       # Shear: moderate shear for perspective variation
            'perspective': 0.0002,  # Perspective: small perspective transform
            
            # Flip augmentations
            'fliplr': 0.5,      # Horizontal flip: default 0.5 (keep)
            'flipud': 0.0,      # Vertical flip: default 0.0 (keep, chimps shouldn't be upside down)
            
            # Advanced augmentations
            'mosaic': 1.0,      # Mosaic: default 1.0 (keep, very useful)
            'mixup': 0.05,      # Mixup: small amount for regularization
        }
    elif aug_level == 'aggressive':
        print("\nUsing AGGRESSIVE augmentation to handle domain shift between cameras...")
        # More aggressive augmentations - reduced from previous version based on results
        aug_params = {
            # Color augmentations (moderate increases, reduced from previous)
            'hsv_h': 0.02,      # Hue: increased for color variation
            'hsv_s': 0.8,       # Saturation: increased for color variation
            'hsv_v': 0.5,       # Value (brightness): increased for lighting variation
            
            # Geometric augmentations (reduced from previous aggressive settings)
            'degrees': 7.0,     # Rotation: reduced from 10.0 to 7.0
            'translate': 0.13,  # Translation: reduced from 0.15 to 0.13
            'scale': 0.58,      # Scaling: reduced from 0.6 to 0.58
            'shear': 3.0,       # Shear: reduced from 5.0 to 3.0
            'perspective': 0.0003,  # Perspective: reduced from 0.0005
            
            # Flip augmentations
            'fliplr': 0.5,      # Horizontal flip: default 0.5 (keep)
            'flipud': 0.0,      # Vertical flip: default 0.0 (keep, chimps shouldn't be upside down)
            
            # Advanced augmentations
            'mosaic': 1.0,      # Mosaic: default 1.0 (keep, very useful)
            'mixup': 0.08,      # Mixup: reduced from 0.1 to 0.08
        }
    else:
        print("\nUsing default augmentation settings...")
    
    # Train the model
    train_kwargs = {
        'data': data_yaml_path,
        'epochs': epochs,
        'imgsz': imgsz,
        'batch': batch_size,
        'device': device,
        'project': 'yolo_training',
        'name': 'bounding_box_model',
        'save': True,
        'plots': True,
        'patience': patience,  # Early stopping patience (epochs with no improvement)
        'save_period': 5  # Save checkpoint every 5 epochs
    }
    
    # Add augmentation parameters if specified
    train_kwargs.update(aug_params)
    
    results = model.train(**train_kwargs)
    
    return model, results


def visualize_predictions_yolo(model, image_dir, num_samples=5):
    """Visualize YOLO model predictions on sample images"""
    if not os.path.isdir(image_dir):
        print(f"Warning: Image directory '{image_dir}' not found. Skipping prediction visualization.")
        return
        
    model.eval()
    
    # Get sample images
    image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    if not image_files:
        print(f"Warning: No sample images found in '{image_dir}'. Skipping prediction visualization.")
        return
        
    sample_images = image_files[:num_samples]
    
    fig, axes = plt.subplots(1, len(sample_images), figsize=(15, 3))
    if len(sample_images) == 1:
        axes = [axes]
    
    for i, img_file in enumerate(sample_images):
        img_path = os.path.join(image_dir, img_file)
        
        # Run prediction
        results = model(img_path)
        
        # Get the first result
        result = results[0]
        
        # Load original image for display
        img = Image.open(img_path)
        axes[i].imshow(img)
        
        # Draw bounding boxes
        if result.boxes is not None:
            boxes = result.boxes.xyxy.cpu().numpy()  # x1, y1, x2, y2 format
            confidences = result.boxes.conf.cpu().numpy()
            
            for box, conf in zip(boxes, confidences):
                x1, y1, x2, y2 = box
                rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, 
                                       linewidth=2, edgecolor='red', facecolor='none')
                axes[i].add_patch(rect)
                axes[i].text(x1, y1-5, f'{conf:.2f}', color='red', fontsize=8)
        
        axes[i].set_title(f'Sample {i+1}')
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig('yolo_predictions.png', dpi=150, bbox_inches='tight')
    print("Saved prediction visualization to 'yolo_predictions.png'")


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train YOLO model on chimp detection dataset')
    parser.add_argument('--model', type=str, default='yolo11n.pt', help='YOLO model to train (default: yolo11n.pt)')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs (default: 50)')
    parser.add_argument('--batch_size', type=int, default=4, help='Training batch size (default: 4)')
    parser.add_argument('--imgsz', type=int, default=640, help='Input image size (default: 640)')
    parser.add_argument('--patience', type=int, default=50,
                        help='Early stopping patience (epochs with no improvement). Use 0 to disable.')
    parser.add_argument('--multi_class', action='store_false', dest='single_class', default=True,
                        help='Train with multiple classes (chimp and occluded_chimp separately). Default: single class mode (merges occluded and non-occluded)')
    parser.add_argument('--annotation_file', type=str, default='data/annotations/annotations.json', help='Path to the COCO format annotation file.')
    parser.add_argument('--image_dir', type=str, default='data/images/', help='Path to the directory containing images.')
    parser.add_argument('--resume_from', type=str, default=None, help='Path to a model checkpoint to resume training from.')
    parser.add_argument('--aug_level', type=str, default='default', choices=['default', 'moderate', 'aggressive'],
                        help='Augmentation level: default (Ultralytics defaults), moderate (balanced), or aggressive (strong)')
    parser.add_argument('--use_nighttime_aug', action='store_true',
                        help='Enable nighttime augmentation: 50% of images will be augmented to simulate IR/nighttime conditions')
    parser.add_argument('--nighttime_aug_prob', type=float, default=0.5,
                        help='Probability of applying nighttime augmentation (default: 0.5 for 50%)')
    
    args = parser.parse_args()
    
    # Ensure single_class defaults to True
    if not hasattr(args, 'single_class'):
        args.single_class = True
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    class_mode = "single class" if args.single_class else "multi-class"
    print(f'Training mode: {class_mode} ({"occluded and non-occluded chimps merged" if args.single_class else "occluded and non-occluded chimps separate"})')
    
    # Convert COCO format to YOLO format
    print(f"Converting COCO format to YOLO format...")
    yolo_data_dir = 'yolo_dataset'
    convert_coco_to_yolo_format(
        annotation_file=args.annotation_file,
        image_dir=args.image_dir,
        output_dir=yolo_data_dir,
        single_class=args.single_class,
        use_nighttime_aug=args.use_nighttime_aug,
        nighttime_aug_prob=args.nighttime_aug_prob
    )
    
    try:
        print('Creating video-disjoint train/val/test split lists...')
        create_train_val_test_split(
            os.path.join(yolo_data_dir, 'images'),
            output_dir=yolo_data_dir,
        )
    except Exception as e:
        print(f"Skipping split creation due to error: {e}")
        for fname in ("train.txt", "val.txt", "test.txt"):
            stale_file = os.path.join(yolo_data_dir, fname)
            if os.path.exists(stale_file):
                print(f"Removing stale split file: {stale_file}")
                os.remove(stale_file)

    # Create YOLO configuration (will reference split files if present)
    config_path = create_yolo_config(yolo_data_dir, num_classes=2, single_class=args.single_class)
    
    # Train YOLO model
    print(f"Training YOLO model using {args.model}...")
    model, results = train_yolo_model(
        data_yaml_path=config_path,
        model_name=args.model,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch_size=args.batch_size,
        resume_from=args.resume_from,
        aug_level=args.aug_level,
        patience=args.patience,
    )
    
    print("Training completed!")
    print(f"Model saved in: {model.ckpt_path}")
    
    # Visualize predictions
    print("Generating prediction visualizations...")
    visualize_predictions_yolo(model, os.path.join(yolo_data_dir, 'images'), num_samples=5)
    
    # Save the best model with a more descriptive name
    model_name_suffix = Path(args.annotation_file).stem.replace("_annotations", "").replace("_cleaned", "")
    best_model_path = f"{Path(args.model).stem}_bounding_box_model_{model_name_suffix}.pt"
    shutil.copy2(model.ckpt_path, best_model_path)
    print(f"Best model saved as: {best_model_path}")


if __name__ == '__main__':
    main()
