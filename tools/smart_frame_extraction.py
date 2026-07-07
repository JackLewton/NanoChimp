#!/usr/bin/env python3
"""
Extract diverse frames from videos.

This script processes videos and extracts frames
based on movement detection using SSIM and MSE. It samples densely during 
active moments and sparsely during rest periods.
"""

import argparse
import os
import cv2
import numpy as np
from pathlib import Path
import json
from tqdm import tqdm
import shutil
from skimage.metrics import structural_similarity as ssim
import logging
from datetime import datetime
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('smart_frame_extraction.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class SmartFrameExtractor:
    """
    Intelligent frame extraction based on movement detection and adaptive sampling
    """
    
    def __init__(self, args):
        self.input_dir = Path(args.input_dir)
        self.output_dir = Path(args.output_dir)
        self.movement_threshold = args.movement_threshold
        self.ssim_threshold = args.ssim_threshold
        self.mse_threshold = args.mse_threshold
        self.active_sampling_rate = args.active_sampling_rate
        self.rest_sampling_rate = args.rest_sampling_rate
        self.max_frames_per_video = args.max_frames_per_video
        self.resize_dims = tuple(args.resize_dims) if args.resize_dims else None
        self.jpeg_quality = args.jpeg_quality
        
        # Statistics
        self.stats = {
            'total_videos': 0,
            'processed_videos': 0,
            'total_frames_extracted': 0,
            'active_moments': 0,
            'rest_moments': 0,
            'failed_videos': []
        }
        
        # Ensure output directory exists
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def calculate_frame_difference(self, frame1, frame2):
        """
        Calculate frame difference using both SSIM and MSE
        
        Args:
            frame1, frame2: OpenCV frames (numpy arrays)
            
        Returns:
            dict: Contains SSIM score, MSE score, and movement detected flag
        """
        # Convert to grayscale for comparison
        gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        
        # Calculate SSIM
        ssim_score = ssim(gray1, gray2)
        
        # Calculate MSE
        mse_score = np.mean((gray1.astype(float) - gray2.astype(float)) ** 2)
        
        # Determine if significant movement detected
        movement_detected = (
            (1 - ssim_score) > self.ssim_threshold or
            mse_score > self.mse_threshold
        )
        
        return {
            'ssim': ssim_score,
            'mse': mse_score,
            'movement_detected': movement_detected,
            'movement_score': (1 - ssim_score) + (mse_score / 10000)  # Combined score
        }
    
    def classify_activity_level(self, movement_scores, window_size=10):
        """
        Classify current activity level based on recent movement scores
        
        Args:
            movement_scores: List of recent movement scores
            window_size: Number of recent frames to consider
            
        Returns:
            str: 'active' or 'rest'
        """
        if len(movement_scores) < window_size:
            return 'rest'
            
        recent_scores = movement_scores[-window_size:]
        avg_movement = np.mean(recent_scores)
        
        return 'active' if avg_movement > self.movement_threshold else 'rest'
    
    def should_extract_frame(self, activity_level, frames_since_last_extract, movement_score):
        """
        Determine if frame should be extracted based on activity level and sampling rules
        
        Args:
            activity_level: 'active' or 'rest'
            frames_since_last_extract: Number of frames since last extraction
            movement_score: Current frame's movement score
            
        Returns:
            bool: Whether to extract this frame
        """
        if activity_level == 'active':
            # Dense sampling during active moments
            return frames_since_last_extract >= self.active_sampling_rate
        else:
            # Sparse sampling during rest, but still capture some variation
            return (frames_since_last_extract >= self.rest_sampling_rate and 
                   movement_score > self.movement_threshold * 0.3)  # Lower threshold for rest
    
    def extract_frames_from_video(self, video_path, relative_path=None):
        """
        Extract frames from a single video using smart sampling
        
        Args:
            video_path: Path to the video file
            relative_path: Relative path from input directory (for preserving subfolder structure)
            
        Returns:
            dict: Extraction results and statistics
        """
        video_name = video_path.stem
        
        # Create output directory structure preserving subfolder hierarchy
        if relative_path:
            # Use relative path to preserve subfolder structure
            video_output_dir = self.output_dir / relative_path.parent / video_name
        else:
            # Fallback to original behavior
            video_output_dir = self.output_dir / video_name
            
        frames_dir = video_output_dir / 'frames'
        
        # Create output directories
        video_output_dir.mkdir(parents=True, exist_ok=True)
        frames_dir.mkdir(parents=True, exist_ok=True)
        
        # Copy original video to output directory for reference
        shutil.copy2(video_path, video_output_dir / video_path.name)
        
        # Open video
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            logger.error(f"Failed to open video: {video_path}")
            return None
            
        # Get video properties
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        logger.info(f"Processing {video_name}: {total_frames} frames, {fps:.2f} FPS, {width}x{height}")
        
        # Initialize tracking variables
        prev_frame = None
        movement_scores = []
        extracted_frames = []
        frames_since_last_extract = 0
        frame_idx = 0
        
        video_stats = {
            'video_name': video_name,
            'total_frames': total_frames,
            'fps': fps,
            'dimensions': f"{width}x{height}",
            'extracted_count': 0,
            'active_moments': 0,
            'rest_moments': 0,
            'avg_movement_score': 0
        }
        
        # Process each frame
        pbar = tqdm(total=total_frames, desc=f"Processing {video_name}")
        
        while cap.isOpened() and frame_idx < total_frames:
            ret, frame = cap.read()
            if not ret:
                break
                
            frames_since_last_extract += 1
            
            # Resize frame if specified
            if self.resize_dims:
                frame = cv2.resize(frame, self.resize_dims)
            
            if prev_frame is not None:
                # Calculate movement
                diff_result = self.calculate_frame_difference(prev_frame, frame)
                movement_scores.append(diff_result['movement_score'])
                
                # Classify activity level
                activity_level = self.classify_activity_level(movement_scores)
                
                # Update statistics
                if activity_level == 'active':
                    video_stats['active_moments'] += 1
                else:
                    video_stats['rest_moments'] += 1
                
                # Decide whether to extract frame
                if self.should_extract_frame(activity_level, frames_since_last_extract, 
                                           diff_result['movement_score']):
                    
                    # Extract frame
                    timestamp = frame_idx / fps
                    frame_filename = f"{video_name}_thresh{self.movement_threshold}_frame_{frame_idx:06d}_t{timestamp:.2f}s_{activity_level}.jpg"
                    frame_path = frames_dir / frame_filename
                    
                    # Save frame with specified quality
                    cv2.imwrite(str(frame_path), frame, 
                              [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality])
                    
                    extracted_frames.append({
                        'frame_idx': frame_idx,
                        'timestamp': timestamp,
                        'activity_level': activity_level,
                        'movement_score': diff_result['movement_score'],
                        'ssim': diff_result['ssim'],
                        'mse': diff_result['mse'],
                        'filename': frame_filename
                    })
                    
                    frames_since_last_extract = 0
                    video_stats['extracted_count'] += 1
                    
                    # Check if we've reached max frames limit
                    if video_stats['extracted_count'] >= self.max_frames_per_video:
                        logger.info(f"Reached maximum frames limit ({self.max_frames_per_video}) for {video_name}")
                        break
            
            prev_frame = frame.copy()
            frame_idx += 1
            pbar.update(1)
            
        pbar.close()
        cap.release()
        
        # Calculate final statistics
        if movement_scores:
            video_stats['avg_movement_score'] = np.mean(movement_scores)
        
        # Save extraction metadata
        metadata = {
            'video_stats': video_stats,
            'extraction_params': {
                'movement_threshold': self.movement_threshold,
                'ssim_threshold': self.ssim_threshold,
                'mse_threshold': self.mse_threshold,
                'active_sampling_rate': self.active_sampling_rate,
                'rest_sampling_rate': self.rest_sampling_rate,
            },
            'extracted_frames': extracted_frames,
            'extraction_timestamp': datetime.now().isoformat()
        }
        
        metadata_path = video_output_dir / f"{video_name}_extraction_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Completed {video_name}: {video_stats['extracted_count']} frames extracted")
        return video_stats
    
    def find_videos_recursively(self, directory):
        """
        Recursively find all video files in directory and subdirectories
        
        Args:
            directory: Path object to search in
            
        Returns:
            list: List of tuples (video_path, relative_path) where relative_path 
                  preserves the subfolder structure
        """
        video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv'}
        video_files = []
        
        for root, dirs, files in os.walk(directory):
            root_path = Path(root)
            for file in files:
                file_path = root_path / file
                if file_path.suffix.lower() in video_extensions:
                    # Calculate relative path from input directory
                    relative_path = file_path.relative_to(self.input_dir)
                    video_files.append((file_path, relative_path))
        
        return video_files

    def process_all_videos(self):
        """
        Process all videos in the input directory and subdirectories
        """
        video_files_with_paths = self.find_videos_recursively(self.input_dir)
        
        if not video_files_with_paths:
            logger.error(f"No video files found in {self.input_dir} and subdirectories")
            return
            
        self.stats['total_videos'] = len(video_files_with_paths)
        logger.info(f"Found {len(video_files_with_paths)} videos to process across all subdirectories")
        
        # Group videos by subfolder for better logging
        subfolder_groups = {}
        for video_path, relative_path in video_files_with_paths:
            subfolder = relative_path.parent
            if subfolder not in subfolder_groups:
                subfolder_groups[subfolder] = []
            subfolder_groups[subfolder].append((video_path, relative_path))
        
        # Process videos by subfolder
        for subfolder, videos in subfolder_groups.items():
            subfolder_name = str(subfolder) if subfolder != Path('.') else 'root'
            logger.info(f"Processing {len(videos)} videos in subfolder: {subfolder_name}")
            
            for video_path, relative_path in videos:
                try:
                    logger.info(f"Starting extraction for: {relative_path}")
                    video_stats = self.extract_frames_from_video(video_path, relative_path)
                    
                    if video_stats:
                        self.stats['processed_videos'] += 1
                        self.stats['total_frames_extracted'] += video_stats['extracted_count']
                        self.stats['active_moments'] += video_stats['active_moments']
                        self.stats['rest_moments'] += video_stats['rest_moments']
                    else:
                        self.stats['failed_videos'].append(str(relative_path))
                        
                except Exception as e:
                    logger.error(f"Error processing {relative_path}: {str(e)}")
                    self.stats['failed_videos'].append(str(relative_path))
        
        # Save overall statistics
        self.save_overall_statistics()
        
    def save_overall_statistics(self):
        """
        Save overall extraction statistics
        """
        overall_stats = {
            'extraction_summary': self.stats,
            'parameters_used': {
                'movement_threshold': self.movement_threshold,
                'ssim_threshold': self.ssim_threshold,
                'mse_threshold': self.mse_threshold,
                'active_sampling_rate': self.active_sampling_rate,
                'rest_sampling_rate': self.rest_sampling_rate,
                'max_frames_per_video': self.max_frames_per_video,
                'resize_dims': self.resize_dims,
                'jpeg_quality': self.jpeg_quality
            },
            'completion_timestamp': datetime.now().isoformat()
        }
        
        stats_path = self.output_dir / 'extraction_summary.json'
        with open(stats_path, 'w') as f:
            json.dump(overall_stats, f, indent=2)
            
        # Print summary
        logger.info("=" * 60)
        logger.info("EXTRACTION COMPLETE - SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Total videos found: {self.stats['total_videos']}")
        logger.info(f"Successfully processed: {self.stats['processed_videos']}")
        logger.info(f"Failed videos: {len(self.stats['failed_videos'])}")
        logger.info(f"Total frames extracted: {self.stats['total_frames_extracted']}")
        logger.info(f"Active moments detected: {self.stats['active_moments']}")
        logger.info(f"Rest moments detected: {self.stats['rest_moments']}")
        
        if self.stats['failed_videos']:
            logger.warning(f"Failed videos: {', '.join(self.stats['failed_videos'])}")
            
        avg_frames_per_video = (self.stats['total_frames_extracted'] / 
                               max(self.stats['processed_videos'], 1))
        logger.info(f"Average frames per video: {avg_frames_per_video:.1f}")
        logger.info(f"Output directory: {self.output_dir}")
        logger.info("=" * 60)

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Smart Frame Extraction for Chimp Dataset Curation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Input/Output paths
    parser.add_argument(
        '--input_dir', 
        type=str, 
        default='temp/tools_input_smart_frame_extraction',
        help='Directory containing input videos'
    )
    parser.add_argument(
        '--output_dir', 
        type=str, 
        default='temp/tools_output_smart_frame_extraction',
        help='Directory for extracted frames output'
    )
    
    # Movement detection parameters
    parser.add_argument(
        '--movement_threshold', 
        type=float, 
        default=0.0235,
        help='Movement threshold for activity classification (a higher threshold makes it less sensitive to motion.)'
    )
    parser.add_argument(
        '--ssim_threshold', 
        type=float, 
        default=0.01,
        help='SSIM difference threshold for movement detection (0-1)'
    )
    parser.add_argument(
        '--mse_threshold', 
        type=float, 
        default=150.0,
        help='MSE threshold for movement detection'
    )
    
    # Sampling rates (in frames)
    parser.add_argument(
        '--active_sampling_rate', 
        type=int, 
        default=4,
        help='Frame interval for sampling during active moments'
    )
    parser.add_argument(
        '--rest_sampling_rate', 
        type=int, 
        default=30,
        help='Frame interval for sampling during rest periods'
    )
    
    # Output constraints
    parser.add_argument(
        '--max_frames_per_video', 
        type=int, 
        default=200,
        help='Maximum number of frames to extract per video'
    )
    parser.add_argument(
        '--resize_dims', 
        type=int, 
        nargs=2, 
        default=None,
        help='Resize frames to specific dimensions [width height]'
    )
    parser.add_argument(
        '--jpeg_quality', 
        type=int, 
        default=95,
        help='JPEG quality for saved frames (1-100)'
    )
    
    return parser.parse_args()

def main():
    """Main execution function"""
    args = parse_args()
    
    logger.info("Starting Smart Frame Extraction")
    logger.info(f"Input directory: {args.input_dir}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"Movement threshold: {args.movement_threshold}")
    logger.info(f"Active sampling rate: {args.active_sampling_rate} frames")
    logger.info(f"Rest sampling rate: {args.rest_sampling_rate} frames")
    
    # Initialize and run extractor
    extractor = SmartFrameExtractor(args)
    extractor.process_all_videos()

if __name__ == "__main__":
    main() 