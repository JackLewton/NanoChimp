#!/usr/bin/env python3
"""
Optional, if you want to train a separate nighttime detection model. 
I used this script to create IR-style images from the existing daytime images.

This module provides a PIL Image transform that applies nighttime/IR-style augmentation
to images during training. Designed to be used with 50% probability in the training pipeline.
"""

import numpy as np
import cv2
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2


class NighttimeAugmentation:
    """
    Apply nighttime/IR-style augmentation to PIL Images.
    
    This transform simulates nighttime IR camera conditions:
    - Grayscale conversion
    - High contrast and brightness adjustments
    - Gamma correction (darker mid-tones)
    - Gaussian noise (sensor grain)
    - Optional blur (image quality reduction)
    - Vignette effect (IR flashlight)
    - Shadows from top light source
    """
    
    def __init__(
        self,
        vignette_range=(0.2, 0.8),
        brightness_limit=(-0.1, 0.1),
        contrast_limit=(0.3, 0.5),
        gamma_limit=(50, 70),
        noise_std_limit=(5, 15),
        blur_limit=(3, 5),
        apply_blur=True,
        shadow_strength=(0.3, 0.6),
        shadow_falloff=0.5
    ):
        """
        Initialize nighttime augmentation transform.
        
        Args:
            vignette_range: Tuple for vignette strength (0.2-0.8 makes edges almost fully black)
            brightness_limit: Tuple for brightness adjustment range
            contrast_limit: Tuple for contrast increase (higher = more contrast)
            gamma_limit: Tuple for gamma correction as percentages (100 = no change, <100 darkens)
            noise_std_limit: Tuple for Gaussian noise standard deviation
            blur_limit: Tuple for blur radius
            apply_blur: Whether to apply blur (optional step)
            shadow_strength: Tuple for shadow strength from top (0.3-0.6 = subtle to strong)
            shadow_falloff: Float for shadow falloff rate (0.0-1.0, higher = faster falloff)
        """
        self.vignette_range = vignette_range
        self.brightness_limit = brightness_limit
        self.contrast_limit = contrast_limit
        self.gamma_limit = gamma_limit
        self.noise_std_limit = noise_std_limit
        self.blur_limit = blur_limit
        self.apply_blur = apply_blur
        self.shadow_strength = shadow_strength
        self.shadow_falloff = shadow_falloff
        
        # Create Albumentations pipeline (will be created per-image for randomness)
        self._create_pipeline()
    
    def _create_pipeline(self):
        """Create the Albumentations pipeline for transformations."""
        transforms = [
            # A. Color Transformation - Remove all color information (IR records intensity only)
            A.ToGray(p=1.0),
            
            # C. Intensity & Contrast Manipulation
            # High contrast to mimic harsh IR reflection
            A.RandomBrightnessContrast(
                p=1.0,
                brightness_limit=self.brightness_limit,
                contrast_limit=self.contrast_limit
            ),
            
            # Lower gamma to simulate rapid light falloff in dark background
            A.RandomGamma(
                p=1.0,
                gamma_limit=self.gamma_limit
            ),
            
            # D. Sensor Noise Simulation - High gain introduces grain/speckle
            A.GaussNoise(
                p=1.0,
                var_limit=(self.noise_std_limit[0]**2, self.noise_std_limit[1]**2)
            ),
        ]
        
        # E. Image Quality Reduction - Mild blur to soften HD look (optional)
        if self.apply_blur:
            transforms.append(
                A.GaussianBlur(
                    p=1.0,
                    blur_limit=self.blur_limit
                )
            )
        
        self.transform = A.Compose(transforms)
    
    def __call__(self, image):
        """
        Apply nighttime augmentation to a PIL Image.
        
        Args:
            image: PIL Image (RGB)
        
        Returns:
            PIL Image with nighttime augmentation applied
        """
        # Convert PIL to numpy array (RGB)
        img_array = np.array(image)
        
        # Apply Albumentations transforms
        transformed = self.transform(image=img_array)
        result = transformed['image']
        
        # Ensure result is 3-channel RGB for consistent processing
        if len(result.shape) == 2:  # Grayscale (single channel)
            result = cv2.cvtColor(result, cv2.COLOR_GRAY2RGB)
        elif len(result.shape) == 3 and result.shape[2] == 1:  # Grayscale with channel dimension
            result = result.squeeze(2)
            result = cv2.cvtColor(result, cv2.COLOR_GRAY2RGB)
        
        h, w, c = result.shape
        
        # B. Lighting Geometry - Apply shadows from top light source (IR LEDs above camera)
        # Create vertical gradient from top (light source) to bottom (shadows)
        y_coords = np.arange(h, dtype=np.float32)
        # Normalize to 0-1 (0 = top, 1 = bottom)
        y_normalized = y_coords / h
        
        # Create shadow mask: darker at bottom, brighter at top
        # Use exponential falloff for more realistic shadow
        shadow_strength_actual = np.random.uniform(self.shadow_strength[0], self.shadow_strength[1])
        shadow_mask = 1.0 - (shadow_strength_actual * (y_normalized ** self.shadow_falloff))
        shadow_mask = np.clip(shadow_mask, 0, 1)
        
        # Expand to match image dimensions
        shadow_mask = shadow_mask[:, np.newaxis]  # Shape: (h, 1)
        shadow_mask = np.repeat(shadow_mask, w, axis=1)  # Shape: (h, w)
        shadow_mask = np.stack([shadow_mask] * c, axis=2)  # Shape: (h, w, c)
        
        # Apply shadow
        result = (result.astype(np.float32) * shadow_mask).astype(np.uint8)
        
        # C. Apply aggressive vignette (IR flashlight effect from center)
        # Create radial gradient from center
        center_x, center_y = w // 2, h // 2
        y, x = np.ogrid[:h, :w]
        # Calculate distance from center (normalized to 0-1)
        mask = np.sqrt((x - center_x)**2 + (y - center_y)**2)
        mask = mask / mask.max()
        
        # Apply vignette strength (random within range)
        vignette_strength = np.random.uniform(self.vignette_range[0], self.vignette_range[1])
        # Invert so center is bright, edges are dark
        vignette = 1 - (mask * vignette_strength)
        vignette = np.clip(vignette, 0, 1)
        
        # Expand vignette to match channels
        vignette = np.stack([vignette] * c, axis=2)
        
        result = (result.astype(np.float32) * vignette).astype(np.uint8)
        
        # Ensure values are in valid range
        result = np.clip(result, 0, 255).astype(np.uint8)
        
        # Convert back to PIL Image
        return Image.fromarray(result)
    
    def __repr__(self):
        return f"NighttimeAugmentation(vignette={self.vignette_range}, shadow={self.shadow_strength})"


class RandomNighttimeAugmentation:
    """
    Randomly apply nighttime augmentation with specified probability.
    
    This is a wrapper that applies NighttimeAugmentation with probability p.
    Use with p=0.5 to get 50% nighttime, 50% daytime images.
    """
    
    def __init__(self, p=0.5, **nighttime_kwargs):
        """
        Initialize random nighttime augmentation.
        
        Args:
            p: Probability of applying nighttime augmentation (default: 0.5 for 50%)
            **nighttime_kwargs: Arguments to pass to NighttimeAugmentation
        """
        self.p = p
        self.nighttime_aug = NighttimeAugmentation(**nighttime_kwargs)
    
    def __call__(self, image):
        """
        Apply nighttime augmentation with probability p.
        
        Args:
            image: PIL Image
        
        Returns:
            PIL Image (augmented or original)
        """
        if np.random.random() < self.p:
            return self.nighttime_aug(image)
        return image
    
    def __repr__(self):
        return f"RandomNighttimeAugmentation(p={self.p})"

