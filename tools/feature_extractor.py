#!/usr/bin/env python3
"""
Feature extraction utility for the Re-ID model.
Wraps the ReIDNet model to extract a 128-dimensional embedding from a PIL Image crop.
"""

import os
import sys
import numpy as np
import torch
from torchvision import transforms
from PIL import Image

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from lib.reid_model import ReIDNet, infer_reid_head_from_state_dict


class FeatureExtractor:
    def __init__(self, model_path, embedding_dim=128, image_size=224):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        state = torch.load(model_path, map_location=self.device)
        head = infer_reid_head_from_state_dict(state)
        self.model = ReIDNet(embedding_dim, head=head).to(self.device)
        self.model.load_state_dict(state)
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def extract_features(self, crop_image: Image.Image) -> np.ndarray:
        """
        Extract a feature embedding from a single cropped image.

        Args:
            crop_image: A PIL Image of the cropped subject.

        Returns:
            A 1-D numpy array representing the feature embedding.
        """
        if not isinstance(crop_image, Image.Image):
            raise TypeError("Input must be a PIL Image.")

        if crop_image.mode != 'RGB':
            crop_image = crop_image.convert('RGB')

        image_tensor = self.transform(crop_image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            embedding = self.model(image_tensor)

        return embedding.cpu().numpy().flatten()
