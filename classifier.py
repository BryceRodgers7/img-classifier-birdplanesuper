"""
Standalone Bird/Plane/Superman/Other Image Classifier

This module provides a self-contained classifier that can be copied to other projects.
Just copy this file and the trained model file (.pth) to use the classifier.

Usage:
    from classifier import BirdPlaneSupermanClassifier
    
    # Load model (temperature calibration applied automatically if temperature.json exists)
    classifier = BirdPlaneSupermanClassifier('models/best_model.pth', confidence_threshold=0.7)
    
    # Single prediction
    pred_class, confidence, probs = classifier.predict('test_image.jpg')
    print(f"Predicted: {pred_class} ({confidence:.2%} confident)")
    
    # Batch prediction
    results = classifier.predict_batch(['img1.jpg', 'img2.jpg', 'img3.jpg'])
    for img_path, pred_class, confidence, probs in results:
        print(f"{img_path}: {pred_class} ({confidence:.2%})")

Temperature calibration:
    Run `python calibrate_temperature.py` after training to fit a temperature scalar T.
    The result is saved to models/temperature.json.  The classifier loads it automatically
    and computes softmax(logits / T) instead of softmax(logits), producing better-
    calibrated confidence scores.  If the file is absent T defaults to 1.0 (no change).
"""

import json
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Union


class BirdPlaneSupermanClassifier:
    """
    Image classifier for Bird/Plane/Superman/Other categories

    Features:
    - Pre-trained ResNet50 backbone
    - Confidence thresholding (low confidence → 'other')
    - Temperature scaling calibration (loaded automatically from temperature.json)
    - Self-contained architecture definition
    - Batch inference support

    Args:
        model_path: Path to the trained model (.pth file)
        confidence_threshold: Minimum confidence to predict main classes (default: 0.7)
                             If max probability < threshold, predicts 'other'
        device: Device to run on ('cuda' or 'cpu', default: auto-detect)
        temperature_path: Path to temperature.json produced by calibrate_temperature.py.
                          Pass None to auto-discover (looks next to the model file and in
                          models/temperature.json).  Pass False to disable calibration.
    """

    def __init__(
        self,
        model_path: str,
        confidence_threshold: float = 0.7,
        device: Optional[str] = None,
        temperature_path: Optional[Union[str, bool]] = None,
    ):
        self.confidence_threshold = confidence_threshold
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")

        # Load model
        self.model, self.classes, self.config = self._load_model(model_path)
        self.model = self.model.to(self.device)
        self.model.eval()

        # Load temperature calibration
        self.temperature = self._load_temperature(model_path, temperature_path)

        # Setup image preprocessing
        self.transform = self._create_transform()

        print(f"Classifier loaded successfully!")
        print(f"  Device              : {self.device}")
        print(f"  Classes             : {self.classes}")
        print(f"  Confidence threshold: {self.confidence_threshold}")
        print(f"  Temperature (T)     : {self.temperature:.4f}"
              + (" (calibrated)" if self.temperature != 1.0 else " (no calibration file found, using default)"))
    
    def _load_model(self, model_path: str):
        """Load trained model from checkpoint."""
        checkpoint = torch.load(model_path, map_location=self.device)

        classes = checkpoint.get("classes", ["bird", "plane", "superman", "other"])
        config = checkpoint.get("config", {})

        model = models.resnet50(pretrained=False)
        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features, len(classes))
        model.load_state_dict(checkpoint["model_state_dict"])

        return model, classes, config

    def _load_temperature(
        self,
        model_path: str,
        temperature_path: Optional[Union[str, bool]],
    ) -> float:
        """
        Load the temperature scalar from a JSON file.

        Discovery order (when temperature_path is None):
          1. <model_dir>/temperature.json  (next to the .pth file)
          2. models/temperature.json       (default output location)

        Returns 1.0 (no calibration) if no file is found or calibration is
        explicitly disabled by passing temperature_path=False.
        """
        if temperature_path is False:
            return 1.0

        candidates: list[Path] = []
        if temperature_path is not None:
            candidates = [Path(temperature_path)]
        else:
            model_dir = Path(model_path).parent
            candidates = [
                model_dir / "temperature.json",
                Path("models") / "temperature.json",
            ]

        for candidate in candidates:
            if candidate.exists():
                try:
                    with open(candidate) as f:
                        data = json.load(f)
                    t = float(data["temperature"])
                    print(f"  Loaded temperature T={t:.4f} from {candidate}")
                    return t
                except Exception as e:
                    print(f"  Warning: failed to load temperature from {candidate}: {e}")

        return 1.0

    def _create_transform(self):
        """Create image preprocessing transform"""
        return transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    
    def _preprocess_image(self, image_path: str) -> torch.Tensor:
        """Load and preprocess an image"""
        image = Image.open(image_path).convert('RGB')
        image_tensor = self.transform(image)
        return image_tensor.unsqueeze(0)  # Add batch dimension
    
    def predict(self, image_path: str) -> Tuple[str, float, Dict[str, float]]:
        """
        Predict class for a single image.

        Probabilities are computed as softmax(logits / T) where T is the
        fitted temperature scalar (T=1.0 when no calibration file was found).

        Args:
            image_path: Path to image file

        Returns:
            pred_class: Predicted class name
            confidence: Calibrated confidence score (0-1) for the predicted class
            all_probs: Dictionary of {class_name: calibrated_probability}
        """
        image_tensor = self._preprocess_image(image_path).to(self.device)

        with torch.no_grad():
            logits = self.model(image_tensor)
            probabilities = torch.softmax(logits / self.temperature, dim=1)[0]

        max_prob, pred_idx = torch.max(probabilities, 0)
        max_prob = max_prob.item()
        pred_idx = pred_idx.item()

        if max_prob < self.confidence_threshold:
            pred_class = "other"
            confidence = max_prob
        else:
            pred_class = self.classes[pred_idx]
            confidence = max_prob

        all_probs = {
            self.classes[i]: float(probabilities[i].item())
            for i in range(len(self.classes))
        }

        return pred_class, confidence, all_probs
    
    def predict_batch(
        self,
        image_paths: List[str],
        batch_size: int = 32,
    ) -> List[Tuple[str, str, float, Dict[str, float]]]:
        """
        Predict classes for multiple images (efficient batch processing).

        Probabilities are computed as softmax(logits / T) where T is the
        fitted temperature scalar (T=1.0 when no calibration file was found).

        Args:
            image_paths: List of image file paths
            batch_size: Number of images to process at once

        Returns:
            List of tuples: (image_path, pred_class, confidence, all_probs)
        """
        results = []

        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i : i + batch_size]

            batch_tensors = []
            valid_paths = []
            for path in batch_paths:
                try:
                    tensor = self._preprocess_image(path)
                    batch_tensors.append(tensor)
                    valid_paths.append(path)
                except Exception as e:
                    print(f"Warning: Failed to load {path}: {e}")
                    continue

            if not batch_tensors:
                continue

            batch = torch.cat(batch_tensors, dim=0).to(self.device)

            with torch.no_grad():
                logits = self.model(batch)
                probabilities = torch.softmax(logits / self.temperature, dim=1)

            for j, path in enumerate(valid_paths):
                probs = probabilities[j]
                max_prob, pred_idx = torch.max(probs, 0)
                max_prob = max_prob.item()
                pred_idx = pred_idx.item()

                if max_prob < self.confidence_threshold:
                    pred_class = "other"
                    confidence = max_prob
                else:
                    pred_class = self.classes[pred_idx]
                    confidence = max_prob

                all_probs = {
                    self.classes[k]: float(probs[k].item())
                    for k in range(len(self.classes))
                }

                results.append((path, pred_class, confidence, all_probs))

        return results
    
    def set_confidence_threshold(self, threshold: float):
        """Update confidence threshold"""
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("Threshold must be between 0.0 and 1.0")
        self.confidence_threshold = threshold
        print(f"Confidence threshold updated to {threshold}")
    
    def get_model_info(self) -> Dict:
        """Get model configuration and metadata."""
        return {
            "classes": self.classes,
            "confidence_threshold": self.confidence_threshold,
            "temperature": self.temperature,
            "calibrated": self.temperature != 1.0,
            "device": self.device,
            "config": self.config,
        }


def demo():
    """Demo usage of the classifier"""
    import sys
    
    if len(sys.argv) < 3:
        print("Usage: python classifier.py <model_path> <image_path_or_folder>")
        print("\nExample:")
        print("  python classifier.py models/best_model.pth test_image.jpg")
        print("  python classifier.py models/best_model.pth test_images/")
        sys.exit(1)
    
    model_path = sys.argv[1]
    input_path = Path(sys.argv[2])
    
    # Load classifier
    print(f"Loading model from {model_path}...")
    classifier = BirdPlaneSupermanClassifier(model_path, confidence_threshold=0.7)
    
    # Get image paths
    if input_path.is_file():
        image_paths = [str(input_path)]
    elif input_path.is_dir():
        image_paths = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
            image_paths.extend([str(p) for p in input_path.glob(ext)])
        print(f"\nFound {len(image_paths)} images in {input_path}")
    else:
        print(f"Error: {input_path} is not a valid file or directory")
        sys.exit(1)
    
    if not image_paths:
        print("No images found!")
        sys.exit(1)
    
    # Run predictions
    print(f"\nRunning predictions...")
    print("=" * 80)
    
    if len(image_paths) == 1:
        # Single image
        pred_class, confidence, probs = classifier.predict(image_paths[0])
        print(f"\nImage: {image_paths[0]}")
        print(f"Predicted class: {pred_class}")
        print(f"Confidence: {confidence:.2%}")
        print(f"\nAll probabilities:")
        for cls, prob in sorted(probs.items(), key=lambda x: x[1], reverse=True):
            print(f"  {cls:10s}: {prob:.2%}")
    else:
        # Batch prediction
        results = classifier.predict_batch(image_paths)
        
        # Summary statistics
        class_counts = {cls: 0 for cls in classifier.classes}
        total_confidence = 0
        
        for path, pred_class, confidence, probs in results:
            print(f"\n{Path(path).name:40s} → {pred_class:10s} ({confidence:.2%})")
            class_counts[pred_class] += 1
            total_confidence += confidence
        
        print("\n" + "=" * 80)
        print("Summary:")
        print(f"  Total images: {len(results)}")
        print(f"  Average confidence: {total_confidence/len(results):.2%}")
        print(f"\n  Predictions by class:")
        for cls, count in sorted(class_counts.items(), key=lambda x: x[1], reverse=True):
            print(f"    {cls:10s}: {count:4d} ({100*count/len(results):.1f}%)")


if __name__ == '__main__':
    demo()
