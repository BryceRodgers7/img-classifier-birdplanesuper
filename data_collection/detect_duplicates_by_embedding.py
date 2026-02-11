"""
Duplicate Image Detector using Model Embeddings

Scans the dataset folder to find duplicate and semantically similar images using 
deep learning embeddings from your trained model. This method can detect:
- Semantically similar images (different photos of same subject)
- Images that "look similar" to the model
- Images that might confuse the model during training/inference

Best used AFTER training the model to find semantically similar images.

Usage:
    python data_collection/detect_duplicates_by_embedding.py
    
Output:
    dataset/duplicate_report_embedding.json - Detailed report of all duplicates
    
Requirements:
    - Trained model at models/best_model.pth
    - PyTorch and torchvision (already in requirements.txt)
"""

import os
import json
from pathlib import Path
from datetime import datetime
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from torchvision import models, transforms
from sklearn.metrics.pairwise import cosine_similarity


def get_image_info(file_path):
    """Get information about an image file."""
    stat = os.stat(file_path)
    parts = file_path.parts
    
    # Determine set (train/val) and class
    dataset_set = None
    image_class = None
    
    if 'train' in parts:
        dataset_set = 'train'
        train_idx = parts.index('train')
        if train_idx + 1 < len(parts):
            image_class = parts[train_idx + 1]
    elif 'val' in parts:
        dataset_set = 'val'
        val_idx = parts.index('val')
        if val_idx + 1 < len(parts):
            image_class = parts[val_idx + 1]
    
    return {
        'path': str(file_path),
        'size': stat.st_size,
        'set': dataset_set,
        'class': image_class,
        'filename': file_path.name
    }


class EmbeddingExtractor:
    """Extract feature embeddings from images using trained model."""
    
    def __init__(self, model_path, device=None):
        """
        Initialize embedding extractor.
        
        Args:
            model_path: Path to trained model checkpoint
            device: Device to use ('cuda' or 'cpu')
        """
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"🤖 Loading model from {model_path}")
        print(f"   Device: {self.device}")
        
        # Load model
        self.model = self._load_model(model_path)
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # Create feature extractor (remove final classification layer)
        self.feature_extractor = nn.Sequential(*list(self.model.children())[:-1])
        self.feature_extractor.eval()
        
        # Image preprocessing
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        print("✅ Model loaded successfully")
    
    def _load_model(self, model_path):
        """Load trained model from checkpoint."""
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Extract metadata
        classes = checkpoint.get('classes', ['bird', 'plane', 'superman', 'other'])
        
        # Recreate model architecture (ResNet50)
        model = models.resnet50(pretrained=False)
        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features, len(classes))
        
        # Load trained weights
        model.load_state_dict(checkpoint['model_state_dict'])
        
        return model
    
    def extract_embedding(self, image_path):
        """
        Extract feature embedding for a single image.
        
        Returns:
            numpy array of shape (2048,) or None if error
        """
        try:
            # Load and preprocess image
            img = Image.open(image_path).convert('RGB')
            img_tensor = self.transform(img).unsqueeze(0).to(self.device)
            
            # Extract features
            with torch.no_grad():
                features = self.feature_extractor(img_tensor)
                # Flatten and normalize
                features = features.squeeze().cpu().numpy()
                # L2 normalize for cosine similarity
                features = features / (np.linalg.norm(features) + 1e-8)
            
            return features
            
        except Exception as e:
            print(f"  ⚠️  Error processing {image_path}: {str(e)[:50]}")
            return None
    
    def extract_embeddings_batch(self, image_paths, batch_size=32):
        """
        Extract embeddings for multiple images efficiently.
        
        Returns:
            List of (image_path, embedding) tuples
        """
        results = []
        
        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i:i+batch_size]
            
            # Load batch
            batch_tensors = []
            valid_paths = []
            
            for path in batch_paths:
                try:
                    img = Image.open(path).convert('RGB')
                    img_tensor = self.transform(img)
                    batch_tensors.append(img_tensor)
                    valid_paths.append(path)
                except Exception as e:
                    print(f"  ⚠️  Error loading {path}: {str(e)[:30]}")
                    continue
            
            if not batch_tensors:
                continue
            
            # Stack into batch
            batch = torch.stack(batch_tensors).to(self.device)
            
            # Extract features
            with torch.no_grad():
                features = self.feature_extractor(batch)
                features = features.squeeze().cpu().numpy()
                
                # Handle single image case
                if len(features.shape) == 1:
                    features = features.reshape(1, -1)
                
                # L2 normalize each feature vector
                norms = np.linalg.norm(features, axis=1, keepdims=True) + 1e-8
                features = features / norms
            
            # Store results
            for path, feat in zip(valid_paths, features):
                results.append((path, feat))
        
        return results


def find_duplicates(dataset_dir, model_path, similarity_threshold=0.95):
    """
    Find duplicate images using model embeddings.
    
    Args:
        dataset_dir: Path to dataset directory
        model_path: Path to trained model checkpoint
        similarity_threshold: Minimum cosine similarity to consider duplicates (0-1)
                            0.95+ = very similar (default), 0.90+ = similar
    
    Returns:
        list: List of duplicate groups
    """
    dataset_path = Path(dataset_dir)
    
    if not dataset_path.exists():
        print(f"❌ Dataset directory not found: {dataset_dir}")
        return []
    
    print(f"🔍 Scanning dataset directory: {dataset_path}")
    print(f"   Using model embeddings with similarity threshold: {similarity_threshold:.2f}")
    
    # Supported image extensions
    image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp'}
    
    # Collect all image paths
    image_paths = []
    for root, dirs, files in os.walk(dataset_path):
        for filename in files:
            file_path = Path(root) / filename
            if file_path.suffix.lower() in image_extensions:
                image_paths.append(file_path)
    
    print(f"📊 Total images found: {len(image_paths)}")
    
    if not image_paths:
        return []
    
    # Initialize extractor
    extractor = EmbeddingExtractor(model_path)
    
    # Extract embeddings
    print("\n📊 Extracting embeddings...")
    embeddings_data = extractor.extract_embeddings_batch(image_paths, batch_size=32)
    
    if not embeddings_data:
        print("❌ No embeddings extracted")
        return []
    
    print(f"✅ Extracted embeddings for {len(embeddings_data)} images")
    
    # Prepare data
    image_infos = []
    embeddings = []
    
    for path, embedding in embeddings_data:
        info = get_image_info(path)
        image_infos.append(info)
        embeddings.append(embedding)
    
    embeddings = np.array(embeddings)
    
    # Calculate pairwise similarities
    print(f"\n🔍 Computing pairwise similarities...")
    similarity_matrix = cosine_similarity(embeddings)
    
    # Find duplicate groups
    print(f"🔍 Finding similar pairs (threshold: {similarity_threshold:.2f})...")
    duplicate_groups = []
    processed = set()
    
    n = len(embeddings)
    for i in range(n):
        if i in processed:
            continue
        
        # Find all images similar to this one
        similar_indices = []
        for j in range(n):
            if i != j and j not in processed:
                if similarity_matrix[i, j] >= similarity_threshold:
                    similar_indices.append(j)
        
        # Create group if similarities found
        if similar_indices:
            group_indices = [i] + similar_indices
            group_files = [image_infos[idx] for idx in group_indices]
            
            # Add similarity scores
            for idx, file_info in zip(group_indices, group_files):
                # Average similarity to all other files in group
                similarities = [similarity_matrix[idx, other_idx] 
                              for other_idx in group_indices if other_idx != idx]
                file_info['avg_similarity'] = float(np.mean(similarities)) if similarities else 1.0
            
            duplicate_groups.append({
                'count': len(group_files),
                'files': group_files
            })
            
            # Mark as processed
            for idx in group_indices:
                processed.add(idx)
    
    print(f"✅ Found {len(duplicate_groups)} groups of similar images")
    
    return duplicate_groups


def analyze_duplicates(duplicate_groups):
    """
    Analyze duplicates to categorize them by whether they're in same class/set.
    
    Returns:
        dict: Categorized duplicate groups
    """
    analysis = {
        'same_class_and_set': [],      # Duplicates (same class + same set)
        'same_class_diff_set': [],     # Same class but different set (train vs val)
        'diff_class_same_set': [],     # Different class but same set
        'diff_class_diff_set': [],     # Different class and different set
        'unknown': []                  # Files without clear class/set
    }
    
    for group in duplicate_groups:
        files = group['files']
        
        # Group files by class and set
        classes = set(f['class'] for f in files if f['class'])
        sets = set(f['set'] for f in files if f['set'])
        
        # Determine category
        if len(classes) == 1 and len(sets) == 1:
            category = 'same_class_and_set'
        elif len(classes) == 1 and len(sets) > 1:
            category = 'same_class_diff_set'
        elif len(classes) > 1 and len(sets) == 1:
            category = 'diff_class_same_set'
        elif len(classes) > 1 and len(sets) > 1:
            category = 'diff_class_diff_set'
        else:
            category = 'unknown'
        
        analysis[category].append(group)
    
    return analysis


def create_report(duplicate_groups, analysis, output_path, similarity_threshold, model_path):
    """Create a JSON report of duplicates."""
    total_duplicates = sum(group['count'] for group in duplicate_groups)
    
    report = {
        'scan_date': datetime.now().isoformat(),
        'method': 'embedding_similarity',
        'model_path': str(model_path),
        'embedding_dimension': 2048,
        'similarity_metric': 'cosine_similarity',
        'similarity_threshold': similarity_threshold,
        'summary': {
            'total_duplicate_files': total_duplicates,
            'total_duplicate_groups': len(duplicate_groups),
            'same_class_and_set': {
                'groups': len(analysis['same_class_and_set']),
                'files': sum(g['count'] for g in analysis['same_class_and_set']),
                'description': 'Semantically similar images within the same class and set'
            },
            'same_class_diff_set': {
                'groups': len(analysis['same_class_diff_set']),
                'files': sum(g['count'] for g in analysis['same_class_diff_set']),
                'description': 'Semantically similar images in same class but different sets (data leakage)'
            },
            'diff_class_same_set': {
                'groups': len(analysis['diff_class_same_set']),
                'files': sum(g['count'] for g in analysis['diff_class_same_set']),
                'description': 'Semantically similar images in different classes but same set'
            },
            'diff_class_diff_set': {
                'groups': len(analysis['diff_class_diff_set']),
                'files': sum(g['count'] for g in analysis['diff_class_diff_set']),
                'description': 'Semantically similar images in different classes and different sets'
            },
            'unknown': {
                'groups': len(analysis['unknown']),
                'files': sum(g['count'] for g in analysis['unknown']),
                'description': 'Files without clear class or set information'
            }
        },
        'duplicate_groups': analysis
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    return report


def print_summary(report):
    """Print a human-readable summary of the duplicate report."""
    print("\n" + "=" * 70)
    print("📊 DUPLICATE DETECTION REPORT (Model Embeddings)")
    print("=" * 70)
    
    summary = report['summary']
    
    print(f"\n🔍 Scan Date: {report['scan_date']}")
    print(f"   Method: {report['method']}")
    print(f"   Model: {report['model_path']}")
    print(f"   Embedding Dimension: {report['embedding_dimension']}")
    print(f"   Similarity Metric: {report['similarity_metric']}")
    print(f"   Similarity Threshold: {report['similarity_threshold']:.2f} (0=different, 1=identical)")
    
    print(f"\n📈 Overall Statistics:")
    print(f"   Total similar images: {summary['total_duplicate_files']}")
    print(f"   Total similarity groups: {summary['total_duplicate_groups']}")
    
    print(f"\n📂 Breakdown by Category:")
    
    categories = [
        ('same_class_and_set', '✅ Same Class & Set', 
         'Semantically similar images in same location'),
        ('same_class_diff_set', '⚠️  Same Class, Diff Set', 
         'CAUTION: Data leakage between train/val!'),
        ('diff_class_same_set', '⚠️  Diff Class, Same Set', 
         'PROBLEM: Model thinks these should be same class'),
        ('diff_class_diff_set', '⚠️  Diff Class, Diff Set', 
         'PROBLEM: Model confused by similar images in different classes'),
        ('unknown', '❓ Unknown Location', 
         'Files without clear classification')
    ]
    
    for key, label, description in categories:
        cat = summary[key]
        if cat['groups'] > 0:
            print(f"\n   {label}:")
            print(f"      Groups: {cat['groups']}")
            print(f"      Files:  {cat['files']}")
            print(f"      → {description}")
    
    print("\n" + "=" * 70)
    
    # Highlight critical issues
    if summary['same_class_diff_set']['files'] > 0:
        print("\n⚠️  WARNING: Data leakage detected!")
        print("   Similar images appear in both training and validation sets.")
        print("   This can artificially inflate validation accuracy.")
        print("   Recommend: Delete duplicates from validation set.")
    
    if summary['diff_class_same_set']['files'] > 0 or summary['diff_class_diff_set']['files'] > 0:
        print("\n⚠️  WARNING: Potential labeling inconsistencies!")
        print("   The model thinks these images are very similar but they have different labels.")
        print("   This might indicate:")
        print("   - Mislabeled images that should be reviewed")
        print("   - Edge cases where classes overlap")
        print("   Recommend: Manually review these groups.")
    
    print("\n💡 Note: Embedding similarity finds images that 'look similar' to your model.")
    print("   This helps identify images that might confuse the model during training.")
    
    print("\n" + "=" * 70)


def main():
    """Main function."""
    print("=" * 70)
    print("🔍 Duplicate Image Detector (Model Embeddings)")
    print("=" * 70)
    
    # Get paths
    script_dir = Path(__file__).parent
    dataset_dir = script_dir.parent / 'dataset'
    model_path = script_dir.parent / 'models' / 'best_model.pth'
    
    print(f"\n📁 Dataset directory: {dataset_dir}")
    print(f"🤖 Model path: {model_path}")
    
    # Check if model exists
    if not model_path.exists():
        print("\n❌ Error: Trained model not found!")
        print("   Please train the model first:")
        print("   python train_classifier.py")
        return
    
    # Get similarity threshold from user
    print("\n⚙️  Cosine Similarity Threshold:")
    print("   0.98-1.00: Nearly identical (very strict)")
    print("   0.95-0.97: Very similar (recommended)")
    print("   0.90-0.94: Similar (more permissive)")
    print("   0.85-0.89: Loosely similar (may include false positives)")
    
    while True:
        try:
            threshold_input = input("\n   Enter threshold (press Enter for default 0.95): ").strip()
            if threshold_input == '':
                similarity_threshold = 0.95
                break
            similarity_threshold = float(threshold_input)
            if 0.0 <= similarity_threshold <= 1.0:
                break
            print("   ⚠️  Please enter a value between 0.0 and 1.0")
        except ValueError:
            print("   ⚠️  Please enter a valid number")
    
    # Find duplicates
    print("\n🔍 Step 1: Extracting embeddings and finding similar images...")
    duplicate_groups = find_duplicates(dataset_dir, model_path, similarity_threshold)
    
    if not duplicate_groups:
        print("\n✅ No similar images found!")
        return
    
    # Analyze duplicates
    print("\n🔍 Step 2: Analyzing similarity groups...")
    analysis = analyze_duplicates(duplicate_groups)
    
    # Create report
    output_path = dataset_dir / 'duplicate_report_embedding.json'
    print(f"\n💾 Step 3: Creating report at {output_path}...")
    report = create_report(duplicate_groups, analysis, output_path, similarity_threshold, model_path)
    
    # Print summary
    print_summary(report)
    
    print(f"\n💾 Full report saved to: {output_path}")
    print(f"\n🔧 Next step:")
    print(f"   python data_collection/remove_duplicates.py")
    print(f"   (Modify it to read 'duplicate_report_embedding.json' instead)")
    
    print("\n" + "=" * 70)


if __name__ == '__main__':
    main()
