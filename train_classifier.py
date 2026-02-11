"""
Bird/Plane/Superman/Other Image Classifier Training Script

Trains a ResNet50-based classifier with:
- Weighted sampling to handle class imbalance
- Class-specific augmentations
- Confidence thresholding support
- Comprehensive metrics tracking and error analysis
- Optional per-image loss tracking to identify mislabeled/problematic images
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms, models
from PIL import Image
import numpy as np
from pathlib import Path
import json
import argparse
from tqdm import tqdm
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix, accuracy_score
import time
from collections import defaultdict
from datetime import datetime


class BirdPlaneSupermanDataset(Dataset):
    """Custom dataset with class-specific augmentations"""
    
    def __init__(self, root_dir, split='train', base_transform=None):
        self.root_dir = Path(root_dir)
        self.split = split
        self.base_transform = base_transform
        
        # Class names and mappings
        self.classes = ['bird', 'plane', 'superman', 'other']
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        
        # Load all images
        self.samples = []
        for class_name in self.classes:
            class_dir = self.root_dir / split / class_name
            if class_dir.exists():
                for img_path in class_dir.glob('*.jpg'):
                    self.samples.append((str(img_path), self.class_to_idx[class_name]))
        
        print(f"  Loaded {len(self.samples)} images for {split} set")
        
        # Class-specific augmentation transforms
        self._setup_augmentations()
    
    def _setup_augmentations(self):
        """Setup class-specific augmentation pipelines"""
        # Base normalization (ImageNet stats)
        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
        
        # Training augmentations (class-specific)
        if self.split == 'train':
            # Bird: NO vertical flip (birds don't fly upside down normally)
            self.bird_transform = transforms.Compose([
                transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(15),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                transforms.ToTensor(),
                normalize
            ])
            
            # Plane: NO vertical flip (planes don't fly upside down normally)
            self.plane_transform = transforms.Compose([
                transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(15),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                transforms.ToTensor(),
                normalize
            ])
            
            # Superman: CAN be vertical flip (comic book poses, flying upside down)
            self.superman_transform = transforms.Compose([
                transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.3),
                transforms.RandomRotation(15),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                transforms.ToTensor(),
                normalize
            ])
            
            # Other: Full augmentation (can be anything)
            self.other_transform = transforms.Compose([
                transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.3),
                transforms.RandomRotation(20),
                transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.15),
                transforms.ToTensor(),
                normalize
            ])
        else:
            # Validation: No augmentation, just resize and normalize
            self.bird_transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize
            ])
            self.plane_transform = self.bird_transform
            self.superman_transform = self.bird_transform
            self.other_transform = self.bird_transform
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        
        # Load image
        image = Image.open(img_path).convert('RGB')
        
        # Apply class-specific transform
        class_name = self.classes[label]
        if class_name == 'bird':
            image = self.bird_transform(image)
        elif class_name == 'plane':
            image = self.plane_transform(image)
        elif class_name == 'superman':
            image = self.superman_transform(image)
        else:  # other
            image = self.other_transform(image)
        
        return image, label, img_path


def create_model(num_classes=4, freeze_until='layer3'):
    """
    Create ResNet50 model with frozen early layers
    
    Args:
        num_classes: Number of output classes (4 for bird/plane/superman/other)
        freeze_until: Freeze layers up to and including this layer
                      Options: 'layer1', 'layer2', 'layer3', None (no freezing)
    """
    # Load pre-trained ResNet50
    model = models.resnet50(pretrained=True)
    
    # Freeze early layers (keep low-level features stable)
    freeze_layers = ['conv1', 'bn1', 'relu', 'maxpool']
    if freeze_until:
        if freeze_until == 'layer1':
            freeze_layers.extend(['layer1'])
        elif freeze_until == 'layer2':
            freeze_layers.extend(['layer1', 'layer2'])
        elif freeze_until == 'layer3':
            freeze_layers.extend(['layer1', 'layer2', 'layer3'])
    
    for name, param in model.named_parameters():
        for freeze_layer in freeze_layers:
            if name.startswith(freeze_layer):
                param.requires_grad = False
                break
    
    # Replace final FC layer
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, num_classes)
    
    return model


def get_class_weights(dataset):
    """Calculate inverse frequency weights for weighted sampling"""
    class_counts = defaultdict(int)
    for _, label, _ in dataset.samples:
        class_counts[label] += 1
    
    total_samples = len(dataset)
    num_classes = len(dataset.classes)
    
    # Calculate weights: total / (num_classes * class_count)
    class_weights = {}
    for class_idx in range(num_classes):
        class_weights[class_idx] = total_samples / (num_classes * class_counts[class_idx])
    
    # Create sample weights for each sample
    sample_weights = [class_weights[label] for _, label, _ in dataset.samples]
    
    return sample_weights, class_counts


def train_one_epoch(model, dataloader, criterion, optimizer, device, track_per_image_loss=False):
    """
    Train for one epoch
    
    Args:
        track_per_image_loss: If True, return per-image loss data for analysis
        
    Returns:
        epoch_loss: Average loss for the epoch
        epoch_acc: Accuracy for the epoch
        per_image_data: List of dicts with per-image loss info (if track_per_image_loss=True)
    """
    model.train()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    per_image_data = [] if track_per_image_loss else None
    
    pbar = tqdm(dataloader, desc='Training')
    for images, labels, paths in pbar:
        images, labels = images.to(device), labels.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(images)
        
        # Calculate per-sample losses if tracking
        if track_per_image_loss:
            # Calculate loss per sample (no reduction)
            loss_per_sample = nn.CrossEntropyLoss(reduction='none')(outputs, labels)
            loss = loss_per_sample.mean()  # Average for backward pass
            
            # Store per-image data
            for i in range(len(paths)):
                per_image_data.append({
                    'path': paths[i],
                    'loss': float(loss_per_sample[i].item()),
                    'label': int(labels[i].item())
                })
        else:
            loss = criterion(outputs, labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Track metrics
        running_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        
        # Update progress bar
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_acc = accuracy_score(all_labels, all_preds)
    
    if track_per_image_loss:
        return epoch_loss, epoch_acc, per_image_data
    return epoch_loss, epoch_acc


def validate(model, dataloader, criterion, device, classes, save_errors=False):
    """
    Validate model and optionally collect error analysis data
    
    Returns:
        metrics: Dictionary with loss, accuracy, precision, recall, f1, confusion matrix
        error_data: List of wrong predictions with confidence scores (if save_errors=True)
    """
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    all_probs = []
    all_paths = []
    
    with torch.no_grad():
        for images, labels, paths in tqdm(dataloader, desc='Validating'):
            images, labels = images.to(device), labels.to(device)
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Get probabilities
            probs = torch.softmax(outputs, dim=1)
            
            # Track metrics
            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            all_paths.extend(paths)
    
    # Calculate metrics
    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_acc = accuracy_score(all_labels, all_preds)
    
    # Per-class metrics
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average=None, labels=range(len(classes)), zero_division=0
    )
    
    # Overall F1 (macro average)
    overall_f1 = np.mean(f1)
    
    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds, labels=range(len(classes)))
    
    metrics = {
        'loss': epoch_loss,
        'accuracy': epoch_acc,
        'precision': precision.tolist(),
        'recall': recall.tolist(),
        'f1': f1.tolist(),
        'overall_f1': overall_f1,
        'confusion_matrix': cm.tolist()
    }
    
    # Error analysis: collect wrong predictions
    error_data = None
    if save_errors:
        error_data = defaultdict(list)
        
        for i in range(len(all_preds)):
            if all_preds[i] != all_labels[i]:
                true_class = classes[all_labels[i]]
                pred_class = classes[all_preds[i]]
                confidence = all_probs[i][all_preds[i]]
                
                error_info = {
                    'image_path': all_paths[i],
                    'true_label': true_class,
                    'predicted_label': pred_class,
                    'confidence': float(confidence),
                    'all_probabilities': {
                        classes[j]: float(all_probs[i][j]) for j in range(len(classes))
                    }
                }
                
                # Group by TRUE label (to see what each class is confused with)
                error_data[true_class].append(error_info)
        
        # Sort by confidence (descending) and keep top 10 per class
        for class_name in error_data:
            error_data[class_name] = sorted(
                error_data[class_name],
                key=lambda x: x['confidence'],
                reverse=True
            )[:10]
    
    return metrics, error_data


def save_error_analysis(error_data, epoch, output_dir):
    """Save error analysis data for an epoch"""
    epoch_dir = output_dir / f'epoch_{epoch:02d}'
    epoch_dir.mkdir(parents=True, exist_ok=True)
    
    for class_name, errors in error_data.items():
        error_file = epoch_dir / f'{class_name}_errors.json'
        with open(error_file, 'w') as f:
            json.dump(errors, f, indent=2)
    
    print(f"  Saved error analysis to {epoch_dir}")


def aggregate_per_image_losses(per_image_data_by_epoch, classes):
    """
    Aggregate per-image losses across multiple epochs
    
    Args:
        per_image_data_by_epoch: List of per-image data from each epoch
        classes: List of class names
        
    Returns:
        dict: Aggregated loss statistics per image path
    """
    image_loss_stats = defaultdict(lambda: {'losses': [], 'label': None, 'class': None})
    
    # Collect all losses per image across epochs
    for epoch_data in per_image_data_by_epoch:
        for item in epoch_data:
            path = item['path']
            image_loss_stats[path]['losses'].append(item['loss'])
            image_loss_stats[path]['label'] = item['label']
            image_loss_stats[path]['class'] = classes[item['label']]
    
    # Calculate statistics for each image
    aggregated = {}
    for path, data in image_loss_stats.items():
        losses = data['losses']
        aggregated[path] = {
            'path': path,
            'class': data['class'],
            'label': data['label'],
            'mean_loss': float(np.mean(losses)),
            'max_loss': float(np.max(losses)),
            'min_loss': float(np.min(losses)),
            'std_loss': float(np.std(losses)),
            'losses_by_epoch': [float(l) for l in losses],
            'num_epochs': len(losses)
        }
    
    return aggregated


def identify_problematic_images(aggregated_stats, top_n=50):
    """
    Identify the top N highest-loss images per class
    
    Args:
        aggregated_stats: Dictionary from aggregate_per_image_losses
        top_n: Number of top images to return per class
        
    Returns:
        dict: Top N problematic images grouped by class
    """
    # Group by class
    by_class = defaultdict(list)
    for path, stats in aggregated_stats.items():
        by_class[stats['class']].append(stats)
    
    # Sort each class by mean loss (descending) and take top N
    problematic_by_class = {}
    for class_name, images in by_class.items():
        sorted_images = sorted(images, key=lambda x: x['mean_loss'], reverse=True)
        problematic_by_class[class_name] = sorted_images[:top_n]
    
    return problematic_by_class


def create_problematic_images_report(problematic_by_class, analysis_epochs, dataset_dir, output_path):
    """
    Create a JSON report of potentially problematic training images
    
    Args:
        problematic_by_class: Dictionary from identify_problematic_images
        analysis_epochs: Number of epochs used for analysis
        dataset_dir: Path to dataset directory
        output_path: Path to save the report
        
    Returns:
        dict: The report data
    """
    # Calculate summary statistics
    total_flagged = sum(len(images) for images in problematic_by_class.values())
    
    summary = {
        'total_flagged_images': total_flagged,
        'num_classes': len(problematic_by_class),
        'images_per_class': {
            class_name: len(images) 
            for class_name, images in problematic_by_class.items()
        }
    }
    
    # Add statistics about loss ranges
    class_loss_stats = {}
    for class_name, images in problematic_by_class.items():
        if images:
            mean_losses = [img['mean_loss'] for img in images]
            class_loss_stats[class_name] = {
                'count': len(images),
                'highest_mean_loss': float(max(mean_losses)),
                'lowest_mean_loss': float(min(mean_losses)),
                'avg_mean_loss': float(np.mean(mean_losses)),
                'description': f'Top {len(images)} highest-loss images from {class_name} class'
            }
    
    # Create report structure similar to duplicate detection reports
    report = {
        'scan_date': datetime.now().isoformat(),
        'method': 'per_image_loss_tracking',
        'analysis_epochs': analysis_epochs,
        'dataset_dir': str(dataset_dir),
        'summary': {
            **summary,
            'class_loss_stats': class_loss_stats,
            'description': (
                f'Images with highest average loss during first {analysis_epochs} epochs. '
                'These may be mislabeled, ambiguous, or non-representative of their class.'
            )
        },
        'problematic_images_by_class': problematic_by_class
    }
    
    # Save report
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    return report


def print_problematic_images_summary(report):
    """Print a human-readable summary of the problematic images report"""
    print("\n" + "=" * 70)
    print("📊 PROBLEMATIC TRAINING IMAGES REPORT")
    print("=" * 70)
    
    print(f"\n🔍 Analysis Date: {report['scan_date']}")
    print(f"   Method: Per-image loss tracking")
    print(f"   Epochs analyzed: {report['analysis_epochs']}")
    print(f"   Dataset: {report['dataset_dir']}")
    
    summary = report['summary']
    
    print(f"\n📈 Overall Statistics:")
    print(f"   Total flagged images: {summary['total_flagged_images']}")
    print(f"   Number of classes: {summary['num_classes']}")
    
    print(f"\n📂 Breakdown by Class:")
    
    for class_name, stats in summary['class_loss_stats'].items():
        print(f"\n   {class_name.upper()}:")
        print(f"      Images flagged: {stats['count']}")
        print(f"      Highest mean loss: {stats['highest_mean_loss']:.4f}")
        print(f"      Lowest mean loss: {stats['lowest_mean_loss']:.4f}")
        print(f"      Average mean loss: {stats['avg_mean_loss']:.4f}")
        print(f"      → {stats['description']}")
    
    print("\n" + "=" * 70)
    print("\n💡 Next Steps:")
    print("   1. Review the flagged images manually to identify:")
    print("      - Mislabeled images (wrong class)")
    print("      - Ambiguous images (could belong to multiple classes)")
    print("      - Non-representative images (poor quality, unusual angles)")
    print("   2. Remove or relabel problematic images")
    print("   3. Re-train the classifier with cleaned data")
    print("\n" + "=" * 70)


def main():
    parser = argparse.ArgumentParser(description='Train Bird/Plane/Superman classifier')
    parser.add_argument('--dataset-dir', type=str, default='dataset',
                        help='Path to dataset directory')
    parser.add_argument('--epochs', type=int, default=15,
                        help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size for training')
    parser.add_argument('--learning-rate', type=float, default=0.0001,
                        help='Learning rate')
    parser.add_argument('--freeze-until', type=str, default='layer3',
                        choices=['layer1', 'layer2', 'layer3', 'none'],
                        help='Freeze layers up to and including this layer')
    parser.add_argument('--confidence-threshold', type=float, default=0.7,
                        help='Confidence threshold for classification (saved in metadata)')
    parser.add_argument('--output-dir', type=str, default='models',
                        help='Output directory for models and metadata')
    
    # Problematic image detection arguments
    parser.add_argument('--detect-problematic', action='store_true',
                        help='Enable per-image loss tracking to identify mislabeled/problematic images')
    parser.add_argument('--analysis-epochs', type=int, default=3,
                        help='Number of epochs to track per-image loss (default: 3)')
    parser.add_argument('--top-n-problematic', type=int, default=50,
                        help='Number of highest-loss images to report per class (default: 50)')
    parser.add_argument('--stop-after-analysis', action='store_true',
                        help='Stop training after analysis epochs (only analyze, do not train full model)')
    
    args = parser.parse_args()
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    dataset_dir = Path(args.dataset_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load datasets
    print("\nLoading datasets...")
    train_dataset = BirdPlaneSupermanDataset(dataset_dir, split='train')
    val_dataset = BirdPlaneSupermanDataset(dataset_dir, split='val')
    
    # Calculate class distribution and weights
    print("\nCalculating sampling weights...")
    sample_weights, class_counts = get_class_weights(train_dataset)
    
    print("\nClass distribution (training set):")
    for class_name, class_idx in train_dataset.class_to_idx.items():
        count = class_counts[class_idx]
        weight = sample_weights[0] if class_idx == 0 else sample_weights[sum(class_counts[i] for i in range(class_idx))]
        print(f"  {class_name:10s}: {count:4d} images (weight: {weight:.4f})")
    
    # Create weighted sampler for training
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(train_dataset),
        replacement=True
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=0,  # Windows compatible
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    # Create model
    print("\nCreating model...")
    freeze_layer = None if args.freeze_until == 'none' else args.freeze_until
    model = create_model(num_classes=len(train_dataset.classes), freeze_until=freeze_layer)
    model = model.to(device)
    
    # Count trainable parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,} ({100*trainable_params/total_params:.1f}%)")
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), 
                           lr=args.learning_rate)
    
    # Training loop
    print(f"\nStarting training for {args.epochs} epochs...")
    if args.detect_problematic:
        print(f"   📊 Per-image loss tracking ENABLED for first {args.analysis_epochs} epochs")
        print(f"   Will report top {args.top_n_problematic} problematic images per class")
        if args.stop_after_analysis:
            print(f"   ⚠️  Will STOP training after {args.analysis_epochs} epochs (analysis-only mode)")
    print("=" * 70)
    
    best_f1 = 0.0
    training_history = {
        'epochs': [],
        'config': {
            'model': 'ResNet50',
            'num_classes': len(train_dataset.classes),
            'classes': train_dataset.classes,
            'epochs': args.epochs,
            'batch_size': args.batch_size,
            'learning_rate': args.learning_rate,
            'freeze_until': args.freeze_until,
            'confidence_threshold': args.confidence_threshold,
            'class_counts_train': dict(class_counts),
            'total_params': total_params,
            'trainable_params': trainable_params,
            'detect_problematic': args.detect_problematic,
            'analysis_epochs': args.analysis_epochs if args.detect_problematic else None,
            'top_n_problematic': args.top_n_problematic if args.detect_problematic else None
        }
    }
    
    error_analysis_dir = output_dir / 'error_analysis'
    per_image_data_by_epoch = [] if args.detect_problematic else None
    
    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        print("-" * 70)
        
        epoch_start = time.time()
        
        # Determine if we should track per-image loss this epoch
        track_loss = args.detect_problematic and epoch <= args.analysis_epochs
        
        # Train
        if track_loss:
            print("   📊 Tracking per-image loss this epoch...")
            train_result = train_one_epoch(model, train_loader, criterion, optimizer, device, 
                                          track_per_image_loss=True)
            train_loss, train_acc, per_image_data = train_result
            per_image_data_by_epoch.append(per_image_data)
        else:
            train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        
        # Validate
        val_metrics, error_data = validate(
            model, val_loader, criterion, device, 
            train_dataset.classes, 
            save_errors=True
        )
        
        epoch_time = time.time() - epoch_start
        
        # Print results
        print(f"\nResults:")
        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"  Val Loss:   {val_metrics['loss']:.4f} | Val Acc:   {val_metrics['accuracy']:.4f}")
        print(f"  Overall F1: {val_metrics['overall_f1']:.4f}")
        print(f"\n  Per-class metrics:")
        for i, class_name in enumerate(train_dataset.classes):
            print(f"    {class_name:10s}: Precision={val_metrics['precision'][i]:.4f}, "
                  f"Recall={val_metrics['recall'][i]:.4f}, F1={val_metrics['f1'][i]:.4f}")
        
        print(f"\n  Confusion Matrix:")
        print(f"    {'':>10s} ", end='')
        for class_name in train_dataset.classes:
            print(f"{class_name:>10s}", end=' ')
        print()
        for i, class_name in enumerate(train_dataset.classes):
            print(f"    {class_name:>10s} ", end='')
            for j in range(len(train_dataset.classes)):
                print(f"{val_metrics['confusion_matrix'][i][j]:>10d}", end=' ')
            print()
        
        print(f"\n  Epoch time: {epoch_time:.1f}s")
        
        # Save error analysis
        save_error_analysis(error_data, epoch, error_analysis_dir)
        
        # Save history
        epoch_data = {
            'epoch': epoch,
            'train_loss': train_loss,
            'train_accuracy': train_acc,
            'val_loss': val_metrics['loss'],
            'val_accuracy': val_metrics['accuracy'],
            'val_precision': val_metrics['precision'],
            'val_recall': val_metrics['recall'],
            'val_f1': val_metrics['f1'],
            'val_overall_f1': val_metrics['overall_f1'],
            'confusion_matrix': val_metrics['confusion_matrix'],
            'epoch_time': epoch_time
        }
        training_history['epochs'].append(epoch_data)
        
        # Save best model
        if val_metrics['overall_f1'] > best_f1:
            best_f1 = val_metrics['overall_f1']
            best_model_path = output_dir / 'best_model.pth'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_f1': best_f1,
                'classes': train_dataset.classes,
                'config': training_history['config']
            }, best_model_path)
            print(f"\n  ✓ Saved best model (F1: {best_f1:.4f}) to {best_model_path}")
        
        # Check if we should stop after analysis
        if args.detect_problematic and args.stop_after_analysis and epoch >= args.analysis_epochs:
            print(f"\n⚠️  Stopping training after {args.analysis_epochs} analysis epochs (as requested)")
            break
    
    # Generate problematic images report if tracking was enabled
    if args.detect_problematic and per_image_data_by_epoch:
        print("\n" + "=" * 70)
        print("📊 ANALYZING PER-IMAGE LOSSES...")
        print("=" * 70)
        
        print(f"\n   Aggregating loss data from {len(per_image_data_by_epoch)} epochs...")
        aggregated_stats = aggregate_per_image_losses(per_image_data_by_epoch, train_dataset.classes)
        print(f"   Analyzed {len(aggregated_stats)} unique training images")
        
        print(f"\n   Identifying top {args.top_n_problematic} problematic images per class...")
        problematic_by_class = identify_problematic_images(aggregated_stats, top_n=args.top_n_problematic)
        
        print(f"\n   Generating report...")
        report_path = dataset_dir / 'problematic_images_report.json'
        report = create_problematic_images_report(
            problematic_by_class,
            len(per_image_data_by_epoch),
            dataset_dir,
            report_path
        )
        
        print(f"\n   ✓ Saved problematic images report to {report_path}")
        
        # Print summary
        print_problematic_images_summary(report)
    
    # Save training metadata
    metadata_path = output_dir / 'training_metadata.json'
    with open(metadata_path, 'w') as f:
        json.dump(training_history, f, indent=2)
    print(f"\n✓ Saved training history to {metadata_path}")
    
    print("\n" + "=" * 70)
    print("Training complete!")
    print("=" * 70)
    print(f"\nBest validation F1: {best_f1:.4f}")
    print(f"Model saved to: {output_dir / 'best_model.pth'}")
    print(f"Metadata saved to: {metadata_path}")
    print(f"Error analysis saved to: {error_analysis_dir}")
    
    if args.detect_problematic:
        print(f"Problematic images report saved to: {dataset_dir / 'problematic_images_report.json'}")
    
    print("\nNext steps:")
    if args.detect_problematic:
        print("  1. Review problematic images report to identify mislabeled/ambiguous images")
        print("  2. Remove or relabel problematic images")
        print("  3. Re-run training with cleaned dataset")
    else:
        print("  1. Review error analysis to identify problematic images")
        print("  2. Use classifier.py for inference on new images")
        print("  3. Adjust confidence threshold if needed")
        print("  4. Consider using --detect-problematic to identify mislabeled training data")


if __name__ == '__main__':
    main()
