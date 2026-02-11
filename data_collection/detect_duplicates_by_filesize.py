"""
Duplicate Image Detector

Scans the dataset folder to find duplicate images based on file size.
Creates a JSON report identifying duplicates and whether they're in the same class/set.

Usage:
    python data_collection/detect_duplicates.py
    
Output:
    dataset/duplicate_report.json - Detailed report of all duplicates
"""

import os
import json
from pathlib import Path
from collections import defaultdict
from datetime import datetime


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


def find_duplicates(dataset_dir):
    """
    Find duplicate images in the dataset folder based on file size.
    
    Returns:
        dict: Dictionary mapping file sizes to lists of file info
    """
    dataset_path = Path(dataset_dir)
    
    if not dataset_path.exists():
        print(f"❌ Dataset directory not found: {dataset_dir}")
        return {}
    
    print(f"🔍 Scanning dataset directory: {dataset_path}")
    
    # Dictionary mapping file size to list of files with that size
    size_map = defaultdict(list)
    
    # Supported image extensions
    image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp'}
    
    # Recursively scan all files
    total_files = 0
    for root, dirs, files in os.walk(dataset_path):
        for filename in files:
            file_path = Path(root) / filename
            
            # Only process image files
            if file_path.suffix.lower() in image_extensions:
                total_files += 1
                info = get_image_info(file_path)
                size_map[info['size']].append(info)
    
    print(f"📊 Total images scanned: {total_files}")
    
    # Filter to only sizes with duplicates
    duplicates = {size: files for size, files in size_map.items() if len(files) > 1}
    
    return duplicates


def analyze_duplicates(duplicates):
    """
    Analyze duplicates to categorize them by whether they're in same class/set.
    
    Returns:
        dict: Categorized duplicate groups
    """
    analysis = {
        'same_class_and_set': [],      # Exact duplicates (same class + same set)
        'same_class_diff_set': [],     # Same class but different set (train vs val)
        'diff_class_same_set': [],     # Different class but same set
        'diff_class_diff_set': [],     # Different class and different set
        'unknown': []                  # Files without clear class/set
    }
    
    for size, files in duplicates.items():
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
        
        duplicate_group = {
            'size_bytes': size,
            'count': len(files),
            'files': files
        }
        
        analysis[category].append(duplicate_group)
    
    return analysis


def create_report(duplicates, analysis, output_path):
    """Create a JSON report of duplicates."""
    total_duplicates = sum(len(group['files']) for groups in analysis.values() for group in groups)
    total_groups = sum(len(groups) for groups in analysis.values())
    
    report = {
        'scan_date': datetime.now().isoformat(),
        'summary': {
            'total_duplicate_files': total_duplicates,
            'total_duplicate_groups': total_groups,
            'same_class_and_set': {
                'groups': len(analysis['same_class_and_set']),
                'files': sum(g['count'] for g in analysis['same_class_and_set']),
                'description': 'Exact duplicates within the same class and set (train or val)'
            },
            'same_class_diff_set': {
                'groups': len(analysis['same_class_diff_set']),
                'files': sum(g['count'] for g in analysis['same_class_diff_set']),
                'description': 'Duplicates in same class but different sets (one in train, one in val)'
            },
            'diff_class_same_set': {
                'groups': len(analysis['diff_class_same_set']),
                'files': sum(g['count'] for g in analysis['diff_class_same_set']),
                'description': 'Duplicates in different classes but same set'
            },
            'diff_class_diff_set': {
                'groups': len(analysis['diff_class_diff_set']),
                'files': sum(g['count'] for g in analysis['diff_class_diff_set']),
                'description': 'Duplicates in different classes and different sets'
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
    print("📊 DUPLICATE DETECTION REPORT")
    print("=" * 70)
    
    summary = report['summary']
    
    print(f"\n🔍 Scan Date: {report['scan_date']}")
    print(f"\n📈 Overall Statistics:")
    print(f"   Total duplicate files: {summary['total_duplicate_files']}")
    print(f"   Total duplicate groups: {summary['total_duplicate_groups']}")
    
    print(f"\n📂 Breakdown by Category:")
    
    categories = [
        ('same_class_and_set', '✅ Same Class & Set', 
         'Safe to delete - exact duplicates in same location'),
        ('same_class_diff_set', '⚠️  Same Class, Diff Set', 
         'CAUTION: Data leakage between train/val!'),
        ('diff_class_same_set', '⚠️  Diff Class, Same Set', 
         'PROBLEM: Same image labeled differently'),
        ('diff_class_diff_set', '⚠️  Diff Class, Diff Set', 
         'PROBLEM: Same image labeled differently across sets'),
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
        print("   Some images appear in both training and validation sets.")
        print("   This can artificially inflate validation accuracy.")
        print("   Recommend: Delete duplicates from validation set.")
    
    if summary['diff_class_same_set']['files'] > 0 or summary['diff_class_diff_set']['files'] > 0:
        print("\n⚠️  WARNING: Labeling inconsistencies detected!")
        print("   Some images are labeled as different classes.")
        print("   This will confuse the model during training.")
        print("   Recommend: Manually review and fix labels.")
    
    print("\n" + "=" * 70)


def main():
    """Main function."""
    print("=" * 70)
    print("🔍 Duplicate Image Detector")
    print("=" * 70)
    
    # Get dataset directory (one level up from this script)
    script_dir = Path(__file__).parent
    dataset_dir = script_dir.parent / 'dataset'
    
    print(f"\n📁 Dataset directory: {dataset_dir}")
    
    # Find duplicates
    print("\n🔍 Step 1: Scanning for duplicate images...")
    duplicates = find_duplicates(dataset_dir)
    
    if not duplicates:
        print("\n✅ No duplicates found!")
        return
    
    print(f"\n✅ Found {len(duplicates)} groups of duplicate files")
    
    # Analyze duplicates
    print("\n🔍 Step 2: Analyzing duplicate categories...")
    analysis = analyze_duplicates(duplicates)
    
    # Create report
    output_path = dataset_dir / 'duplicate_report.json'
    print(f"\n💾 Step 3: Creating report at {output_path}...")
    report = create_report(duplicates, analysis, output_path)
    
    # Print summary
    print_summary(report)
    
    print(f"\n💾 Full report saved to: {output_path}")
    print(f"\n🔧 Next step:")
    print(f"   python data_collection/remove_duplicates.py")
    print(f"   (This will help you remove duplicate files)")
    
    print("\n" + "=" * 70)


if __name__ == '__main__':
    main()
