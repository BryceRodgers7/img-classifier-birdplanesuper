"""
Duplicate Image Detector using Perceptual Hashing (pHash)

Scans the dataset folder to find duplicate and near-duplicate images using perceptual hashing.
This method is more robust than file size comparison and can detect images that are:
- Saved with different compression levels
- Converted between formats (PNG <-> JPG)
- Slightly edited or cropped

Best used BEFORE training the model to clean the dataset.

Usage:
    python data_collection/detect_duplicates_by_phash.py
    
Output:
    dataset/duplicate_report_phash.json - Detailed report of all duplicates
    
Requirements:
    pip install imagehash
"""

import os
import json
from pathlib import Path
from collections import defaultdict
from datetime import datetime
from PIL import Image
import imagehash
import time


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


def compute_perceptual_hash(image_path):
    """
    Compute perceptual hash for an image.
    
    Returns:
        imagehash.ImageHash or None if error
    """
    try:
        img = Image.open(image_path)
        # Convert to RGB if needed
        if img.mode != 'RGB':
            img = img.convert('RGB')
        # Use pHash (perceptual hash) - good balance of speed and accuracy
        return imagehash.phash(img, hash_size=8)
    except Exception as e:
        print(f"  ⚠️  Error processing {image_path}: {str(e)[:50]}")
        return None


def find_duplicates(dataset_dir, hamming_threshold=5):
    """
    Find duplicate images in the dataset folder using perceptual hashing.
    
    Args:
        dataset_dir: Path to dataset directory
        hamming_threshold: Maximum Hamming distance to consider duplicates (0-64)
                          0 = exact match, 5 = very similar (default), 10 = similar
    
    Returns:
        dict: Dictionary mapping hash groups to lists of similar files
    """
    dataset_path = Path(dataset_dir)
    
    if not dataset_path.exists():
        print(f"❌ Dataset directory not found: {dataset_dir}")
        return {}
    
    print(f"🔍 Scanning dataset directory: {dataset_path}")
    print(f"   Using perceptual hash with Hamming distance threshold: {hamming_threshold}")
    
    # Supported image extensions
    image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp'}
    
    # First pass: compute hashes for all images
    print("\n📊 Computing perceptual hashes...")
    image_hashes = []
    total_files = 0
    
    for root, dirs, files in os.walk(dataset_path):
        for filename in files:
            file_path = Path(root) / filename
            
            # Only process image files
            if file_path.suffix.lower() in image_extensions:
                total_files += 1
                if total_files % 100 == 0:
                    print(f"  Processed {total_files} images...")
                
                phash = compute_perceptual_hash(file_path)
                if phash is not None:
                    info = get_image_info(file_path)
                    info['phash'] = str(phash)
                    image_hashes.append((phash, info))
    
    print(f"✅ Total images scanned: {total_files}")
    print(f"   Successfully computed hashes: {len(image_hashes)}")
    
    # Second pass: find duplicates by comparing hashes
    print(f"\n🔍 Finding duplicates (threshold: {hamming_threshold})...")
    duplicate_groups = []
    processed = set()
    
    for i, (hash1, info1) in enumerate(image_hashes):
        if i in processed:
            continue
        
        # Find all images similar to this one
        similar_group = [info1]
        processed.add(i)
        
        for j, (hash2, info2) in enumerate(image_hashes[i+1:], start=i+1):
            if j in processed:
                continue
            
            # Calculate Hamming distance
            distance = hash1 - hash2
            
            if distance <= hamming_threshold:
                similar_group.append(info2)
                processed.add(j)
        
        # Only keep groups with 2+ images
        if len(similar_group) > 1:
            duplicate_groups.append({
                'representative_hash': str(hash1),
                'count': len(similar_group),
                'files': similar_group
            })
    
    print(f"✅ Found {len(duplicate_groups)} groups of similar images")
    
    return duplicate_groups


def analyze_duplicates(duplicate_groups):
    """
    Analyze duplicates to categorize them by whether they're in same class/set.
    
    Returns:
        dict: Categorized duplicate groups
    """
    print(f"   Analyzing {len(duplicate_groups)} duplicate groups...")
    
    analysis = {
        'same_class_and_set': [],      # Exact duplicates (same class + same set)
        'same_class_diff_set': [],     # Same class but different set (train vs val)
        'diff_class_same_set': [],     # Different class but same set
        'diff_class_diff_set': [],     # Different class and different set
        'unknown': []                  # Files without clear class/set
    }
    
    total_groups = len(duplicate_groups)
    
    for idx, group in enumerate(duplicate_groups, 1):
        # Show progress every 100 groups or at the end
        if idx % 100 == 0 or idx == total_groups:
            print(f"   Progress: {idx}/{total_groups} groups analyzed ({100*idx/total_groups:.1f}%)")
        
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
    
    print(f"   ✅ Categorization complete!")
    
    return analysis


def create_report(duplicate_groups, analysis, output_path, hamming_threshold):
    """Create a JSON report of duplicates."""
    total_duplicates = sum(group['count'] for group in duplicate_groups)
    
    report = {
        'scan_date': datetime.now().isoformat(),
        'method': 'perceptual_hash',
        'algorithm': 'pHash (8x8)',
        'hamming_threshold': hamming_threshold,
        'summary': {
            'total_duplicate_files': total_duplicates,
            'total_duplicate_groups': len(duplicate_groups),
            'same_class_and_set': {
                'groups': len(analysis['same_class_and_set']),
                'files': sum(g['count'] for g in analysis['same_class_and_set']),
                'description': 'Near-duplicates within the same class and set (train or val)'
            },
            'same_class_diff_set': {
                'groups': len(analysis['same_class_diff_set']),
                'files': sum(g['count'] for g in analysis['same_class_diff_set']),
                'description': 'Near-duplicates in same class but different sets (data leakage)'
            },
            'diff_class_same_set': {
                'groups': len(analysis['diff_class_same_set']),
                'files': sum(g['count'] for g in analysis['diff_class_same_set']),
                'description': 'Near-duplicates in different classes but same set'
            },
            'diff_class_diff_set': {
                'groups': len(analysis['diff_class_diff_set']),
                'files': sum(g['count'] for g in analysis['diff_class_diff_set']),
                'description': 'Near-duplicates in different classes and different sets'
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
    print("📊 DUPLICATE DETECTION REPORT (Perceptual Hash)")
    print("=" * 70)
    
    summary = report['summary']
    
    print(f"\n🔍 Scan Date: {report['scan_date']}")
    print(f"   Method: {report['method']}")
    print(f"   Algorithm: {report['algorithm']}")
    print(f"   Hamming Threshold: {report['hamming_threshold']} (0=exact, 64=max difference)")
    
    print(f"\n📈 Overall Statistics:")
    print(f"   Total duplicate files: {summary['total_duplicate_files']}")
    print(f"   Total duplicate groups: {summary['total_duplicate_groups']}")
    
    print(f"\n📂 Breakdown by Category:")
    
    categories = [
        ('same_class_and_set', '✅ Same Class & Set', 
         'Safe to delete - near-duplicates in same location'),
        ('same_class_diff_set', '⚠️  Same Class, Diff Set', 
         'CAUTION: Data leakage between train/val!'),
        ('diff_class_same_set', '⚠️  Diff Class, Same Set', 
         'PROBLEM: Similar images labeled differently'),
        ('diff_class_diff_set', '⚠️  Diff Class, Diff Set', 
         'PROBLEM: Similar images labeled differently across sets'),
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
        print("\n⚠️  WARNING: Labeling inconsistencies detected!")
        print("   Similar images are labeled as different classes.")
        print("   This will confuse the model during training.")
        print("   Recommend: Manually review and fix labels.")
    
    print("\n💡 Note: Perceptual hash finds near-duplicates (similar images)")
    print("   even if they have different compression or minor edits.")
    
    print("\n" + "=" * 70)


def main():
    """Main function."""
    # Start overall timer
    start_time = time.time()
    
    print("=" * 70)
    print("🔍 Duplicate Image Detector (Perceptual Hash)")
    print("=" * 70)
    
    # Get dataset directory (one level up from this script)
    script_dir = Path(__file__).parent
    dataset_dir = script_dir.parent / 'dataset'
    
    print(f"\n📁 Dataset directory: {dataset_dir}")
    
    # Get Hamming threshold from user
    print("\n⚙️  Hamming Distance Threshold:")
    print("   0-2:  Only nearly identical images (strictest)")
    print("   3-5:  Very similar images (recommended)")
    print("   6-10: Similar images (may include some false positives)")
    print("   11+:  Loosely similar (many false positives)")
    
    while True:
        try:
            threshold_input = input("\n   Enter threshold (press Enter for default 5): ").strip()
            if threshold_input == '':
                hamming_threshold = 5
                break
            hamming_threshold = int(threshold_input)
            if 0 <= hamming_threshold <= 64:
                break
            print("   ⚠️  Please enter a value between 0 and 64")
        except ValueError:
            print("   ⚠️  Please enter a valid number")
    
    # Find duplicates
    print("\n🔍 Step 1: Computing perceptual hashes and finding duplicates...")
    step1_start = time.time()
    duplicate_groups = find_duplicates(dataset_dir, hamming_threshold)
    step1_time = time.time() - step1_start
    print(f"   ⏱️  Step 1 completed in {step1_time:.1f} seconds")
    
    if not duplicate_groups:
        elapsed = time.time() - start_time
        print(f"\n✅ No duplicates found!")
        print(f"⏱️  Total time: {elapsed:.1f} seconds ({elapsed/60:.1f} minutes)")
        return
    
    # Analyze duplicates
    print("\n🔍 Step 2: Analyzing duplicate categories...")
    step2_start = time.time()
    analysis = analyze_duplicates(duplicate_groups)
    step2_time = time.time() - step2_start
    print(f"   ⏱️  Step 2 completed in {step2_time:.1f} seconds")
    
    # Create report
    output_path = dataset_dir / 'duplicate_report_phash.json'
    print(f"\n💾 Step 3: Creating report at {output_path}...")
    step3_start = time.time()
    report = create_report(duplicate_groups, analysis, output_path, hamming_threshold)
    step3_time = time.time() - step3_start
    print(f"   ⏱️  Step 3 completed in {step3_time:.1f} seconds")
    
    # Print summary
    print_summary(report)
    
    # Calculate total elapsed time
    elapsed = time.time() - start_time
    
    print(f"\n💾 Full report saved to: {output_path}")
    print(f"\n⏱️  Total Processing Time: {elapsed:.1f} seconds ({elapsed/60:.1f} minutes)")
    print(f"\n🔧 Next step:")
    print(f"   python data_collection/remove_duplicates.py")
    print(f"   (Modify it to read 'duplicate_report_phash.json' instead)")
    
    print("\n" + "=" * 70)


if __name__ == '__main__':
    main()
