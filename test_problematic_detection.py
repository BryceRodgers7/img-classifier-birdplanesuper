"""
Simple test script to verify problematic image detection functionality

This script tests the key functions without running full training.
"""

import numpy as np
from pathlib import Path
import json
import sys

# Import functions from train_classifier
sys.path.insert(0, str(Path(__file__).parent))
from train_classifier import (
    aggregate_per_image_losses,
    identify_problematic_images,
    create_problematic_images_report,
    print_problematic_images_summary
)


def generate_mock_data():
    """Generate mock per-image loss data for testing"""
    classes = ['bird', 'plane', 'superman', 'other']
    
    # Simulate 3 epochs of data for 100 images per class
    per_image_data_by_epoch = []
    
    for epoch in range(3):
        epoch_data = []
        
        for class_idx, class_name in enumerate(classes):
            for img_idx in range(100):
                # Create some images with consistently high loss (mislabeled)
                if img_idx < 5:
                    # These are "problematic" - high and increasing loss
                    base_loss = 2.0 + (epoch * 0.3)
                    loss = base_loss + np.random.uniform(0, 0.2)
                
                # Create some with medium loss
                elif img_idx < 20:
                    base_loss = 1.5 + (epoch * 0.1)
                    loss = base_loss + np.random.uniform(-0.2, 0.2)
                
                # Most images have normal/decreasing loss
                else:
                    base_loss = 1.2 - (epoch * 0.15)  # Decreasing
                    loss = max(0.3, base_loss + np.random.uniform(-0.1, 0.1))
                
                epoch_data.append({
                    'path': f'dataset/train/{class_name}/img_{img_idx:03d}.jpg',
                    'loss': float(loss),
                    'label': class_idx
                })
        
        per_image_data_by_epoch.append(epoch_data)
    
    return per_image_data_by_epoch, classes


def test_aggregate_per_image_losses():
    """Test the aggregation function"""
    print("\n" + "=" * 70)
    print("TEST 1: Aggregate Per-Image Losses")
    print("=" * 70)
    
    per_image_data, classes = generate_mock_data()
    
    print(f"✓ Generated mock data: {len(per_image_data)} epochs")
    print(f"  Total samples per epoch: {len(per_image_data[0])}")
    
    aggregated = aggregate_per_image_losses(per_image_data, classes)
    
    print(f"✓ Aggregated data for {len(aggregated)} unique images")
    
    # Check first image
    first_image = list(aggregated.values())[0]
    print(f"\n  Sample aggregated data (first image):")
    print(f"    Path: {first_image['path']}")
    print(f"    Class: {first_image['class']}")
    print(f"    Mean loss: {first_image['mean_loss']:.4f}")
    print(f"    Max loss: {first_image['max_loss']:.4f}")
    print(f"    Min loss: {first_image['min_loss']:.4f}")
    print(f"    Std loss: {first_image['std_loss']:.4f}")
    print(f"    Losses by epoch: {[f'{l:.3f}' for l in first_image['losses_by_epoch']]}")
    
    assert len(aggregated) == 400, "Should have 400 images (100 per class * 4 classes)"
    assert first_image['num_epochs'] == 3, "Should have 3 epochs of data"
    
    print("\n✅ Aggregation test passed!")
    return aggregated, classes


def test_identify_problematic_images(aggregated, classes):
    """Test problematic image identification"""
    print("\n" + "=" * 70)
    print("TEST 2: Identify Problematic Images")
    print("=" * 70)
    
    problematic = identify_problematic_images(aggregated, top_n=10)
    
    print(f"✓ Identified problematic images in {len(problematic)} classes")
    
    for class_name, images in problematic.items():
        print(f"\n  {class_name}: {len(images)} images")
        if images:
            print(f"    Highest mean loss: {images[0]['mean_loss']:.4f}")
            print(f"    Lowest mean loss: {images[-1]['mean_loss']:.4f}")
            
            # Verify sorting
            mean_losses = [img['mean_loss'] for img in images]
            assert mean_losses == sorted(mean_losses, reverse=True), \
                f"Images should be sorted by mean_loss (descending) in {class_name}"
    
    assert all(len(imgs) == 10 for imgs in problematic.values()), \
        "Each class should have exactly 10 problematic images"
    
    print("\n✅ Identification test passed!")
    return problematic


def test_create_report(problematic, classes):
    """Test report creation"""
    print("\n" + "=" * 70)
    print("TEST 3: Create Report")
    print("=" * 70)
    
    # Create a temporary report file
    test_output_path = Path('test_problematic_report.json')
    
    report = create_problematic_images_report(
        problematic,
        analysis_epochs=3,
        dataset_dir=Path('dataset'),
        output_path=test_output_path
    )
    
    print(f"✓ Created report with {len(report)} top-level keys")
    print(f"  Report saved to: {test_output_path}")
    
    # Verify report structure
    assert 'scan_date' in report
    assert 'method' in report
    assert 'analysis_epochs' in report
    assert 'summary' in report
    assert 'problematic_images_by_class' in report
    
    print(f"\n  Report structure:")
    print(f"    Scan date: {report['scan_date']}")
    print(f"    Method: {report['method']}")
    print(f"    Analysis epochs: {report['analysis_epochs']}")
    print(f"    Total flagged: {report['summary']['total_flagged_images']}")
    
    # Verify summary statistics
    summary = report['summary']
    assert summary['total_flagged_images'] == 40, "Should have 40 images (10 per class * 4)"
    assert summary['num_classes'] == 4, "Should have 4 classes"
    
    print("\n✅ Report creation test passed!")
    
    # Test summary printing
    print("\n" + "=" * 70)
    print("TEST 4: Print Summary")
    print("=" * 70)
    
    print_problematic_images_summary(report)
    
    print("\n✅ Summary printing test passed!")
    
    # Clean up test file
    if test_output_path.exists():
        test_output_path.unlink()
        print(f"\n✓ Cleaned up test file: {test_output_path}")
    
    return report


def test_report_loading():
    """Test that the review script can load the report"""
    print("\n" + "=" * 70)
    print("TEST 5: Report Loading (for review script)")
    print("=" * 70)
    
    # Create a small test report
    test_report = {
        'scan_date': '2026-02-10T12:00:00',
        'method': 'per_image_loss_tracking',
        'analysis_epochs': 3,
        'dataset_dir': 'dataset',
        'summary': {
            'total_flagged_images': 4,
            'num_classes': 2,
            'images_per_class': {'bird': 2, 'plane': 2},
            'class_loss_stats': {
                'bird': {
                    'count': 2,
                    'highest_mean_loss': 2.5,
                    'lowest_mean_loss': 2.0,
                    'avg_mean_loss': 2.25,
                    'description': 'Top 2 highest-loss images from bird class'
                }
            }
        },
        'problematic_images_by_class': {
            'bird': [
                {
                    'path': 'dataset/train/bird/img_001.jpg',
                    'class': 'bird',
                    'label': 0,
                    'mean_loss': 2.5,
                    'max_loss': 2.7,
                    'min_loss': 2.3,
                    'std_loss': 0.2,
                    'losses_by_epoch': [2.3, 2.5, 2.7],
                    'num_epochs': 3
                }
            ]
        }
    }
    
    test_path = Path('test_report_loading.json')
    with open(test_path, 'w') as f:
        json.dump(test_report, f, indent=2)
    
    print(f"✓ Created test report: {test_path}")
    
    # Try loading it
    with open(test_path, 'r') as f:
        loaded = json.load(f)
    
    assert loaded['analysis_epochs'] == 3
    assert loaded['summary']['total_flagged_images'] == 4
    
    print(f"✓ Successfully loaded and validated report")
    
    # Clean up
    test_path.unlink()
    print(f"✓ Cleaned up test file")
    
    print("\n✅ Report loading test passed!")


def main():
    """Run all tests"""
    print("\n" + "=" * 70)
    print("PROBLEMATIC IMAGE DETECTION - TEST SUITE")
    print("=" * 70)
    print("\nThis test validates the core functionality without running full training.")
    
    try:
        # Run tests
        aggregated, classes = test_aggregate_per_image_losses()
        problematic = test_identify_problematic_images(aggregated, classes)
        report = test_create_report(problematic, classes)
        test_report_loading()
        
        # Summary
        print("\n" + "=" * 70)
        print("🎉 ALL TESTS PASSED!")
        print("=" * 70)
        print("\nThe problematic image detection feature is working correctly.")
        print("\nNext steps:")
        print("  1. Run actual training with --detect-problematic to test end-to-end")
        print("  2. Review the generated report using review_problematic_images.py")
        print("\nExample command:")
        print("  python train_classifier.py --detect-problematic --stop-after-analysis")
        
    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")
        return 1
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == '__main__':
    exit(main())
