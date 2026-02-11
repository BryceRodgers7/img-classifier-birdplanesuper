"""
Problematic Images Reviewer

Reads the problematic_images_report.json and displays the flagged images
for manual review. Helps identify mislabeled, ambiguous, or non-representative images.

Usage:
    python review_problematic_images.py
    python review_problematic_images.py --report-path dataset/problematic_images_report.json
    python review_problematic_images.py --class bird --limit 10
"""

import json
import argparse
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


def load_report(report_path):
    """Load the problematic images report"""
    with open(report_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def display_images_grid(images_data, class_name, num_cols=5):
    """
    Display a grid of images with their loss statistics
    
    Args:
        images_data: List of image dictionaries from report
        class_name: Name of the class being displayed
        num_cols: Number of columns in the grid
    """
    if not images_data:
        print(f"No images to display for class: {class_name}")
        return
    
    num_images = len(images_data)
    num_rows = (num_images + num_cols - 1) // num_cols
    
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(num_cols * 3, num_rows * 3.5))
    fig.suptitle(f'Problematic Images - Class: {class_name.upper()}', 
                 fontsize=16, fontweight='bold')
    
    # Flatten axes for easier iteration
    if num_rows == 1 and num_cols == 1:
        axes = [axes]
    else:
        axes = axes.flatten() if num_rows > 1 else [axes] if num_cols == 1 else axes
    
    for idx, (ax, img_data) in enumerate(zip(axes, images_data)):
        try:
            # Load and display image
            img = Image.open(img_data['path'])
            ax.imshow(img)
            
            # Create title with loss statistics
            title = (
                f"#{idx+1} | Mean Loss: {img_data['mean_loss']:.3f}\n"
                f"Max: {img_data['max_loss']:.3f} | Min: {img_data['min_loss']:.3f}\n"
                f"Std: {img_data['std_loss']:.3f}"
            )
            ax.set_title(title, fontsize=8, pad=5)
            
            # Color code by loss severity
            if img_data['mean_loss'] > 2.0:
                color = 'red'
                severity = 'High'
            elif img_data['mean_loss'] > 1.5:
                color = 'orange'
                severity = 'Medium'
            else:
                color = 'yellow'
                severity = 'Low'
            
            # Add colored border
            for spine in ax.spines.values():
                spine.set_edgecolor(color)
                spine.set_linewidth(3)
            
            # Add filename at bottom
            filename = Path(img_data['path']).name
            ax.text(0.5, -0.1, filename, transform=ax.transAxes,
                   fontsize=6, ha='center', wrap=True)
            
        except Exception as e:
            ax.text(0.5, 0.5, f"Error loading image:\n{str(e)[:50]}", 
                   ha='center', va='center', fontsize=8, color='red')
            ax.set_title(f"#{idx+1} | Error", fontsize=8)
        
        ax.axis('off')
    
    # Hide unused subplots
    for idx in range(num_images, len(axes)):
        axes[idx].axis('off')
    
    # Add legend for severity colors
    red_patch = mpatches.Patch(color='red', label='High Loss (>2.0)')
    orange_patch = mpatches.Patch(color='orange', label='Medium Loss (1.5-2.0)')
    yellow_patch = mpatches.Patch(color='yellow', label='Low Loss (<1.5)')
    plt.legend(handles=[red_patch, orange_patch, yellow_patch], 
              loc='upper center', bbox_to_anchor=(0.5, -0.02),
              ncol=3, frameon=True, fontsize=10)
    
    plt.tight_layout()
    plt.show()


def display_loss_trajectory(images_data, class_name, num_images=10):
    """
    Display loss trajectories across epochs for top N images
    
    Args:
        images_data: List of image dictionaries from report
        class_name: Name of the class being displayed
        num_images: Number of images to plot
    """
    if not images_data:
        print(f"No images to display for class: {class_name}")
        return
    
    plt.figure(figsize=(12, 6))
    
    # Plot top N images
    for idx, img_data in enumerate(images_data[:num_images]):
        epochs = list(range(1, len(img_data['losses_by_epoch']) + 1))
        losses = img_data['losses_by_epoch']
        
        filename = Path(img_data['path']).name
        plt.plot(epochs, losses, marker='o', label=f"#{idx+1}: {filename[:20]}...", 
                linewidth=2, markersize=6)
    
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title(f'Loss Trajectories - Top {num_images} Problematic Images\nClass: {class_name.upper()}', 
             fontsize=14, fontweight='bold')
    plt.legend(loc='best', fontsize=8)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def print_detailed_summary(report):
    """Print a detailed summary of the report"""
    print("\n" + "=" * 80)
    print("PROBLEMATIC IMAGES REPORT SUMMARY")
    print("=" * 80)
    
    print(f"\nReport Date: {report['scan_date']}")
    print(f"Analysis Method: {report['method']}")
    print(f"Epochs Analyzed: {report['analysis_epochs']}")
    print(f"Dataset: {report['dataset_dir']}")
    
    summary = report['summary']
    print(f"\n📊 OVERALL STATISTICS:")
    print(f"   Total flagged images: {summary['total_flagged_images']}")
    print(f"   Number of classes: {summary['num_classes']}")
    
    print(f"\n📂 CLASS BREAKDOWN:")
    for class_name, stats in summary['class_loss_stats'].items():
        print(f"\n   {class_name.upper()}:")
        print(f"      Count: {stats['count']}")
        print(f"      Highest mean loss: {stats['highest_mean_loss']:.4f}")
        print(f"      Lowest mean loss: {stats['lowest_mean_loss']:.4f}")
        print(f"      Average mean loss: {stats['avg_mean_loss']:.4f}")
    
    print("\n" + "=" * 80)


def interactive_review(report, class_name=None):
    """
    Interactive review mode - shows images and prompts for actions
    
    Args:
        report: The loaded report dictionary
        class_name: Optional class to focus on (None = all classes)
    """
    problematic_by_class = report['problematic_images_by_class']
    
    if class_name:
        if class_name not in problematic_by_class:
            print(f"Class '{class_name}' not found in report.")
            return
        classes_to_review = {class_name: problematic_by_class[class_name]}
    else:
        classes_to_review = problematic_by_class
    
    for cls, images in classes_to_review.items():
        print(f"\n{'=' * 80}")
        print(f"Reviewing class: {cls.upper()} ({len(images)} images)")
        print(f"{'=' * 80}\n")
        
        for idx, img_data in enumerate(images, 1):
            print(f"\n--- Image {idx}/{len(images)} ---")
            print(f"Path: {img_data['path']}")
            print(f"Mean Loss: {img_data['mean_loss']:.4f}")
            print(f"Max Loss: {img_data['max_loss']:.4f}")
            print(f"Min Loss: {img_data['min_loss']:.4f}")
            print(f"Std Loss: {img_data['std_loss']:.4f}")
            print(f"Losses by Epoch: {[f'{l:.3f}' for l in img_data['losses_by_epoch']]}")
            
            try:
                img = Image.open(img_data['path'])
                img.show()
            except Exception as e:
                print(f"⚠️  Error displaying image: {e}")
            
            print("\nActions:")
            print("  [n] Next image")
            print("  [d] Mark for deletion")
            print("  [r] Mark for relabeling")
            print("  [k] Keep (no action)")
            print("  [s] Skip remaining images in this class")
            print("  [q] Quit review")
            
            action = input("\nYour choice: ").lower().strip()
            
            if action == 'q':
                print("\n👋 Exiting review...")
                return
            elif action == 's':
                print(f"\n⏭️  Skipping remaining images in {cls}")
                break
            elif action == 'd':
                print(f"   ✓ Marked for deletion: {img_data['path']}")
                # TODO: Add to deletion list
            elif action == 'r':
                print(f"   ✓ Marked for relabeling: {img_data['path']}")
                # TODO: Add to relabeling list
            elif action == 'k':
                print(f"   ✓ Keeping: {img_data['path']}")
            else:
                print(f"   → Next image")
    
    print("\n" + "=" * 80)
    print("Review Complete!")
    print("=" * 80)
    print("\n💡 Note: This is a preview mode. Actual deletion/relabeling")
    print("   functionality can be added based on your workflow preferences.")


def main():
    parser = argparse.ArgumentParser(
        description='Review problematic images identified during training'
    )
    parser.add_argument(
        '--report-path',
        type=str,
        default='dataset/problematic_images_report.json',
        help='Path to problematic images report JSON file'
    )
    parser.add_argument(
        '--class',
        dest='class_name',
        type=str,
        default=None,
        help='Review only this class (e.g., bird, plane, superman, other)'
    )
    parser.add_argument(
        '--limit',
        type=int,
        default=None,
        help='Limit number of images to display per class'
    )
    parser.add_argument(
        '--mode',
        type=str,
        choices=['grid', 'trajectory', 'interactive', 'summary'],
        default='grid',
        help='Display mode: grid (image grid), trajectory (loss plots), interactive (step-by-step), summary (text only)'
    )
    
    args = parser.parse_args()
    
    # Load report
    report_path = Path(args.report_path)
    if not report_path.exists():
        print(f"❌ Report not found: {report_path}")
        print("\nRun training with --detect-problematic first:")
        print("  python train_classifier.py --detect-problematic --stop-after-analysis")
        return
    
    print(f"📂 Loading report from: {report_path}")
    report = load_report(report_path)
    
    # Print summary first
    print_detailed_summary(report)
    
    # Select mode
    if args.mode == 'summary':
        return
    
    problematic_by_class = report['problematic_images_by_class']
    
    # Filter by class if specified
    if args.class_name:
        if args.class_name not in problematic_by_class:
            print(f"\n❌ Class '{args.class_name}' not found in report.")
            print(f"   Available classes: {', '.join(problematic_by_class.keys())}")
            return
        classes_to_display = {args.class_name: problematic_by_class[args.class_name]}
    else:
        classes_to_display = problematic_by_class
    
    # Apply limit if specified
    if args.limit:
        classes_to_display = {
            cls: images[:args.limit] 
            for cls, images in classes_to_display.items()
        }
    
    # Display based on mode
    if args.mode == 'interactive':
        interactive_review(report, args.class_name)
    
    elif args.mode == 'grid':
        for class_name, images in classes_to_display.items():
            print(f"\n📸 Displaying {len(images)} images for class: {class_name}")
            display_images_grid(images, class_name)
    
    elif args.mode == 'trajectory':
        for class_name, images in classes_to_display.items():
            print(f"\n📈 Displaying loss trajectories for class: {class_name}")
            num_to_plot = min(10, len(images))
            display_loss_trajectory(images, class_name, num_to_plot)


if __name__ == '__main__':
    main()
