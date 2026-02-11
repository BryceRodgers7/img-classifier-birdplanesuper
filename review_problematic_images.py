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
from PIL import Image, ImageTk
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import tkinter as tk
from tkinter import messagebox
import os


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
    Interactive review mode - shows images in a GUI with action buttons
    
    Args:
        report: The loaded report dictionary
        class_name: Optional class to focus on (None = all classes)
    """
    reviewer = ImageReviewerGUI(report, class_name)
    reviewer.run()
    
    print("\n" + "=" * 80)
    print("Review Complete!")
    print("=" * 80)
    print("\n💡 Note: This is a preview mode. Actual deletion/relabeling")
    print("   functionality can be added based on your workflow preferences.")


class ImageReviewerGUI:
    """GUI for interactive image review"""
    
    def __init__(self, report, class_name=None):
        self.report = report
        self.problematic_by_class = report['problematic_images_by_class']
        
        # Filter by class if specified
        if class_name:
            if class_name not in self.problematic_by_class:
                print(f"Class '{class_name}' not found in report.")
                return
            self.classes_to_review = {class_name: self.problematic_by_class[class_name]}
        else:
            self.classes_to_review = self.problematic_by_class
        
        # Flatten all images to review
        self.all_images = []
        for cls, images in self.classes_to_review.items():
            for img_data in images:
                self.all_images.append((cls, img_data))
        
        # State tracking
        self.current_index = 0
        self.deletion_list = []
        self.relabel_list = []
        self.kept_list = []
        self.skip_current_class = False
        
        # Create main window
        self.root = tk.Tk()
        self.root.title("Image Reviewer - Problematic Images")
        self.root.geometry("1000x800")
        
        # Configure grid
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(1, weight=1)
        
        self.setup_ui()
        self.load_image()
        
    def setup_ui(self):
        """Setup the UI components"""
        
        # Top info panel
        info_frame = tk.Frame(self.root, bg='#f0f0f0', pady=10)
        info_frame.grid(row=0, column=0, sticky='ew', padx=10, pady=5)
        
        self.class_label = tk.Label(info_frame, text="", font=('Arial', 12, 'bold'), bg='#f0f0f0')
        self.class_label.pack()
        
        self.progress_label = tk.Label(info_frame, text="", font=('Arial', 10), bg='#f0f0f0')
        self.progress_label.pack()
        
        # Image display area
        image_frame = tk.Frame(self.root, bg='white')
        image_frame.grid(row=1, column=0, sticky='nsew', padx=10, pady=5)
        
        self.image_label = tk.Label(image_frame, bg='white')
        self.image_label.pack(expand=True, fill='both')
        
        # Stats panel
        stats_frame = tk.Frame(self.root, bg='#e8e8e8', pady=5)
        stats_frame.grid(row=2, column=0, sticky='ew', padx=10, pady=5)
        
        self.stats_label = tk.Label(stats_frame, text="", font=('Arial', 9), 
                                    bg='#e8e8e8', justify='left')
        self.stats_label.pack(padx=10, pady=5)
        
        self.path_label = tk.Label(stats_frame, text="", font=('Arial', 8), 
                                   bg='#e8e8e8', fg='#555', wraplength=900)
        self.path_label.pack(padx=10, pady=2)
        
        # Action buttons panel
        button_frame = tk.Frame(self.root, bg='#f0f0f0', pady=15)
        button_frame.grid(row=3, column=0, sticky='ew', padx=10, pady=5)
        
        # Configure button frame grid
        for i in range(5):
            button_frame.columnconfigure(i, weight=1)
        
        # Create buttons with colors
        tk.Button(button_frame, text="✓ Keep", command=self.action_keep,
                 bg='#4CAF50', fg='white', font=('Arial', 11, 'bold'),
                 width=12, height=2).grid(row=0, column=0, padx=5)
        
        tk.Button(button_frame, text="🗑 Delete", command=self.action_delete,
                 bg='#f44336', fg='white', font=('Arial', 11, 'bold'),
                 width=12, height=2).grid(row=0, column=1, padx=5)
        
        tk.Button(button_frame, text="🏷 Relabel", command=self.action_relabel,
                 bg='#FF9800', fg='white', font=('Arial', 11, 'bold'),
                 width=12, height=2).grid(row=0, column=2, padx=5)
        
        tk.Button(button_frame, text="⏭ Skip Class", command=self.action_skip_class,
                 bg='#2196F3', fg='white', font=('Arial', 11, 'bold'),
                 width=12, height=2).grid(row=0, column=3, padx=5)
        
        tk.Button(button_frame, text="❌ Quit", command=self.action_quit,
                 bg='#9E9E9E', fg='white', font=('Arial', 11, 'bold'),
                 width=12, height=2).grid(row=0, column=4, padx=5)
        
        # Summary label
        self.summary_label = tk.Label(button_frame, text="", font=('Arial', 9), 
                                      bg='#f0f0f0', fg='#333')
        self.summary_label.grid(row=1, column=0, columnspan=5, pady=10)
        
        # Bind keyboard shortcuts
        self.root.bind('k', lambda e: self.action_keep())
        self.root.bind('d', lambda e: self.action_delete())
        self.root.bind('r', lambda e: self.action_relabel())
        self.root.bind('s', lambda e: self.action_skip_class())
        self.root.bind('q', lambda e: self.action_quit())
        self.root.bind('<Right>', lambda e: self.action_keep())
        self.root.bind('<Delete>', lambda e: self.action_delete())
    
    def load_image(self):
        """Load and display the current image"""
        if self.current_index >= len(self.all_images):
            self.finish_review()
            return
        
        cls, img_data = self.all_images[self.current_index]
        
        # Check if we should skip this class
        if self.skip_current_class:
            # Find next image from different class
            current_class = cls
            while self.current_index < len(self.all_images):
                cls, img_data = self.all_images[self.current_index]
                if cls != current_class:
                    self.skip_current_class = False
                    break
                self.current_index += 1
            
            if self.current_index >= len(self.all_images):
                self.finish_review()
                return
        
        # Update labels
        self.class_label.config(text=f"Class: {cls.upper()}")
        self.progress_label.config(
            text=f"Image {self.current_index + 1} of {len(self.all_images)}"
        )
        
        # Load and display image
        try:
            img = Image.open(img_data['path'])
            
            # Resize to fit window while maintaining aspect ratio
            max_size = (900, 500)
            img.thumbnail(max_size, Image.Resampling.LANCZOS)
            
            photo = ImageTk.PhotoImage(img)
            self.image_label.config(image=photo)
            self.image_label.image = photo  # Keep a reference
            
        except Exception as e:
            self.image_label.config(text=f"Error loading image:\n{str(e)}", 
                                   fg='red', font=('Arial', 12))
        
        # Update stats
        stats_text = (
            f"Mean Loss: {img_data['mean_loss']:.4f}  |  "
            f"Max Loss: {img_data['max_loss']:.4f}  |  "
            f"Min Loss: {img_data['min_loss']:.4f}  |  "
            f"Std Loss: {img_data['std_loss']:.4f}\n"
            f"Losses by Epoch: {', '.join([f'{l:.3f}' for l in img_data['losses_by_epoch']])}"
        )
        self.stats_label.config(text=stats_text)
        self.path_label.config(text=f"Path: {img_data['path']}")
        
        # Update summary
        summary = (
            f"Marked for deletion: {len(self.deletion_list)}  |  "
            f"Marked for relabeling: {len(self.relabel_list)}  |  "
            f"Kept: {len(self.kept_list)}"
        )
        self.summary_label.config(text=summary)
    
    def next_image(self):
        """Move to next image"""
        self.current_index += 1
        self.load_image()
    
    def action_keep(self):
        """Mark current image as keep"""
        if self.current_index < len(self.all_images):
            cls, img_data = self.all_images[self.current_index]
            self.kept_list.append((cls, img_data['path']))
            print(f"✓ Kept: {img_data['path']}")
            self.next_image()
    
    def action_delete(self):
        """Mark current image for deletion"""
        if self.current_index < len(self.all_images):
            cls, img_data = self.all_images[self.current_index]
            self.deletion_list.append((cls, img_data['path']))
            print(f"🗑 Marked for deletion: {img_data['path']}")
            self.next_image()
    
    def action_relabel(self):
        """Mark current image for relabeling"""
        if self.current_index < len(self.all_images):
            cls, img_data = self.all_images[self.current_index]
            self.relabel_list.append((cls, img_data['path']))
            print(f"🏷 Marked for relabeling: {img_data['path']}")
            self.next_image()
    
    def action_skip_class(self):
        """Skip remaining images in current class"""
        if self.current_index < len(self.all_images):
            cls, img_data = self.all_images[self.current_index]
            print(f"⏭ Skipping remaining images in class: {cls}")
            self.skip_current_class = True
            self.next_image()
    
    def action_quit(self):
        """Quit the review"""
        if messagebox.askyesno("Quit Review", 
                              "Are you sure you want to quit?\nAny pending actions will still be processed."):
            self.finish_review()
    
    def finish_review(self):
        """Complete the review and handle deletions"""
        print("\n" + "=" * 80)
        print("REVIEW COMPLETE!")
        print("=" * 80)
        
        # Show summary
        print(f"\n📊 Summary:")
        print(f"   Images reviewed: {self.current_index}")
        print(f"   Kept: {len(self.kept_list)}")
        print(f"   Marked for relabeling: {len(self.relabel_list)}")
        print(f"   Marked for deletion: {len(self.deletion_list)}")
        
        # Handle relabeling list
        if self.relabel_list:
            print(f"\n🏷 Images marked for relabeling:")
            for cls, path in self.relabel_list:
                print(f"   - [{cls}] {path}")
            print("\n   Note: Please manually relabel these images.")
        
        # Handle deletion list
        if self.deletion_list:
            print(f"\n🗑 Images marked for deletion:")
            for cls, path in self.deletion_list:
                print(f"   - [{cls}] {path}")
            
            # Show confirmation dialog
            message = (
                f"You have marked {len(self.deletion_list)} image(s) for deletion.\n\n"
                f"Do you want to permanently delete these files?\n\n"
                f"⚠️ This action cannot be undone!"
            )
            
            if messagebox.askyesno("Confirm Deletion", message, icon='warning'):
                self.perform_deletions()
            else:
                print("\n❌ Deletion cancelled. Files were NOT deleted.")
                print("   Files that were marked for deletion:")
                for cls, path in self.deletion_list:
                    print(f"   - [{cls}] {path}")
        
        self.root.destroy()
    
    def perform_deletions(self):
        """Actually delete the files in the deletion list"""
        print("\n🗑 Deleting files...")
        deleted_count = 0
        failed_count = 0
        
        for cls, path in self.deletion_list:
            try:
                if os.path.exists(path):
                    os.remove(path)
                    print(f"   ✓ Deleted: {path}")
                    deleted_count += 1
                else:
                    print(f"   ⚠️  File not found: {path}")
                    failed_count += 1
            except Exception as e:
                print(f"   ❌ Failed to delete {path}: {e}")
                failed_count += 1
        
        print(f"\n✓ Deletion complete!")
        print(f"   Successfully deleted: {deleted_count}")
        if failed_count > 0:
            print(f"   Failed: {failed_count}")
        
        messagebox.showinfo("Deletion Complete", 
                          f"Successfully deleted {deleted_count} file(s).\n"
                          f"Failed: {failed_count}")
    
    def run(self):
        """Start the GUI"""
        self.root.mainloop()


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
