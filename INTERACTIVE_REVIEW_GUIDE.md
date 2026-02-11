# Interactive Review Mode - GUI Guide

## Overview

The interactive review mode has been upgraded from a terminal-based interface to a Windows GUI application. This provides a better user experience with visual image display and button-based actions.

## Features

### Visual Interface
- **Image Display**: Shows the problematic image directly in the window (resized to fit)
- **Statistics Panel**: Displays loss metrics (mean, max, min, std) and per-epoch losses
- **Progress Tracking**: Shows current image number and class being reviewed
- **Action Summary**: Real-time counter of kept, deleted, and relabeled images

### Action Buttons

Five action buttons are available:

1. **✓ Keep** (Green) - Mark the image as okay, no action needed
   - Keyboard shortcut: `K` or `Right Arrow`

2. **🗑 Delete** (Red) - Mark the image for deletion
   - Keyboard shortcut: `D` or `Delete`

3. **🏷 Relabel** (Orange) - Mark the image for manual relabeling
   - Keyboard shortcut: `R`

4. **⏭ Skip Class** (Blue) - Skip all remaining images in the current class
   - Keyboard shortcut: `S`

5. **❌ Quit** (Gray) - Exit the review (with confirmation)
   - Keyboard shortcut: `Q`

### Deletion Workflow

The deletion feature now includes proper safety measures:

1. **Accumulation**: Images marked for deletion are added to a list during review
2. **Summary**: At the end of review, all marked images are listed
3. **Confirmation Dialog**: A popup asks for final confirmation before any deletion
4. **Execution**: Only after confirmation are files actually deleted
5. **Feedback**: Success/failure status for each deletion is shown

## Usage

### Basic Usage

Review all problematic images with GUI:
```bash
python review_problematic_images.py --mode interactive
```

### Review Specific Class

Review only images from one class:
```bash
python review_problematic_images.py --mode interactive --class bird
```

### Other Options

The script still supports the original modes:

- **Grid View**: `--mode grid` - Shows image grid with matplotlib
- **Trajectory View**: `--mode trajectory` - Shows loss trajectory plots
- **Summary Only**: `--mode summary` - Text summary without images

## Technical Details

### Dependencies

The GUI uses Python's built-in `tkinter` library, which is included with standard Python installations on Windows. No additional installation required.

### File Structure

- **ImageReviewerGUI Class**: Encapsulates all GUI logic
  - `setup_ui()`: Creates the window layout and buttons
  - `load_image()`: Loads and displays current image
  - `action_*()`: Handler methods for each button
  - `finish_review()`: Handles end-of-review summary and deletion
  - `perform_deletions()`: Executes actual file deletions

### Lists Maintained

During review, the following lists are maintained:

- `deletion_list`: Files marked for deletion
- `relabel_list`: Files marked for relabeling (manual action required)
- `kept_list`: Files explicitly marked as okay

## Safety Features

1. **Non-destructive until confirmed**: Files are only deleted after explicit confirmation
2. **Cancel option**: User can cancel deletion even after reviewing all images
3. **Detailed logging**: All actions are logged to console for record-keeping
4. **Error handling**: If a deletion fails, it's reported without crashing
5. **Quit confirmation**: Quitting mid-review requires confirmation

## Tips

- Use keyboard shortcuts for faster navigation
- The "Keep" button can be used to just move to the next image
- Console output provides a permanent log of all actions
- Images marked for relabeling are listed at the end but not automatically processed
- You can quit at any time and still get the deletion confirmation dialog

## Example Workflow

1. Run: `python review_problematic_images.py --mode interactive`
2. Review each image using the GUI buttons
3. Mark problematic images for deletion or relabeling
4. Complete the review (or quit early)
5. Review the summary of actions
6. Confirm (or cancel) the deletion of marked files
7. Check console output for detailed logs
