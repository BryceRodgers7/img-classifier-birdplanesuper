# Bird/Plane/Superman Image Classifier

A PyTorch-based image classifier that distinguishes between birds, planes, Superman, and other objects using transfer learning with ResNet50.

## Features

- **Pre-trained ResNet50** backbone with fine-tuned layers
- **Weighted sampling** to handle class imbalance
- **Class-specific augmentations** (e.g., no vertical flips for birds/planes)
- **Confidence thresholding** for robust predictions
- **Comprehensive metrics tracking** (precision, recall, F1 per class)
- **Error analysis** - saves top-10 misclassified images per class per epoch
- **Mislabeled image detection** - per-image loss tracking to identify problematic training data
- **Standalone classifier** module for easy deployment

## Project Structure

```
img_classifier_birdplanesuper/
├── data_collection/
│   ├── __init__.py
│   ├── download_images.py                # Automated image downloading
│   ├── search_queries.py                 # Search queries for each class
│   ├── detect_duplicates_by_filesize.py  # Detect exact duplicates (file size)
│   ├── detect_duplicates_by_phash.py     # Detect near-duplicates (perceptual hash)
│   ├── detect_duplicates_by_embedding.py # Detect semantic duplicates (model embeddings)
│   └── remove_duplicates.py              # Remove duplicate images
├── train_classifier.py                   # Main training script
├── classifier.py                         # Standalone inference module
├── review_problematic_images.py          # Visual review tool for problematic images
├── requirements.txt                      # Python dependencies
├── USAGE_EXAMPLES.md                     # Detailed usage examples
├── dataset/                              # Training data (created by download_images.py)
│   ├── train/
│   │   ├── bird/
│   │   ├── plane/
│   │   ├── superman/
│   │   └── other/
│   ├── val/
│   │   ├── bird/
│   │   ├── plane/
│   │   ├── superman/
│   │   └── other/
│   ├── duplicate_report_filesize.json    # (Optional) Created by detect_duplicates_by_filesize.py
│   ├── duplicate_report_phash.json       # (Optional) Created by detect_duplicates_by_phash.py
│   ├── duplicate_report_embedding.json   # (Optional) Created by detect_duplicates_by_embedding.py
│   ├── problematic_images_report.json    # (Optional) Created by train_classifier.py --detect-problematic
│   └── deletion_log_*.json               # (Optional) Created by remove_duplicates.py
└── models/                               # Created during training
    ├── best_model.pth
    ├── training_metadata.json
    └── error_analysis/
        ├── epoch_01/
        ├── epoch_02/
        └── ...
```

## Installation

1. Clone or download this repository
2. Install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

### Step 1: Download Training Images

Download images for all 4 classes (bird, plane, superman, other):

```bash
python data_collection/download_images.py --duckduckgo
```

This will:
- Search for ~2000 images across all categories
- Download and organize them into train/val splits (80/20)
- Save to `dataset/train/` and `dataset/val/`

### Step 2: (Optional) Detect and Remove Duplicates

After downloading images, you may want to check for and remove duplicates. There are three methods available:

#### Method 1: File Size (Fastest - Exact Duplicates Only)

```bash
python data_collection/detect_duplicates_by_filesize.py
```

- **Speed**: ⚡⚡⚡ Instant
- **Detects**: Exact byte-for-byte duplicates only
- **Output**: `dataset/duplicate_report_filesize.json`
- **Best for**: Quick initial cleanup

#### Method 2: Perceptual Hash (Recommended - Before Training)

```bash
python data_collection/detect_duplicates_by_phash.py
```

- **Speed**: ⚡⚡ Fast (~1 second per image)
- **Detects**: Near-duplicates (different compression, formats, minor edits)
- **Output**: `dataset/duplicate_report_phash.json`
- **Best for**: Pre-training dataset cleanup
- **Adjustable threshold**: Hamming distance 0-64 (default: 5)

#### Method 3: Model Embeddings (Advanced - After Training)

```bash
python data_collection/detect_duplicates_by_embedding.py
```

- **Speed**: ⚡ Slower (requires GPU for efficiency)
- **Detects**: Semantically similar images (model thinks they're similar)
- **Output**: `dataset/duplicate_report_embedding.json`
- **Best for**: Finding images that confuse the model
- **Requires**: Trained model at `models/best_model.pth`
- **Adjustable threshold**: Cosine similarity 0-1 (default: 0.95)

**Comparison:**

| Method | Speed | Finds | Use Case |
|--------|-------|-------|----------|
| File Size | ⚡⚡⚡ Instant | Exact copies | Quick check |
| Perceptual Hash | ⚡⚡ Fast | Near-duplicates | **Pre-training cleanup** |
| Embeddings | ⚡ Slower | Semantic similarity | Post-training analysis |

**Then remove duplicates:**

```bash
# Interactive removal with multiple strategies
python data_collection/remove_duplicates.py
```

Note: You'll need to modify `remove_duplicates.py` to read the specific report file you want to use (e.g., `duplicate_report_phash.json` or `duplicate_report_embedding.json`).

**Available Deletion Strategies:**

1. **Safe Delete Only (Recommended)**: Only deletes exact duplicates in same class+set
2. **Prefer Training Set**: Keeps training copies, deletes validation copies (fixes data leakage)
3. **Prefer Validation Set**: Keeps validation copies, deletes training copies
4. **Keep First Alphabetically**: Keeps file that comes first alphabetically
5. **Keep Shortest Path**: Keeps file with shortest path name
6. **Interactive Mode**: Review and decide for each duplicate group manually
7. **Delete All Duplicates**: Deletes all duplicates keeping only one per group (use with caution)

The removal tool always performs a dry run first to show what will be deleted before making actual changes.

### Step 3: (Optional) Detect Mislabeled/Problematic Training Images

Before investing time in full training, you can run a quick analysis to identify potentially mislabeled, ambiguous, or non-representative images in your training set:

```bash
python train_classifier.py \
    --detect-problematic \
    --analysis-epochs 3 \
    --top-n-problematic 50 \
    --stop-after-analysis
```

**How it works:**
1. Trains for N epochs (default: 3) while tracking loss for each individual image
2. Identifies images with consistently high loss across these epochs
3. Reports the top N highest-loss images per class (default: 50)
4. Saves a detailed JSON report to `dataset/problematic_images_report.json`

**Problematic Image Detection Parameters:**
- `--detect-problematic`: Enable per-image loss tracking (toggleable feature)
- `--analysis-epochs`: Number of epochs to track losses (default: 3)
- `--top-n-problematic`: Number of high-loss images to report per class (default: 50)
- `--stop-after-analysis`: Stop training after analysis (don't continue to full training)

**Why use this?**
- **Mislabeled images**: Images labeled as "bird" but actually containing planes will have high loss
- **Ambiguous images**: Images that could fit multiple classes (e.g., bird-shaped plane)
- **Non-representative images**: Poor quality, extreme angles, unusual perspectives
- **Save training time**: Identify and fix issues before committing to full 15-epoch training

**Output:**
- `dataset/problematic_images_report.json` - Detailed report with:
  - Top N highest-loss images per class
  - Mean, max, min, and std deviation of loss per image
  - Loss values for each tracked epoch
  - Summary statistics by class

**Next steps after analysis:**
1. Review the flagged images manually (see review tool below)
2. Remove or relabel problematic images
3. Run the analysis again to verify improvements
4. Proceed with full training on the cleaned dataset

#### Visual Review Tool

Use the included review script to visually inspect problematic images:

```bash
# Display all problematic images in a grid (default)
python review_problematic_images.py

# Display only bird class images
python review_problematic_images.py --class bird

# Show top 10 images per class
python review_problematic_images.py --limit 10

# Show loss trajectories across epochs
python review_problematic_images.py --mode trajectory

# Interactive step-by-step review
python review_problematic_images.py --mode interactive

# Text summary only
python review_problematic_images.py --mode summary
```

The review tool provides:
- **Grid mode**: Visual grid of images with loss statistics and color-coded severity
- **Trajectory mode**: Line plots showing how loss changed across epochs
- **Interactive mode**: Step-by-step review with prompts to mark for deletion/relabeling
- **Summary mode**: Text-only statistics (no GUI)

### Step 4: Train the Classifier

Train the model with default settings:

```bash
python train_classifier.py
```

Or customize training parameters:

```bash
python train_classifier.py \
    --epochs 15 \
    --batch-size 32 \
    --learning-rate 0.0001 \
    --confidence-threshold 0.7 \
    --freeze-until layer3
```

You can also combine problematic image detection with full training (will continue training after analysis):

```bash
python train_classifier.py \
    --detect-problematic \
    --analysis-epochs 3 \
    --epochs 15
```

**Training Parameters:**
- `--epochs`: Number of training epochs (default: 15)
- `--batch-size`: Batch size for training (default: 32)
- `--learning-rate`: Learning rate (default: 0.0001)
- `--confidence-threshold`: Confidence threshold for predictions (default: 0.7)
- `--freeze-until`: Freeze layers up to this layer (choices: layer1, layer2, layer3, none)
- `--dataset-dir`: Path to dataset directory (default: dataset)
- `--output-dir`: Output directory for models (default: models)

**Training Features:**
- Weighted sampling automatically handles class imbalance
- Class-specific augmentations:
  - Bird/Plane: No vertical flips (they don't fly upside down)
  - Superman: Can include vertical flips (comic book poses)
  - Other: Full augmentation suite
- Saves best model based on validation F1 score
- Tracks per-class precision, recall, F1 for each epoch
- Saves top-10 highest-confidence errors per class for analysis
- Optional per-image loss tracking to identify mislabeled data

**Training Output:**
- `models/best_model.pth` - Best model checkpoint
- `models/training_metadata.json` - Complete training history
- `models/error_analysis/epoch_XX/` - Misclassified images per epoch
- `dataset/problematic_images_report.json` - (Optional) Problematic training images report

### Step 5: Use the Classifier

#### Python API

```python
from classifier import BirdPlaneSupermanClassifier

# Load model
classifier = BirdPlaneSupermanClassifier('models/best_model.pth', confidence_threshold=0.7)

# Predict single image
pred_class, confidence, probs = classifier.predict('test_image.jpg')
print(f"Predicted: {pred_class} ({confidence:.2%} confident)")
print(f"All probabilities: {probs}")

# Batch prediction (efficient for multiple images)
image_paths = ['img1.jpg', 'img2.jpg', 'img3.jpg']
results = classifier.predict_batch(image_paths)
for path, pred_class, confidence, probs in results:
    print(f"{path}: {pred_class} ({confidence:.2%})")

# Adjust confidence threshold
classifier.set_confidence_threshold(0.8)
```

#### Command Line

```bash
# Single image
python classifier.py models/best_model.pth test_image.jpg

# Entire folder
python classifier.py models/best_model.pth test_images/
```

## Model Architecture

**Base Model:** ResNet50 (pre-trained on ImageNet)

**Modifications:**
- Freeze early layers (conv1, bn1, layer1, layer2, layer3)
- Unfreeze layer4 and classifier head for fine-tuning
- Replace final FC layer: 2048 → 4 classes

**Why This Architecture?**
- ResNet50 provides strong feature extraction from ImageNet
- Freezing early layers preserves low-level features (edges, textures)
- Fine-tuning later layers adapts to our specific classes
- ~25M parameters, ~5M trainable

## Confidence Thresholding

The classifier uses confidence thresholding to handle uncertain predictions:

- If `max_probability < confidence_threshold` → predict **'other'**
- Otherwise → predict the class with highest probability

This prevents false positives when the model is unsure. The threshold (default: 0.7) can be adjusted based on your use case:
- **Higher threshold (0.8-0.9)**: More conservative, fewer false positives
- **Lower threshold (0.5-0.6)**: More aggressive, may have more false positives

## Error Analysis

After each epoch, the training script saves the top-10 highest-confidence misclassifications per class:

```json
{
  "image_path": "dataset/val/bird/img_0042.jpg",
  "true_label": "bird",
  "predicted_label": "plane",
  "confidence": 0.92,
  "all_probabilities": {
    "bird": 0.05,
    "plane": 0.92,
    "superman": 0.02,
    "other": 0.01
  }
}
```

**Use this to:**
1. Identify and remove bad training images
2. Understand model confusion patterns
3. Improve dataset quality
4. Adjust class definitions if needed

## Deployment to Other Projects

To use the trained classifier in another project:

1. Copy `classifier.py` to your new project
2. Copy `models/best_model.pth` to your new project
3. Install dependencies: `torch`, `torchvision`, `Pillow`, `numpy`
4. Use the classifier:

```python
from classifier import BirdPlaneSupermanClassifier

classifier = BirdPlaneSupermanClassifier('best_model.pth')
pred_class, confidence, probs = classifier.predict('image.jpg')
```

The `classifier.py` module is fully self-contained and has no dependencies on the training code.

## Training Tips

1. **Start with downloaded images**: The automated downloader gets you started quickly
2. **Check for duplicates BEFORE training**: 
   - Use **perceptual hash** detector (recommended for pre-training cleanup)
   - Catches near-duplicates, different compressions, format conversions
   - Prevents data leakage and labeling inconsistencies
3. **Review error analysis AFTER training**: Check `models/error_analysis/` to find problematic images
4. **Use embedding detector for advanced analysis**: 
   - Find semantically similar images that confuse the model
   - Identify edge cases where classes overlap
5. **Clean your dataset**: Remove mislabeled or ambiguous images
6. **Retrain**: After cleaning, retrain for better accuracy
7. **Adjust confidence threshold**: Based on validation performance and your use case

## Data Quality Best Practices

**Why Remove Duplicates?**

1. **Data Leakage**: Same/similar image in both train and val sets artificially inflates validation accuracy
2. **Labeling Errors**: Similar images labeled as different classes confuses the model
3. **Training Efficiency**: Duplicates waste training time and storage

**Three Detection Methods:**

1. **File Size** (Instant): Exact byte-for-byte copies only
2. **Perceptual Hash** (Fast): Near-duplicates, different compression/formats - **Use before training**
3. **Model Embeddings** (Advanced): Semantic similarity - **Use after training for analysis**

**Recommended Workflow:**

```bash
# 1. Download images
python data_collection/download_images.py --duckduckgo

# 2. Check for near-duplicates (RECOMMENDED before training)
python data_collection/detect_duplicates_by_phash.py

# 3. Remove duplicates (use "Prefer Training Set" to fix data leakage)
python data_collection/remove_duplicates.py

# 4. Train the model
python train_classifier.py

# 5. Check for semantic duplicates (optional - advanced analysis)
python data_collection/detect_duplicates_by_embedding.py

# 6. Review error analysis
# Check models/error_analysis/ for misclassified images

# 7. Manually remove bad images and retrain
python train_classifier.py
```

**When to Use Each Method:**

- **Before Training**: Use perceptual hash to catch near-duplicates and format variations
- **After Training**: Use embeddings to find images that confuse your specific model
- **Quick Check**: Use file size if you just want to find exact copies

## Metrics Interpretation

**Per-epoch metrics saved:**
- **Accuracy**: Overall correct predictions / total predictions
- **Precision**: True positives / (true positives + false positives) per class
- **Recall**: True positives / (true positives + false negatives) per class
- **F1 Score**: Harmonic mean of precision and recall
- **Confusion Matrix**: Shows which classes are confused with each other

**Best model selection:** Based on macro-averaged F1 score (equal weight to all classes)

## Advanced Usage

### Custom Dataset

If you have your own images, organize them like this:

```
dataset/
├── train/
│   ├── bird/
│   │   ├── img001.jpg
│   │   ├── img002.jpg
│   │   └── ...
│   ├── plane/
│   ├── superman/
│   └── other/
└── val/
    ├── bird/
    ├── plane/
    ├── superman/
    └── other/
```

Then run training as normal.

### Fine-tuning Hyperparameters

Experiment with different settings:

```bash
# More aggressive fine-tuning (unfreeze more layers)
python train_classifier.py --freeze-until layer2

# Longer training
python train_classifier.py --epochs 25

# Different learning rate
python train_classifier.py --learning-rate 0.0005

# Different confidence threshold (saved in metadata)
python train_classifier.py --confidence-threshold 0.8
```

## System Requirements

- Python 3.7+
- GPU recommended but not required (CPU training works but is slower)
- ~4GB RAM minimum
- ~2GB disk space for dataset + models

## Troubleshooting

**Out of memory during training:**
- Reduce batch size: `--batch-size 16`
- Use CPU instead of GPU (slower but uses less memory)

**Low accuracy on specific class:**
- Check error analysis for that class
- Add more diverse training images
- Review class-specific augmentations

**Model predicts 'other' too often:**
- Lower confidence threshold: `classifier.set_confidence_threshold(0.6)`
- Check if training data quality is good
- Ensure model trained long enough

**Model too confident on wrong predictions:**
- Review error analysis high-confidence errors
- Clean dataset of ambiguous images
- Increase confidence threshold

## License

This project is provided as-is for educational and research purposes.

## Credits

- PyTorch and torchvision for deep learning framework
- ResNet50 architecture from "Deep Residual Learning for Image Recognition" (He et al., 2015)
- Pre-trained weights from ImageNet
