"""
Data Collection Utilities

This package contains utilities for collecting and managing the image dataset:

Data Collection:
- download_images.py: Download images from various sources (DuckDuckGo, etc.)
- search_queries.py: Define search queries for each class

Duplicate Detection (3 methods):
- detect_duplicates_by_filesize.py: Detect exact duplicates by file size (fastest)
- detect_duplicates_by_phash.py: Detect near-duplicates using perceptual hashing (recommended before training)
- detect_duplicates_by_embedding.py: Detect semantic duplicates using model embeddings (use after training)

Duplicate Management:
- remove_duplicates.py: Remove duplicate images with various strategies
"""

__version__ = '1.0.0'
