"""
Automated Image Downloader for Training Dataset

Fully automated image downloading for training your classifier.
No manual downloading required - just run and go!

Methods supported:
1. Bing Image Search API (best quality, requires free Azure API key)
2. DuckDuckGo Search (no API key needed, fully automated)
3. Unsplash API (high quality, requires free API key)

Usage:
    # Bing (recommended if you have API key)
    python download_images.py --bing
    
    # DuckDuckGo (no API key needed!)
    python download_images.py --duckduckgo
    
    # Unsplash (highest quality)
    python download_images.py --unsplash
    
    # Auto-detect (tries available methods)
    python download_images.py --auto
"""

import requests
from pathlib import Path
import time
import json
from urllib.parse import quote, urlencode
import hashlib
from PIL import Image
import io
import os
import sys
import shutil
import random
from ddgs import DDGS
from search_queries import load_search_queries

class ImageDownloader:
    def __init__(self, output_dir):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
    
    def download_image(self, url, filename):
        """Download a single image from URL"""
        try:
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            
            # Verify it's actually an image
            img = Image.open(io.BytesIO(response.content))
            
            # Convert to RGB if needed
            if img.mode in ('RGBA', 'LA', 'P'):
                img = img.convert('RGB')
            
            # Save image
            img.save(filename, 'JPEG', quality=95)
            return True
            
        except Exception as e:
            print(f"  ⚠️  Failed to download {url}: {str(e)[:50]}")
            return False
    
    def download_from_urls_file(self, urls_file, category, query_name='manual'):
        """Download images from a text file containing URLs
        
        Args:
            urls_file: Path to file containing URLs (one per line)
            category: Category name (bird, plane, superman, other)
            query_name: Query name for filename (default: 'manual')
        """
        urls_path = Path(urls_file)
        if not urls_path.exists():
            print(f"❌ File not found: {urls_file}")
            return 0
        
        with open(urls_path, 'r') as f:
            urls = [line.strip() for line in f if line.strip()]
        
        print(f"\n📥 Downloading {len(urls)} images for '{category}'...")
        
        # Sanitize query name for filename
        query_sanitized = query_name.replace(' ', '_').replace('-', '_')
        query_sanitized = ''.join(c for c in query_sanitized if c.isalnum() or c == '_')
        
        downloaded = 0
        for i, url in enumerate(urls, 1):
            # Format: class_query_number.jpg
            filename = self.output_dir / f"{category}_{query_sanitized}_{i:03d}.jpg"
            
            if filename.exists():
                print(f"  ⏭️  Skipping (exists): {filename.name}")
                continue
            
            print(f"  [{i}/{len(urls)}] Downloading...", end='')
            if self.download_image(url, filename):
                downloaded += 1
                print(f" ✓ Saved as {filename.name}")
            else:
                print(f" ✗ Failed")
            
            # Be nice to servers
            time.sleep(0.5)
        
        return downloaded
    
    def search_duckduckgo(self, query, max_images=100):
        """
        Search DuckDuckGo for images (no API key needed!)
        Uses the ddgs library
        """
        print(f"\n🔍 Searching DuckDuckGo for '{query}'...")
        
        all_urls = []
        
        try:
            # Use DDGS library for reliable image search
            ddgs = DDGS()
            results = ddgs.images(
                query=query,
                max_results=max_images
            )
            
            for result in results:
                if 'image' in result:
                    all_urls.append(result['image'])
                elif 'url' in result:
                    all_urls.append(result['url'])
            
            print(f"  Found {len(all_urls)} images for '{query}'")
                    
        except Exception as e:
            print(f"  ⚠️  Search error: {str(e)}")
            print("  💡 Tip: If DuckDuckGo doesn't work, try Bing API (free at Azure)")
        
        return all_urls[:max_images]

def download_category_images(category, queries, method='duckduckgo', api_key=None, 
                             base_dir=None, train_pct=80, val_pct=20, images_per_query=100):
    """Download images for a single category using specified method"""
    
    print(f"\n{'='*70}")
    print(f"📥 Downloading {category.upper()} images")
    print(f"{'='*70}")
    
    # Create temporary download directory
    temp_dir = base_dir / 'temp' / category
    downloader = ImageDownloader(temp_dir)
    
    # Step 1: Collect URLs from ALL queries and download per query
    print(f"\n🔍 Searching for {images_per_query} images per query ({len(queries)} queries)...")
    print(f"   Will split into {train_pct}% for training and {val_pct}% for validation")
    
    # Store downloaded files with metadata (query info)
    downloaded_files_with_metadata = []
    
    for query_idx, query in enumerate(queries, 1):
        print(f"\n   Query {query_idx}/{len(queries)}: '{query}'")
        
        # Sanitize query for filename (replace spaces with underscores, remove special chars)
        query_sanitized = query.replace(' ', '_').replace('-', '_')
        query_sanitized = ''.join(c for c in query_sanitized if c.isalnum() or c == '_')
        
        if method == 'duckduckgo':
            urls = downloader.search_duckduckgo(query, images_per_query)
        else:
            print(f"❌ Unknown method: {method}")
            return 0
        
        print(f"   → Collected {len(urls)} URLs for query '{query}'")
        
        # Download images for this query
        print(f"   💾 Downloading {len(urls)} images for this query...")
        for img_idx, url in enumerate(urls, 1):
            # Format: class_query_number.jpg
            filename = temp_dir / f"{category}_{query_sanitized}_{img_idx:03d}.jpg"
            
            if downloader.download_image(url, filename):
                downloaded_files_with_metadata.append({
                    'file': filename,
                    'query': query,
                    'query_sanitized': query_sanitized
                })
            
            # Rate limiting
            time.sleep(0.5)
        
        print(f"   ✓ Downloaded {len([f for f in downloaded_files_with_metadata if query_sanitized in f['file'].name])} images for '{query}'")
    
    print(f"\n✅ Successfully downloaded {len(downloaded_files_with_metadata)} total images for {category}")
    
    # Step 2: Split into train and val sets
    print(f"\n📊 Splitting into {train_pct}% for training and {val_pct}% for validation")
    
    train_dir = base_dir / 'train' / category
    val_dir = base_dir / 'val' / category
    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)
    
    # Randomize order before splitting to avoid ordering bias
    random.shuffle(downloaded_files_with_metadata)
    
    # Split: first train_pct% images to train, rest to val
    split_idx = int(len(downloaded_files_with_metadata) * train_pct / 100)
    train_files = downloaded_files_with_metadata[:split_idx]
    val_files = downloaded_files_with_metadata[split_idx:]
    
    # Move images to train set (keeping original names)
    for file_info in train_files:
        src_file = file_info['file']
        dest_file = train_dir / src_file.name  # Keep original name
        shutil.move(str(src_file), str(dest_file))
    
    # Move images to val set (keeping original names)
    for file_info in val_files:
        src_file = file_info['file']
        dest_file = val_dir / src_file.name  # Keep original name
        shutil.move(str(src_file), str(dest_file))
    
    # Clean up temp directory
    if temp_dir.exists():
        shutil.rmtree(temp_dir)
    
    print(f"  ✓ Moved {len(train_files)} images to train set ({train_pct}%)")
    print(f"  ✓ Moved {len(val_files)} images to val set ({100-train_pct}%)")
    print(f"  ✓ Total in dataset: {len(train_files) + len(val_files)} images")
    
    return len(train_files) + len(val_files)


def main():
    """Main function"""
    print("=" * 70)
    print("🤖 FULLY AUTOMATED Image Downloader")
    print("=" * 70)
    print("\n✨ No manual downloading required - just choose your method!\n")
    
    base_dir = Path(__file__).parent.parent / 'dataset'
    
    # Define search queries for each category
    search_queries = load_search_queries()
    
    # Check command line arguments
    # args = sys.argv[1:]
    
    # Determine method
    method = None
    # bing_api_key = os.environ.get('BING_API_KEY')
    # unsplash_api_key = os.environ.get('UNSPLASH_API_KEY')
    
    method = 'duckduckgo'
    api_key = None
    print("🎯 Using DuckDuckGo (no API key needed!)")
    print("💡 Tip: For better results, get a free Bing API key")
    
    print(f"\n📁 Dataset directory: {base_dir}")
    print("\n⚙️  Download Strategy:")
    print("   1. Search ALL queries and collect URLs")
    print("   2. Download ALL available images from all queries")
    print("   3. Randomly sample best images for train/val splits")
    print("\n📊 Expected Downloads:")
    print("   Bird/Plane/Superman (each):")
    print("     - Search: 100 images/query × 5 queries = ~500 images")
    print("     - Final dataset: 100 train + 25 val = 125 images")
    print("   Other:")
    print("     - Search: 10 images/query × 50 queries = ~500 images")
    print("     - Final dataset: 100 train + 25 val = 125 images")
    print("\n   📥 Total images to download: ~2000")
    print("   💾 Final dataset size: 500 images (125 per category)")
    
    response = input("\n▶️  Start downloading? (y/n): ")
    if response.lower() != 'y':
        print("❌ Cancelled.")
        return
    
    print("\n" + "=" * 70)
    print("🚀 Starting automated download...")
    print("=" * 70)
    print("\n⏱️  This will download ~2000 images (could take 30-60 minutes)...")
    print("☕ Perfect time for a coffee break!\n")
    print("💡 Progress updates will show every 50 images downloaded\n")
    
    # Download images for each category
    start_time = time.time()
    total_downloaded = 0
    
    for category, queries in search_queries.items():
        # Set images_per_query based on category
        if category == 'other':
            images_per_query = 100  # 10 images per query for 'other'
        else:
            images_per_query = 500  # 100 images per query for bird/plane/superman
        
        downloaded = download_category_images(
            category=category,
            queries=queries,
            method=method,
            api_key=api_key,
            base_dir=base_dir,
            train_pct=80,
            val_pct=20,
            images_per_query=images_per_query
        )
        total_downloaded += downloaded
    
    elapsed = time.time() - start_time
    
    print("\n" + "=" * 70)
    print("✅ DOWNLOAD COMPLETE!")
    print("=" * 70)
    print(f"\n📊 Summary:")
    print(f"   - Time elapsed: {elapsed/60:.1f} minutes")
    print(f"   - Total images: {total_downloaded}")
    print(f"   - Location: {base_dir}")
    
    print("\n🎯 Next steps:")
    print("   1. Train the model:")
    print("      python train_classifier.py")
    print("\n   2. After training, analyze errors:")
    print("      python analyze_training_errors.py")
    print("\n   3. Review high-loss images and delete bad ones")
    print("\n   4. Retrain for better accuracy!")
    
    print("\n💡 Tip: Error analysis helps you identify and remove")
    print("   problematic images - this is the key to high accuracy!")
    print("\n" + "=" * 70)

if __name__ == '__main__':
    main()

