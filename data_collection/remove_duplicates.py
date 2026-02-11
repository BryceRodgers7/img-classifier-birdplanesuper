"""
Duplicate Image Remover

Reads the duplicate report and removes duplicate images based on user-selected strategy.
Provides multiple strategies for deciding which duplicate to keep when files are not in 
the same exact class + set.

Usage:
    python data_collection/remove_duplicates.py
    
Requires:
    dataset/duplicate_report.json (created by detect_duplicates.py)
"""

import os
import json
from pathlib import Path
from datetime import datetime


class DuplicateRemover:
    """Handles removal of duplicate images with various strategies."""
    
    def __init__(self, report_path):
        """Initialize with a duplicate report."""
        self.report_path = Path(report_path)
        
        if not self.report_path.exists():
            raise FileNotFoundError(f"Report not found: {report_path}")
        
        with open(self.report_path, 'r', encoding='utf-8') as f:
            self.report = json.load(f)
    
    def get_available_strategies(self):
        """Return list of available deletion strategies."""
        return [
            {
                'id': 'same_only',
                'name': 'Safe Delete Only (Recommended)',
                'description': 'Only delete exact duplicates in same class+set. No risk.'
            },
            {
                'id': 'keep_train',
                'name': 'Prefer Training Set',
                'description': 'Keep training set copies, delete validation set copies (fixes data leakage).'
            },
            {
                'id': 'keep_val',
                'name': 'Prefer Validation Set',
                'description': 'Keep validation set copies, delete training set copies.'
            },
            {
                'id': 'keep_first_alphabetical',
                'name': 'Keep First Alphabetically',
                'description': 'Keep the file that comes first alphabetically, delete others.'
            },
            {
                'id': 'keep_shortest_path',
                'name': 'Keep Shortest Path',
                'description': 'Keep file with shortest path name, delete others.'
            },
            {
                'id': 'interactive',
                'name': 'Interactive Mode',
                'description': 'Review each duplicate group and decide manually.'
            },
            {
                'id': 'delete_all_duplicates',
                'name': 'Delete All Duplicates (Dangerous)',
                'description': 'Delete ALL duplicate files, keeping only one per group. Use with caution!'
            }
        ]
    
    def apply_strategy_same_only(self, dry_run=True):
        """Delete only exact duplicates (same class and set)."""
        groups = self.report['duplicate_groups']['same_class_and_set']
        deleted = []
        
        print(f"\n🔍 Processing {len(groups)} groups of exact duplicates...")
        
        for group in groups:
            files = group['files']
            # Keep first file, delete rest
            keep_file = files[0]
            delete_files = files[1:]
            
            print(f"\n  Group (size: {group['size_bytes']} bytes, {len(files)} files):")
            print(f"    ✅ Keep: {keep_file['path']}")
            
            for file_info in delete_files:
                print(f"    🗑️  Delete: {file_info['path']}")
                if not dry_run:
                    try:
                        os.remove(file_info['path'])
                        deleted.append(file_info['path'])
                    except Exception as e:
                        print(f"       ⚠️  Error deleting: {e}")
        
        return deleted
    
    def apply_strategy_keep_train(self, dry_run=True):
        """Keep training set copies, delete validation set copies."""
        all_groups = []
        for category in ['same_class_diff_set', 'diff_class_diff_set', 'diff_class_same_set', 'same_class_and_set']:
            all_groups.extend(self.report['duplicate_groups'][category])
        
        deleted = []
        
        print(f"\n🔍 Processing {len(all_groups)} duplicate groups...")
        print("   Strategy: Keep training set, delete validation set\n")
        
        for group in all_groups:
            files = group['files']
            
            # Separate by set
            train_files = [f for f in files if f['set'] == 'train']
            val_files = [f for f in files if f['set'] == 'val']
            other_files = [f for f in files if f['set'] not in ['train', 'val']]
            
            print(f"  Group (size: {group['size_bytes']} bytes, {len(files)} files):")
            
            # Keep train files
            if train_files:
                print(f"    ✅ Keep (train): {train_files[0]['path']}")
                for f in train_files[1:]:
                    print(f"       ✅ Keep (train): {f['path']}")
            
            # Delete val files
            for file_info in val_files:
                print(f"    🗑️  Delete (val): {file_info['path']}")
                if not dry_run:
                    try:
                        os.remove(file_info['path'])
                        deleted.append(file_info['path'])
                    except Exception as e:
                        print(f"       ⚠️  Error deleting: {e}")
            
            # Keep other files
            for file_info in other_files:
                print(f"    ✅ Keep (unknown set): {file_info['path']}")
        
        return deleted
    
    def apply_strategy_keep_val(self, dry_run=True):
        """Keep validation set copies, delete training set copies."""
        all_groups = []
        for category in ['same_class_diff_set', 'diff_class_diff_set', 'diff_class_same_set', 'same_class_and_set']:
            all_groups.extend(self.report['duplicate_groups'][category])
        
        deleted = []
        
        print(f"\n🔍 Processing {len(all_groups)} duplicate groups...")
        print("   Strategy: Keep validation set, delete training set\n")
        
        for group in all_groups:
            files = group['files']
            
            # Separate by set
            train_files = [f for f in files if f['set'] == 'train']
            val_files = [f for f in files if f['set'] == 'val']
            other_files = [f for f in files if f['set'] not in ['train', 'val']]
            
            print(f"  Group (size: {group['size_bytes']} bytes, {len(files)} files):")
            
            # Keep val files
            if val_files:
                print(f"    ✅ Keep (val): {val_files[0]['path']}")
                for f in val_files[1:]:
                    print(f"       ✅ Keep (val): {f['path']}")
            
            # Delete train files
            for file_info in train_files:
                print(f"    🗑️  Delete (train): {file_info['path']}")
                if not dry_run:
                    try:
                        os.remove(file_info['path'])
                        deleted.append(file_info['path'])
                    except Exception as e:
                        print(f"       ⚠️  Error deleting: {e}")
            
            # Keep other files
            for file_info in other_files:
                print(f"    ✅ Keep (unknown set): {file_info['path']}")
        
        return deleted
    
    def apply_strategy_keep_first_alphabetical(self, dry_run=True):
        """Keep the file that comes first alphabetically."""
        all_groups = []
        for category in self.report['duplicate_groups'].values():
            all_groups.extend(category)
        
        deleted = []
        
        print(f"\n🔍 Processing {len(all_groups)} duplicate groups...")
        print("   Strategy: Keep first alphabetically\n")
        
        for group in all_groups:
            files = sorted(group['files'], key=lambda f: f['path'])
            keep_file = files[0]
            delete_files = files[1:]
            
            print(f"  Group (size: {group['size_bytes']} bytes, {len(files)} files):")
            print(f"    ✅ Keep: {keep_file['path']}")
            
            for file_info in delete_files:
                print(f"    🗑️  Delete: {file_info['path']}")
                if not dry_run:
                    try:
                        os.remove(file_info['path'])
                        deleted.append(file_info['path'])
                    except Exception as e:
                        print(f"       ⚠️  Error deleting: {e}")
        
        return deleted
    
    def apply_strategy_keep_shortest_path(self, dry_run=True):
        """Keep the file with the shortest path."""
        all_groups = []
        for category in self.report['duplicate_groups'].values():
            all_groups.extend(category)
        
        deleted = []
        
        print(f"\n🔍 Processing {len(all_groups)} duplicate groups...")
        print("   Strategy: Keep shortest path\n")
        
        for group in all_groups:
            files = sorted(group['files'], key=lambda f: len(f['path']))
            keep_file = files[0]
            delete_files = files[1:]
            
            print(f"  Group (size: {group['size_bytes']} bytes, {len(files)} files):")
            print(f"    ✅ Keep: {keep_file['path']}")
            
            for file_info in delete_files:
                print(f"    🗑️  Delete: {file_info['path']}")
                if not dry_run:
                    try:
                        os.remove(file_info['path'])
                        deleted.append(file_info['path'])
                    except Exception as e:
                        print(f"       ⚠️  Error deleting: {e}")
        
        return deleted
    
    def apply_strategy_interactive(self, dry_run=True):
        """Interactive mode - user decides for each group."""
        all_groups = []
        for category in self.report['duplicate_groups'].values():
            all_groups.extend(category)
        
        deleted = []
        
        print(f"\n🔍 Interactive mode: {len(all_groups)} duplicate groups to review")
        print("   For each group, you'll choose which file(s) to keep\n")
        
        for i, group in enumerate(all_groups, 1):
            files = group['files']
            
            print(f"\n{'=' * 70}")
            print(f"Group {i}/{len(all_groups)} - Size: {group['size_bytes']} bytes, {len(files)} files")
            print('=' * 70)
            
            for j, file_info in enumerate(files, 1):
                print(f"  [{j}] {file_info['path']}")
                print(f"      Class: {file_info['class']}, Set: {file_info['set']}")
            
            print("\nOptions:")
            print("  - Enter number(s) to KEEP (e.g., '1' or '1,3' for multiple)")
            print("  - Enter 'a' to keep all (skip this group)")
            print("  - Enter 'q' to quit interactive mode")
            
            while True:
                choice = input("\n  Your choice: ").strip().lower()
                
                if choice == 'q':
                    print("\n❌ Exiting interactive mode.")
                    return deleted
                
                if choice == 'a':
                    print("  → Keeping all files in this group")
                    break
                
                try:
                    # Parse selection
                    keep_indices = [int(x.strip()) - 1 for x in choice.split(',')]
                    
                    # Validate indices
                    if not all(0 <= idx < len(files) for idx in keep_indices):
                        print("  ⚠️  Invalid selection. Try again.")
                        continue
                    
                    # Delete non-selected files
                    for j, file_info in enumerate(files):
                        if j in keep_indices:
                            print(f"  ✅ Keeping: {file_info['path']}")
                        else:
                            print(f"  🗑️  Deleting: {file_info['path']}")
                            if not dry_run:
                                try:
                                    os.remove(file_info['path'])
                                    deleted.append(file_info['path'])
                                except Exception as e:
                                    print(f"     ⚠️  Error deleting: {e}")
                    
                    break
                    
                except ValueError:
                    print("  ⚠️  Invalid input. Enter numbers separated by commas, 'a' for all, or 'q' to quit.")
        
        return deleted
    
    def apply_strategy_delete_all_duplicates(self, dry_run=True):
        """Delete all duplicates, keeping only one per group (dangerous!)."""
        all_groups = []
        for category in self.report['duplicate_groups'].values():
            all_groups.extend(category)
        
        deleted = []
        
        print(f"\n⚠️  WARNING: This will delete ALL duplicates!")
        print(f"   Processing {len(all_groups)} duplicate groups...\n")
        
        for group in all_groups:
            files = group['files']
            keep_file = files[0]
            delete_files = files[1:]
            
            print(f"  Group (size: {group['size_bytes']} bytes, {len(files)} files):")
            print(f"    ✅ Keep: {keep_file['path']} ({keep_file['class']}/{keep_file['set']})")
            
            for file_info in delete_files:
                print(f"    🗑️  Delete: {file_info['path']} ({file_info['class']}/{file_info['set']})")
                if not dry_run:
                    try:
                        os.remove(file_info['path'])
                        deleted.append(file_info['path'])
                    except Exception as e:
                        print(f"       ⚠️  Error deleting: {e}")
        
        return deleted
    
    def apply_strategy(self, strategy_id, dry_run=True):
        """Apply a specific deletion strategy."""
        strategies = {
            'same_only': self.apply_strategy_same_only,
            'keep_train': self.apply_strategy_keep_train,
            'keep_val': self.apply_strategy_keep_val,
            'keep_first_alphabetical': self.apply_strategy_keep_first_alphabetical,
            'keep_shortest_path': self.apply_strategy_keep_shortest_path,
            'interactive': self.apply_strategy_interactive,
            'delete_all_duplicates': self.apply_strategy_delete_all_duplicates,
        }
        
        if strategy_id not in strategies:
            raise ValueError(f"Unknown strategy: {strategy_id}")
        
        return strategies[strategy_id](dry_run=dry_run)


def main():
    """Main function."""
    print("=" * 70)
    print("🗑️  Duplicate Image Remover")
    print("=" * 70)
    
    # Get report path
    script_dir = Path(__file__).parent
    dataset_dir = script_dir.parent / 'dataset'
    report_path = dataset_dir / 'duplicate_report.json'
    
    print(f"\n📁 Report path: {report_path}")
    
    # Check if report exists
    if not report_path.exists():
        print("\n❌ Error: duplicate_report.json not found!")
        print("   Please run detect_duplicates.py first:")
        print("   python data_collection/detect_duplicates.py")
        return
    
    # Load report
    try:
        remover = DuplicateRemover(report_path)
    except Exception as e:
        print(f"\n❌ Error loading report: {e}")
        return
    
    # Show summary
    summary = remover.report['summary']
    print(f"\n📊 Duplicate Summary:")
    print(f"   Total duplicate files: {summary['total_duplicate_files']}")
    print(f"   Total duplicate groups: {summary['total_duplicate_groups']}")
    
    # Show available strategies
    strategies = remover.get_available_strategies()
    
    print("\n🎯 Available Deletion Strategies:")
    print("=" * 70)
    for i, strategy in enumerate(strategies, 1):
        print(f"\n{i}. {strategy['name']}")
        print(f"   {strategy['description']}")
    
    print("\n" + "=" * 70)
    
    # Get user selection
    while True:
        try:
            choice = input("\nSelect strategy (1-7) or 'q' to quit: ").strip()
            
            if choice.lower() == 'q':
                print("❌ Cancelled.")
                return
            
            choice_num = int(choice)
            if 1 <= choice_num <= len(strategies):
                selected_strategy = strategies[choice_num - 1]
                break
            else:
                print(f"⚠️  Please enter a number between 1 and {len(strategies)}")
        except ValueError:
            print("⚠️  Please enter a valid number or 'q' to quit")
    
    print(f"\n✅ Selected: {selected_strategy['name']}")
    
    # Dry run first
    print("\n" + "=" * 70)
    print("🔍 DRY RUN (no files will be deleted)")
    print("=" * 70)
    
    deleted = remover.apply_strategy(selected_strategy['id'], dry_run=True)
    
    print("\n" + "=" * 70)
    print("📊 DRY RUN COMPLETE")
    print("=" * 70)
    print(f"\nWould delete {len(deleted)} files")
    
    # Confirm actual deletion
    print("\n⚠️  This will permanently delete files!")
    confirm = input("Proceed with actual deletion? (yes/no): ").strip().lower()
    
    if confirm != 'yes':
        print("❌ Cancelled. No files were deleted.")
        return
    
    # Actual deletion
    print("\n" + "=" * 70)
    print("🗑️  DELETING FILES")
    print("=" * 70)
    
    deleted = remover.apply_strategy(selected_strategy['id'], dry_run=False)
    
    print("\n" + "=" * 70)
    print("✅ DELETION COMPLETE")
    print("=" * 70)
    print(f"\n📊 Deleted {len(deleted)} files")
    
    # Save deletion log
    log_path = dataset_dir / f'deletion_log_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    log_data = {
        'deletion_date': datetime.now().isoformat(),
        'strategy': selected_strategy['name'],
        'deleted_files': deleted
    }
    
    with open(log_path, 'w', encoding='utf-8') as f:
        json.dump(log_data, f, indent=2)
    
    print(f"\n💾 Deletion log saved to: {log_path}")
    
    print("\n🔧 Recommended next step:")
    print("   python data_collection/detect_duplicates.py")
    print("   (Re-run to verify all duplicates are removed)")
    
    print("\n" + "=" * 70)


if __name__ == '__main__':
    main()
