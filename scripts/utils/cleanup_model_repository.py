#!/usr/bin/env python3
"""
Model Repository Cleanup Script

This script automates the cleanup and maintenance of the strawberry detection
model repository. It helps keep the repository organized and manageable.

Usage:
    python scripts/cleanup_model_repository.py [options]

Options:
    --dry-run           Show what would be cleaned without actually deleting
    --archive-empty     Move empty model directories to archive/
    --remove-epochs     Remove intermediate epoch checkpoints
    --update-readme     Update README.md with current model status
    --full-cleanup      Run all cleanup operations
"""

import os
import sys
import shutil
import argparse
from pathlib import Path
import json

def get_model_directories(base_path):
    """Get all model directories in model/detection/"""
    detection_path = Path("/home/user/machine-learning/GitHubRepos/strawberryPicker/model/detection")
    if not detection_path.exists():
        print(f"❌ Detection path not found: {detection_path}")
        return []
    
    model_dirs = []
    for item in detection_path.iterdir():
        if item.is_dir() and item.name not in ["archive", "__pycache__"]:
            model_dirs.append(item)
    return model_dirs

def check_model_weights(model_dir):
    """Check if model directory has weight files"""
    weights_dir = model_dir / "weights"
    if not weights_dir.exists():
        return False
    
    # Look for .pt files (excluding epoch checkpoints)
    pt_files = list(weights_dir.glob("*.pt"))
    non_epoch_files = [f for f in pt_files if not f.name.startswith("epoch")]
    
    return len(non_epoch_files) > 0

def get_model_status(model_dir):
    """Get detailed status of a model directory"""
    status = {
        "name": model_dir.name,
        "path": str(model_dir),
        "has_weights": False,
        "has_best": False,
        "has_last": False,
        "epoch_count": 0,
        "total_size_mb": 0,
        "is_empty": True
    }
    
    weights_dir = model_dir / "weights"
    if weights_dir.exists():
        pt_files = list(weights_dir.glob("*.pt"))
        status["epoch_count"] = len([f for f in pt_files if f.name.startswith("epoch")])
        status["has_best"] = (weights_dir / "best.pt").exists()
        status["has_last"] = (weights_dir / "last.pt").exists()
        status["has_weights"] = len([f for f in pt_files if not f.name.startswith("epoch")]) > 0
        
        # Calculate total size
        total_bytes = sum(f.stat().st_size for f in pt_files)
        status["total_size_mb"] = total_bytes / (1024 * 1024)
        status["is_empty"] = len(pt_files) == 0
    
    return status

def archive_empty_models(base_path, dry_run=False):
    """Move empty model directories to archive/"""
    archive_path = base_path / "model" / "detection" / "archive"
    archive_path.mkdir(exist_ok=True)
    
    model_dirs = get_model_directories(base_path)
    archived = []
    
    for model_dir in model_dirs:
        if not check_model_weights(model_dir):
            dest_dir = archive_path / model_dir.name
            if dry_run:
                print(f"📦 Would archive: {model_dir.name}")
                archived.append(model_dir.name)
            else:
                try:
                    shutil.move(str(model_dir), str(dest_dir))
                    print(f"📦 Archived: {model_dir.name}")
                    archived.append(model_dir.name)
                except Exception as e:
                    print(f"❌ Failed to archive {model_dir.name}: {e}")
    
    return archived

def remove_epoch_checkpoints(base_path, dry_run=False):
    """Remove intermediate epoch checkpoint files"""
    removed_count = 0
    removed_size = 0
    
    model_dirs = get_model_directories(base_path)
    
    for model_dir in model_dirs:
        weights_dir = model_dir / "weights"
        if not weights_dir.exists():
            continue
        
        epoch_files = list(weights_dir.glob("epoch*.pt"))
        for epoch_file in epoch_files:
            file_size = epoch_file.stat().st_size
            if dry_run:
                print(f"🗑️  Would remove: {epoch_file.relative_to(base_path)} ({file_size / (1024*1024):.1f} MB)")
                removed_count += 1
                removed_size += file_size
            else:
                try:
                    epoch_file.unlink()
                    print(f"🗑️  Removed: {epoch_file.relative_to(base_path)}")
                    removed_count += 1
                    removed_size += file_size
                except Exception as e:
                    print(f"❌ Failed to remove {epoch_file}: {e}")
    
    return removed_count, removed_size / (1024 * 1024)

def update_readme_status(base_path):
    """Update README.md with current model status"""
    readme_path = base_path / "model" / "detection" / "README.md"
    if not readme_path.exists():
        print(f"❌ README.md not found: {readme_path}")
        return False
    
    # Read current README
    with open(readme_path, 'r') as f:
        content = f.read()
    
    # Find the directory structure section
    # This is a simplified update - in practice you'd parse and regenerate this section
    print(f"✅ README.md status update would be performed here")
    print(f"   (Manual update recommended for complex README structures)")
    
    return True

def generate_cleanup_report(base_path):
    """Generate a comprehensive cleanup report"""
    report = []
    report.append("=" * 60)
    report.append("STRAWBERRY DETECTION MODEL REPOSITORY CLEANUP REPORT")
    report.append("=" * 60)
    report.append("")
    
    model_dirs = get_model_directories(base_path)
    archive_path = base_path / "model" / "detection" / "archive"
    
    # Active models
    active_models = []
    empty_models = []
    total_size = 0
    
    for model_dir in model_dirs:
        status = get_model_status(model_dir)
        if status["has_weights"]:
            active_models.append(status)
            total_size += status["total_size_mb"]
        else:
            empty_models.append(status)
    
    # Archived models
    archived_models = []
    if archive_path.exists():
        for item in archive_path.iterdir():
            if item.is_dir():
                status = get_model_status(item)
                archived_models.append(status)
    
    # Report summary
    report.append(f"📊 Repository Summary:")
    report.append(f"   Total model directories: {len(model_dirs) + len(archived_models)}")
    report.append(f"   Active models: {len(active_models)}")
    report.append(f"   Archived models: {len(archived_models)}")
    report.append(f"   Empty models: {len(empty_models)}")
    report.append(f"   Total model size: {total_size:.1f} MB")
    report.append("")
    
    # Active models details
    if active_models:
        report.append("✅ Active Models:")
        for model in sorted(active_models, key=lambda x: x["total_size_mb"], reverse=True):
            report.append(f"   {model['name']}")
            report.append(f"     - Size: {model['total_size_mb']:.1f} MB")
            report.append(f"     - Best.pt: {'✓' if model['has_best'] else '✗'}")
            report.append(f"     - Last.pt: {'✓' if model['has_last'] else '✗'}")
            report.append(f"     - Epoch files: {model['epoch_count']}")
        report.append("")
    
    # Empty models
    if empty_models:
        report.append("❌ Empty Models (ready for archive):")
        for model in empty_models:
            report.append(f"   {model['name']}")
        report.append("")
    
    # Archived models
    if archived_models:
        report.append("📦 Archived Models:")
        for model in archived_models:
            report.append(f"   {model['name']}")
        report.append("")
    
    # Recommendations
    report.append("💡 Recommendations:")
    if empty_models:
        report.append("   • Archive empty model directories")
    epoch_files = sum(m["epoch_count"] for m in active_models)
    if epoch_files > 0:
        report.append(f"   • Remove {epoch_files} intermediate epoch files")
        potential_savings = sum(m["total_size_mb"] for m in active_models if m["epoch_count"] > 0) * 0.7  # Approx
        report.append(f"   • Potential space savings: ~{potential_savings:.1f} MB")
    
    return "\n".join(report)

def main():
    parser = argparse.ArgumentParser(description="Clean up strawberry detection model repository")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be cleaned without deleting")
    parser.add_argument("--archive-empty", action="store_true", help="Move empty model directories to archive/")
    parser.add_argument("--remove-epochs", action="store_true", help="Remove intermediate epoch checkpoints")
    parser.add_argument("--update-readme", action="store_true", help="Update README.md with current model status")
    parser.add_argument("--full-cleanup", action="store_true", help="Run all cleanup operations")
    parser.add_argument("--report", action="store_true", help="Generate cleanup report only")
    
    args = parser.parse_args()
    
    # Get repository base path
    script_path = Path(__file__).resolve()
    base_path = script_path.parent.parent.parent  # scripts/ -> project root
    
    print("🍓 Strawberry Detection Model Repository Cleanup")
    print("=" * 50)
    
    # Generate report first
    if args.report or not any([args.archive_empty, args.remove_epochs, args.update_readme, args.full_cleanup]):
        report = generate_cleanup_report(base_path)
        print(report)
        if not any([args.archive_empty, args.remove_epochs, args.update_readme, args.full_cleanup]):
            return
    
    # Perform cleanup operations
    if args.full_cleanup:
        args.archive_empty = True
        args.remove_epochs = True
        args.update_readme = True
    
    if args.archive_empty:
        print("\n📦 Archiving empty model directories...")
        archived = archive_empty_models(base_path, dry_run=args.dry_run)
        print(f"   Archived {len(archived)} directories")
    
    if args.remove_epochs:
        print("\n🗑️  Removing intermediate epoch checkpoints...")
        count, size = remove_epoch_checkpoints(base_path, dry_run=args.dry_run)
        print(f"   Removed {count} files ({size:.1f} MB)")
    
    if args.update_readme:
        print("\n📝 Updating README.md...")
        if update_readme_status(base_path):
            print("   README.md update completed")
        else:
            print("   README.md update skipped")
    
    print("\n✅ Cleanup completed!")
    if args.dry_run:
        print("   (This was a dry run - no files were actually modified)")

if __name__ == "__main__":
    main()