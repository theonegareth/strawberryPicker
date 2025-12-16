#!/usr/bin/env python3
"""
Sync Strawberry Picker Models to HuggingFace Repository

This script automates the process of syncing trained models from strawberryPicker
to the HuggingFace strawberryPicker repository.

Usage:
    python sync_to_huggingface.py [--dry-run]
"""

import os
import sys
import json
import shutil
import argparse
import subprocess
from pathlib import Path
from datetime import datetime
import hashlib

def calculate_file_hash(file_path):
    """Calculate SHA256 hash of a file."""
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()

def find_updated_models(repo_path):
    """Find models that have been updated."""
    updated_models = []

    # Check detection model
    detection_pt = Path(repo_path) / "detection" / "best.pt"
    if detection_pt.exists():
        detection_hash = calculate_file_hash(detection_pt)
        updated_models.append({
            'component': 'detection',
            'path': detection_pt,
            'hash': detection_hash
        })

    # Check classification model
    classification_pth = Path(repo_path) / "classification" / "best_enhanced_classifier.pth"
    if classification_pth.exists():
        classification_hash = calculate_file_hash(classification_pth)
        updated_models.append({
            'component': 'classification',
            'path': classification_pth,
            'hash': classification_hash
        })

    return updated_models

def export_detection_to_onnx(model_path, output_dir):
    """Export detection model to ONNX format."""
    try:
        cmd = [
            "yolo", "export",
            f"model={model_path}",
            f"dir={output_dir}",
            "format=onnx",
            "opset=12"
        ]

        print(f"Exporting detection model to ONNX...")
        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode == 0:
            print("Successfully exported detection model to ONNX")
            return True
        else:
            print(f"Export failed: {result.stderr}")
            return False

    except Exception as e:
        print(f"Error during ONNX export: {e}")
        return False

def update_model_metadata(repo_path, models_info):
    """Update metadata files with sync information."""
    metadata = {
        "last_sync": datetime.now().isoformat(),
        "models": {}
    }

    for model in models_info:
        metadata["models"][model['component']] = {
            "hash": model['hash'],
            "path": str(model['path']),
            "last_updated": datetime.now().isoformat()
        }

    metadata_file = Path(repo_path) / "sync_metadata.json"
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"Updated metadata file: {metadata_file}")
    return True

def sync_repository(repo_path, dry_run=False):
    """Main sync function for strawberryPicker repository."""
    print(f"Syncing strawberryPicker repository at {repo_path}")

    if dry_run:
        print("DRY RUN MODE - No changes will be made")

    # Find updated models
    updated_models = find_updated_models(repo_path)

    if not updated_models:
        print("No models found to sync")
        return True

    print(f"Found {len(updated_models)} model components:")
    for model in updated_models:
        print(f"  - {model['component']} (hash: {model['hash'][:8]}...)")

    if dry_run:
        return True

    # Export detection model to ONNX if needed
    detection_model = next((m for m in updated_models if m['component'] == 'detection'), None)
    if detection_model:
        detection_dir = Path(repo_path) / "detection"
        export_detection_to_onnx(detection_model['path'], detection_dir)

    # Update metadata
    update_model_metadata(repo_path, updated_models)

    print("\nSync completed successfully!")
    print("Remember to:")
    print("1. Review and commit changes to git")
    print("2. Push to HuggingFace: git push origin main")
    print("3. Update READMEs with any new performance metrics")

    return True

def main():
    parser = argparse.ArgumentParser(description="Sync strawberryPicker models to HuggingFace")
    parser.add_argument("--repo-path",
                        default="/home/user/machine-learning/HuggingfaceModels/strawberryPicker",
                        help="Path to strawberryPicker repository")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would be done without making changes")

    args = parser.parse_args()

    # Validate path
    if not os.path.exists(args.repo_path):
        print(f"Error: Repository path {args.repo_path} does not exist")
        sys.exit(1)

    # Run sync
    success = sync_repository(args.repo_path, args.dry_run)

    if not success:
        sys.exit(1)

if __name__ == "__main__":
    main()