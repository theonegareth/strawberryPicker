#!/usr/bin/env python3
"""
Update training registry with actual metrics from results.csv files
"""

import sys
from pathlib import Path
import json
import pandas as pd
from datetime import datetime

# Add scripts to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from validation.training_registry import TrainingRegistry

def extract_metrics_from_csv(csv_path):
    """Extract final metrics from results.csv file"""
    try:
        df = pd.read_csv(csv_path)

        # Get the last row (final epoch)
        last_row = df.iloc[-1]

        metrics = {
            'training_time_minutes': last_row.get('time', 0) / 60,  # Convert seconds to minutes
            'val_precision': last_row.get('metrics/precision(B)', 0.0),
            'val_recall': last_row.get('metrics/recall(B)', 0.0),
            'val_map50': last_row.get('metrics/mAP50(B)', 0.0),
            'val_map50_95': last_row.get('metrics/mAP50-95(B)', 0.0),
            'train_box_loss': last_row.get('train/box_loss', 0.0),
            'train_cls_loss': last_row.get('train/cls_loss', 0.0),
            'train_dfl_loss': last_row.get('train/dfl_loss', 0.0),
            'val_box_loss': last_row.get('val/box_loss', 0.0),
            'val_cls_loss': last_row.get('val/cls_loss', 0.0),
            'val_dfl_loss': last_row.get('val/dfl_loss', 0.0),
            'epochs_completed': len(df),
            'best_epoch': len(df)  # Assume last epoch is best
        }

        return metrics

    except Exception as e:
        print(f"Error reading {csv_path}: {e}")
        return None

def find_results_csv_for_model(model_path):
    """Find the corresponding results.csv for a model"""
    model_path = Path(model_path)

    # Try different possible locations for results.csv
    possible_locations = [
        model_path.parent.parent / "results.csv",  # Same level as weights dir
        model_path.parent / "results.csv",         # In weights dir
        model_path.parent.parent.parent / "results.csv",  # Up two levels
    ]

    for csv_path in possible_locations:
        if csv_path.exists():
            return csv_path

    # Try to find results.csv in the model directory structure
    model_dir = model_path.parent
    while model_dir != model_dir.parent:  # Not root
        csv_path = model_dir / "results.csv"
        if csv_path.exists():
            return csv_path
        model_dir = model_dir.parent

    return None

def update_registry_metrics():
    """Update all registry entries with actual metrics from results.csv files"""

    registry = TrainingRegistry("model/training_registry.json")
    runs = registry.get_all_runs()

    print(f"🔄 Updating metrics for {len(runs)} training runs...")
    print("="*60)

    updated_count = 0

    for run in runs:
        run_id = run['run_id']
        model_path = run.get('model_path', '')

        if not model_path:
            print(f"⚠️  No model path for {run_id}")
            continue

        # Find results.csv
        csv_path = find_results_csv_for_model(model_path)

        if not csv_path:
            print(f"❌ No results.csv found for {run_id} (model: {model_path})")
            continue

        # Extract metrics
        metrics = extract_metrics_from_csv(csv_path)

        if not metrics:
            print(f"❌ Failed to extract metrics for {run_id}")
            continue

        # Update the run with real metrics
        updated_run = run.copy()
        updated_run.update(metrics)

        # Update in registry
        registry.update_run(run_id, updated_run)

        print(f"✅ Updated {run_id}:")
        print(".3f")
        print(".1f")
        print()

        updated_count += 1

    print("="*60)
    print(f"SUMMARY: Updated {updated_count} training runs with real metrics")
    print(f"Total runs in registry: {len(registry.get_all_runs())}")

    return updated_count

if __name__ == "__main__":
    update_registry_metrics()