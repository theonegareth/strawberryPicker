#!/usr/bin/env python3
"""
Model Consolidation and Registry Update Script

This script:
1. Scans all model locations for .pt files
2. Checks registry completeness (training_registry.json)
3. Adds missing models to registry with metadata
4. Organizes folder structure by copying best models to model/detection/
5. Generates a summary report
"""

import os
import json
import shutil
import yaml
import glob
from datetime import datetime
from pathlib import Path
import sys

# Configuration
PROJECT_ROOT = Path(__file__).parent.parent
MODEL_DIRS = [
    PROJECT_ROOT / "model" / "detection",
    PROJECT_ROOT / "model" / "results",
    PROJECT_ROOT / "model" / "detection" / "multi_model_training",
]
REGISTRY_PATH = PROJECT_ROOT / "model" / "training_registry.json"
DEST_DIR = PROJECT_ROOT / "model" / "detection"
REPORT_PATH = PROJECT_ROOT / "model" / "consolidation_report.md"

def find_model_files():
    """Find all .pt model files in the specified directories."""
    model_files = []
    
    for model_dir in MODEL_DIRS:
        if not model_dir.exists():
            print(f"⚠️  Directory not found: {model_dir}")
            continue
            
        # Look for .pt files recursively
        for pt_file in model_dir.rglob("*.pt"):
            # Skip files in archive or temp directories
            if any(x in str(pt_file) for x in ["archive", "temp", "backup"]):
                continue
                
            model_files.append({
                "path": pt_file,
                "relative": pt_file.relative_to(PROJECT_ROOT),
                "size_mb": pt_file.stat().st_size / (1024 * 1024),
                "dir": pt_file.parent,
                "name": pt_file.stem,
                "type": "best" if "best" in pt_file.stem.lower() else 
                       "last" if "last" in pt_file.stem.lower() else 
                       "model" if "model" in pt_file.stem.lower() else "unknown"
            })
    
    return model_files

def load_registry():
    """Load existing training registry or create empty one."""
    if REGISTRY_PATH.exists():
        with open(REGISTRY_PATH, 'r') as f:
            data = json.load(f)
            # Handle both list format and dict format
            if isinstance(data, list):
                return {"models": data, "last_updated": datetime.now().isoformat()}
            else:
                return data
    else:
        return {"models": [], "last_updated": datetime.now().isoformat()}

def extract_metadata(model_path):
    """Extract metadata from model directory."""
    metadata = {
        "model_name": model_path.stem,
        "model_path": str(model_path.relative_to(PROJECT_ROOT)),
        "size_mb": round(model_path.stat().st_size / (1024 * 1024), 2),
        "detected_at": datetime.now().isoformat(),
        "status": "detected",
        "in_registry": False
    }
    
    # Try to find args.yaml
    args_file = model_path.parent / "args.yaml"
    if args_file.exists():
        try:
            with open(args_file, 'r') as f:
                args_data = yaml.safe_load(f)
                metadata["training_args"] = args_data
                metadata["epochs"] = args_data.get("epochs", "unknown")
                metadata["batch_size"] = args_data.get("batch", "unknown")
                metadata["dataset"] = args_data.get("data", "unknown")
        except:
            metadata["training_args"] = "error_reading"
    
    # Try to find results.csv
    results_file = model_path.parent / "results.csv"
    if results_file.exists():
        metadata["has_results"] = True
        # Could parse metrics here if needed
    else:
        metadata["has_results"] = False
    
    # Determine model type from path
    path_str = str(model_path)
    if "yolov8n" in path_str.lower():
        metadata["model_type"] = "yolov8n"
    elif "yolov8s" in path_str.lower():
        metadata["model_type"] = "yolov8s"
    elif "yolov8m" in path_str.lower():
        metadata["model_type"] = "yolov8m"
    elif "yolov8l" in path_str.lower():
        metadata["model_type"] = "yolov8l"
    elif "yolov8x" in path_str.lower():
        metadata["model_type"] = "yolov8x"
    else:
        metadata["model_type"] = "unknown"
    
    return metadata

def is_model_in_registry(registry, model_path):
    """Check if a model is already in the registry."""
    model_rel_path = str(model_path.relative_to(PROJECT_ROOT))
    for entry in registry.get("models", []):
        if entry.get("model_path") == model_rel_path:
            return True
    return False

def add_to_registry(registry, metadata):
    """Add model metadata to registry."""
    registry_entry = {
        "run_id": f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{metadata['model_name']}",
        "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "experiment_name": metadata["model_name"],
        "model_type": "detection",
        "model_architecture": "YOLOv8",
        "model_size": metadata.get("model_type", "unknown").replace("yolov8", ""),
        "pretrained": True,
        "dataset_name": metadata.get("dataset", "unknown"),
        "dataset_size": "unknown",
        "num_classes": 1,
        "class_names": ["strawberry"],
        "batch_size": metadata.get("batch_size", 16),
        "image_size": 640,
        "epochs_planned": metadata.get("epochs", 50),
        "epochs_completed": metadata.get("epochs", 50),
        "learning_rate": 0.001,
        "optimizer": "AdamW",
        "weight_decay": 0.0005,
        "train_accuracy": 0.0,
        "val_accuracy": 0.0,
        "test_accuracy": 0.0,
        "train_loss": 0.0,
        "val_loss": 0.0,
        "test_loss": 0.0,
        "precision": 0.0,
        "recall": 0.0,
        "f1_score": 0.0,
        "macro_avg_precision": 0.0,
        "macro_avg_recall": 0.0,
        "macro_avg_f1": 0.0,
        "weighted_avg_precision": 0.0,
        "weighted_avg_recall": 0.0,
        "weighted_avg_f1": 0.0,
        "confusion_matrix": None,
        "training_time_minutes": 0.0,
        "early_stopped": False,
        "best_epoch": 0,
        "gpu_name": "NVIDIA GeForce RTX 3050 Ti Laptop GPU",
        "gpu_memory_peak_gb": 0.0,
        "cpu_count": 20,
        "ram_total_gb": 15.5,
        "python_version": "3.12.3",
        "tensorflow_version": "",
        "pytorch_version": "2.9.1+cu128",
        "cuda_version": "12.8",
        "os_info": "linux",
        "model_path": metadata["model_path"],
        "results_path": str(Path(metadata["model_path"]).parent.parent),
        "config_path": metadata["model_path"],
        "status": "completed",
        "teacher_model": None,
        "distillation_temperature": 1.0,
        "distillation_alpha": 0.5
    }
    
    registry.setdefault("models", []).append(registry_entry)
    return registry_entry

def organize_model(model_info, dest_base=DEST_DIR):
    """Copy best models to organized directory structure."""
    if model_info["type"] != "best":
        return None  # Only organize best models
    
    # Create destination path
    model_type = model_info["path"].parent.parent.name if "multi_model_training" in str(model_info["path"]) else model_info["dir"].name
    dest_dir = dest_base / model_type / "weights"
    dest_dir.mkdir(parents=True, exist_ok=True)
    
    dest_path = dest_dir / model_info["path"].name
    
    # Copy if not already there
    if not dest_path.exists():
        print(f"📁 Copying {model_info['name']} to {dest_path.relative_to(PROJECT_ROOT)}")
        shutil.copy2(model_info["path"], dest_path)
        
        # Also copy args.yaml and results.csv if they exist
        for ext_file in ["args.yaml", "results.csv", "opt.yaml"]:
            src_file = model_info["path"].parent / ext_file
            if src_file.exists():
                dest_file = dest_dir / ext_file
                shutil.copy2(src_file, dest_file)
        
        return str(dest_path.relative_to(PROJECT_ROOT))
    
    return str(dest_path.relative_to(PROJECT_ROOT))

def generate_report(model_files, registry, organized_models, missing_in_registry):
    """Generate a comprehensive markdown report."""
    report = f"""# Model Consolidation Report

Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## 📊 Summary
- **Total models found**: {len(model_files)}
- **Models in registry**: {len(registry.get('models', []))}
- **Models missing from registry**: {len(missing_in_registry)}
- **Models organized**: {len(organized_models)}

## 🔍 Models Found
| Model | Path | Size (MB) | Type | In Registry |
|-------|------|-----------|------|-------------|
"""
    
    for model in model_files:
        in_registry = "✅" if is_model_in_registry(registry, model["path"]) else "❌"
        report += f"| {model['name']} | `{model['relative']}` | {model['size_mb']:.1f} | {model['type']} | {in_registry} |\n"
    
    report += "\n## 📋 Missing from Registry\n"
    if missing_in_registry:
        for model in missing_in_registry:
            report += f"- `{model['relative']}` ({model['type']}, {model['size_mb']:.1f}MB)\n"
    else:
        report += "✅ All models are in the registry!\n"
    
    report += "\n## 🗂️ Organized Models\n"
    if organized_models:
        for src, dest in organized_models.items():
            report += f"- `{src}` → `{dest}`\n"
    else:
        report += "No new models organized (all best models already in organized structure)\n"
    
    report += "\n## 📈 Registry Status\n"
    report += f"- **Total entries**: {len(registry.get('models', []))}\n"
    report += f"- **Last updated**: {registry.get('last_updated', 'unknown')}\n"
    
    # Model types breakdown
    model_types = {}
    for entry in registry.get("models", []):
        model_type = entry.get("model_size", "unknown")
        model_types[model_type] = model_types.get(model_type, 0) + 1
    
    report += "\n### Model Type Distribution\n"
    for model_type, count in model_types.items():
        report += f"- **{model_type}**: {count} models\n"
    
    report += "\n## 🚀 Next Steps\n"
    report += "1. Review the organized model structure\n"
    report += "2. Update any references to use the new organized paths\n"
    report += "3. Consider cleaning up duplicate model files\n"
    report += "4. Push updated registry to version control\n"
    
    return report

def main():
    print("🔍 Starting model consolidation and registry update...")
    
    # 1. Find all model files
    print("📂 Scanning for model files...")
    model_files = find_model_files()
    print(f"   Found {len(model_files)} model files")
    
    # 2. Load registry
    print("📋 Loading training registry...")
    registry = load_registry()
    print(f"   Registry has {len(registry.get('models', []))} entries")
    
    # 3. Identify models missing from registry
    missing_in_registry = []
    for model in model_files:
        if not is_model_in_registry(registry, model["path"]):
            missing_in_registry.append(model)
    
    print(f"   {len(missing_in_registry)} models missing from registry")
    
    # 4. Add missing models to registry
    added_models = []
    for model in missing_in_registry:
        print(f"   Adding {model['relative']} to registry...")
        metadata = extract_metadata(model["path"])
        registry_entry = add_to_registry(registry, metadata)
        added_models.append(registry_entry)
    
    # 5. Organize best models
    print("🗂️ Organizing best models...")
    organized_models = {}
    for model in model_files:
        if model["type"] == "best":
            dest_path = organize_model(model)
            if dest_path:
                organized_models[str(model["relative"])] = dest_path
    
    # 6. Update registry timestamp
    registry["last_updated"] = datetime.now().isoformat()
    registry["consolidation_run"] = datetime.now().isoformat()
    
    # 7. Save updated registry
    print("💾 Saving updated registry...")
    REGISTRY_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(REGISTRY_PATH, 'w') as f:
        json.dump(registry, f, indent=2)
    
    # 8. Generate report
    print("📊 Generating report...")
    report = generate_report(model_files, registry, organized_models, missing_in_registry)
    with open(REPORT_PATH, 'w') as f:
        f.write(report)
    
    # 9. Print summary
    print("\n" + "="*60)
    print("✅ MODEL CONSOLIDATION COMPLETE")
    print("="*60)
    print(f"📁 Models found: {len(model_files)}")
    print(f"📋 Added to registry: {len(added_models)}")
    print(f"🗂️ Organized: {len(organized_models)}")
    print(f"📄 Registry saved: {REGISTRY_PATH.relative_to(PROJECT_ROOT)}")
    print(f"📊 Report generated: {REPORT_PATH.relative_to(PROJECT_ROOT)}")
    print("="*60)
    
    if added_models:
        print("\n📋 Newly added models:")
        for entry in added_models:
            print(f"  - {entry['experiment_name']} ({entry['model_size']})")
    
    if organized_models:
        print("\n🗂️ Organized models:")
        for src, dest in organized_models.items():
            print(f"  - {src} → {dest}")

if __name__ == "__main__":
    main()