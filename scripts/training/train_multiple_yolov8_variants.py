#!/usr/bin/env python3
"""
Multi-Model YOLOv8 Training Script
Trains multiple YOLOv8 variants (s, m, l) for different accuracy/speed trade-offs
Optimized for strawberry detection with comprehensive validation and ONNX conversion
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Any

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from ultralytics import YOLO
import yaml

class MultiModelTrainer:
    """Train multiple YOLOv8 variants for comprehensive model comparison"""
    
    def __init__(self, dataset_path: str, output_base: str):
        self.dataset_path = Path(dataset_path)
        self.output_base = Path(output_base)
        self.results = {}
        
        # Model configurations for different variants
        self.model_configs = {
            'yolov8s': {
                'model_size': 's',
                'epochs': 100,
                'batch_size': 32,
                'imgsz': 640,
                'lr0': 0.01,
                'weight_decay': 0.0005,
                'description': 'Balanced speed/accuracy for general use'
            },
            'yolov8m': {
                'model_size': 'm', 
                'epochs': 120,
                'batch_size': 16,
                'imgsz': 640,
                'lr0': 0.01,
                'weight_decay': 0.0005,
                'description': 'Higher accuracy for critical applications'
            },
            'yolov8l': {
                'model_size': 'l',
                'epochs': 150,
                'batch_size': 8,
                'imgsz': 640,
                'lr0': 0.005,
                'weight_decay': 0.001,
                'description': 'Maximum accuracy for research/validation'
            }
        }
    
    def train_model_variant(self, model_name: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """Train a single YOLOv8 model variant"""
        print(f"\n🚀 Training {model_name.upper()}")
        print(f"📋 Configuration: {config['description']}")
        print(f"⏱️  Epochs: {config['epochs']}, Batch: {config['batch_size']}, Img Size: {config['imgsz']}")
        
        # Create output directory for this model
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_output_dir = self.output_base / f"{model_name}_{timestamp}"
        model_output_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            # Load and train model using simplified approach
            model = YOLO(f'yolov8{config["model_size"]}.pt')
            
            start_time = time.time()
            
            # Use simplified training arguments that work with current YOLO version
            results = model.train(
                data=str(self.dataset_path / 'data.yaml'),
                epochs=config['epochs'],
                batch=config['batch_size'],
                imgsz=config['imgsz'],
                project=str(model_output_dir),
                name='train',
                exist_ok=True,
                device=0 if torch.cuda.is_available() else 'cpu',
                verbose=True,
                save=True,
                save_period=10,
                cache=True,
                amp=True,
                lr0=config['lr0'],
                weight_decay=config['weight_decay'],
                warmup_epochs=3.0,
                momentum=0.937,
                box=7.5,
                cls=0.5,
                dfl=1.5,
                hsv_h=0.015,
                hsv_s=0.7,
                hsv_v=0.4,
                degrees=0.0,
                translate=0.1,
                scale=0.5,
                shear=0.0,
                perspective=0.0,
                flipud=0.0,
                fliplr=0.5,
                mosaic=1.0,
                mixup=0.0,
                erasing=0.4,
                crop_fraction=1.0,
                plots=True,
                val=True
            )
            
            training_time = time.time() - start_time
            
            # Get best model path
            best_model_path = model_output_dir / 'runs' / 'detect' / 'train' / 'weights' / 'best.pt'
            
            # Validate the trained model
            print(f"🔍 Validating {model_name} model...")
            val_results = model.val(
                data=str(self.dataset_path / 'data.yaml'),
                split='val',
                batch=8,
                imgsz=config['imgsz'],
                device=0 if torch.cuda.is_available() else 'cpu',
                plots=True,
                save_dir=str(model_output_dir / 'validation'),
                verbose=False
            )
            
            # Extract key metrics
            metrics = {
                'model_name': model_name,
                'model_size': config['model_size'],
                'training_time_hours': training_time / 3600,
                'epochs_trained': config['epochs'],
                'final_train_loss': getattr(results, 'loss', None),
                'final_val_loss': getattr(val_results, 'loss', None),
                'mAP50': getattr(val_results, 'map50', None),
                'mAP50_95': getattr(val_results, 'map', None),
                'precision': getattr(val_results, 'mp', None),
                'recall': getattr(val_results, 'mr', None),
                'f1_score': getattr(val_results, 'mf1', None),
                'model_path': str(best_model_path),
                'output_directory': str(model_output_dir),
                'config': config,
                'training_timestamp': timestamp
            }
            
            print(f"✅ {model_name} training completed!")
            print(f"📊 mAP@50: {metrics['mAP50']:.3f}" if metrics['mAP50'] else "mAP@50: N/A")
            print(f"⏱️  Training time: {training_time/3600:.1f} hours")
            
            return metrics
            
        except Exception as e:
            print(f"❌ Error training {model_name}: {str(e)}")
            return {
                'model_name': model_name,
                'error': str(e),
                'config': config,
                'training_timestamp': timestamp
            }
    
    def train_all_variants(self) -> Dict[str, Any]:
        """Train all YOLOv8 model variants"""
        print(f"🎯 Starting Multi-Model YOLOv8 Training")
        print(f"📁 Dataset: {self.dataset_path}")
        print(f"💾 Output Base: {self.output_base}")
        print(f"🖥️  Device: {'GPU' if torch.cuda.is_available() else 'CPU'}")
        
        all_results = {}
        
        for model_name, config in self.model_configs.items():
            print(f"\n{'='*60}")
            result = self.train_model_variant(model_name, config)
            all_results[model_name] = result
            
            # Small delay between models to prevent GPU memory issues
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                time.sleep(5)
        
        self.results = all_results
        return all_results
    
    def create_comparison_report(self) -> str:
        """Create a comprehensive comparison report of all trained models"""
        if not self.results:
            return "No training results available."
        
        report_lines = [
            "# 🍓 Multi-Model YOLOv8 Training Report",
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "## 📊 Model Performance Comparison",
            ""
        ]
        
        # Create comparison table
        report_lines.extend([
            "| Model | mAP@50 | mAP@50:95 | Precision | Recall | F1 | Training Time (h) | Model Size |",
            "|-------|--------|-----------|-----------|--------|----|-------------------|------------|"
        ])
        
        for model_name, result in self.results.items():
            if 'error' in result:
                report_lines.append(f"| {model_name} | ERROR | - | - | - | - | - | - |")
                continue
                
            mAP50 = f"{result.get('mAP50', 0):.3f}" if result.get('mAP50') else "N/A"
            mAP50_95 = f"{result.get('mAP50_95', 0):.3f}" if result.get('mAP50_95') else "N/A"
            precision = f"{result.get('precision', 0):.3f}" if result.get('precision') else "N/A"
            recall = f"{result.get('recall', 0):.3f}" if result.get('recall') else "N/A"
            f1 = f"{result.get('f1_score', 0):.3f}" if result.get('f1_score') else "N/A"
            training_time = f"{result.get('training_time_hours', 0):.1f}"
            model_size = result.get('config', {}).get('model_size', 'N/A')
            
            report_lines.append(f"| {model_name} | {mAP50} | {mAP50_95} | {precision} | {recall} | {f1} | {training_time} | {model_size} |")
        
        report_lines.extend([
            "",
            "## 🎯 Recommendations",
            ""
        ])
        
        # Add recommendations based on results
        successful_results = {k: v for k, v in self.results.items() if 'error' not in v}
        
        if successful_results:
            # Find best models for different use cases
            best_map50 = max(successful_results.items(), key=lambda x: x[1].get('mAP50', 0))
            fastest_training = min(successful_results.items(), key=lambda x: x[1].get('training_time_hours', float('inf')))
            
            report_lines.extend([
                f"- **Highest Accuracy**: {best_map50[0]} (mAP@50: {best_map50[1].get('mAP50', 0):.3f})",
                f"- **Fastest Training**: {fastest_training[0]} ({fastest_training[1].get('training_time_hours', 0):.1f} hours)",
                "",
                "### Model Selection Guide:",
                "- **YOLOv8n**: Ultra-fast inference, suitable for real-time applications on limited hardware",
                "- **YOLOv8s**: Balanced speed/accuracy, recommended for most production deployments", 
                "- **YOLOv8m**: Higher accuracy for critical applications where precision is paramount",
                "- **YOLOv8l**: Maximum accuracy for research and validation purposes",
                "",
                "## 📁 Output Locations",
                ""
            ])
            
            for model_name, result in self.results.items():
                if 'error' not in result:
                    report_lines.append(f"- **{model_name}**: `{result.get('output_directory', 'N/A')}`")
        
        report_lines.extend([
            "",
            "## 🔧 Next Steps",
            "",
            "1. **Model Optimization**: Convert best models to ONNX and FP16 formats",
            "2. **Hardware Testing**: Test optimized models on target Raspberry Pi hardware", 
            "3. **Performance Benchmarking**: Compare inference speed across all variants",
            "4. **Production Deployment**: Deploy the most suitable model for your use case",
            "",
            "---",
            "*Report generated by Multi-Model YOLOv8 Training Pipeline*"
        ])
        
        report_content = "\n".join(report_lines)
        
        # Save report
        report_path = self.output_base / f"multi_model_training_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
        with open(report_path, 'w') as f:
            f.write(report_content)
        
        print(f"\n📋 Training report saved to: {report_path}")
        return report_content
    
    def save_results_json(self) -> str:
        """Save training results as JSON for programmatic access"""
        results_path = self.output_base / f"training_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        # Convert results to JSON-serializable format
        json_results = {}
        for model_name, result in self.results.items():
            json_results[model_name] = result.copy()
            # Convert Path objects to strings
            if 'model_path' in json_results[model_name]:
                json_results[model_name]['model_path'] = str(json_results[model_name]['model_path'])
            if 'output_directory' in json_results[model_name]:
                json_results[model_name]['output_directory'] = str(json_results[model_name]['output_directory'])
        
        with open(results_path, 'w') as f:
            json.dump(json_results, f, indent=2, default=str)
        
        print(f"💾 Results JSON saved to: {results_path}")
        return str(results_path)

def main():
    parser = argparse.ArgumentParser(description='Train multiple YOLOv8 model variants')
    parser.add_argument('--dataset', type=str, required=True,
                       help='Path to dataset directory containing data.yaml')
    parser.add_argument('--output', type=str, default='model/detection/multi_model_training',
                       help='Output directory for training results')
    parser.add_argument('--models', nargs='+', 
                       choices=['yolov8s', 'yolov8m', 'yolov8l'],
                       default=['yolov8s', 'yolov8m', 'yolov8l'],
                       help='Model variants to train')
    parser.add_argument('--skip-existing', action='store_true',
                       help='Skip training if model directory already exists')
    
    args = parser.parse_args()
    
    # Validate dataset path
    dataset_path = Path(args.dataset)
    if not (dataset_path / 'data.yaml').exists():
        print(f"❌ Dataset not found: {dataset_path}")
        print("Please ensure the dataset directory contains a data.yaml file")
        return
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize trainer
    trainer = MultiModelTrainer(str(dataset_path), str(output_dir))
    
    # Filter model configs based on user selection
    if args.models != ['yolov8s', 'yolov8m', 'yolov8l']:
        trainer.model_configs = {k: v for k, v in trainer.model_configs.items() if k in args.models}
    
    # Check for existing models if skip-existing is enabled
    if args.skip_existing:
        existing_models = []
        for model_name in trainer.model_configs.keys():
            model_pattern = f"{model_name}_*"
            if list(output_dir.glob(model_pattern)):
                existing_models.append(model_name)
        
        if existing_models:
            print(f"⚠️  Found existing models: {existing_models}")
            print("Use --skip-existing to skip these models")
            return
    
    # Train all selected models
    results = trainer.train_all_variants()
    
    # Generate reports
    print("\n📊 Generating comparison reports...")
    trainer.create_comparison_report()
    trainer.save_results_json()
    
    # Print summary
    print(f"\n🎉 Multi-Model Training Complete!")
    print(f"📁 Results saved to: {output_dir}")
    print(f"🏆 Best performing model: {max(results.items(), key=lambda x: x[1].get('mAP50', 0))[0] if results else 'None'}")

if __name__ == "__main__":
    import torch
    main()