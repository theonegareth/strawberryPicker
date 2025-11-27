#!/usr/bin/env python3
"""
Validation Logger for Strawberry Detection Models
Tracks model performance across different training runs and validations
"""

import json
import csv
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional
import platform
import psutil
from training_registry import TrainingRun, TrainingRegistry

@dataclass
class ValidationResult:
    """Data class for storing validation results for a model"""
    
    # Identification
    validation_id: str
    validation_date: str
    model_id: str
    model_path: str
    
    # Model Information
    model_architecture: str  # yolov8n, yolov8s, yolov8m, etc.
    model_size_mb: float = 0.0
    
    # Dataset Information
    dataset_name: str = "strawberry-detect.v1"
    dataset_size: int = 0
    test_images: int = 0
    
    # Validation Metrics
    mAP_50: float = 0.0
    mAP_50_95: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    f1_score: float = 0.0
    
    # Detection Statistics
    total_detections: int = 0
    avg_confidence: float = 0.0
    high_confidence_ratio: float = 0.0  # detections > 0.5 confidence
    images_with_detections: int = 0
    images_without_detections: int = 0
    avg_detections_per_image: float = 0.0
    
    # Performance Metrics
    avg_inference_time_ms: float = 0.0
    model_load_time_ms: float = 0.0
    
    # System Information
    gpu_name: str = "Unknown"
    gpu_memory_peak_gb: float = 0.0
    cpu_count: int = 0
    ram_total_gb: float = 0.0
    python_version: str = ""
    ultralytics_version: str = ""
    torch_version: str = ""
    cuda_version: str = ""
    
    # Validation Metadata
    validation_duration_minutes: float = 0.0
    validation_config_path: str = ""
    results_path: str = ""
    
    # Best/Worst Cases
    best_case_examples: Optional[List[str]] = None
    worst_case_examples: Optional[List[str]] = None
    failure_patterns: Optional[List[str]] = None
    
    def __post_init__(self):
        """Initialize default lists"""
        if self.best_case_examples is None:
            self.best_case_examples = []
        if self.worst_case_examples is None:
            self.worst_case_examples = []
        if self.failure_patterns is None:
            self.failure_patterns = []

class ValidationLogger:
    """Logger for tracking validation results across models"""
    
    def __init__(self, registry_path: str = "model/validation_registry.json"):
        self.registry_path = Path(registry_path)
        self.registry_path.parent.mkdir(parents=True, exist_ok=True)
        self.results = self._load_registry()
        self.training_registry = TrainingRegistry("model/training_registry.json")
    
    def _load_registry(self) -> List[Dict[str, Any]]:
        """Load validation registry from JSON file"""
        if self.registry_path.exists():
            try:
                with open(self.registry_path, 'r') as f:
                    return json.load(f)
            except Exception as e:
                print(f"Warning: Could not load validation registry: {e}")
                return []
        return []
    
    def _save_registry(self):
        """Save validation registry to JSON file"""
        try:
            with open(self.registry_path, 'w') as f:
                json.dump(self.results, f, indent=2, default=str)
        except Exception as e:
            print(f"Error saving validation registry: {e}")
    
    def log_validation(self, result: ValidationResult):
        """Log a validation result"""
        result_dict = asdict(result)
        self.results.append(result_dict)
        self._save_registry()
        print(f"✓ Validation logged: {result.validation_id} for model {result.model_id}")
    
    def get_all_validations(self) -> List[Dict[str, Any]]:
        """Get all validation results"""
        return self.results
    
    def get_validations_for_model(self, model_id: str) -> List[Dict[str, Any]]:
        """Get all validations for a specific model"""
        return [r for r in self.results if r['model_id'] == model_id]
    
    def get_best_model(self, metric: str = 'mAP_50') -> Optional[Dict[str, Any]]:
        """Get the best performing model based on a metric"""
        if not self.results:
            return None
        
        # Filter models that have the metric
        valid_results = [r for r in self.results if r.get(metric, 0) > 0]
        if not valid_results:
            return None
        
        return max(valid_results, key=lambda x: x.get(metric, 0))
    
    def export_to_csv(self, csv_path: str = "model/validation_history.csv"):
        """Export validation history to CSV"""
        if not self.results:
            print("No validation results to export")
            return
        
        csv_path = Path(csv_path)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Define CSV columns
        columns = [
            'validation_id', 'validation_date', 'model_id', 'model_architecture',
            'dataset_name', 'dataset_size', 'test_images',
            'mAP_50', 'mAP_50_95', 'precision', 'recall', 'f1_score',
            'total_detections', 'avg_confidence', 'high_confidence_ratio',
            'images_with_detections', 'images_without_detections',
            'avg_detections_per_image', 'avg_inference_time_ms',
            'validation_duration_minutes', 'gpu_name'
        ]
        
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=columns)
            writer.writeheader()
            
            for result in self.results:
                # Filter to only include columns we defined
                row = {col: result.get(col, '') for col in columns}
                writer.writerow(row)
        
        print(f"✓ Validation history exported to: {csv_path}")
    
    def generate_comparison_report(self, report_path: str = "model/validation_comparison.md"):
        """Generate a comparison report of all models"""
        if not self.results:
            print("No validation results to report")
            return
        
        report_path = Path(report_path)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(report_path, 'w') as f:
            f.write("# Strawberry Detection Model - Validation Comparison Report\n\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Summary table
            f.write("## Model Performance Summary\n\n")
            f.write("| Model ID | Architecture | mAP@0.5 | Precision | Recall | F1-Score | Avg Conf. | Speed (ms) |\n")
            f.write("|----------|--------------|---------|-----------|--------|----------|-----------|------------|\n")
            
            for result in sorted(self.results, key=lambda x: x.get('mAP_50', 0), reverse=True):
                f.write(f"| {result['model_id']} | {result['model_architecture']} | "
                       f"{result.get('mAP_50', 0):.3f} | {result.get('precision', 0):.3f} | "
                       f"{result.get('recall', 0):.3f} | {result.get('f1_score', 0):.3f} | "
                       f"{result.get('avg_confidence', 0):.3f} | {result.get('avg_inference_time_ms', 0):.1f} |\n")
            
            f.write("\n## Best Performing Model\n\n")
            best = self.get_best_model('mAP_50')
            if best:
                f.write(f"- **Model**: {best['model_id']} ({best['model_architecture']})\n")
                f.write(f"- **mAP@0.5**: {best.get('mAP_50', 0):.3f}\n")
                f.write(f"- **Test Images**: {best.get('test_images', 0)}\n")
                f.write(f"- **Dataset**: {best.get('dataset_name', 'Unknown')}\n")
                f.write(f"- **Validation Date**: {best.get('validation_date', 'Unknown')}\n")
            
            f.write("\n## All Validation Runs\n\n")
            for idx, result in enumerate(self.results, 1):
                f.write(f"### {idx}. {result['model_id']}\n")
                f.write(f"- **Architecture**: {result['model_architecture']}\n")
                f.write(f"- **Validation**: {result['validation_date']}\n")
                f.write(f"- **Dataset**: {result['dataset_name']} ({result['dataset_size']} images)\n")
                f.write(f"- **Test Set**: {result['test_images']} images\n")
                f.write(f"- **mAP@0.5**: {result.get('mAP_50', 0):.3f}\n")
                f.write(f"- **mAP@0.5:0.95**: {result.get('mAP_50_95', 0):.3f}\n")
                f.write(f"- **Precision**: {result.get('precision', 0):.3f}\n")
                f.write(f"- **Recall**: {result.get('recall', 0):.3f}\n")
                f.write(f"- **F1-Score**: {result.get('f1_score', 0):.3f}\n")
                f.write(f"- **Avg Confidence**: {result.get('avg_confidence', 0):.3f}\n")
                f.write(f"- **Inference Speed**: {result.get('avg_inference_time_ms', 0):.1f} ms\n")
                f.write(f"- **GPU**: {result.get('gpu_name', 'Unknown')}\n\n")
        
        print(f"✓ Comparison report generated: {report_path}")

def create_validation_result_from_metrics(
    model_id: str,
    model_path: str,
    metrics: Dict[str, Any],
    validation_config: Dict[str, Any]
) -> ValidationResult:
    """
    Create a ValidationResult from metrics dictionary
    
    Args:
        model_id: Unique ID for the model
        model_path: Path to the model file
        metrics: Dictionary of validation metrics
        validation_config: Configuration used for validation
    
    Returns:
        ValidationResult object
    """
    import torch
    
    # Get system info
    gpu_name = "CPU"
    gpu_memory_peak = 0.0
    
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory_peak = torch.cuda.max_memory_allocated() / 1e9  # Convert to GB
    
    # Create validation result
    result = ValidationResult(
        validation_id=f"val_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        validation_date=datetime.now().isoformat(),
        model_id=model_id,
        model_path=str(Path(model_path).absolute()),
        model_architecture=validation_config.get('architecture', 'unknown'),
        model_size_mb=Path(model_path).stat().st_size / (1024 * 1024),
        dataset_name=validation_config.get('dataset_name', 'strawberry-detect.v1'),
        dataset_size=validation_config.get('dataset_size', 0),
        test_images=metrics.get('test_images', 0),
        mAP_50=metrics.get('mAP_50', 0.0),
        mAP_50_95=metrics.get('mAP_50_95', 0.0),
        precision=metrics.get('precision', 0.0),
        recall=metrics.get('recall', 0.0),
        f1_score=metrics.get('f1_score', 0.0),
        total_detections=metrics.get('total_detections', 0),
        avg_confidence=metrics.get('avg_confidence', 0.0),
        high_confidence_ratio=metrics.get('high_confidence_ratio', 0.0),
        images_with_detections=metrics.get('images_with_detections', 0),
        images_without_detections=metrics.get('images_without_detections', 0),
        avg_detections_per_image=metrics.get('avg_detections_per_image', 0.0),
        avg_inference_time_ms=metrics.get('avg_inference_time_ms', 0.0),
        gpu_name=gpu_name,
        gpu_memory_peak_gb=gpu_memory_peak,
        cpu_count=psutil.cpu_count(),
        ram_total_gb=psutil.virtual_memory().total / (1024**3),
        python_version=platform.python_version(),
        ultralytics_version="8.3.229",  # Can be detected dynamically
        torch_version=torch.__version__,
        cuda_version=torch.version.cuda if torch.cuda.is_available() else "N/A",
        validation_duration_minutes=metrics.get('validation_duration_minutes', 0.0),
        results_path=str(Path("model/validation_results").absolute())
    )
    
    # Add best/worst cases if provided
    if 'best_case_examples' in metrics:
        result.best_case_examples = metrics['best_case_examples']
    if 'worst_case_examples' in metrics:
        result.worst_case_examples = metrics['worst_case_examples']
    if 'failure_patterns' in metrics:
        result.failure_patterns = metrics['failure_patterns']
    
    return result

# Global instance
validation_logger = ValidationLogger()

if __name__ == "__main__":
    # Example usage
    print("Validation Logger initialized")
    print(f"Registry path: {validation_logger.registry_path}")
    print(f"Training registry path: {validation_logger.training_registry.registry_path}")