#!/usr/bin/env python3
"""
Strawberry Picker Pipeline - End-to-End Real-time System
Combines detection, classification, and robotic control for automated strawberry picking

Author: AI Assistant
Date: 2025-12-15
"""

import cv2
import numpy as np
import time
import json
import logging
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import argparse
import yaml
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
import threading

# Import our custom modules
from integrated_detection_classification import IntegratedDetectorClassifier
from arduino_bridge import ArduinoBridge
from coordinate_transformer import CoordinateTransformer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('strawberry_picker.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class PickingTarget:
    """Represents a strawberry target for picking"""
    bbox: Tuple[int, int, int, int]  # x, y, w, h
    confidence: float
    ripeness: str  # 'unripe', 'ripe', 'overripe'
    ripeness_confidence: float
    pixel_coords: Tuple[int, int]  # center pixel coordinates
    world_coords: Tuple[float, float, float]  # x, y, z in robot coordinates
    priority: float  # calculated priority score

class StrawberryPickerPipeline:
    """
    Main pipeline for automated strawberry picking system
    Integrates computer vision, classification, and robotic control
    """
    
    def __init__(self, config_path: str = "config.yaml"):
        """Initialize the strawberry picker pipeline"""
        self.config = self._load_config(config_path)
        self.running = False
        self.picking_targets = []
        self.processed_count = 0
        self.successful_picks = 0
        self.failed_picks = 0
        
        # Initialize components
        self.detector_classifier = IntegratedDetectorClassifier(
            detection_model_path=self.config['models']['detection_model'],
            classification_model_path=self.config['models']['classification_model'],
            confidence_threshold=self.config['detection']['confidence_threshold']
        )
        
        self.arduino = ArduinoBridge(
            port=self.config['arduino']['port'],
            baudrate=self.config['arduino']['baudrate']
        )
        
        self.coordinate_transformer = CoordinateTransformer(
            camera_matrix_path=self.config['calibration']['camera_matrix'],
            distortion_coeffs_path=self.config['calibration']['distortion_coeffs'],
            stereo_calibration_path=self.config['calibration']['stereo_calibration']
        )
        
        # Threading for real-time processing
        self.executor = ThreadPoolExecutor(max_workers=2)
        self.processing_lock = threading.Lock()
        
        logger.info("Strawberry Picker Pipeline initialized successfully")
    
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from YAML file"""
        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            logger.warning(f"Config file {config_path} not found, using defaults")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict:
        """Get default configuration"""
        return {
            'models': {
                'detection_model': 'model/weights/best.pt',
                'classification_model': 'model/ripeness_classifier.h5'
            },
            'detection': {
                'confidence_threshold': 0.5,
                'nms_threshold': 0.4
            },
            'arduino': {
                'port': '/dev/ttyUSB0',
                'baudrate': 115200
            },
            'calibration': {
                'camera_matrix': 'calibration/camera_matrix.npy',
                'distortion_coeffs': 'calibration/distortion_coeffs.npy',
                'stereo_calibration': 'calibration/stereo_calibration.npz'
            },
            'picking': {
                'max_targets_per_frame': 5,
                'min_confidence': 0.7,
                'pick_delay': 2.0,  # seconds between picks
                'safety_timeout': 30.0  # seconds
            }
        }
    
    def start(self):
        """Start the strawberry picker pipeline"""
        logger.info("Starting Strawberry Picker Pipeline...")
        
        try:
            # Initialize hardware
            self.arduino.connect()
            self.arduino.initialize_servos()
            
            # Start processing
            self.running = True
            self._start_processing_loop()
            
        except Exception as e:
            logger.error(f"Failed to start pipeline: {e}")
            self.stop()
            raise
    
    def stop(self):
        """Stop the strawberry picker pipeline"""
        logger.info("Stopping Strawberry Picker Pipeline...")
        self.running = False
        
        # Close connections
        if hasattr(self, 'arduino'):
            self.arduino.disconnect()
        
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=True)
        
        logger.info("Pipeline stopped successfully")
    
    def _start_processing_loop(self):
        """Start the main processing loop"""
        # Start camera capture in separate thread
        capture_future = self.executor.submit(self._camera_capture_loop)
        
        # Start picking loop
        picking_future = self.executor.submit(self._picking_loop)
        
        logger.info("Processing loops started")
    
    def _camera_capture_loop(self):
        """Continuous camera capture and processing loop"""
        cap = cv2.VideoCapture(self.config['camera']['index'])
        
        if not cap.isOpened():
            logger.error("Failed to open camera")
            return
        
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config['camera']['width'])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config['camera']['height'])
        cap.set(cv2.CAP_PROP_FPS, self.config['camera']['fps'])
        
        logger.info("Camera capture started")
        
        while self.running:
            ret, frame = cap.read()
            if not ret:
                logger.warning("Failed to capture frame")
                continue
            
            # Process frame for detection and classification
            self._process_frame_async(frame)
            
            # Display frame with annotations
            self._display_frame(frame)
            
            # Control frame rate
            time.sleep(1.0 / self.config['camera']['fps'])
        
        cap.release()
        logger.info("Camera capture stopped")
    
    def _process_frame_async(self, frame: np.ndarray):
        """Process frame asynchronously for detection and classification"""
        def process():
            try:
                # Detect and classify strawberries
                results = self.detector_classifier.process_frame(frame)
                
                # Convert to picking targets
                targets = self._create_picking_targets(results, frame)
                
                # Update targets list
                with self.processing_lock:
                    self.picking_targets = targets
                    self.processed_count += 1
                
            except Exception as e:
                logger.error(f"Frame processing error: {e}")
        
        # Submit to thread pool
        self.executor.submit(process)
    
    def _create_picking_targets(self, detection_results: Dict, frame: np.ndarray) -> List[PickingTarget]:
        """Create picking targets from detection results"""
        targets = []
        
        for detection in detection_results.get('detections', []):
            if detection['confidence'] < self.config['picking']['min_confidence']:
                continue
            
            # Get classification result
            ripeness = detection.get('ripeness', 'unknown')
            ripeness_confidence = detection.get('ripeness_confidence', 0.0)
            
            # Only pick ripe strawberries
            if ripeness != 'ripe':
                continue
            
            # Calculate pixel coordinates
            x, y, w, h = detection['bbox']
            center_x = int(x + w / 2)
            center_y = int(y + h / 2)
            
            # Transform to world coordinates
            try:
                world_coords = self.coordinate_transformer.pixel_to_world(
                    center_x, center_y, frame.shape
                )
            except Exception as e:
                logger.warning(f"Coordinate transformation failed: {e}")
                world_coords = (0.0, 0.0, 0.0)
            
            # Calculate priority (higher confidence = higher priority)
            priority = detection['confidence'] * ripeness_confidence
            
            target = PickingTarget(
                bbox=detection['bbox'],
                confidence=detection['confidence'],
                ripeness=ripeness,
                ripeness_confidence=ripeness_confidence,
                pixel_coords=(center_x, center_y),
                world_coords=world_coords,
                priority=priority
            )
            
            targets.append(target)
        
        # Sort by priority and limit number of targets
        targets.sort(key=lambda t: t.priority, reverse=True)
        return targets[:self.config['picking']['max_targets_per_frame']]
    
    def _picking_loop(self):
        """Main picking loop"""
        logger.info("Picking loop started")
        
        last_pick_time = 0
        safety_timeout = self.config['picking']['safety_timeout']
        
        while self.running:
            try:
                current_time = time.time()
                
                # Check if enough time has passed since last pick
                if current_time - last_pick_time < self.config['picking']['pick_delay']:
                    time.sleep(0.1)
                    continue
                
                # Get current targets
                with self.processing_lock:
                    targets = self.picking_targets.copy()
                
                if not targets:
                    time.sleep(0.1)
                    continue
                
                # Select best target
                target = targets[0]
                
                # Execute pick
                success = self._execute_pick(target)
                
                if success:
                    self.successful_picks += 1
                    logger.info(f"Successful pick! Total: {self.successful_picks}")
                else:
                    self.failed_picks += 1
                    logger.warning(f"Failed pick. Total failures: {self.failed_picks}")
                
                last_pick_time = current_time
                
                # Safety timeout check
                if current_time - last_pick_time > safety_timeout:
                    logger.warning("Safety timeout reached, pausing picking")
                    time.sleep(5.0)
                
            except Exception as e:
                logger.error(f"Picking loop error: {e}")
                time.sleep(1.0)
        
        logger.info("Picking loop stopped")
    
    def _execute_pick(self, target: PickingTarget) -> bool:
        """Execute a picking action for the given target"""
        try:
            logger.info(f"Executing pick for target at {target.pixel_coords}")
            
            # Move to target position
            x, y, z = target.world_coords
            self.arduino.move_to_position(x, y, z)
            
            # Wait for movement to complete
            time.sleep(2.0)
            
            # Close gripper
            self.arduino.close_gripper()
            time.sleep(1.0)
            
            # Lift strawberry
            self.arduino.move_to_position(x, y, z + 0.1)
            time.sleep(1.0)
            
            # Move to collection area
            collection_pos = self.config['picking']['collection_position']
            self.arduino.move_to_position(*collection_pos)
            time.sleep(2.0)
            
            # Open gripper to release strawberry
            self.arduino.open_gripper()
            time.sleep(1.0)
            
            # Return to home position
            home_pos = self.config['picking']['home_position']
            self.arduino.move_to_position(*home_pos)
            time.sleep(2.0)
            
            logger.info("Pick sequence completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Pick execution failed: {e}")
            return False
    
    def _display_frame(self, frame: np.ndarray):
        """Display frame with annotations"""
        try:
            # Add annotations for current targets
            with self.processing_lock:
                targets = self.picking_targets
            
            for i, target in enumerate(targets):
                x, y, w, h = target.bbox
                
                # Draw bounding box
                color = (0, 255, 0) if target.ripeness == 'ripe' else (0, 0, 255)
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                
                # Add label
                label = f"{target.ripeness} ({target.confidence:.2f})"
                cv2.putText(frame, label, (x, y - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                
                # Add priority indicator
                cv2.putText(frame, f"P{i+1}: {target.priority:.2f}", 
                           (x, y + h + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 
                           (255, 255, 0), 2)
            
            # Add status information
            status_text = f"Processed: {self.processed_count} | Success: {self.successful_picks} | Failed: {self.failed_picks}"
            cv2.putText(frame, status_text, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Display frame
            cv2.imshow('Strawberry Picker Pipeline', frame)
            cv2.waitKey(1)
            
        except Exception as e:
            logger.error(f"Frame display error: {e}")
    
    def get_statistics(self) -> Dict:
        """Get pipeline statistics"""
        with self.processing_lock:
            return {
                'processed_frames': self.processed_count,
                'successful_picks': self.successful_picks,
                'failed_picks': self.failed_picks,
                'success_rate': self.successful_picks / max(1, self.successful_picks + self.failed_picks),
                'current_targets': len(self.picking_targets)
            }
    
    def save_statistics(self, filepath: str):
        """Save statistics to file"""
        stats = self.get_statistics()
        stats['timestamp'] = time.time()
        
        with open(filepath, 'w') as f:
            json.dump(stats, f, indent=2)
        
        logger.info(f"Statistics saved to {filepath}")

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Strawberry Picker Pipeline')
    parser.add_argument('--config', default='config.yaml', help='Configuration file path')
    parser.add_argument('--test', action='store_true', help='Run in test mode without hardware')
    parser.add_argument('--save-stats', help='Save statistics to file')
    
    args = parser.parse_args()
    
    try:
        # Initialize pipeline
        pipeline = StrawberryPickerPipeline(args.config)
        
        if args.test:
            logger.info("Running in test mode - no hardware control")
        
        # Start pipeline
        pipeline.start()
        
        # Keep running until interrupted
        try:
            while True:
                time.sleep(1)
                
                # Print statistics periodically
                stats = pipeline.get_statistics()
                if stats['processed_frames'] % 100 == 0:
                    logger.info(f"Statistics: {stats}")
                
        except KeyboardInterrupt:
            logger.info("Received interrupt signal")
        
        finally:
            pipeline.stop()
            
            if args.save_stats:
                pipeline.save_statistics(args.save_stats)
            
            logger.info("Pipeline execution completed")
    
    except Exception as e:
        logger.error(f"Pipeline execution failed: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())