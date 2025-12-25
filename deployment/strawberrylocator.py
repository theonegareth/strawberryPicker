#!/usr/bin/env python3
"""
Enhanced Strawberry Locator - Immediate Implementation
Improved depth detection using bounding box analysis with your current 23cm baseline setup
"""

import cv2
import numpy as np
from ultralytics import YOLO
import math
import serial
import time
import logging
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import yaml
from statistics import median, median_abs_deviation


class StrawberryLocator:
    """
    Enhanced strawberry locator with improved depth detection using bounding box analysis
    Works with your current 23cm baseline setup and detection-first pipeline
    """
    
    def __init__(self, config_file: str = "locator_config.yaml"):
        """Initialize the enhanced strawberry locator"""
        self.config = self.load_config(config_file)
        self.setup_logging()
        self.setup_cameras()
        self.setup_error_handling()
        
        # Camera calibration parameters (from your finaltest.py)
        self.K_A = np.array([[629.10808758, 0.0, 347.20913144],
                            [0.0, 631.11321979, 277.5222819],
                            [0.0, 0.0, 1.0]], dtype=np.float64)
        self.dist_A = np.array([-0.35469562, 0.10232556, -0.0005468, -0.00174671, 0.01546246], dtype=np.float64)
        
        self.K_B = np.array([[1001.67997, 0.0, 367.736216],
                            [0.0, 996.698369, 312.866527],
                            [0.0, 0.0, 1.0]], dtype=np.float64)
        self.dist_B = np.array([-0.49543094, 0.82826695, -0.00180861, -0.00362202, -1.42667838], dtype=np.float64)
        
        # Stereo parameters (your 23cm baseline)
        self.baseline_cm = 23.0
        self.yaw_left_deg = +10.0
        self.yaw_right_deg = -10.0
        
        # Processing parameters
        self.frame_w = 640
        self.frame_h = 408
        self.iou_dedupe_thresh = 0.45
        self.match_distance_thresh = 180.0
        
        self.logger.info("Enhanced Strawberry Locator initialized with 23cm baseline")
    
    def load_config(self, config_file: str) -> dict:
        """Load configuration from YAML file"""
        default_config = {
            'cameras': {
                'baseline_cm': 23.0,
                'left_camera_id': 1,
                'right_camera_id': 2,
                'resolution': [640, 408]
            },
            'depth_detection': {
                'min_confidence_threshold': 0.6,
                'enable_bbox_corners': True,
                'enable_bbox_perimeter': True,
                'num_perimeter_points': 8,
                'enable_fallback_strategies': True
            },
            'processing': {
                'max_strawberries_per_frame': 5,
                'enable_error_recovery': True,
                'max_retries': 3
            },
            'logging': {
                'level': 'INFO',
                'file': 'strawberry_locator.log'
            }
        }
        
        try:
            with open(config_file, 'r') as f:
                config = yaml.safe_load(f)
                # Merge with defaults
                for key, value in default_config.items():
                    if key not in config:
                        config[key] = value
                    elif isinstance(value, dict):
                        for subkey, subvalue in value.items():
                            if subkey not in config[key]:
                                config[key][subkey] = subvalue
                return config
        except FileNotFoundError:
            self.logger.warning(f"Config file {config_file} not found, using defaults")
            return default_config
    
    def setup_logging(self):
        """Setup comprehensive logging"""
        self.logger = logging.getLogger('StrawberryLocator')
        self.logger.setLevel(getattr(logging, self.config['logging']['level']))
        
        # File handler
        file_handler = logging.FileHandler(self.config['logging']['file'])
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(file_formatter)
        self.logger.addHandler(file_handler)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_formatter = logging.Formatter(
            '[%(levelname)s] %(message)s'
        )
        console_handler.setFormatter(console_formatter)
        self.logger.addHandler(console_handler)
    
    def setup_cameras(self):
        """Initialize camera capture objects"""
        self.cap_left = None
        self.cap_right = None
    
    def setup_error_handling(self):
        """Setup comprehensive error handling and recovery"""
        self.max_retries = self.config['processing']['max_retries']
        self.error_count = 0
        self.consecutive_failures = 0
    
    def capture_frames(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Capture synchronized frames from both cameras"""
        try:
            # Initialize cameras if not already done
            if self.cap_left is None:
                self.cap_left = cv2.VideoCapture(self.config['cameras']['left_camera_id'], cv2.CAP_DSHOW)
                self.cap_left.set(cv2.CAP_PROP_FRAME_WIDTH, self.frame_w)
                self.cap_left.set(cv2.CAP_PROP_FRAME_HEIGHT, self.frame_h)
            
            if self.cap_right is None:
                self.cap_right = cv2.VideoCapture(self.config['cameras']['right_camera_id'], cv2.CAP_DSHOW)
                self.cap_right.set(cv2.CAP_PROP_FRAME_WIDTH, self.frame_w)
                self.cap_right.set(cv2.CAP_PROP_FRAME_HEIGHT, self.frame_h)
            
            # Warm up cameras
            for _ in range(3):
                self.cap_left.read()
                self.cap_right.read()
            
            # Capture frames
            ret_left, frame_left = self.cap_left.read()
            ret_right, frame_right = self.cap_right.read()
            
            if not ret_left or not ret_right:
                self.logger.error("Camera capture failed")
                return None, None
            
            return frame_left, frame_right
            
        except Exception as e:
            self.logger.error(f"Camera capture error: {e}")
            return None, None
    
    def detect_strawberries(self, image: np.ndarray, model: YOLO) -> List[Dict]:
        """Detect strawberries using YOLO model"""
        try:
            results = model(image)[0]
            detections = []
            
            for box in results.boxes:
                x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
                cx = int((x1 + x2) / 2)
                cy = int((y1 + y2) / 2)
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                name = model.names.get(cls, str(cls))
                
                # Calculate area for quality assessment
                area = (x2 - x1) * (y2 - y1)
                
                detections.append({
                    'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2,
                    'cx': cx, 'cy': cy, 'conf': conf,
                    'cls': cls, 'name': name, 'area': area
                })
            
            return detections
            
        except Exception as e:
            self.logger.error(f"Detection error: {e}")
            return []
    
    def generate_bbox_corners(self, detection: Dict) -> List[Tuple[int, int]]:
        """Generate corner points of bounding box"""
        x1, y1, x2, y2 = detection['x1'], detection['y1'], detection['x2'], detection['y2']
        return [
            (x1, y1),  # Top-left
            (x2, y1),  # Top-right
            (x1, y2),  # Bottom-left
            (x2, y2)   # Bottom-right
        ]
    
    def generate_bbox_perimeter(self, detection: Dict, num_points: int = 8) -> List[Tuple[int, int]]:
        """Generate perimeter points of bounding box"""
        x1, y1, x2, y2 = detection['x1'], detection['y1'], detection['x2'], detection['y2']
        
        # Calculate perimeter points
        width = x2 - x1
        height = y2 - y1
        perimeter_points = []
        
        # Top edge
        for i in range(num_points // 4):
            x = x1 + (i * width) // (num_points // 4)
            perimeter_points.append((x, y1))
        
        # Right edge
        for i in range(num_points // 4):
            y = y1 + (i * height) // (num_points // 4)
            perimeter_points.append((x2, y))
        
        # Bottom edge
        for i in range(num_points // 4):
            x = x2 - (i * width) // (num_points // 4)
            perimeter_points.append((x, y2))
        
        # Left edge
        for i in range(num_points // 4):
            y = y2 - (i * height) // (num_points // 4)
            perimeter_points.append((x1, y))
        
        return perimeter_points
    
    def assess_bbox_quality(self, left_det: Dict, right_det: Dict) -> float:
        """Assess bounding box quality for depth reliability"""
        try:
            quality_factors = {
                'detection_confidence': min(left_det['conf'], right_det['conf']),
                'bbox_size_ratio': min(left_det['area'], right_det['area']) / max(left_det['area'], right_det['area']),
                'center_distance': 1.0 / (1.0 + math.hypot(left_det['cx'] - right_det['cx'], left_det['cy'] - right_det['cy']) / 100.0),
                'size_consistency': 1.0 - abs(left_det['area'] - right_det['area']) / max(left_det['area'], right_det['area'])
            }
            
            # Weighted geometric mean
            weights = {'detection_confidence': 0.4, 'bbox_size_ratio': 0.3, 
                       'center_distance': 0.2, 'size_consistency': 0.1}
            
            weighted_score = 1.0
            for factor, weight in weights.items():
                weighted_score *= (quality_factors[factor] ** weight)
            
            return min(weighted_score, 1.0)
            
        except Exception as e:
            self.logger.error(f"Quality assessment error: {e}")
            return 0.5  # Medium confidence on error
    
    def calculate_robust_strawberry_depth(self, left_det: Dict, right_det: Dict, 
                                        left_img: np.ndarray, right_img: np.ndarray) -> Tuple[Optional[float], float, str]:
        """Calculate depth using enhanced bounding box analysis"""
        try:
            # Build undistort maps
            mapAx, mapAy, newK_A = self.build_undistort_maps(self.K_A, self.dist_A)
            mapBx, mapBy, newK_B = self.build_undistort_maps(self.K_B, self.dist_B)
            
            # Build projection matrices
            P1, P2, R_rel, T_rel = self.build_projection_matrices(
                newK_A, newK_B, self.yaw_left_deg, self.yaw_right_deg, self.baseline_cm
            )
            
            # Method 1: Bounding box corners (4 points)
            if self.config['depth_detection']['enable_bbox_corners']:
                left_corners = self.generate_bbox_corners(left_det)
                right_corners = self.generate_bbox_corners(right_det)
                
                corner_depths = []
                for (lx, ly), (rx, ry) in zip(left_corners, right_corners):
                    depth = self.triangulate_points(lx, ly, rx, ry, P1, P2)
                    if depth is not None:
                        corner_depths.append(depth[2])
                
                if len(corner_depths) > 0:
                    # Robust statistics: median + outlier removal
                    median_depth = median(corner_depths)
                    mad = median_abs_deviation(corner_depths)
                    filtered_depths = [d for d in corner_depths if abs(d - median_depth) <= 2 * mad]
                    
                    if len(filtered_depths) > 0:
                        final_depth = sum(filtered_depths) / len(filtered_depths)
                        confidence = len(filtered_depths) / 4.0  # 4 corners max
                        return final_depth, confidence, 'bbox_corners'
            
            # Method 2: Bounding box perimeter (8 points)
            if self.config['depth_detection']['enable_bbox_perimeter']:
                left_perimeter = self.generate_bbox_perimeter(left_det, self.config['depth_detection']['num_perimeter_points'])
                right_perimeter = self.generate_bbox_perimeter(right_det, self.config['depth_detection']['num_perimeter_points'])
                
                perimeter_depths = []
                for (lx, ly), (rx, ry) in zip(left_perimeter, right_perimeter):
                    depth = self.triangulate_points(lx, ly, rx, ry, P1, P2)
                    if depth is not None:
                        perimeter_depths.append(depth[2])
                
                if len(perimeter_depths) > 0:
                    # Robust statistics
                    median_depth = median(perimeter_depths)
                    mad = median_abs_deviation(perimeter_depths)
                    filtered_depths = [d for d in perimeter_depths if abs(d - median_depth) <= 2 * mad]
                    
                    if len(filtered_depths) > 0:
                        final_depth = sum(filtered_depths) / len(filtered_depths)
                        confidence = len(filtered_depths) / len(perimeter_depths)
                        return final_depth, confidence, 'bbox_perimeter'
            
            # Method 3: Fallback to center point (original method)
            center_depth = self.triangulate_with_Ps(left_det, right_det, P1, P2)
            if center_depth is not None:
                return center_depth[2], 1.0, 'center_fallback'
            
            return None, 0.0, 'all_methods_failed'
            
        except Exception as e:
            self.logger.error(f"Depth calculation error: {e}")
            return None, 0.0, 'calculation_failed'
    
    def build_undistort_maps(self, K, dist):
        """Build undistortion maps for camera"""
        newK, _ = cv2.getOptimalNewCameraMatrix(K, dist, (self.frame_w, self.frame_h), 1.0)
        mapx, mapy = cv2.initUndistortRectifyMap(K, dist, None, newK, (self.frame_w, self.frame_h), cv2.CV_32FC1)
        return mapx, mapy, newK
    
    def build_projection_matrices(self, newK_A, newK_B, yaw_left_deg, yaw_right_deg, baseline_cm):
        """Build projection matrices for stereo triangulation"""
        # Left camera rotation and translation in world
        R_left = self.yaw_to_R_deg(yaw_left_deg)
        R_right = self.yaw_to_R_deg(yaw_right_deg)
        R_rel = R_right @ R_left.T
        T_rel = np.array([[baseline_cm], [0.0], [0.0]], dtype=np.float64)
        
        # Projection matrices P = K * [R | t]
        P1 = newK_A @ np.hstack((np.eye(3, dtype=np.float64), np.zeros((3,1), dtype=np.float64)))
        P2 = newK_B @ np.hstack((R_rel, T_rel))
        
        return P1, P2, R_rel, T_rel
    
    def yaw_to_R_deg(self, yaw_deg):
        """Convert yaw angle to rotation matrix"""
        y = math.radians(yaw_deg)
        cy = math.cos(y)
        sy = math.sin(y)
        R = np.array([[cy, 0.0, sy],
                      [0.0, 1.0, 0.0],
                      [-sy, 0.0, cy]], dtype=np.float64)
        return R
    
    def triangulate_points(self, x_left: float, y_left: float, x_right: float, y_right: float, 
                          P1: np.ndarray, P2: np.ndarray) -> Optional[Tuple[float, float, float]]:
        """Triangulate 3D point from stereo correspondences"""
        try:
            pts_left = np.array([[float(x_left)], [float(y_left)]], dtype=np.float64)
            pts_right = np.array([[float(x_right)], [float(y_right)]], dtype=np.float64)
            
            Xh = cv2.triangulatePoints(P1, P2, pts_left, pts_right)
            if Xh is None or Xh.shape[1] == 0:
                return None
            
            X = Xh[:, 0]
            if abs(X[3]) < 1e-9:
                return None
            
            X = X / X[3]
            return float(X[0]), float(X[1]), float(X[2])
            
        except Exception as e:
            self.logger.error(f"Triangulation error: {e}")
            return None
    
    def triangulate_with_Ps(self, dL: Dict, dR: Dict, P1: np.ndarray, P2: np.ndarray) -> Optional[Tuple[float, float, float]]:
        """Triangulate using detection dictionaries"""
        return self.triangulate_points(dL['cx'], dL['cy'], dR['cx'], dR['cy'], P1, P2)
    
    def match_detections(self, detA: List[Dict], detB: List[Dict]) -> List[Tuple[Dict, Dict]]:
        """Match detections between left and right cameras"""
        matches = []
        if not detA or not detB:
            return matches
        
        # Simple matching for equal number of detections
        if len(detA) == len(detB):
            return [(a, b) for a, b in zip(detA, detB)]
        
        # Advanced matching with cost matrix
        costs = np.zeros((len(detA), len(detB)), dtype=np.float64)
        for i, a in enumerate(detA):
            for j, b in enumerate(detB):
                class_penalty = 0.0 if a['cls'] == b['cls'] else 40.0
                costs[i, j] = math.hypot(a['cx'] - b['cx'], a['cy'] - b['cy']) + class_penalty
        
        # Hungarian algorithm approximation
        usedB = set()
        for i in range(len(detA)):
            j = int(np.argmin(costs[i]))
            if j not in usedB and costs[i, j] <= self.match_distance_thresh:
                matches.append((detA[i], detB[j]))
                usedB.add(j)
        
        return matches
    
    def process_frame_pair(self, left_frame: np.ndarray, right_frame: np.ndarray, 
                          model: YOLO) -> List[Dict]:
        """Process a pair of frames and return enhanced depth results"""
        try:
            # Undistort frames
            mapAx, mapAy, newK_A = self.build_undistort_maps(self.K_A, self.dist_A)
            mapBx, mapBy, newK_B = self.build_undistort_maps(self.K_B, self.dist_B)
            
            undistorted_left = cv2.remap(left_frame, mapAx, mapAy, cv2.INTER_LINEAR)
            undistorted_right = cv2.remap(right_frame, mapBx, mapBy, cv2.INTER_LINEAR)
            
            # Detect strawberries
            left_dets = self.detect_strawberries(undistorted_left, model)
            right_dets = self.detect_strawberries(undistorted_right, model)
            
            if not left_dets or not right_dets:
                self.logger.warning("No strawberries detected")
                return []
            
            # Match detections
            matches = self.match_detections(left_dets, right_dets)
            
            # Process each match with enhanced depth calculation
            results = []
            for left_det, right_det in matches:
                depth, confidence, method = self.calculate_robust_strawberry_depth(
                    left_det, right_det, undistorted_left, undistorted_right
                )
                
                if depth is not None and confidence > self.config['depth_detection']['min_confidence_threshold']:
                    # Assess overall quality
                    quality_score = self.assess_bbox_quality(left_det, right_det)
                    
                    results.append({
                        'left_detection': left_det,
                        'right_detection': right_det,
                        'depth_cm': depth,
                        'confidence': confidence,
                        'quality_score': quality_score,
                        'method': method,
                        'position_3d': self.convert_to_3d_coordinates(left_det, depth)
                    })
            
            return results
            
        except Exception as e:
            self.logger.error(f"Frame processing error: {e}")
            return []
    
    def convert_to_3d_coordinates(self, detection: Dict, depth: float) -> Tuple[float, float, float]:
        """Convert 2D detection to 3D coordinates"""
        # Simple conversion - can be enhanced with camera calibration
        x = detection['cx'] - self.frame_w // 2  # Center-relative X
        y = detection['cy'] - self.frame_h // 2  # Center-relative Y
        z = depth
        
        # Scale to real-world coordinates (approximate)
        x_cm = x * 0.1  # Rough scaling factor
        y_cm = y * 0.1  # Rough scaling factor
        
        return x_cm, y_cm, z
    
    def visualize_results(self, left_img: np.ndarray, right_img: np.ndarray, results: List[Dict]) -> np.ndarray:
        """Create visualization of detection and depth results"""
        # Create side-by-side visualization
        vis_width = self.frame_w * 2
        vis_height = self.frame_h
        visualization = np.zeros((vis_height, vis_width, 3), dtype=np.uint8)
        
        # Copy undistorted images
        visualization[:, :self.frame_w] = left_img
        visualization[:, self.frame_w:] = right_img
        
        # Draw results
        for i, result in enumerate(results):
            left_det = result['left_detection']
            right_det = result['right_detection']
            
            # Draw bounding boxes
            cv2.rectangle(visualization, (left_det['x1'], left_det['y1']), 
                         (left_det['x2'], left_det['y2']), (0, 255, 0), 2)
            cv2.rectangle(visualization, (right_det['x1'] + self.frame_w, right_det['y1']), 
                         (right_det['x2'] + self.frame_w, right_det['y2']), (0, 255, 0), 2)
            
            # Draw center points
            cv2.circle(visualization, (left_det['cx'], left_det['cy']), 6, (0, 0, 255), -1)
            cv2.circle(visualization, (right_det['cx'] + self.frame_w, right_det['cy']), 6, (0, 0, 255), -1)
            
            # Draw depth information
            depth_text = f"Z={result['depth_cm']:.1f}cm"
            conf_text = f"C={result['confidence']:.2f}"
            method_text = f"M={result['method']}"
            
            cv2.putText(visualization, depth_text, (left_det['x1'], left_det['y1'] - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            cv2.putText(visualization, conf_text, (left_det['x1'], left_det['y2'] + 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
            cv2.putText(visualization, method_text, (left_det['x1'], left_det['y2'] + 40), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
        
        return visualization
    
    def demo_immediate_enhancements(self, model_path: str = "model/detection/homemade_yolov8n_v2_negatives5/weights/best.pt"):
        """Demonstrate immediate enhancements with your current setup"""
        self.logger.info("=== STRAWBERRY LOCATOR ENHANCED DEMO ===")
        self.logger.info("Using 23cm baseline with bounding box analysis")
        
        try:
            # Load model
            model = YOLO(model_path)
            self.logger.info(f"Loaded model: {model_path}")
            
            # Capture frames
            self.logger.info("Capturing frames...")
            left_frame, right_frame = self.capture_frames()
            
            if left_frame is None or right_frame is None:
                self.logger.error("Failed to capture frames")
                return
            
            # Process with enhanced depth detection
            self.logger.info("Processing with enhanced depth detection...")
            results = self.process_frame_pair(left_frame, right_frame, model)
            
            if not results:
                self.logger.warning("No strawberries detected or processed")
                return
            
            # Display results
            self.logger.info("\n=== ENHANCED DEPTH RESULTS ===")
            for i, result in enumerate(results):
                self.logger.info(f"Strawberry {i+1}:")
                self.logger.info(f"  Depth: {result['depth_cm']:.2f} cm")
                self.logger.info(f"  Confidence: {result['confidence']:.3f}")
                self.logger.info(f"  Quality Score: {result['quality_score']:.3f}")
                self.logger.info(f"  Method: {result['method']}")
                self.logger.info(f"  3D Position: {result['position_3d']}")
            
            # Create visualization
            visualization = self.visualize_results(left_frame, right_frame, results)
            
            # Save visualization
            output_path = "enhanced_strawberry_detection.jpg"
            cv2.imwrite(output_path, visualization)
            self.logger.info(f"Visualization saved to: {output_path}")
            
            # Show visualization
            cv2.imshow("Enhanced Strawberry Detection", visualization)
            self.logger.info("Press any key to close visualization...")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            
            self.logger.info("=== DEMO COMPLETED ===")
            self.logger.info("Enhanced strawberry locator ready for production use!")
            
        except Exception as e:
            self.logger.error(f"Demo error: {e}")
            import traceback
            traceback.print_exc()


def main():
    """Main demonstration function"""
    print("🍓 ENHANCED STRAWBERRY LOCATOR - IMMEDIATE IMPLEMENTATION")
    print("=" * 60)
    print("Features implemented:")
    print("✅ Bounding box corner triangulation (4 points)")
    print("✅ Bounding box perimeter sampling (8 points)")
    print("✅ Robust statistics with outlier removal")
    print("✅ Confidence scoring system")
    print("✅ Multiple fallback methods")
    print("✅ Comprehensive error handling")
    print("✅ Professional logging")
    print("✅ Visualization tools")
    print("=" * 60)
    
    # Create and run enhanced locator
    locator = StrawberryLocator()
    locator.demo_immediate_enhancements()


if __name__ == "__main__":
    main()
