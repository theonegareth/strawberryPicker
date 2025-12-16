#!/usr/bin/env python3
"""
Coordinate Transformer - Pixel to Robot Coordinate System
Handles camera calibration, stereo vision, and coordinate transformations

Author: AI Assistant
Date: 2025-12-15
"""

import numpy as np
import cv2
import logging
from pathlib import Path
from typing import Tuple, Optional, Dict, List
from dataclasses import dataclass
import json

logger = logging.getLogger(__name__)

@dataclass
class CameraCalibration:
    """Camera calibration parameters"""
    camera_matrix: np.ndarray
    distortion_coeffs: np.ndarray
    image_width: int
    image_height: int
    focal_length: float
    principal_point: Tuple[float, float]

@dataclass
class StereoCalibration:
    """Stereo camera calibration parameters"""
    left_camera_matrix: np.ndarray
    right_camera_matrix: np.ndarray
    left_distortion: np.ndarray
    right_distortion: np.ndarray
    rotation_matrix: np.ndarray
    translation_vector: np.ndarray
    essential_matrix: np.ndarray
    fundamental_matrix: np.ndarray
    rectification_matrix_left: np.ndarray
    rectification_matrix_right: np.ndarray
    projection_matrix_left: np.ndarray
    projection_matrix_right: np.ndarray
    disparity_to_depth_map: np.ndarray

class CoordinateTransformer:
    """
    Handles coordinate transformations between pixel coordinates and robot world coordinates
    Supports both monocular and stereo camera systems
    """
    
    def __init__(self, 
                 camera_matrix_path: Optional[str] = None,
                 distortion_coeffs_path: Optional[str] = None,
                 stereo_calibration_path: Optional[str] = None):
        """Initialize coordinate transformer"""
        
        self.camera_calibration: Optional[CameraCalibration] = None
        self.stereo_calibration: Optional[StereoCalibration] = None
        self.stereo_matcher: Optional[cv2.StereoSGBM] = None
        
        # Robot coordinate system parameters
        self.robot_origin = (0.0, 0.0, 0.0)  # Robot base position
        self.camera_to_robot_transform = np.eye(4)  # 4x4 transformation matrix
        
        # Load calibrations if provided
        if camera_matrix_path and distortion_coeffs_path:
            self.load_camera_calibration(camera_matrix_path, distortion_coeffs_path)
        
        if stereo_calibration_path:
            self.load_stereo_calibration(stereo_calibration_path)
        
        logger.info("Coordinate Transformer initialized")
    
    def load_camera_calibration(self, camera_matrix_path: str, distortion_coeffs_path: str):
        """Load single camera calibration"""
        try:
            camera_matrix = np.load(camera_matrix_path)
            distortion_coeffs = np.load(distortion_coeffs_path)
            
            # Extract calibration parameters
            fx, fy = camera_matrix[0, 0], camera_matrix[1, 1]
            cx, cy = camera_matrix[0, 2], camera_matrix[1, 2]
            
            self.camera_calibration = CameraCalibration(
                camera_matrix=camera_matrix,
                distortion_coeffs=distortion_coeffs,
                image_width=int(camera_matrix[0, 2] * 2),  # Approximate
                image_height=int(camera_matrix[1, 2] * 2),  # Approximate
                focal_length=(fx + fy) / 2,
                principal_point=(cx, cy)
            )
            
            logger.info("Camera calibration loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load camera calibration: {e}")
            raise
    
    def load_stereo_calibration(self, stereo_calibration_path: str):
        """Load stereo camera calibration"""
        try:
            calibration_data = np.load(stereo_calibration_path)
            
            self.stereo_calibration = StereoCalibration(
                left_camera_matrix=calibration_data['left_camera_matrix'],
                right_camera_matrix=calibration_data['right_camera_matrix'],
                left_distortion=calibration_data['left_distortion'],
                right_distortion=calibration_data['right_distortion'],
                rotation_matrix=calibration_data['rotation_matrix'],
                translation_vector=calibration_data['translation_vector'],
                essential_matrix=calibration_data['essential_matrix'],
                fundamental_matrix=calibration_data['fundamental_matrix'],
                rectification_matrix_left=calibration_data['rectification_matrix_left'],
                rectification_matrix_right=calibration_data['rectification_matrix_right'],
                projection_matrix_left=calibration_data['projection_matrix_left'],
                projection_matrix_right=calibration_data['projection_matrix_right'],
                disparity_to_depth_map=calibration_data['disparity_to_depth_map']
            )
            
            # Initialize stereo matcher for real-time depth calculation
            self._initialize_stereo_matcher()
            
            logger.info("Stereo calibration loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load stereo calibration: {e}")
            raise
    
    def _initialize_stereo_matcher(self):
        """Initialize stereo matching for depth calculation"""
        if not self.stereo_calibration:
            return
        
        # Create stereo block matcher
        self.stereo_matcher = cv2.StereoSGBM_create(
            minDisparity=0,
            numDisparities=64,
            blockSize=9,
            P1=8 * 9 * 9,
            P2=32 * 9 * 9,
            disp12MaxDiff=1,
            uniquenessRatio=10,
            speckleWindowSize=100,
            speckleRange=32
        )
        
        logger.info("Stereo matcher initialized")
    
    def pixel_to_world(self, 
                      pixel_x: int, 
                      pixel_y: int, 
                      image_shape: Tuple[int, int, int],
                      depth: Optional[float] = None) -> Tuple[float, float, float]:
        """
        Convert pixel coordinates to world coordinates
        
        Args:
            pixel_x, pixel_y: Pixel coordinates
            image_shape: Shape of the image (height, width, channels)
            depth: Depth in meters (if None, uses stereo or default depth)
        
        Returns:
            Tuple of (x, y, z) world coordinates in meters
        """
        if not self.camera_calibration:
            raise ValueError("Camera calibration not loaded")
        
        # Get depth if not provided
        if depth is None:
            depth = self._estimate_depth(pixel_x, pixel_y, image_shape)
        
        # Convert pixel to normalized coordinates
        fx, fy = self.camera_calibration.camera_matrix[0, 0], self.camera_calibration.camera_matrix[1, 1]
        cx, cy = self.camera_calibration.principal_point
        
        # Convert to camera coordinates
        x_cam = (pixel_x - cx) * depth / fx
        y_cam = (pixel_y - cy) * depth / fy
        z_cam = depth
        
        # Transform to robot world coordinates
        camera_point = np.array([x_cam, y_cam, z_cam, 1.0])
        world_point = self.camera_to_robot_transform @ camera_point
        
        return (float(world_point[0]), float(world_point[1]), float(world_point[2]))
    
    def world_to_pixel(self, 
                      world_x: float, 
                      world_y: float, 
                      world_z: float) -> Tuple[int, int]:
        """
        Convert world coordinates to pixel coordinates
        
        Returns:
            Tuple of (pixel_x, pixel_y) coordinates
        """
        if not self.camera_calibration:
            raise ValueError("Camera calibration not loaded")
        
        # Transform world point to camera coordinates
        world_point = np.array([world_x, world_y, world_z, 1.0])
        camera_point = np.linalg.inv(self.camera_to_robot_transform) @ world_point
        
        x_cam, y_cam, z_cam = camera_point[:3]
        
        # Convert to pixel coordinates
        fx, fy = self.camera_calibration.camera_matrix[0, 0], self.camera_calibration.camera_matrix[1, 1]
        cx, cy = self.camera_calibration.principal_point
        
        pixel_x = int(x_cam * fx / z_cam + cx)
        pixel_y = int(y_cam * fy / z_cam + cy)
        
        return (pixel_x, pixel_y)
    
    def _estimate_depth(self, pixel_x: int, pixel_y: int, image_shape: Tuple[int, int, int]) -> float:
        """Estimate depth using stereo vision or default depth"""
        
        # If stereo calibration is available, try to get depth from disparity
        if self.stereo_calibration and len(image_shape) == 3:
            # This would require stereo images - simplified for now
            pass
        
        # Default depth estimation based on image position
        # Assume strawberries are typically 20-50cm from camera
        image_height = image_shape[0]
        
        # Simple depth estimation based on vertical position
        # Lower in image = closer to camera
        normalized_y = pixel_y / image_height
        estimated_depth = 0.2 + (0.3 * (1.0 - normalized_y))  # 20-50cm range
        
        return estimated_depth
    
    def calculate_depth_from_stereo(self, 
                                   left_image: np.ndarray, 
                                   right_image: np.ndarray,
                                   pixel_x: int, 
                                   pixel_y: int) -> Optional[float]:
        """Calculate depth from stereo images"""
        if not self.stereo_matcher or not self.stereo_calibration:
            return None
        
        try:
            # Compute disparity map
            disparity = self.stereo_matcher.compute(left_image, right_image)
            
            # Get disparity at specific pixel
            if 0 <= pixel_y < disparity.shape[0] and 0 <= pixel_x < disparity.shape[1]:
                disp_value = disparity[pixel_y, pixel_x]
                
                if disp_value > 0:
                    # Convert disparity to depth
                    depth = self.stereo_calibration.disparity_to_depth_map[disp_value]
                    return float(depth)
            
            return None
            
        except Exception as e:
            logger.error(f"Error calculating stereo depth: {e}")
            return None
    
    def undistort_point(self, pixel_x: int, pixel_y: int) -> Tuple[int, int]:
        """Undistort pixel coordinates using camera calibration"""
        if not self.camera_calibration:
            return pixel_x, pixel_y
        
        try:
            # Create point array
            points = np.array([[pixel_x, pixel_y]], dtype=np.float32)
            
            # Undistort points
            undistorted_points = cv2.undistortPoints(
                points,
                self.camera_calibration.camera_matrix,
                self.camera_calibration.distortion_coeffs
            )
            
            return int(undistorted_points[0][0][0]), int(undistorted_points[0][0][1])
            
        except Exception as e:
            logger.error(f"Error undistorting point: {e}")
            return pixel_x, pixel_y
    
    def set_camera_to_robot_transform(self, transform_matrix: np.ndarray):
        """Set the transformation matrix from camera to robot coordinates"""
        if transform_matrix.shape != (4, 4):
            raise ValueError("Transform matrix must be 4x4")
        
        self.camera_to_robot_transform = transform_matrix
        logger.info("Camera to robot transform updated")
    
    def calibrate_camera_to_robot(self, 
                                 world_points: List[Tuple[float, float, float]],
                                 pixel_points: List[Tuple[int, int]]) -> bool:
        """
        Calibrate camera to robot transformation using known correspondences
        
        Args:
            world_points: List of (x, y, z) world coordinates
            pixel_points: List of (pixel_x, pixel_y) pixel coordinates
        
        Returns:
            True if calibration successful
        """
        if len(world_points) != len(pixel_points) or len(world_points) < 4:
            logger.error("Need at least 4 corresponding points for calibration")
            return False
        
        try:
            # Prepare point correspondences
            world_points_3d = np.array(world_points, dtype=np.float32)
            pixel_points_2d = np.array(pixel_points, dtype=np.float32)
            
            # Solve PnP problem
            success, rotation_vector, translation_vector, inliers = cv2.solvePnPRansac(
                world_points_3d,
                pixel_points_2d,
                self.camera_calibration.camera_matrix,
                self.camera_calibration.distortion_coeffs
            )
            
            if success:
                # Convert rotation vector to rotation matrix
                rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
                
                # Create 4x4 transformation matrix
                transform_matrix = np.eye(4)
                transform_matrix[:3, :3] = rotation_matrix
                transform_matrix[:3, 3] = translation_vector.flatten()
                
                self.set_camera_to_robot_transform(transform_matrix)
                
                logger.info("Camera to robot calibration successful")
                return True
            else:
                logger.error("PnP calibration failed")
                return False
                
        except Exception as e:
            logger.error(f"Calibration error: {e}")
            return False
    
    def get_workspace_bounds(self) -> Dict[str, Tuple[float, float]]:
        """Get the bounds of the robot workspace in world coordinates"""
        # This should be calibrated for your specific robot setup
        return {
            'x_min': -0.5, 'x_max': 0.5,
            'y_min': -0.5, 'y_max': 0.5,
            'z_min': 0.0, 'z_max': 0.5
        }
    
    def is_point_in_workspace(self, x: float, y: float, z: float) -> bool:
        """Check if a point is within the robot workspace"""
        bounds = self.get_workspace_bounds()
        
        return (bounds['x_min'] <= x <= bounds['x_max'] and
                bounds['y_min'] <= y <= bounds['y_max'] and
                bounds['z_min'] <= z <= bounds['z_max'])
    
    def project_3d_to_2d(self, 
                        world_points: List[Tuple[float, float, float]],
                        image_shape: Tuple[int, int]) -> List[Tuple[int, int]]:
        """Project 3D world points to 2D image coordinates"""
        if not self.camera_calibration:
            raise ValueError("Camera calibration not loaded")
        
        projected_points = []
        
        for world_point in world_points:
            pixel_x, pixel_y = self.world_to_pixel(*world_point)
            
            # Check if point is within image bounds
            if (0 <= pixel_x < image_shape[1] and 0 <= pixel_y < image_shape[0]):
                projected_points.append((pixel_x, pixel_y))
            else:
                projected_points.append((-1, -1))  # Outside image
        
        return projected_points
    
    def save_calibration(self, filepath: str):
        """Save current calibration to file"""
        calibration_data = {
            'camera_calibration': {
                'camera_matrix': self.camera_calibration.camera_matrix.tolist() if self.camera_calibration else None,
                'distortion_coeffs': self.camera_calibration.distortion_coeffs.tolist() if self.camera_calibration else None,
                'image_width': self.camera_calibration.image_width if self.camera_calibration else None,
                'image_height': self.camera_calibration.image_height if self.camera_calibration else None,
                'focal_length': self.camera_calibration.focal_length if self.camera_calibration else None,
                'principal_point': self.camera_calibration.principal_point if self.camera_calibration else None
            },
            'stereo_calibration': {
                'left_camera_matrix': self.stereo_calibration.left_camera_matrix.tolist() if self.stereo_calibration else None,
                'right_camera_matrix': self.stereo_calibration.right_camera_matrix.tolist() if self.stereo_calibration else None,
                'rotation_matrix': self.stereo_calibration.rotation_matrix.tolist() if self.stereo_calibration else None,
                'translation_vector': self.stereo_calibration.translation_vector.tolist() if self.stereo_calibration else None
            },
            'camera_to_robot_transform': self.camera_to_robot_transform.tolist(),
            'robot_origin': self.robot_origin
        }
        
        with open(filepath, 'w') as f:
            json.dump(calibration_data, f, indent=2)
        
        logger.info(f"Calibration saved to {filepath}")
    
    def load_calibration(self, filepath: str):
        """Load calibration from file"""
        try:
            with open(filepath, 'r') as f:
                calibration_data = json.load(f)
            
            # Load camera calibration
            if calibration_data['camera_calibration']['camera_matrix']:
                cam_data = calibration_data['camera_calibration']
                self.camera_calibration = CameraCalibration(
                    camera_matrix=np.array(cam_data['camera_matrix']),
                    distortion_coeffs=np.array(cam_data['distortion_coeffs']),
                    image_width=cam_data['image_width'],
                    image_height=cam_data['image_height'],
                    focal_length=cam_data['focal_length'],
                    principal_point=tuple(cam_data['principal_point'])
                )
            
            # Load stereo calibration
            if calibration_data['stereo_calibration']['left_camera_matrix']:
                stereo_data = calibration_data['stereo_calibration']
                # Note: This is simplified - you'd need to load all stereo parameters
                logger.warning("Stereo calibration loading not fully implemented")
            
            # Load transformation matrix
            self.camera_to_robot_transform = np.array(calibration_data['camera_to_robot_transform'])
            self.robot_origin = tuple(calibration_data['robot_origin'])
            
            logger.info(f"Calibration loaded from {filepath}")
            
        except Exception as e:
            logger.error(f"Failed to load calibration: {e}")
            raise

def main():
    """Test coordinate transformer functionality"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Test Coordinate Transformer')
    parser.add_argument('--camera-matrix', help='Camera matrix file path')
    parser.add_argument('--distortion', help='Distortion coefficients file path')
    parser.add_argument('--stereo', help='Stereo calibration file path')
    parser.add_argument('--test-pixel', nargs=2, type=int, metavar=('X', 'Y'), 
                       help='Test pixel coordinates')
    
    args = parser.parse_args()
    
    try:
        # Create transformer
        transformer = CoordinateTransformer(
            args.camera_matrix,
            args.distortion,
            args.stereo
        )
        
        print("Coordinate Transformer initialized")
        
        if args.test_pixel:
            pixel_x, pixel_y = args.test_pixel
            world_coords = transformer.pixel_to_world(pixel_x, pixel_y, (480, 640, 3))
            print(f"Pixel ({pixel_x}, {pixel_y}) -> World {world_coords}")
            
            # Convert back
            pixel_x_back, pixel_y_back = transformer.world_to_pixel(*world_coords)
            print(f"World {world_coords} -> Pixel ({pixel_x_back}, {pixel_y_back})")
        
        # Print workspace bounds
        bounds = transformer.get_workspace_bounds()
        print(f"Workspace bounds: {bounds}")
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()