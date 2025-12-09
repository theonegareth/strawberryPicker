from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        # YOLOv8 detection publisher (from yolov8_cam package)
        Node(
            package='yolov8_cam',
            executable='detection_publisher',
            name='detection_publisher',
            output='screen',
            parameters=[{
                'camera_url': 'http://10.24.150.100:5000/video_feed',  # adjust as needed
                'model_path': 'best.pt',
                'confidence_threshold': 0.5,
            }]
        ),
        # Arm controller node
        Node(
            package='strawberry_picker_control',
            executable='arm_controller',
            name='arm_controller',
            output='screen',
            parameters=[
                {'serial_port': '/dev/ttyUSB0'},
                {'baud_rate': 9600},
                {'confidence_threshold': 0.5},
                {'image_width': 640},
                {'image_height': 640},
                {'workspace_x_min': -10.0},
                {'workspace_x_max': 10.0},
                {'workspace_y_min': 0.0},
                {'workspace_y_max': 20.0},
                {'pick_height': 5.0}
            ]
        ),
    ])