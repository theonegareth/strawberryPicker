from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
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
        # Optionally include the detection publisher (yolov8_cam) if needed
        # Node(
        #     package='yolov8_cam',
        #     executable='detection_publisher',
        #     name='detection_publisher',
        #     output='screen'
        # ),
        # Or use mock detection for testing
        Node(
            package='strawberry_picker_control',
            executable='test_arm_controller',
            name='mock_detection_publisher',
            output='screen'
        )
    ])