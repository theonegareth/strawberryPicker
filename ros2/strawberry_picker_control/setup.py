from setuptools import find_packages, setup

package_name = 'strawberry_picker_control'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', ['launch/picker.launch.py']),
    ],
    install_requires=['setuptools', 'pyserial'],
    zip_safe=True,
    maintainer='user',
    maintainer_email='user@todo.todo',
    description='ROS2 package for controlling strawberry picking robotic arm based on YOLOv8 detections.',
    license='MIT',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'arm_controller = strawberry_picker_control.arm_controller_node:main',
            'detection_listener = strawberry_picker_control.detection_listener:main',
            'test_arm_controller = strawberry_picker_control.test_arm_controller:main',
        ],
    },
)
