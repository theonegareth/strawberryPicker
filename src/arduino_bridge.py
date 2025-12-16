#!/usr/bin/env python3
"""
Arduino Bridge - Serial Communication for Robotic Arm Control
Handles communication between Python pipeline and Arduino microcontroller

Author: AI Assistant
Date: 2025-12-15
"""

import serial
import serial.tools.list_ports
import time
import logging
import threading
from typing import Optional, Tuple, List, Dict
from dataclasses import dataclass
import json
import re

logger = logging.getLogger(__name__)

@dataclass
class ServoPosition:
    """Represents a servo position"""
    servo_id: int
    angle: float
    timestamp: float

@dataclass
class SensorData:
    """Represents sensor data from Arduino"""
    limit_switches: Dict[int, bool]
    force_sensor: float
    temperature: float
    timestamp: float

class ArduinoBridge:
    """
    Bridge for communication with Arduino-based robotic arm controller
    Handles servo control, sensor reading, and safety monitoring
    """
    
    def __init__(self, port: str = '/dev/ttyUSB0', baudrate: int = 115200):
        """Initialize Arduino bridge"""
        self.port = port
        self.baudrate = baudrate
        self.serial_connection: Optional[serial.Serial] = None
        self.connected = False
        self.running = False
        
        # Command queue for async communication
        self.command_queue = []
        self.response_queue = []
        self.queue_lock = threading.Lock()
        
        # Current servo positions
        self.current_positions = {}
        self.target_positions = {}
        
        # Sensor data
        self.latest_sensor_data: Optional[SensorData] = None
        self.sensor_history = []
        
        # Safety parameters
        self.max_servo_angle = 180.0
        self.min_servo_angle = 0.0
        self.movement_timeout = 10.0  # seconds
        self.emergency_stop_flag = False
        
        # Communication thread
        self.comm_thread: Optional[threading.Thread] = None
        
        logger.info(f"Arduino Bridge initialized for port {port} at {baudrate} baud")
    
    def connect(self) -> bool:
        """Connect to Arduino via serial port"""
        try:
            # Auto-detect port if not specified
            if self.port == '/dev/ttyUSB0':
                self.port = self._auto_detect_port()
                if not self.port:
                    logger.error("Could not auto-detect Arduino port")
                    return False
            
            # Establish serial connection
            self.serial_connection = serial.Serial(
                port=self.port,
                baudrate=self.baudrate,
                timeout=1.0,
                write_timeout=1.0
            )
            
            # Wait for Arduino to initialize
            time.sleep(2.0)
            
            # Test connection
            if self._test_connection():
                self.connected = True
                self.running = True
                
                # Start communication thread
                self.comm_thread = threading.Thread(target=self._communication_loop, daemon=True)
                self.comm_thread.start()
                
                logger.info(f"Successfully connected to Arduino on {self.port}")
                return True
            else:
                logger.error("Arduino connection test failed")
                self.disconnect()
                return False
                
        except Exception as e:
            logger.error(f"Failed to connect to Arduino: {e}")
            return False
    
    def disconnect(self):
        """Disconnect from Arduino"""
        logger.info("Disconnecting from Arduino...")
        
        self.running = False
        self.connected = False
        
        # Stop communication thread
        if self.comm_thread and self.comm_thread.is_alive():
            self.comm_thread.join(timeout=2.0)
        
        # Close serial connection
        if self.serial_connection and self.serial_connection.is_open:
            self.serial_connection.close()
        
        logger.info("Arduino disconnected")
    
    def _auto_detect_port(self) -> Optional[str]:
        """Auto-detect Arduino port"""
        ports = serial.tools.list_ports.comports()
        
        for port in ports:
            # Look for common Arduino identifiers
            if any(keyword in port.description.lower() for keyword in 
                   ['arduino', 'ch340', 'cp2102', 'ftdi']):
                logger.info(f"Auto-detected Arduino on port: {port.device}")
                return port.device
        
        # If no Arduino found, return first available port
        if ports:
            logger.warning(f"No Arduino detected, using first available port: {ports[0].device}")
            return ports[0].device
        
        logger.error("No serial ports available")
        return None
    
    def _test_connection(self) -> bool:
        """Test connection with Arduino"""
        try:
            # Send test command
            self._send_command("PING")
            
            # Wait for response
            start_time = time.time()
            while time.time() - start_time < 3.0:
                if self._check_response("PONG"):
                    return True
                time.sleep(0.1)
            
            return False
            
        except Exception as e:
            logger.error(f"Connection test failed: {e}")
            return False
    
    def _communication_loop(self):
        """Main communication loop for async processing"""
        logger.info("Arduino communication loop started")
        
        while self.running and self.connected:
            try:
                # Process outgoing commands
                self._process_command_queue()
                
                # Read incoming data
                self._read_serial_data()
                
                # Process sensor data
                self._process_sensor_data()
                
                time.sleep(0.01)  # 10ms loop delay
                
            except Exception as e:
                logger.error(f"Communication loop error: {e}")
                time.sleep(0.1)
        
        logger.info("Arduino communication loop stopped")
    
    def _process_command_queue(self):
        """Process commands in the queue"""
        with self.queue_lock:
            if not self.command_queue:
                return
            
            command = self.command_queue.pop(0)
        
        try:
            self._send_raw_command(command)
        except Exception as e:
            logger.error(f"Failed to send command {command}: {e}")
    
    def _read_serial_data(self):
        """Read and process incoming serial data"""
        if not self.serial_connection or not self.serial_connection.is_open:
            return
        
        try:
            if self.serial_connection.in_waiting > 0:
                line = self.serial_connection.readline().decode('utf-8').strip()
                self._process_serial_line(line)
        except Exception as e:
            logger.error(f"Error reading serial data: {e}")
    
    def _process_serial_line(self, line: str):
        """Process a single line of serial data"""
        try:
            # Parse different message types
            if line.startswith("SENSOR:"):
                self._parse_sensor_data(line)
            elif line.startswith("STATUS:"):
                self._parse_status_data(line)
            elif line.startswith("ERROR:"):
                logger.error(f"Arduino error: {line}")
            elif line.startswith("DEBUG:"):
                logger.debug(f"Arduino debug: {line}")
            else:
                # Add to response queue
                with self.queue_lock:
                    self.response_queue.append(line)
                
        except Exception as e:
            logger.error(f"Error processing serial line '{line}': {e}")
    
    def _parse_sensor_data(self, line: str):
        """Parse sensor data from Arduino"""
        try:
            # Format: SENSOR:limit_sw1:0,limit_sw2:1,force:45.2,temp:23.5
            data_part = line[7:]  # Remove "SENSOR:" prefix
            
            sensor_data = {}
            for item in data_part.split(','):
                key, value = item.split(':')
                if key.startswith('limit_sw'):
                    sensor_data[key] = bool(int(value))
                elif key == 'force':
                    sensor_data[key] = float(value)
                elif key == 'temp':
                    sensor_data[key] = float(value)
            
            self.latest_sensor_data = SensorData(
                limit_switches={k: v for k, v in sensor_data.items() if k.startswith('limit_sw')},
                force_sensor=sensor_data.get('force', 0.0),
                temperature=sensor_data.get('temp', 0.0),
                timestamp=time.time()
            )
            
            # Add to history
            self.sensor_history.append(self.latest_sensor_data)
            if len(self.sensor_history) > 100:  # Keep last 100 readings
                self.sensor_history.pop(0)
                
        except Exception as e:
            logger.error(f"Error parsing sensor data: {e}")
    
    def _parse_status_data(self, line: str):
        """Parse status data from Arduino"""
        try:
            # Format: STATUS:servo1:90.5,servo2:45.0
            data_part = line[7:]  # Remove "STATUS:" prefix
            
            for item in data_part.split(','):
                servo_id, angle = item.split(':')
                servo_num = int(servo_id.replace('servo', ''))
                self.current_positions[servo_num] = float(angle)
                
        except Exception as e:
            logger.error(f"Error parsing status data: {e}")
    
    def _send_command(self, command: str):
        """Send command and wait for response"""
        with self.queue_lock:
            self.command_queue.append(command)
        
        # Wait for response
        start_time = time.time()
        while time.time() - start_time < 5.0:
            if self._check_response(command):
                return True
            time.sleep(0.1)
        
        logger.warning(f"No response received for command: {command}")
        return False
    
    def _send_raw_command(self, command: str):
        """Send raw command without queuing"""
        if not self.serial_connection or not self.serial_connection.is_open:
            raise Exception("Serial connection not available")
        
        self.serial_connection.write(f"{command}\n".encode('utf-8'))
        self.serial_connection.flush()
    
    def _check_response(self, expected_command: str) -> bool:
        """Check if expected response is in queue"""
        with self.queue_lock:
            for i, response in enumerate(self.response_queue):
                if expected_command in response:
                    self.response_queue.pop(i)
                    return True
        return False
    
    def initialize_servos(self):
        """Initialize all servos to home position"""
        logger.info("Initializing servos...")
        
        # Home positions for each servo (adjust based on your robot design)
        home_positions = {
            1: 90.0,   # Base rotation
            2: 45.0,   # Shoulder
            3: 90.0,   # Elbow
            4: 90.0,   # Wrist
            5: 0.0,    # Gripper
        }
        
        for servo_id, position in home_positions.items():
            self.move_servo(servo_id, position)
            time.sleep(0.5)  # Small delay between servo movements
        
        logger.info("Servos initialized to home positions")
    
    def move_servo(self, servo_id: int, angle: float, speed: float = 1.0) -> bool:
        """Move a specific servo to target angle"""
        if not self.connected:
            logger.error("Not connected to Arduino")
            return False
        
        # Validate angle
        if not (self.min_servo_angle <= angle <= self.max_servo_angle):
            logger.error(f"Invalid angle {angle} for servo {servo_id}")
            return False
        
        # Check emergency stop
        if self.emergency_stop_flag:
            logger.warning("Emergency stop active, ignoring servo command")
            return False
        
        try:
            command = f"MOVE:{servo_id}:{angle:.1f}:{speed:.2f}"
            success = self._send_command(command)
            
            if success:
                self.target_positions[servo_id] = angle
                logger.debug(f"Moved servo {servo_id} to {angle} degrees")
            
            return success
            
        except Exception as e:
            logger.error(f"Failed to move servo {servo_id}: {e}")
            return False
    
    def move_to_position(self, x: float, y: float, z: float) -> bool:
        """Move robotic arm to 3D position using inverse kinematics"""
        if not self.connected:
            logger.error("Not connected to Arduino")
            return False
        
        try:
            # Convert 3D coordinates to servo angles
            # This is a simplified version - implement proper inverse kinematics
            servo_angles = self._inverse_kinematics(x, y, z)
            
            # Move all servos simultaneously
            success = True
            for servo_id, angle in servo_angles.items():
                if not self.move_servo(servo_id, angle):
                    success = False
            
            if success:
                logger.info(f"Moved to position ({x:.2f}, {y:.2f}, {z:.2f})")
            
            return success
            
        except Exception as e:
            logger.error(f"Failed to move to position: {e}")
            return False
    
    def _inverse_kinematics(self, x: float, y: float, z: float) -> Dict[int, float]:
        """Simple inverse kinematics calculation"""
        # This is a placeholder implementation
        # Replace with proper IK calculations for your robot design
        
        # Simplified mapping (adjust based on your robot geometry)
        base_angle = (np.arctan2(y, x) * 180 / np.pi) + 90
        shoulder_angle = 45 + (z * 10)  # Simplified mapping
        elbow_angle = 90 - (z * 5)
        wrist_angle = 90
        
        return {
            1: np.clip(base_angle, 0, 180),
            2: np.clip(shoulder_angle, 0, 180),
            3: np.clip(elbow_angle, 0, 180),
            4: np.clip(wrist_angle, 0, 180),
        }
    
    def open_gripper(self) -> bool:
        """Open the gripper"""
        return self.move_servo(5, 0.0)
    
    def close_gripper(self) -> bool:
        """Close the gripper"""
        return self.move_servo(5, 90.0)
    
    def emergency_stop(self):
        """Activate emergency stop"""
        logger.warning("EMERGENCY STOP ACTIVATED")
        self.emergency_stop_flag = True
        
        # Send emergency stop command
        try:
            self._send_command("ESTOP")
        except:
            pass
        
        # Move all servos to safe positions
        safe_positions = {1: 90, 2: 45, 3: 90, 4: 90, 5: 0}
        for servo_id, position in safe_positions.items():
            try:
                self.move_servo(servo_id, position, speed=2.0)
            except:
                pass
    
    def reset_emergency_stop(self):
        """Reset emergency stop"""
        self.emergency_stop_flag = False
        logger.info("Emergency stop reset")
    
    def get_sensor_data(self) -> Optional[SensorData]:
        """Get latest sensor data"""
        return self.latest_sensor_data
    
    def get_servo_positions(self) -> Dict[int, float]:
        """Get current servo positions"""
        return self.current_positions.copy()
    
    def is_movement_complete(self, servo_id: int, tolerance: float = 2.0) -> bool:
        """Check if servo movement is complete"""
        if servo_id not in self.target_positions:
            return True
        
        current = self.current_positions.get(servo_id, 0)
        target = self.target_positions[servo_id]
        
        return abs(current - target) <= tolerance
    
    def wait_for_movement(self, servo_ids: List[int], timeout: float = 10.0) -> bool:
        """Wait for servo movements to complete"""
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            all_complete = True
            for servo_id in servo_ids:
                if not self.is_movement_complete(servo_id):
                    all_complete = False
                    break
            
            if all_complete:
                return True
            
            time.sleep(0.1)
        
        logger.warning(f"Movement timeout after {timeout} seconds")
        return False
    
    def get_status(self) -> Dict:
        """Get Arduino bridge status"""
        return {
            'connected': self.connected,
            'port': self.port,
            'baudrate': self.baudrate,
            'emergency_stop': self.emergency_stop_flag,
            'current_positions': self.get_servo_positions(),
            'target_positions': self.target_positions.copy(),
            'latest_sensor_data': self.latest_sensor_data.__dict__ if self.latest_sensor_data else None,
            'queue_size': len(self.command_queue)
        }

def main():
    """Test Arduino bridge functionality"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Test Arduino Bridge')
    parser.add_argument('--port', default='/dev/ttyUSB0', help='Arduino port')
    parser.add_argument('--baudrate', type=int, default=115200, help='Baud rate')
    
    args = parser.parse_args()
    
    # Create bridge
    bridge = ArduinoBridge(args.port, args.baudrate)
    
    try:
        # Connect
        if bridge.connect():
            print("Connected to Arduino successfully")
            
            # Initialize servos
            bridge.initialize_servos()
            time.sleep(2)
            
            # Test movements
            print("Testing servo movements...")
            bridge.move_servo(1, 45)
            time.sleep(1)
            bridge.move_servo(1, 135)
            time.sleep(1)
            bridge.move_servo(1, 90)
            
            # Test gripper
            print("Testing gripper...")
            bridge.open_gripper()
            time.sleep(1)
            bridge.close_gripper()
            time.sleep(1)
            bridge.open_gripper()
            
            # Print status
            print("\nArduino Bridge Status:")
            print(json.dumps(bridge.get_status(), indent=2, default=str))
            
        else:
            print("Failed to connect to Arduino")
    
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    
    finally:
        bridge.disconnect()
        print("Arduino bridge disconnected")

if __name__ == "__main__":
    main()