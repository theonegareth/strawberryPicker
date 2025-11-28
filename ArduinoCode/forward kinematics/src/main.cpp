#include <Arduino.h>
#include <Wire.h>
#include <Adafruit_PWMServoDriver.h>

Adafruit_PWMServoDriver pwm = Adafruit_PWMServoDriver();

#define SERVOMIN  150
#define SERVOMAX  600

// Arm parameters
#define L1 20.0  // Lower arm length in cm
#define L2 14.5  // Upper arm length in cm
#define L3 8.0   // hand length in cm (adjust this value)

// Servo indices
#define SERVO_BASE 0
#define SERVO_SHOULDER 1
#define SERVO_ELBOW 2
#define SERVO_WRIST 3
#define SERVO_SCISSOR 4

// Current angles for all servos
float currentAngles[5] = {90, 90, 90, 90, 180};
float targetAngles[5] = {90, 90, 90, 90, 180};

int angleToPulse(float angle) {
  return map((int)angle, 0, 180, SERVOMIN, SERVOMAX);
}

// Forward Kinematics for 2-link planar arm with rotating base
// Computes end effector position (x, y, z) from joint angles
void computeForwardKinematics(float theta0, float theta1, float theta2, float theta3, float &x, float &y, float &z) {
  // Convert degrees to radians
  
  // Apply servo offset and convert to radians
  // Servo 90° = 85° actual angle from horizontal (5° offset due to servo teeth)
  float theta0_rad = theta0 * PI / 180.0;  // Base rotation
  float theta1_rad = (theta1 - 5.0) * PI / 180.0;  // Shoulder absolute angle from horizontal
  
  // Elbow servo angle conversion:
  // Elbow servo 90° = horizontal (0° from horizontal) - due to 5° servo teeth offset
  // Elbow servo 180° = vertical up (90° from horizontal)
  // So: absolute_angle_from_horizontal = 85° - (servo_angle - 5°)
  // This makes servo 90° give 0° (horizontal), servo 95° give 85° (vertical up)
  float elbow_servo_angle = theta2;  // Use raw servo angle
  float theta2_rad = (85.0 * PI / 180.0) - ((elbow_servo_angle - 5.0) * PI / 180.0);  // Convert to absolute angle from horizontal

  // Calculate arm position in the vertical plane
  float plane_x = L1 * cos(theta1_rad) + L2 * cos(theta2_rad);  // Distance in the plane
  float plane_z = L1 * sin(theta1_rad) + L2 * sin(theta2_rad);  // Height

  // Add hand/gripper length (L3) in direction of end effector
  // The hand points in same direction as forearm (elbow angle)
  plane_x += L3 * cos(theta2_rad);
  plane_z += L3 * sin(theta2_rad);
  
  // Wrist servo (theta3) - always 90° to align with forearm
  // If you want to add wrist angle: wrist_servo 0° = up, 90° = forward
  // float wrist_servo_rad = (theta3 - 15.0) * PI / 180.0;
  // float theta3_rad = (PI/2.0) - wrist_servo_rad;
  // plane_x += L3 * cos(theta3_rad);
  // plane_z += L3 * sin(theta3_rad);

  // Convert to 3D coordinates
  // When theta0=0°: arm points along positive X axis
  // When theta0=90°: arm points along positive Y axis
  x = plane_x * cos(theta0_rad);  // Left/right distance (right=positive)
  y = plane_x * sin(theta0_rad);  // Forward distance (forward=positive)
  z = plane_z;                    // Height (toward sky)
}

void moveToTargetAngles() {
  float step = 0.125;    // step size in degrees
  int delayTime = 0.25;  // delay between steps

  // Find max angle change
  float maxChange = 0;
  for (int i = 0; i < 5; i++) {
    float change = abs(targetAngles[i] - currentAngles[i]);
    if (change > maxChange) maxChange = change;
  }

  int steps = maxChange / step;
  if (steps < 1) steps = 1;

  for (int s = 0; s <= steps; s++) {
    for (int i = 0; i < 5; i++) {
      float pos = currentAngles[i] + (targetAngles[i] - currentAngles[i]) * s / steps;
      int pulse = angleToPulse(pos);
      pwm.setPWM(i, 0, pulse);
    }
    delay(delayTime);
  }

  // Update current angles
  for (int i = 0; i < 5; i++) {
    currentAngles[i] = targetAngles[i];
  }
}

void setup() {
  Serial.begin(9600);
  Serial.println("Forward Kinematics Calculator with Servo Control");
  Serial.println("Enter four angles separated by spaces: theta0 theta1 theta2 theta3");
  Serial.println("Example: 90 135 90 90");

  pwm.begin();
  pwm.setPWMFreq(50);
  delay(10);

  // Initialize all servos to 90 degrees
  for (int i = 0; i < 5; i++) {
    int pulse = angleToPulse(currentAngles[i]);
    pwm.setPWM(i, 0, pulse);
  }
}

void loop() {
  if (Serial.available()) {
    String input = Serial.readStringUntil('\n');
    input.trim();

    // Parse four floats separated by spaces: theta0 theta1 theta2 theta3
    int space1 = input.indexOf(' ');
    int space2 = input.indexOf(' ', space1 + 1);
    int space3 = input.indexOf(' ', space2 + 1);
    int space4 = input.indexOf(' ', space3 + 1);  // Should be -1 for exactly 4 values

    if (space1 > 0 && space2 > space1 && space3 > space2 && space4 == -1) {
      float theta0 = input.substring(0, space1).toFloat();
      float theta1 = input.substring(space1 + 1, space2).toFloat();
      float theta2 = input.substring(space2 + 1, space3).toFloat();
      float theta3 = input.substring(space3 + 1).toFloat();

      // Set servo angles
      targetAngles[SERVO_BASE] = theta0;
      targetAngles[SERVO_SHOULDER] = theta1;
      targetAngles[SERVO_ELBOW] = theta2;
      targetAngles[SERVO_WRIST] = theta3;
      // Scissor remains at 180

      moveToTargetAngles();

      // Compute and print position
      float x, y, z;
      computeForwardKinematics(theta0, theta1, theta2, theta3, x, y, z);

      Serial.print("Moved to angles and computed position: (");
      Serial.print(x);
      Serial.print(", ");
      Serial.print(y);
      Serial.print(", ");
      Serial.print(z);
      Serial.println(")");
    } else {
      Serial.println("Invalid input. Enter four angles separated by spaces: theta0 theta1 theta2 theta3");
    }
  }
}