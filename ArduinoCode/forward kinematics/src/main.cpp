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
float currentAngles[5] = {90, 95, 80, 90, 180};
float targetAngles[5] = {90, 95, 80, 90, 180};

int angleToPulse(float angle) {
  return map((int)angle, 0, 180, SERVOMIN, SERVOMAX);
}

// Forward Kinematics for 2-link planar arm with rotating base
// Computes end effector position (x, y, z) from joint angles
void computeForwardKinematics(float theta0, float theta1, float theta2, float &x, float &y, float &z) {
  float theta0_rad = theta0 * PI / 180.0;
  float theta1_rad = (theta1) * PI / 180.0;
  float theta2_rad = (theta2 - theta1) * PI / 180.0;

  float y_arm = L1 * cos(theta1_rad) + L2 * cos(theta2_rad) + L3;
  float z_arm = L1 * sin(theta1_rad) - L2 * sin(theta2_rad);

  y = y_arm * sin(theta0_rad);
  x = y_arm * cos(theta0_rad);
  z = z_arm;
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

void moveScissorOnce(float angle) {
  targetAngles[SERVO_SCISSOR] = angle;
  moveToTargetAngles();
}

void setup() {
  Serial.begin(9600);
  Serial.println("Forward Kinematics Calculator with Servo Control");
  Serial.println("Enter four angles separated by spaces: theta0 theta1 theta2");
  Serial.println("Example: 90 135 90");

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

    if (space1 > 0 && space2 > space1 && input.indexOf(' ', space2 + 1) == -1) {
      float theta0 = input.substring(0, space1).toFloat();
      float theta1 = input.substring(space1 + 1, space2).toFloat();
      float theta2 = input.substring(space2 + 1).toFloat();

      // Set servo angles
      targetAngles[SERVO_BASE] = theta0;
      targetAngles[SERVO_SHOULDER] = theta1 + 5.0;  // Adjust for servo offset
      targetAngles[SERVO_ELBOW] = theta2 - 10.0;    // Adjust for servo offset
      targetAngles[SERVO_WRIST] = theta2 - (80 - theta2) * 2; // Wrist to maintain end effector orientation
      targetAngles[SERVO_SCISSOR] = 120;

      //Move arm first
      moveToTargetAngles();

      // Then move scissor
      moveScissorOnce(80);
      moveScissorOnce(120);

      // Compute and print position
      float x, y, z;
      computeForwardKinematics(theta0, theta1, theta2, x, y, z);

      Serial.print("Moved to angles and computed position: (");
      Serial.print(x);
      Serial.print(", ");
      Serial.print(y);
      Serial.print(", ");
      Serial.print(z);
      Serial.println(")");
    } else {
      Serial.println("Invalid input. Enter four angles separated by spaces: theta0 theta1 theta2");
    }
  }
}