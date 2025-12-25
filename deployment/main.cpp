#include <Arduino.h>
#include <Wire.h>
#include <Adafruit_PWMServoDriver.h>

Adafruit_PWMServoDriver pwm = Adafruit_PWMServoDriver();

#define SERVOMIN  150
#define SERVOMAX  600

// Arm lengths in cm
#define L1 20.0
#define L2 13.2
#define L3 7.0

// Servos
#define SERVO_BASE 0
#define SERVO_SHOULDER 1
#define SERVO_ELBOW 2
#define SERVO_WRIST 3
#define SERVO_SCISSOR 4

// Safety limits
#define SHOULDER_MAX 160
#define SHOULDER_MIN 10
#define ELBOW_MAX 140
#define WRIST_MIN_WHEN_ELBOW_MAX 90

// ======================================================
// PID PARAMETERS - ADJUST THESE VALUES
// ======================================================
float Kp = 1.0;  // Proportional gain - CHANGE THIS
float Ki = 0.0;  // Integral gain - CHANGE THIS
float Kd = 0.0;  // Derivative gain - CHANGE THIS

// ======================================================
// PID STRUCTURE FOR EACH SERVO
// ======================================================
struct PIDController {
  float setpoint;           // Target position
  float estimatedPosition;  // Estimated current position
  float integral;           // Accumulated error
  float previousError;      // Last error for derivative
  unsigned long lastTime;   // Last update time
  
  PIDController() {
    setpoint = 0;
    estimatedPosition = 0;
    integral = 0;
    previousError = 0;
    lastTime = 0;
  }
};

PIDController pidControllers[5];

float currentAngles[5] = {90, 125, 110, 120, 180};
float targetAngles[5]  = {90, 125, 110, 120, 180};

int angleToPulse(float angle) {
  return map((int)angle, 0, 180, SERVOMIN, SERVOMAX);
}

// ======================================================
// PID COMPUTATION FUNCTION
// ======================================================
float computePID(int servoIndex, float targetAngle) {
  PIDController &pid = pidControllers[servoIndex];
  
  unsigned long currentTime = millis();
  float dt = (currentTime - pid.lastTime) / 1000.0; // Convert to seconds
  
  if (dt <= 0) dt = 0.001; // Prevent division by zero
  
  // Update setpoint
  pid.setpoint = targetAngle;
  
  // Simple first-order model: servo moves toward target exponentially
  // Typical servo response time constant (adjust based on your servos)
  float tau = 0.15; // Time constant in seconds (150ms typical for hobby servos)
  float alpha = dt / (tau + dt);
  pid.estimatedPosition += alpha * (pid.setpoint - pid.estimatedPosition);
  
  // Calculate error
  float error = pid.setpoint - pid.estimatedPosition;
  
  // Proportional term
  float P = Kp * error;
  
  // Integral term (with anti-windup)
  pid.integral += error * dt;
  pid.integral = constrain(pid.integral, -50, 50); // Prevent integral windup
  float I = Ki * pid.integral;
  
  // Derivative term
  float derivative = (error - pid.previousError) / dt;
  float D = Kd * derivative;
  
  // Save for next iteration
  pid.previousError = error;
  pid.lastTime = currentTime;
  
  // PID output (control signal)
  float output = P + I + D;
  
  // The output is added to the estimated position to get commanded position
  float commandedAngle = pid.estimatedPosition + output;
  
  // Constrain to valid servo range
  commandedAngle = constrain(commandedAngle, 0, 180);
  
  return commandedAngle;
}

// ======================================================
// INITIALIZE PID CONTROLLERS
// ======================================================
void initializePID() {
  for (int i = 0; i < 5; i++) {
    pidControllers[i].estimatedPosition = currentAngles[i];
    pidControllers[i].setpoint = currentAngles[i];
    pidControllers[i].integral = 0;
    pidControllers[i].previousError = 0;
    pidControllers[i].lastTime = millis();
  }
}

// ======================================================
// SAFETY CHECK FUNCTION
// ======================================================
bool isSafeToMove(float shoulder, float elbow, float wrist) {
  // Check shoulder limits
  if (shoulder > SHOULDER_MAX) {
    Serial.print("UNSAFE: Shoulder angle ");
    Serial.print(shoulder);
    Serial.print(" exceeds maximum of ");
    Serial.println(SHOULDER_MAX);
    return false;
  }
  
  if (shoulder < SHOULDER_MIN) {
    Serial.print("UNSAFE: Shoulder angle ");
    Serial.print(shoulder);
    Serial.print(" is below minimum of ");
    Serial.println(SHOULDER_MIN);
    return false;
  }
  
  // Check elbow limits
  if (elbow > ELBOW_MAX) {
    Serial.print("UNSAFE: Elbow angle ");
    Serial.print(elbow);
    Serial.print(" exceeds maximum of ");
    Serial.println(ELBOW_MAX);
    return false;
  }
  
  // Check wrist limits when elbow is at max
  if (elbow >= ELBOW_MAX && wrist < WRIST_MIN_WHEN_ELBOW_MAX) {
    Serial.print("UNSAFE: Wrist angle ");
    Serial.print(wrist);
    Serial.print(" is below minimum of ");
    Serial.print(WRIST_MIN_WHEN_ELBOW_MAX);
    Serial.print(" when elbow is at ");
    Serial.println(elbow);
    return false;
  }
  
  return true;
}

// ======================================================
// FORWARD KINEMATICS
// ======================================================
void computeForwardKinematics(float theta0, float theta1, float theta2, float &x, float &y, float &z){
  float t0 = theta0 * PI / 180.0;
  float t1 = theta1 * PI / 180.0;
  float t2 = (theta2 - theta1) * PI / 180.0;

  float base_offset = 1.4;
  float base_offset_x = base_offset * cos(t0);
  float base_offset_y = base_offset * sin(t0);

  float y_arm = L1 * cos(t1) + L2 * cos(t2) + L3;
  float z_arm = L1 * sin(t1) - L2 * sin(t2);

  x = y_arm * cos(t0) + base_offset_x;
  y = y_arm * sin(t0) + base_offset_y;
  z = z_arm + 7.5;
}

// ======================================================
// INVERSE KINEMATICS
// ======================================================
bool computeInverseKinematics(float x, float y, float z,
                              float &theta0, float &theta1, float &theta2, float &theta3,
                              float &dbg_shoulderTrigDeg,
                              float &dbg_shoulderRightDeg,
                              float &dbg_elbowTrigDeg,
                              float &dbg_wristTrigDeg)
{
  float arm_length = sqrt(x*x + y*y);
  float Z = z - 9.0;

  float L3_offset = arm_length - L3;

  float C = sqrt(L3_offset * L3_offset + Z*Z);
  if (C > L1 + L2 || C < fabs(L1 - L2))
    return false;

  float a = L1;
  float b = L2;

  theta0 = atan2(y, x) * 180.0 / PI;

  float elbowtrig = (a*a + b*b - C*C) / (2*a*b);
  elbowtrig = constrain(elbowtrig, -1, 1);
  float elbowtrig_angle = acos(elbowtrig);

  theta2 = elbowtrig_angle * 180.0 / PI;

  dbg_elbowTrigDeg = elbowtrig_angle * 180.0 / PI;

  float shouldertrig = (a*a + C*C - b*b) / (2*a*C);
  shouldertrig = constrain(shouldertrig, -1, 1);
  float shouldertrig_angle = acos(shouldertrig);

  if (L3_offset > 0){
    float shoulder_rightangle = atan(Z / L3_offset);
    float shoulder = shouldertrig_angle + shoulder_rightangle;
    theta1 = (shoulder * 180.0 / PI);
    dbg_shoulderTrigDeg  = shouldertrig_angle * 180.0 / PI;
    dbg_shoulderRightDeg = shoulder_rightangle * 180.0 / PI;
  }
  else if (L3_offset == 0){
    float shoulder = shouldertrig_angle + PI/2;
    theta1 = (shoulder * 180.0 / PI);
    dbg_shoulderTrigDeg  = shouldertrig_angle * 180.0 / PI;
  }
  else if (L3_offset < 0){
    float shoulder_rightangle = PI - atan(Z / L3_offset);
    float shoulder = shouldertrig_angle - shoulder_rightangle;
    theta1 = (shoulder * 180.0 / PI);
    dbg_shoulderTrigDeg  = shouldertrig_angle * 180.0 / PI;
    dbg_shoulderRightDeg = shoulder_rightangle * 180.0 / PI;
  }

  float L1_Z = a * fabs(sin(theta1 * PI / 180.0));
  float Wrist_Y = L3 + b * cos((fabs(theta2 - ((theta2 - 90) * 2) - theta1)) * PI / 180.0);
  float Wrist_Z = b * sin((fabs(theta2 - ((theta2 - 90) * 2) - theta1)) * PI / 180.0);

  if (Wrist_Z == 0){
    theta3 = 90;
    dbg_wristTrigDeg = theta3;
    return true;
  }

  float Wrist_ZY = sqrt(Wrist_Y*Wrist_Y + Wrist_Z*Wrist_Z);
  float wristtrig = (L3*L3 + b*b - Wrist_ZY * Wrist_ZY) / (2*L3*b);
  wristtrig = constrain(wristtrig, -1, 1);
  float wristtrig_angle = acos(wristtrig);

  if (L1_Z <= Z){
    theta3 = (wristtrig_angle - PI/2) * 180.0 / PI;
  }
  else if (L1_Z > Z){
    float theta3_before = (wristtrig_angle - PI/2) * 180.0 / PI;
    theta3 = theta3_before - ((theta3_before - 90) * 2);
  }

  dbg_wristTrigDeg = theta3;

  return true;
}

// ======================================================
// SMOOTH MOVE WITH PID CONTROL
// ======================================================
void moveToTargetAnglesWithPID(int durationMs) {
  unsigned long startTime = millis();
  unsigned long currentTime = startTime;
  
  // Set target angles for all PID controllers
  for (int i = 0; i < 5; i++) {
    targetAngles[i] = constrain(targetAngles[i], 0, 180);
  }
  
  // Move servos with PID control until duration expires
  while ((currentTime - startTime) < durationMs) {
    for (int i = 0; i < 5; i++) {
      // Compute PID output for each servo
      float commandedAngle = computePID(i, targetAngles[i]);
      
      // Send commanded angle to servo
      pwm.setPWM(i, 0, angleToPulse(commandedAngle));
      
      // Update current angle tracking
      currentAngles[i] = pidControllers[i].estimatedPosition;
    }
    
    delay(10); // 10ms update rate (100Hz)
    currentTime = millis();
  }
  
  // Final position update
  for (int i = 0; i < 5; i++) {
    currentAngles[i] = targetAngles[i];
    pidControllers[i].estimatedPosition = targetAngles[i];
  }
}

// ======================================================
// LEGACY SMOOTH MOVE (WITHOUT PID)
// ======================================================
void moveToTargetAngles(float step, int delayTime) {
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
      pwm.setPWM(i, 0, angleToPulse(pos));
    }
    delay(delayTime);
  }

  for (int i = 0; i < 5; i++)
    currentAngles[i] = targetAngles[i];
}

void moveScissorOnce(float angle) {
  targetAngles[SERVO_SCISSOR] = angle;
  moveToTargetAngles(4, 0.01);
}

void moveScissorSecond(float angle) {
  targetAngles[SERVO_SCISSOR] = angle;
  moveToTargetAngles(0.125, 0.25);
}

void moveToDefaultAngles() {
  targetAngles[SERVO_BASE]     = 90;
  targetAngles[SERVO_SHOULDER] = 165;
  targetAngles[SERVO_ELBOW]    = 150;
  targetAngles[SERVO_WRIST]    = 160;
  targetAngles[SERVO_SCISSOR]  = 180;
  moveToTargetAnglesWithPID(2000); // 2 second movement with PID
}

// ======================================================
// SETUP
// ======================================================
void setup() {
  Serial.begin(9600);
  Serial.println("READY");
  Serial.println("Enter coordinates in format: x,y,z");

  pwm.begin();
  pwm.setPWMFreq(50);
  delay(10);

  for (int i = 0; i < 5; i++){
    pwm.setPWM(i, 0, angleToPulse(currentAngles[i]));
  }
  
  // Initialize PID controllers
  initializePID();
  
  Serial.println("PID Control Initialized");
  Serial.print("Kp="); Serial.print(Kp);
  Serial.print(" Ki="); Serial.print(Ki);
  Serial.print(" Kd="); Serial.println(Kd);
}

// ======================================================
// LOOP
// ======================================================
void loop() {
  static bool armBusy = false;
  if (armBusy) {
    while (Serial.available()) Serial.read();
    return;
  }
  
  if (!Serial.available()) return;

  String input = Serial.readStringUntil('\n');
  input.trim();
  
  Serial.print("DEBUG: Raw input received: '");
  Serial.print(input);
  Serial.println("'");
  
  if (input.length() == 0) {
    Serial.println("DEBUG: Empty input, ignoring");
    return;
  }

  armBusy = true;
  Serial.println("BUSY");

  int c1 = input.indexOf(',');
  int c2 = input.indexOf(',', c1 + 1);
  
  Serial.print("DEBUG: First comma at position: ");
  Serial.println(c1);
  Serial.print("DEBUG: Second comma at position: ");
  Serial.println(c2);
  
  if (c1 < 0 || c2 < 0) {
    Serial.println("ERROR: Invalid format! Use: x,y,z");
    Serial.println("DONE");
    armBusy = false;
    Serial.println("READY");
    return;
  }

  float x = input.substring(0, c1).toFloat();
  float z = input.substring(c1 + 1, c2).toFloat();
  float y = input.substring(c2 + 1).toFloat();

  Serial.println("DEBUG: Successfully parsed coordinates:");
  Serial.print("  X: ");
  Serial.println(x);
  Serial.print("  Y: ");
  Serial.println(y);
  Serial.print("  Z: ");
  Serial.println(z);

  float t0, t1, t2, t3;
  float dbgTrig, dbgRight, dbgElbow, dbgWrist;
  if (!computeInverseKinematics(x, y, z, t0, t1, t2, t3, dbgTrig, dbgRight, dbgElbow, dbgWrist)) {
    Serial.println("IK unreachable.");
    Serial.println("DONE");
    armBusy = false;
    Serial.println("READY");
    return;
  }

  Serial.print("Shoulder trig angle (deg): ");
  Serial.println(dbgTrig);
  Serial.print("Shoulder right angle (deg): ");
  Serial.println(dbgRight);
  Serial.print("Elbow angle (deg): ");
  Serial.println(dbgElbow);
  Serial.print("Wrist angle (deg): ");
  Serial.println(dbgWrist);

  float proposed_shoulder = t1 + 5;
  float proposed_elbow = t2 - ((t2 - 90) * 2) - 10;
  float proposed_wrist = t3 - ((t3 - 90) * 2);

  if (!isSafeToMove(proposed_shoulder, proposed_elbow, proposed_wrist)) {
    Serial.println("MOVEMENT BLOCKED - Unsafe angles detected!");
    Serial.println("Arm will not move.");
    Serial.println("DONE");
    armBusy = false;
    Serial.println("READY");
    return;
  }

  targetAngles[SERVO_BASE]     = t0;
  targetAngles[SERVO_SHOULDER] = proposed_shoulder;
  targetAngles[SERVO_ELBOW]    = proposed_elbow;
  targetAngles[SERVO_WRIST]    = proposed_wrist;

  // Move arm with PID control (2 second movement)
  Serial.println("Moving with PID control...");
  moveToTargetAnglesWithPID(2000);

  // FK check of the IK actual angles 
  float base_math   = currentAngles[SERVO_BASE];
  float shoulder_math = (currentAngles[SERVO_SHOULDER] - 5);
  float elbow_math  = currentAngles[SERVO_ELBOW] + 10;
  float fkx, fky, fkz;
  computeForwardKinematics(base_math, shoulder_math, elbow_math, fkx, fky, fkz);

  Serial.println("=== IK RESULTS ==="); 
  Serial.print("IK Angles: "); 
  Serial.print(t0); Serial.print(" "); 
  Serial.print(t1); Serial.print(" "); 
  Serial.println(t2); 

  Serial.print("Actual Servo Angles: ");
  Serial.print(currentAngles[0]); Serial.print(" ");
  Serial.print(currentAngles[1]); Serial.print(" ");
  Serial.println(currentAngles[2]);

  Serial.print("FK of IK angles: ");
  Serial.print(fkx); Serial.print(" ");
  Serial.print(fky); Serial.print(" ");
  Serial.println(fkz);

  Serial.println("===================");
  Serial.println(" ");

  Serial.println("DONE");
  armBusy = false;
  Serial.println("READY");
}
