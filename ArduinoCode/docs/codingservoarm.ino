#include <Wire.h>
#include <Adafruit_PWMServoDriver.h>

Adafruit_PWMServoDriver pwm = Adafruit_PWMServoDriver();

// Servo pulse limits (tune for your specific servos)
#define SERVOMIN  150  // 0 degrees
#define SERVOMAX  600  // 180 degrees

float currentAngle = 0;  // Start at 90 degrees
float targetAngle = 0;

int angleToPulse(float angle) {
  return map((int)angle, 0, 180, SERVOMIN, SERVOMAX);
}

void setup() {
  Serial.begin(9600);
  Serial.println("Type an angle (0–180) and press Enter:");

  pwm.begin();
  pwm.setPWMFreq(50);
  delay(10);

  // Initialize both servos to center position
  int pulse = angleToPulse(currentAngle);
  pwm.setPWM(0, 0, pulse);
  pwm.setPWM(1, 0, pulse);
}

void loop() {
  if (Serial.available()) {
    int angle = Serial.parseInt();

    if (angle >= 0 && angle <= 180) {
      targetAngle = angle;
      Serial.print("Slowly moving servos to ");
      Serial.print(targetAngle);
      Serial.println(" degrees...");

      // Smoothly step toward target angle
      float step = 0.10;     // smaller = smoother and slower
      int delayTime = 5;   // larger = slower (try 50 or 70 for extra slow)

      if (targetAngle > currentAngle) {
        for (float pos = currentAngle; pos <= targetAngle; pos += step) {
          int pulse = angleToPulse(pos);
          pwm.setPWM(0, 0, pulse);
          pwm.setPWM(1, 0, pulse);
          delay(delayTime);
        }
      } else {
        for (float pos = currentAngle; pos >= targetAngle; pos -= step) {
          int pulse = angleToPulse(pos);
          pwm.setPWM(0, 0, pulse);
          pwm.setPWM(1, 0, pulse);
          delay(delayTime);
        }
      }

      currentAngle = targetAngle;
      Serial.println("Done!");
    } else {
      Serial.println("Please enter an angle between 0 and 180.");
    }

    // Clear serial buffer
    while (Serial.available()) Serial.read();
  }
}
