# FK/IK Angle Convention Analysis

## Current Problem
- FK: theta2_rel = (theta2 - theta1), z = L1*sin(t1) - L2*sin(theta2_rel)  
- IK: Computes theta2 as absolute joint angle from triangle geometry
- This creates inconsistency when theta2 ≠ theta1 + 90°

## Test Case: theta1=90°, theta2=100°
- FK expects: theta2_rel = (100° - 90°) = 10°
- FK result: z = 20*sin(90°) - 14.5*sin(10°) = 17.48 cm ✓
- IK computes: theta2 = acos(...) = ~80° (triangle angle)
- Current IK result: z = 20*sin(90°) - 14.5*sin(80°) = 5.71 cm ✗

## Solution: Convert IK angle to relative convention