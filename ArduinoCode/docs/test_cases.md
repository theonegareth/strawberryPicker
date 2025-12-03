# Test Cases for FK/IK Validation

## Expected Test Results (Manual Calculations)

### Test Case 1: θ0=90°, θ1=90°, θ2=80° (Your Working Case)
- **FK Expected Output**: x=0, y=22.28, z=22.28
- **IK Should Return**: θ0=90°, θ1=90°, θ2=80° (exact match)
- **FK-IK-FK Error**: < 0.01 cm

### Test Case 2: θ0=90°, θ1=90°, θ2=100° (Your Problematic Case)
- **FK Expected Output**: x=0, y=22.28, z=17.48
- **IK Should Return**: θ0=90°, θ1=90°, θ2=100° (exact match)
- **FK-IK-FK Error**: < 0.01 cm

### Test Case 3: θ0=0°, θ1=45°, θ2=135°
- **FK Calculation**:
  - t0=0°, t1=45°, t2_rel=(135°-45°)=90°
  - y_arm = 20*cos(45°) + 14.5*cos(90°) + 8 = 14.14 + 0 + 8 = 22.14
  - z_arm = 20*sin(45°) - 14.5*sin(90°) = 14.14 - 14.5 = -0.36
  - x = 22.14*cos(0°) = 22.14
  - y = 22.14*sin(0°) = 0
  - z = -0.36
- **IK Should Return**: θ0=0°, θ1=45°, θ2=135°

### Test Case 4: θ0=0°, θ1=120°, θ2=150°
- **FK Calculation**:
  - t0=0°, t1=120°, t2_rel=(150°-120°)=30°
  - y_arm = 20*cos(120°) + 14.5*cos(30°) + 8 = -10 + 12.56 + 8 = 10.56
  - z_arm = 20*sin(120°) - 14.5*sin(30°) = 17.32 - 7.25 = 10.07
  - x = 10.56*cos(0°) = 10.56
  - y = 10.56*sin(0°) = 0
  - z = 10.07
- **IK Should Return**: θ0=0°, θ1=120°, θ2=150°

## Validation Protocol

### Step 1: FK→IK→FK Consistency Test
For each test case:
1. Run FK with input angles → get xyz coordinates
2. Run IK with xyz coordinates → get angles back
3. Run FK with IK angles → get xyz coordinates
4. Check: |original_xyz - final_xyz| < tolerance

### Step 2: Elbow Configuration Test
- Test with `elbow_down=true` (your desired config)
- Test with `elbow_down=false` (should produce different angles but same xyz)
- Verify both configurations reach same target position

### Step 3: Edge Case Tests
- **Minimum reach**: θ1=0°, θ2=0° → C = |L1 - L2| = 5.5 cm
- **Maximum reach**: θ1=180°, θ2=0° → C = L1 + L2 = 34.5 cm
- **Unreachable targets**: C > L1+L2 or C < |L1-L2|

## Expected Behavior Changes

### Current Implementation Issues:
- θ2=100° case produces wrong coordinates (25.98, 25.98, 19.7)
- θ1 shifts unexpectedly when θ2 changes
- Elbow points up instead of down for θ2 > 90°

### Corrected Implementation Expected:
- θ2=100° case produces correct coordinates (0, 22.28, 17.48)
- θ1 remains at 90° when expected
- Elbow consistently points down for elbow_down=true configuration
- FK→IK→FK loop maintains < 0.01 cm accuracy

## Serial Commands for Testing

Add this validation function to your main.cpp:

```cpp
void runValidationTests() {
  Serial.println("Running FK/IK Validation Tests...");
  
  float test_angles[4][3] = {
    {90, 90, 80},   // Working case
    {90, 90, 100},  // Problematic case
    {0, 45, 135},   // Different config
    {0, 120, 150}   // Another test
  };
  
  for (int i = 0; i < 4; i++) {
    float t0 = test_angles[i][0];
    float t1 = test_angles[i][1]; 
    float t2 = test_angles[i][2];
    
    // Forward kinematics
    float fx, fy, fz;
    computeForwardKinematics(t0, t1, t2, fx, fy, fz);
    
    // Inverse kinematics (corrected)
    float ik_t0, ik_t1, ik_t2, ik_t3;
    computeInverseKinematicsCorrected(fx, fy, fz, ik_t0, ik_t1, ik_t2, ik_t3, true);
    
    // Validation FK
    float val_fx, val_fy, val_fz;
    computeForwardKinematics(ik_t0, ik_t1, ik_t2, val_fx, val_fy, val_fz);
    
    Serial.print("Test ");
    Serial.print(i+1);
    Serial.print(": FK(");
    Serial.print(t0); Serial.print(",");
    Serial.print(t1); Serial.print(",");
    Serial.print(t2);
    Serial.print(") -> IK(");
    Serial.print(ik_t0); Serial.print(",");
    Serial.print(ik_t1); Serial.print(",");
    Serial.print(ik_t2);
    Serial.print(") -> FK(");
    Serial.print(val_fx); Serial.print(",");
    Serial.print(val_fy); Serial.print(",");
    Serial.print(val_fz);
    Serial.println(")");
    
    float error = sqrt(pow(fx - val_fx, 2) + pow(fy - val_fy, 2) + pow(fz - val_fz, 2));
    Serial.print("  Error: ");
    Serial.println(error);
  }
}
```

Add this command to your serial interface:
- **Command**: `v` (for validation)
- **Action**: Runs all test cases and reports errors