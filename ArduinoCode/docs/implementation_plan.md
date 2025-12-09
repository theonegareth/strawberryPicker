# Implementation Plan for Corrected IK

## Code Integration Steps

### Step 1: Replace computeInverseKinematics function
In `src/main.cpp`, replace the existing `computeInverseKinematics()` function with:

```cpp
bool computeInverseKinematics(float x, float y, float z,
                              float &theta0, float &theta1, float &theta2, float &theta3)
{
  return computeInverseKinematicsCorrected(x, y, z, theta0, theta1, theta2, theta3, true);
}
```

### Step 2: Add the corrected function
Add the complete `computeInverseKinematicsCorrected()` function before the existing one:

```cpp
// Add this BEFORE the existing computeInverseKinematics() function
bool computeInverseKinematicsCorrected(float x, float y, float z,
                                      float &theta0, float &theta1, float &theta2, float &theta3,
                                      bool elbow_down = true)
{
  // Copy the corrected implementation from corrected_solution.md
  // [Full implementation provided earlier]
}
```

### Step 3: Update Serial Command Interface
Add validation command to `loop()`:

```cpp
else if (input.startsWith("v")) {
  runValidationTests();
}
```

### Step 4: Add Validation Function
Add the `runValidationTests()` function before `setup()`:

```cpp
void runValidationTests() {
  // Copy the test validation code from test_cases.md
  // [Full implementation provided earlier]
}
```

## Expected Results After Implementation

### Your Specific Test Case: θ0=90°, θ1=90°, θ2=100°
**Before (Current broken)**:
- FK input: θ0=90°, θ1=90°, θ2=100°
- FK output: (0, 22.28, 17.48)
- IK output: θ0=90°, θ1=95°, θ2=110°
- FK of IK: (0, 25.98, 19.7) ❌ Wrong!

**After (Corrected)**:
- FK input: θ0=90°, θ1=90°, θ2=100°  
- FK output: (0, 22.28, 17.48)
- IK output: θ0=90°, θ1=90°, θ2=100°
- FK of IK: (0, 22.28, 17.48) ✅ Correct!

### FK-IK-FK Consistency Test
After implementation, running the validation (`v` command) should show:
```
Test 1: Error: < 0.01
Test 2: Error: < 0.01  
Test 3: Error: < 0.01
Test 4: Error: < 0.01
```

### Servo Offset Compatibility
The corrected IK will work with your existing servo offset mapping:
```cpp
targetAngles[SERVO_SHOULDER] = (t1 - (t1 - 90) * 2) + 5;
targetAngles[SERVO_ELBOW]    = t2 - 10;
```

## Migration Path

1. **Backup current implementation**
2. **Replace computeInverseKinematics()** function
3. **Test with working case first** (θ0=90°, θ1=90°, θ2=80°)
4. **Test problematic case** (θ0=90°, θ1=90°, θ2=100°)  
5. **Run validation tests** (`v` command)
6. **Verify servo motion** matches your expected elbow-down configuration

## Key Benefits After Fix

- ✅ FK→IK→FK produces identical coordinates (within <0.01cm)
- ✅ θ2=100° case produces correct z-coordinate (17.48 cm vs wrong 19.7 cm)
- ✅ Elbow consistently points down for θ2 > 90°
- ✅ θ1 remains stable when only θ2 changes
- ✅ Backward compatible with existing servo offset mapping

## Files to Modify

1. **src/main.cpp**:
   - Replace `computeInverseKinematics()` function
   - Add `computeInverseKinematicsCorrected()` function  
   - Add `runValidationTests()` function
   - Update serial command handler for 'v' command

2. **Testing**:
   - Run: `f 90 90 100` (should give coordinates: 0 22.28 17.48)
   - Run: `i 0 22.28 17.48` (should return angles: 90 90 100)
   - Run: `v` (should show all test errors < 0.01)

The solution maintains your existing angle conventions while fixing the core mathematical inconsistency that caused the elbow to flip and θ1 to shift.