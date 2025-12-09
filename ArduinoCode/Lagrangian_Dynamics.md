# Lagrangian Dynamics for 5-DOF Strawberry Picker Robotic Arm

## System Overview

This document provides the complete Lagrangian dynamics derivation for the 5-DOF robotic arm used in the strawberry picker project. The dynamics equations enable computed torque control, gravity compensation, and accurate trajectory tracking.

### Arm Configuration
- **Joint 0 (Base)**: Rotation about z-axis (θ₀)
- **Joint 1 (Shoulder)**: Rotation about y-axis (θ₁)
- **Joint 2 (Elbow)**: Rotation about y-axis (θ₂)
- **Joint 3 (Wrist)**: Rotation about y-axis (θ₃)
- **Joint 4 (Scissor)**: Gripper mechanism (θ₄)

### Link Parameters
```mermaid
graph TD
    subgraph Link Dimensions
        L1[Link 1: L₁ = 20.0 cm]
        L2[Link 2: L₂ = 14.5 cm]
        L3[Link 3: L₃ = 8.0 cm]
    end
    
    subgraph Dynamic Parameters
        m1[m₁: Mass of link 1]
        m2[m₂: Mass of link 2]
        m3[m₃: Mass of link 3]
        m4[m₄: Mass of link 4 (gripper)]
        I1[I₁: Inertia of link 1]
        I2[I₂: Inertia of link 2]
        I3[I₃: Inertia of link 3]
        I4[I₄: Inertia of link 4]
    end
```

## Step 1: Kinetic Energy Calculation

The total kinetic energy is the sum of translational and rotational energy for all links:

**K = Σ(K_translational + K_rotational)**

### Link 0 (Base Rotation)

Only rotational kinetic energy about the z-axis:

```
K₀ = ½ I₀z θ̇₀²
```

### Link 1 (Shoulder + Upper Arm)

**Position of Center of Mass (COM):**
```
r₁ = [0, 0, L₁/2] in link 1 frame
```

**Velocity in base frame:**
```
v₁ = [ -L₁/2 · θ̇₀ · sin(θ₁), 
        L₁/2 · θ̇₀ · cos(θ₁), 
        L₁/2 · θ̇₁ ]
```

**Kinetic Energy:**
```
K₁ = ½ m₁[(L₁/2)²(θ̇₀² + θ̇₁²)] + ½ I₁y θ̇₁² + ½ I₁z θ̇₀² · sin²(θ₁)
```

### Link 2 (Elbow + Forearm)

**Position of COM:**
```
r₂ = [L₁·sin(θ₁), 0, L₁·cos(θ₁) + L₂/2] in base frame
```

**Velocity:**
```
v₂ = [L₁·θ̇₁·cos(θ₁), 
      -L₁·θ̇₀·sin(θ₁), 
      -L₁·θ̇₁·sin(θ₁)]
```

**Kinetic Energy:**
```
K₂ = ½ m₂[L₁²θ̇₁² + (L₁² + L₂²/4 + L₁L₂cos(θ₂))θ̇₀² + (L₂²/4)θ̇₂² + L₁L₂cos(θ₂)θ̇₀θ̇₂]
   + ½ I₂y(θ̇₁ + θ̇₂)² + ½ I₂z[θ̇₀²sin²(θ₁ + θ₂)]
```

### Link 3 (Wrist + Hand)

**Position of COM:**
```
r₃ = [L₁·sin(θ₁) + L₂·sin(θ₁+θ₂), 
      0, 
      L₁·cos(θ₁) + L₂·cos(θ₁+θ₂) + L₃/2]
```

**Kinetic Energy:**
```
K₃ = ½ m₃[(L₁² + L₂² + 2L₁L₂cos(θ₂))θ̇₁² 
          + (L₁² + L₂² + 2L₁L₂cos(θ₂))θ̇₀²
          + (L₃²/4)(θ̇₁ + θ̇₂ + θ̇₃)²
          + cross coupling terms]
   + ½ I₃y(θ̇₁ + θ̇₂ + θ̇₃)² + rotational terms
```

### Link 4 (Scissor Mechanism)

```
K₄ = ½ m₄ v₄² + ½ I₄ θ̇₄²
```

## Step 2: Potential Energy Calculation

Total potential energy due to gravity: **P = Σ m_i g h_i**

### Link 1:
```
P₁ = m₁ g (L₁/2) sin(θ₁)
```

### Link 2:
```
P₂ = m₂ g [L₁ sin(θ₁) + (L₂/2) sin(θ₁ + θ₂)]
```

### Link 3:
```
P₃ = m₃ g [L₁ sin(θ₁) + L₂ sin(θ₁ + θ₂) + (L₃/2) sin(θ₁ + θ₂ + θ₃)]
```

### Link 4:
```
P₄ = m₄ g [L₁ sin(θ₁) + L₂ sin(θ₁ + θ₂) + L₃ sin(θ₁ + θ₂ + θ₃)]
```

## Step 3: The Lagrangian

The Lagrangian is defined as the difference between kinetic and potential energy:

```
L = K - P = Σ(K_i) - Σ(P_i)
```

In matrix form:
```
L(θ, θ̇) = ½ Σᵢ Σⱼ Mᵢⱼ(θ) θ̇ᵢ θ̇ⱼ + Σᵢ cᵢ(θ, θ̇) - Σᵢ gᵢ(θ)
```

## Step 4: Euler-Lagrange Equations

For each joint i (i = 0 to 4), the Euler-Lagrange equation is:

```
d/dt(∂L/∂θ̇ᵢ) - ∂L/∂θᵢ = τᵢ
```

This yields the **equations of motion**:

```
Σⱼ Mᵢⱼ(θ) θ̈ⱼ + Σⱼ Σₖ Cᵢⱼₖ(θ) θ̇ⱼ θ̇ₖ + Gᵢ(θ) = τᵢ
```

### Expanded Form for Joint 0 (Base):

```
M₀₀(θ)θ̈₀ + M₀₁(θ)θ̈₁ + M₀₂(θ)θ̈₂ + M₀₃(θ)θ̈₃ + M₀₄(θ)θ̈₄
+ C₀₀₀θ̇₀² + C₀₁₁θ̇₁² + C₀₂₂θ̇₂² + C₀₃₃θ̇₃² + C₀₄₄θ̇₄²
+ 2C₀₀₁θ̇₀θ̇₁ + 2C₀₀₂θ̇₀θ̇₂ + ... + G₀(θ) = τ₀
```

## Step 5: Matrix Form of Dynamics

The complete dynamics can be expressed as:

```
τ = M(θ)θ̈ + C(θ, θ̇)θ̇ + G(θ) + F(θ̇) + τ_ext
```

### Mass Matrix M(θ):

```
M(θ) = [M₀₀ M₀₁ M₀₂ M₀₃ M₀₄
        M₁₀ M₁₁ M₁₂ M₁₃ M₁₄
        M₂₀ M₂₁ M₂₂ M₂₃ M₂₄
        M₃₀ M₃₁ M₃₂ M₃₃ M₃₄
        M₄₀ M₄₁ M₄₂ M₄₃ M₄₄]
```

**Key elements:**

```
M₀₀ = I₀z + I₁z·sin²(θ₁) + m₁(L₁/2)²sin²(θ₁) 
      + I₂z·sin²(θ₁+θ₂) + m₂[L₁² + L₂² + 2L₁L₂cos(θ₂)]sin²(θ₁+θ₂)
      + similar terms for link 3

M₁₁ = I₁y + m₁(L₁/2)² + I₂y + m₂[L₁² + L₂²/4 + L₁L₂cos(θ₂)]
      + I₃y + m₃[(L₁² + L₂² + 2L₁L₂cos(θ₂)) + L₃²/4] + ...

M₂₂ = I₂y + m₂(L₂²)/4 + I₃y + m₃[L₂² + L₃²/4 + L₂L₃cos(θ₃)] + ...

M₁₂ = M₂₁ = m₂[L₁L₂cos(θ₂) + L₂²/2] + m₃[L₁L₂cos(θ₂) + L₂² + L₂L₃cos(θ₃)] + ...
```

### Coriolis/Centrifugal Matrix C(θ, θ̇):

```
C(θ, θ̇)θ̇ = [c₀(θ, θ̇), c₁(θ, θ̇), c₂(θ, θ̇), c₃(θ, θ̇), c₄(θ, θ̇)]ᵀ

c₀ = -[I₁z + m₁(L₁/2)²]·θ̇₀θ̇₁·sin(2θ₁)
     -[I₂z + m₂(L₁² + L₂² + 2L₁L₂cosθ₂)]·θ̇₀(θ̇₁+θ̇₂)·sin[2(θ₁+θ₂)]
     + m₂L₁L₂sin(θ₂)·θ̇₀θ̇₂·sin(θ₁+θ₂) + ...
```

### Gravity Vector G(θ):

```
G₀ = 0  (base rotation doesn't affect potential energy)

G₁ = m₁g(L₁/2)cos(θ₁) 
     + m₂g[L₁cos(θ₁) + (L₂/2)cos(θ₁+θ₂)]
     + m₃g[L₁cos(θ₁) + L₂cos(θ₁+θ₂) + (L₃/2)cos(θ₁+θ₂+θ₃)]
     + m₄g[L₁cos(θ₁) + L₂cos(θ₁+θ₂) + L₃cos(θ₁+θ₂+θ₃)]

G₂ = m₂g(L₂/2)cos(θ₁+θ₂)
     + m₃g[L₂cos(θ₁+θ₂) + (L₃/2)cos(θ₁+θ₂+θ₃)]
     + m₄g[L₂cos(θ₁+θ₂) + L₃cos(θ₁+θ₂+θ₃)]

G₃ = m₃g(L₃/2)cos(θ₁+θ₂+θ₃) + m₄g L₃cos(θ₁+θ₂+θ₃)

G₄ = 0  (scissor gripper, minimal vertical movement)
```

## Step 6: Practical Arduino Implementation

### Simplified Dynamics for Real-Time Control

For Arduino implementation, use these simplified equations:

```cpp
// Dynamic parameters (MEASURE THESE!)
const float m1 = 0.15;    // kg, link 1 mass
const float m2 = 0.12;    // kg, link 2 mass
const float m3 = 0.08;    // kg, link 3 mass
const float m4 = 0.05;    // kg, gripper mass

const float L1 = 0.20;    // m, link 1 length
const float L2 = 0.145;   // m, link 2 length
const float L3 = 0.08;    // m, link 3 length

const float g = 9.81;     // m/s², gravity

// Inertia values (estimate or measure)
const float I1y = 0.0005; // kg·m²
const float I2y = 0.0003;
const float I3y = 0.0002;

// Gravity compensation (MOST IMPORTANT)
float computeGravityCompensation(int joint, float theta1, float theta2, float theta3) {
  switch(joint) {
    case 1: // Shoulder
      return m1 * g * (L1/2) * cos(theta1) 
           + m2 * g * (L1 * cos(theta1) + (L2/2) * cos(theta1 + theta2))
           + m3 * g * (L1 * cos(theta1) + L2 * cos(theta1 + theta2) + (L3/2) * cos(theta1 + theta2 + theta3))
           + m4 * g * (L1 * cos(theta1) + L2 * cos(theta1 + theta2) + L3 * cos(theta1 + theta2 + theta3));
    
    case 2: // Elbow
      return m2 * g * (L2/2) * cos(theta1 + theta2)
           + m3 * g * (L2 * cos(theta1 + theta2) + (L3/2) * cos(theta1 + theta2 + theta3))
           + m4 * g * (L2 * cos(theta1 + theta2) + L3 * cos(theta1 + theta2 + theta3));
    
    case 3: // Wrist
      return m3 * g * (L3/2) * cos(theta1 + theta2 + theta3)
           + m4 * g * L3 * cos(theta1 + theta2 + theta3);
    
    default:
      return 0.0;
  }
}

// Diagonal mass matrix (simplified)
float computeMassMatrixElement(int i, int j, float theta1, float theta2, float theta3) {
  if (i == 1 && j == 1) { // Shoulder
    return I1y + m1 * L1 * L1 / 4 
         + I2y + m2 * (L1 * L1 + L2 * L2 / 4 + L1 * L2 * cos(theta2))
         + I3y + m3 * (L1 * L1 + L2 * L2 + 2 * L1 * L2 * cos(theta2) + L3 * L3 / 4);
  }
  else if (i == 2 && j == 2) { // Elbow
    return I2y + m2 * L2 * L2 / 4 
         + I3y + m3 * (L2 * L2 + L3 * L3 / 4 + L2 * L3 * cos(theta3));
  }
  else if (i == 3 && j == 3) { // Wrist
    return I3y + m3 * L3 * L3 / 4;
  }
  return 0.0;
}

// Coriolis terms (simplified)
float computeCoriolisTerm(int joint, float theta1, float theta2, float theta3, 
                         float theta1_dot, float theta2_dot, float theta3_dot) {
  switch(joint) {
    case 1: // Shoulder
      return -m2 * L1 * L2 * sin(theta2) * theta2_dot * theta2_dot
           - m3 * L1 * L2 * sin(theta2) * theta2_dot * theta2_dot;
    
    case 2: // Elbow
      return m2 * L1 * L2 * sin(theta2) * theta1_dot * theta1_dot
           + m3 * L1 * L2 * sin(theta2) * theta1_dot * theta1_dot;
    
    default:
      return 0.0;
  }
}

// Complete torque calculation
void computeJointTorques(float theta[5], float theta_dot[5], float theta_ddot[5], float torque[5]) {
  // Gravity compensation (most important for holding position)
  torque[1] = computeGravityCompensation(1, theta[1], theta[2], theta[3]);
  torque[2] = computeGravityCompensation(2, theta[1], theta[2], theta[3]);
  torque[3] = computeGravityCompensation(3, theta[1], theta[2], theta[3]);
  
  // Add inertial terms (for acceleration)
  torque[1] += computeMassMatrixElement(1, 1, theta[1], theta[2], theta[3]) * theta_ddot[1];
  torque[2] += computeMassMatrixElement(2, 2, theta[1], theta[2], theta[3]) * theta_ddot[2];
  torque[3] += computeMassMatrixElement(3, 3, theta[1], theta[2], theta[3]) * theta_ddot[3];
  
  // Add Coriolis terms (for multi-joint motion)
  float coriolis = computeCoriolisTerm(1, theta[1], theta[2], theta[3], theta_dot[1], theta_dot[2], theta_dot[3]);
  torque[1] += coriolis;
  
  coriolis = computeCoriolisTerm(2, theta[1], theta[2], theta[3], theta_dot[1], theta_dot[2], theta_dot[3]);
  torque[2] += coriolis;
  
  // Base joint (minimal dynamics)
  torque[0] = 0.1 * theta_ddot[0]; // Simplified inertia
  
  // Gripper (minimal dynamics)
  torque[4] = 0.05 * theta_ddot[4];
}
```

### Integration with Existing Code

Add to your `main.cpp`:

```cpp
// Global variables for dynamics
float joint_torques[5] = {0, 0, 0, 0, 0};
float joint_accelerations[5] = {0, 0, 0, 0, 0};

// Call this before moveToTargetAngles()
void updateDynamics() {
  // Estimate joint velocities (simple difference)
  static float prev_angles[5] = {90, 95, 80, 90, 180};
  static unsigned long prev_time = 0;
  
  unsigned long current_time = millis();
  float dt = (current_time - prev_time) / 1000.0; // seconds
  
  if (dt > 0.01) { // Update at ~100Hz
    for (int i = 0; i < 5; i++) {
      joint_velocities[i] = (currentAngles[i] - prev_angles[i]) / dt;
      joint_accelerations[i] = 0.0; // Simplified - could use second derivative
      prev_angles[i] = currentAngles[i];
    }
    prev_time = current_time;
  }
  
  // Compute required torques
  computeJointTorques(currentAngles, joint_velocities, joint_accelerations, joint_torques);
  
  // Print torque values for debugging
  Serial.print("Torques: ");
  for (int i = 0; i < 5; i++) {
    Serial.print(joint_torques[i]);
    Serial.print(" ");
  }
  Serial.println();
}
```

## Step 7: Parameter Measurement Guide

### Required Measurements

| Parameter | Symbol | How to Measure | Typical Value |
|-----------|--------|----------------|---------------|
| Link 1 mass | m₁ | Weigh on scale | 0.10-0.20 kg |
| Link 2 mass | m₂ | Weigh on scale | 0.08-0.15 kg |
| Link 3 mass | m₃ | Weigh on scale | 0.05-0.10 kg |
| Gripper mass | m₄ | Weigh on scale | 0.03-0.08 kg |
| Link 1 length | L₁ | Measure/CAD | 0.20 m |
| Link 2 length | L₂ | Measure/CAD | 0.145 m |
| Link 3 length | L₃ | Measure/CAD | 0.08 m |
| Link 1 COM | r₁ | Balance point | ~L₁/2 |
| Link 2 COM | r₂ | Balance point | ~L₂/2 |
| Link 3 COM | r₃ | Balance point | ~L₃/2 |
| Link 1 inertia | I₁y | CAD or pendulum test | 0.0003-0.0008 kg·m² |
| Link 2 inertia | I₂y | CAD or pendulum test | 0.0002-0.0005 kg·m² |
| Link 3 inertia | I₃y | CAD or pendulum test | 0.0001-0.0003 kg·m² |

### Measuring Inertia using Pendulum Method

For each link:
1. Suspend link from pivot near joint
2. Measure period of small oscillations: T = 2π√(I/mgd)
3. Solve for I: I = mgd(T/2π)²
4. Where d = distance from pivot to COM

## Step 8: Control Implementation

### Gravity Compensation (Essential)

```cpp
// Add this to your moveToTargetAngles() function
void moveToTargetAnglesWithGravityComp() {
  // ... existing interpolation code ...
  
  // Add gravity compensation torque
  float grav_comp[5];
  computeJointTorques(currentAngles, joint_velocities, joint_accelerations, grav_comp);
  
  // Apply compensation (convert torque to PWM offset if needed)
  for (int i = 0; i < 5; i++) {
    // Your servo control code here with grav_comp[i] added
  }
}
```

### Computed Torque Control (Advanced)

```cpp
// Full dynamics control
void computedTorqueControl(float desired_pos[5], float desired_vel[5], float desired_acc[5]) {
  float pos_error[5], vel_error[5];
  float feedback_torque[5], feedforward_torque[5];
  
  // Position and velocity errors
  for (int i = 0; i < 5; i++) {
    pos_error[i] = desired_pos[i] - currentAngles[i];
    vel_error[i] = desired_vel[i] - joint_velocities[i];
  }
  
  // PD feedback (tune Kp and Kv)
  float Kp = 0.5; // Proportional gain
  float Kv = 0.1; // Derivative gain
  
  for (int i = 0; i < 5; i++) {
    feedback_torque[i] = Kp * pos_error[i] + Kv * vel_error[i];
  }
  
  // Dynamics feedforward
  computeJointTorques(desired_pos, desired_vel, desired_acc, feedforward_torque);
  
  // Total torque command
  float total_torque[5];
  for (int i = 0; i < 5; i++) {
    total_torque[i] = feedforward_torque[i] + feedback_torque[i];
  }
  
  // Send to servos (implementation depends on servo type)
  // sendTorqueToServos(total_torque);
}
```

## Summary

The Lagrangian dynamics provide the complete model:

```
τ = M(θ)θ̈ + C(θ, θ̇)θ̇ + G(θ)
```

**For strawberry picking:**
1. **Gravity compensation (G)**: Prevents arm sagging when stationary
2. **Mass matrix (M)**: Enables accurate acceleration control
3. **Coriolis terms (C)**: Improves multi-joint coordinated motion

**Implementation priority:**
1. Start with gravity compensation only (biggest impact)
2. Add diagonal mass matrix terms
3. Add Coriolis terms for fast movements
4. Implement full computed torque control

This dynamics model enables your strawberry picker to move more accurately, handle varying payloads, and maintain position without constant servo correction.