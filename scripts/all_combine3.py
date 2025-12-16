# dual_camera_15cm_tuned.py
"""
FINAL TUNED VERSION
 - Baseline: 15.0 cm
 - Range: 20-40 cm
 - Calibration: Tuned to 0.82 based on your latest log (48.5cm -> 30cm).
"""
import cv2
import numpy as np
from ultralytics import YOLO
import math

# ----------------------------
# USER SETTINGS
# ----------------------------
CAM_A_ID = 2         # LEFT Camera
CAM_B_ID = 1         # RIGHT Camera

FRAME_W = 640
FRAME_H = 408
YOLO_MODEL_PATH = "strawberry.pt"

# --- GEOMETRY & CALIBRATION ---
BASELINE_CM = 15.0   
FOCUS_DIST_CM = 30.0 

# RE-CALIBRATED SCALAR
# Your Raw Reading is approx 36.6cm. Real is 30.0cm.
# 30.0 / 36.6 = ~0.82
DEPTH_SCALAR = 0.82 

# Auto-calculate angle
val = (BASELINE_CM / 2.0) / FOCUS_DIST_CM
calc_yaw_deg = math.degrees(math.atan(val))
YAW_LEFT_DEG = calc_yaw_deg     
YAW_RIGHT_DEG = -calc_yaw_deg   

print(f"--- CONFIGURATION ---")
print(f"1. Baseline: {BASELINE_CM} cm")
print(f"2. Scalar: x{DEPTH_SCALAR} (Reducing raw estimate to match reality)")
print(f"3. REQUIRED ANGLE: +/- {calc_yaw_deg:.2f} degrees")
print(f"---------------------")

# Intrinsics
K_A = np.array([[629.10808758, 0.0, 347.20913144],
                [0.0, 631.11321979, 277.5222819],
                [0.0, 0.0, 1.0]], dtype=np.float64)
dist_A = np.array([-0.35469562, 0.10232556, -0.0005468, -0.00174671, 0.01546246], dtype=np.float64)

K_B = np.array([[1001.67997, 0.0, 367.736216],
                [0.0, 996.698369, 312.866527],
                [0.0, 0.0, 1.0]], dtype=np.float64)
dist_B = np.array([-0.49543094, 0.82826695, -0.00180861, -0.00362202, -1.42667838], dtype=np.float64)

# ----------------------------
# HELPERS
# ----------------------------
def capture_single(cam_id):
    cap = cv2.VideoCapture(cam_id, cv2.CAP_DSHOW)
    if not cap.isOpened(): return None
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)
    for _ in range(5): cap.read() # Warmup
    ret, frame = cap.read()
    cap.release()
    return frame

def build_undistort_maps(K, dist):
    newK, _ = cv2.getOptimalNewCameraMatrix(K, dist, (FRAME_W, FRAME_H), 1.0)
    mapx, mapy = cv2.initUndistortRectifyMap(K, dist, None, newK, (FRAME_W, FRAME_H), cv2.CV_32FC1)
    return mapx, mapy, newK

def detect_on_image(model, img):
    results = model(img, verbose=False)[0]
    dets = []
    for box in results.boxes:
        x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
        cx = int((x1 + x2) / 2)
        cy = int((y1 + y2) / 2)
        conf = float(box.conf[0])
        cls = int(box.cls[0])
        name = model.names.get(cls, str(cls))
        dets.append({'x1':x1,'y1':y1,'x2':x2,'y2':y2,'cx':cx,'cy':cy,'conf':conf,'cls':cls,'name':name})
    return sorted(dets, key=lambda d: d['cx'])

def match_stereo(detL, detR):
    matches = []
    usedR = set()
    for l in detL:
        best_idx = -1
        best_score = 9999
        for i, r in enumerate(detR):
            if i in usedR: continue
            if l['cls'] != r['cls']: continue
            dy = abs(l['cy'] - r['cy'])
            if dy > 60: continue 
            if dy < best_score:
                best_score = dy
                best_idx = i
        if best_idx != -1:
            matches.append((l, detR[best_idx]))
            usedR.add(best_idx)
    return matches

# --- 3D MATH ---
def yaw_to_R_deg(yaw_deg):
    y = math.radians(yaw_deg)
    cy = math.cos(y); sy = math.sin(y)
    return np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], dtype=np.float64)

def build_projection_matrices(newK_A, newK_B, yaw_L, yaw_R, baseline):
    R_L = yaw_to_R_deg(yaw_L)
    R_R = yaw_to_R_deg(yaw_R)
    R_W2A = R_L.T
    t_W2A = np.zeros((3,1)) 
    R_W2B = R_R.T
    C_B_world = np.array([[baseline], [0.0], [0.0]])
    t_W2B = -R_W2B @ C_B_world
    P1 = newK_A @ np.hstack((R_W2A, t_W2A))
    P2 = newK_B @ np.hstack((R_W2B, t_W2B))
    return P1, P2

def triangulate_matrix(dL, dR, P1, P2):
    ptsL = np.array([[float(dL['cx'])],[float(dL['cy'])]], dtype=np.float64)
    ptsR = np.array([[float(dR['cx'])],[float(dR['cy'])]], dtype=np.float64)
    
    Xh = cv2.triangulatePoints(P1, P2, ptsL, ptsR)
    Xh /= Xh[3] 
    
    X = float(Xh[0].item())
    Y = float(Xh[1].item())
    Z_raw = float(Xh[2].item())
    
    return X, Y, Z_raw * DEPTH_SCALAR

def main():
    print("[INFO] Loading YOLO...")
    model = YOLO(YOLO_MODEL_PATH)

    print(f"[INFO] Capturing...")
    frameA = capture_single(CAM_A_ID)
    frameB = capture_single(CAM_B_ID)
    if frameA is None or frameB is None: return

    mapAx, mapAy, newKA = build_undistort_maps(K_A, dist_A)
    mapBx, mapBy, newKB = build_undistort_maps(K_B, dist_B)
    undA = cv2.remap(frameA, mapAx, mapAy, cv2.INTER_LINEAR)
    undB = cv2.remap(frameB, mapBx, mapBy, cv2.INTER_LINEAR)

    detA = detect_on_image(model, undA)
    detB = detect_on_image(model, undB)
    
    matches = match_stereo(detA, detB)
    print(f"--- Matches found: {len(matches)} ---")

    P1, P2 = build_projection_matrices(newKA, newKB, YAW_LEFT_DEG, YAW_RIGHT_DEG, BASELINE_CM)

    combo = np.hstack((undA, undB))
    
    for l, r in matches:
        XYZ = triangulate_matrix(l, r, P1, P2)
        X,Y,Z = XYZ
        
        label = f"Z={Z:.1f}cm"
        print(f"Target ({l['name']}): {label} (X={X:.1f}, Y={Y:.1f})")
        
        cv2.line(combo, (l['cx'], l['cy']), (r['cx']+FRAME_W, r['cy']), (0,255,0), 2)
        cv2.rectangle(combo, (l['x1'], l['y1']), (l['x2'], l['y2']), (0,0,255), 2)
        cv2.rectangle(combo, (r['x1']+FRAME_W, r['y1']), (r['x2']+FRAME_W, r['y2']), (0,0,255), 2)
        cv2.putText(combo, label, (l['cx'], l['cy']-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2)
        
    cv2.imshow("Tuned Depth Result", combo)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()