# dual_camera_capture_yolo_triangulation_final_proj.py
"""
Corrected single-snapshot triangulation:
 - Undistort frames first
 - Run YOLO on undistorted images (so boxes and centers align visually)
 - Dedupe / label / match
 - Build projection matrices from intrinsics + yaw + baseline
 - Triangulate with cv2.triangulatePoints (correct for rotated cameras)
 - Draw boxes + exact center dot and triangulated Z
Units: baseline in cm -> X,Y,Z in cm
"""
import cv2
import numpy as np
from ultralytics import YOLO
import math
import sys
import serial
import time


# ---------- USER SETTINGS ----------
CAM_A_ID = 1         # left (wide)
CAM_B_ID = 2         # right (normal)
FRAME_W = 640
FRAME_H = 408
YOLO_MODEL_PATH = "face.pt"
BASELINE_CM = 23.0
SERIAL_PORT = "COM4"   # CHANGE THIS
BAUDRATE = 9600



# If you need to change yaw signs, change these two values (degrees)
# Assumption: +yaw = rotate camera to the RIGHT (around camera Y axis)
YAW_LEFT_DEG = +10.0   # left camera rotated inward toward center
YAW_RIGHT_DEG = -10.0  # right camera rotated inward toward center

# Intrinsics (from you)
K_A = np.array([[629.10808758, 0.0, 347.20913144],
                [0.0, 631.11321979, 277.5222819],
                [0.0, 0.0, 1.0]], dtype=np.float64)
dist_A = np.array([-0.35469562, 0.10232556, -0.0005468, -0.00174671, 0.01546246], dtype=np.float64)

K_B = np.array([[1001.67997, 0.0, 367.736216],
                [0.0, 996.698369, 312.866527],
                [0.0, 0.0, 1.0]], dtype=np.float64)
dist_B = np.array([-0.49543094, 0.82826695, -0.00180861, -0.00362202, -1.42667838], dtype=np.float64)

IOU_DEDUPE_THRESH = 0.45
MATCH_DISTANCE_THRESH = 180.0  # px

# ---------- HELPERS ----------
def capture_single(cam_id):
    cap = cv2.VideoCapture(cam_id, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print(f"[ERROR] Camera {cam_id} failed to open.")
        return None
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)
    for _ in range(4):
        cap.read()
    ret, frame = cap.read()
    cap.release()
    if not ret:
        print(f"[ERROR] Camera {cam_id} failed to capture.")
        return None
    return frame

def build_undistort_maps(K, dist):
    newK, _ = cv2.getOptimalNewCameraMatrix(K, dist, (FRAME_W, FRAME_H), 1.0)
    mapx, mapy = cv2.initUndistortRectifyMap(K, dist, None, newK, (FRAME_W, FRAME_H), cv2.CV_32FC1)
    return mapx, mapy, newK

def iou(a,b):
    xA = max(a['x1'], b['x1']); yA = max(a['y1'], b['y1'])
    xB = min(a['x2'], b['x2']); yB = min(a['y2'], b['y2'])
    interW = max(0, xB - xA); interH = max(0, yB - yA)
    inter = interW * interH
    areaA = max(1, (a['x2']-a['x1'])*(a['y2']-a['y1']))
    areaB = max(1, (b['x2']-b['x1'])*(b['y2']-b['y1']))
    return inter / (areaA + areaB - inter + 1e-9)

def merge_overlapping(dets, iou_thresh=IOU_DEDUPE_THRESH):
    if not dets:
        return []
    dets = dets.copy()
    used = [False]*len(dets)
    final = []
    for i in range(len(dets)):
        if used[i]:
            continue
        keep = i
        for j in range(i+1, len(dets)):
            if used[j]:
                continue
            if iou(dets[i], dets[j]) > iou_thresh:
                if dets[j]['conf'] > dets[keep]['conf']:
                    keep = j
                used[j] = True
        used[keep] = True
        final.append(dets[keep])
    return final

def detect_on_image(model, img):
    r = model(img)[0]
    dets = []
    for box in r.boxes:
        x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
        cx = int((x1 + x2) / 2)
        cy = int((y1 + y2) / 2)
        conf = float(box.conf[0])
        cls = int(box.cls[0])
        name = model.names.get(cls, str(cls))
        dets.append({'x1':x1,'y1':y1,'x2':x2,'y2':y2,'cx':cx,'cy':cy,'conf':conf,'cls':cls,'name':name})
    return dets

def label_left_to_right(dets):
    dets_sorted = sorted(dets, key=lambda d: d['cx'])
    for i,d in enumerate(dets_sorted, 1):
        d['label'] = i
    return dets_sorted

def draw_all(img, dets):
    out = img.copy()
    for d in dets:
        cv2.rectangle(out, (d['x1'], d['y1']), (d['x2'], d['y2']), (0,0,255), 2)
        cv2.circle(out, (d['cx'], d['cy']), 6, (0,255,0), -1)
        txt = f"{d.get('label','?')}. {d['name']} {d['conf']:.2f}"
        cv2.putText(out, txt, (d['x1'], max(d['y1']-8,12)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,255,0), 2)
    return out

def match_detections(detA, detB):
    matches=[]
    if not detA or not detB:
        return matches
    if len(detA) == len(detB):
        return [(a,b) for a,b in zip(detA, detB)]
    costs = np.zeros((len(detA), len(detB)), dtype=np.float64)
    for i,a in enumerate(detA):
        for j,b in enumerate(detB):
            class_penalty = 0.0 if a['cls']==b['cls'] else 40.0
            costs[i,j] = math.hypot(a['cx']-b['cx'], a['cy']-b['cy']) + class_penalty
    usedB=set()
    for i in range(len(detA)):
        j = int(np.argmin(costs[i]))
        if j in usedB:
            sorted_idx = np.argsort(costs[i])
            found=False
            for idx in sorted_idx:
                if idx not in usedB and costs[i,idx] <= MATCH_DISTANCE_THRESH:
                    j=int(idx); found=True; break
            if not found:
                continue
        if costs[i,j] <= MATCH_DISTANCE_THRESH:
            matches.append((detA[i], detB[j])); usedB.add(j)
    return matches

# Build rotation matrix for yaw (around Y axis) with convention + = rotate to RIGHT
def yaw_to_R_deg(yaw_deg):
    y = math.radians(yaw_deg)
    cy = math.cos(y); sy = math.sin(y)
    # Rotation about Y axis
    R = np.array([[ cy, 0.0, sy],
                  [ 0.0,1.0, 0.0],
                  [-sy, 0.0, cy]], dtype=np.float64)
    return R

# Build P1 and P2 (3x4) from new intrinsics and yaw/baseline assumptions
def build_projection_matrices(newK_A, newK_B, yaw_left_deg, yaw_right_deg, baseline_cm):
    # Left camera rotation and translation in world (choose left camera frame as world)
    R_left = yaw_to_R_deg(yaw_left_deg)
    R_right = yaw_to_R_deg(yaw_right_deg)
    # Relative rotation from left to right
    R_rel = R_right @ R_left.T
    # translation: right camera center expressed in left camera coordinates
    T_rel = np.array([ [baseline_cm], [0.0], [0.0] ], dtype=np.float64)  # cm
    # Projection matrices P = K * [R | t]
    P1 = newK_A @ np.hstack((np.eye(3, dtype=np.float64), np.zeros((3,1), dtype=np.float64)))
    P2 = newK_B @ np.hstack((R_rel, T_rel))
    return P1, P2, R_rel, T_rel

def triangulate_with_Ps(dL, dR, P1, P2):
    # points must be 2xN floats
    ptsL = np.array([[float(dL['cx'])],[float(dL['cy'])]], dtype=np.float64)
    ptsR = np.array([[float(dR['cx'])],[float(dR['cy'])]], dtype=np.float64)
    Xh = cv2.triangulatePoints(P1, P2, ptsL, ptsR)  # 4xN
    if Xh is None or Xh.shape[1]==0:
        return None
    X = Xh[:,0]
    if abs(X[3]) < 1e-9:
        return None
    X = X / X[3]
    # X now in same linear units as T_rel (cm)
    return float(X[0]), float(X[1]), float(X[2])

# ---------- SERIAL IK SENDING ----------
def open_serial():
    ser = serial.Serial(SERIAL_PORT, BAUDRATE, timeout=1)
    time.sleep(2)  # Arduino reset
    return ser

def wait_for_ready(ser):
    while True:
        line = ser.readline().decode(errors="ignore").strip()
        if line:
            print("[ARDUINO]", line)
        if line == "READY":
            return

def send_ik(ser, x, y, z):
    wait_for_ready(ser)
    cmd = f"i {x:.1f} {y:.1f} {z:.1f}\n"
    print("[SEND]", cmd.strip())
    ser.write(cmd.encode())


# ---------- MAIN ----------
def main():
    print("[INFO] Loading YOLO model...")
    try:
        model = YOLO(YOLO_MODEL_PATH)
    except Exception as e:
        print("[ERROR] YOLO failed to load:", e); return
    
    print("[INFO] Opening serial...")
    ser = open_serial()
    wait_for_ready(ser)

    print("[INFO] Capturing frames...")
    frameA = capture_single(CAM_A_ID)
    frameB = capture_single(CAM_B_ID)
    if frameA is None or frameB is None:
        print("[FATAL] Camera capture failed"); return

    # Undistort (and get new intrinsics)
    mapAx, mapAy, newK_A = build_undistort_maps(K_A, dist_A)
    mapBx, mapBy, newK_B = build_undistort_maps(K_B, dist_B)
    undA = cv2.remap(frameA, mapAx, mapAy, cv2.INTER_LINEAR)
    undB = cv2.remap(frameB, mapBx, mapBy, cv2.INTER_LINEAR)

    # Detect on undistorted images
    detA_raw = detect_on_image(model, undA)
    detB_raw = detect_on_image(model, undB)

    # Dedupe / label
    detA = label_left_to_right(merge_overlapping(detA_raw))
    detB = label_left_to_right(merge_overlapping(detB_raw))

    print("\n--- DET A ---"); print(detA)
    print("\n--- DET B ---"); print(detB)

    # Matching
    matches = match_detections(detA, detB)
    print("\n--- MATCHES ---")
    for a,b in matches:
        print(f"L{a['label']} <-> R{b['label']}  centers: A({a['cx']},{a['cy']}) B({b['cx']},{b['cy']})")

    # Build projection matrices (using yaw assumptions)
    P1, P2, Rrel, Trel = build_projection_matrices(newK_A, newK_B, YAW_LEFT_DEG, YAW_RIGHT_DEG, BASELINE_CM)
    print("\nP1:\n", P1); print("\nP2:\n", P2)
    print("\nR_rel:\n", Rrel); print("T_rel (cm):", Trel.ravel())

    # combo for visualization
    combo = np.zeros((FRAME_H, FRAME_W*2, 3), dtype=np.uint8)
    combo[:, :FRAME_W] = undA
    combo[:, FRAME_W:] = undB
    undA_vis = draw_all(undA, detA)
    undB_vis = draw_all(undB, detB)

    # draw match lines
    for a,b in matches:
        x1,y1 = int(a['cx']), int(a['cy'])
        x2,y2 = int(b['cx'])+FRAME_W, int(b['cy'])
        cv2.line(combo, (x1,y1), (x2,y2), (255,200,0), 2)
        cv2.putText(combo, f"L{a['label']}", (x1-10,y1-12), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
        cv2.putText(combo, f"R{b['label']}", (x2-10,y2-12), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

    # Triangulate with P1,P2
    print("\n--- 3D RESULTS (units = cm) ---")
    any_valid=False
    for a,b in matches:
        XYZ = triangulate_with_Ps(a,b,P1,P2)
        if XYZ is None:
            print(f"L{a['label']} -> triangulation failed")
            continue
        X,Y,Z = XYZ
        any_valid=True
        print(f"L{a['label']} ({a['name']}): X={X:.2f} cm, Y={Y:.2f} cm, Z={Z:.2f} cm") 
        cv2.putText(undA_vis, f"Z={Z:.1f}cm", (a['x1'], a['y2']+18), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255),2) 
        cv2.putText(undB_vis, f"Z={Z:.1f}cm", (b['x1'], b['y2']+18), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255),2) 
        cv2.putText(combo, f"Z={Z:.1f}cm", (a['x1'], a['y2']+18), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255),2) 
        cv2.putText(combo, f"Z={Z:.1f}cm", (b['x1']+FRAME_W, b['y2']+18), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255),2)
        print(f"[TARGET] X={X:.1f} Y={Y:.1f} Z={Z:.1f}")
        send_ik(ser, X, Y, Z)
        break   # IMPORTANT: send only ONE target

    if not any_valid:
        print("[WARN] No valid triangulations (check matches/disparity/yaw/baseline).")

    # show
    cv2.imshow("Left undistorted (Cam A)", undA_vis)
    cv2.imshow("Right undistorted (Cam B)", undB_vis)
    cv2.imshow("Matches (A|B)", combo)
    print("[INFO] Press any key to exit.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    ser.close()


if __name__ == "__main__":
    main()
