# dual_camera_capture_yolo_triangulation_final_corrected.py
"""
Single-run snapshot triangulation:
 - Undistort frames first
 - Run YOLO on undistorted images
 - Keep boxes (NMS/IoU dedupe)
 - Compute centers from those undistorted boxes (exact)
 - Draw boxes + center dot exactly where detection is
 - Match L<->R (labels if same count, else nearest-center fallback)
 - Triangulate using pinhole formula with LEFT=Cam A intrinsics
"""
import cv2
import numpy as np
from ultralytics import YOLO
import math
import itertools
import sys
import time

# ----------------------------
# User settings
# ----------------------------
CAM_A_ID = 1         # left (wide)
CAM_B_ID = 2         # right (normal)
FRAME_W = 640
FRAME_H = 408
YOLO_MODEL_PATH = "face.pt"   # your model
BASELINE_CM = 30.0

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

# ----------------------------
# Helpers
# ----------------------------
def open_cam(cam_id):
    cap = cv2.VideoCapture(cam_id, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)
    return cap

def capture_single(cam_id):
    cap = cv2.VideoCapture(cam_id, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print(f"[ERROR] Camera {cam_id} failed to open.")
        return None
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)
    # warmup
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

# Intersection over union
def iou(a,b):
    xA = max(a['x1'], b['x1']); yA = max(a['y1'], b['y1'])
    xB = min(a['x2'], b['x2']); yB = min(a['y2'], b['y2'])
    interW = max(0, xB - xA); interH = max(0, yB - yA)
    inter = interW * interH
    areaA = max(1, (a['x2']-a['x1']) * (a['y2']-a['y1']))
    areaB = max(1, (b['x2']-b['x1']) * (b['y2']-b['y1']))
    return inter / (areaA + areaB - inter + 1e-9)

# Remove overlapping duplicates (keep higher-conf)
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
                # keep highest conf
                if dets[j]['conf'] > dets[keep]['conf']:
                    keep = j
                used[j] = True
        used[keep] = True
        final.append(dets[keep])
    return final

# Detect on *undistorted* image and return boxes (x1,y1,x2,y2) and centers
def detect_on_image(model, img):
    r = model(img)[0]
    dets = []
    for box in r.boxes:
        # box.xyxy is a tensor [[x1,y1,x2,y2]]
        x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
        cx = int((x1 + x2) / 2)
        cy = int((y1 + y2) / 2)
        conf = float(box.conf[0])
        cls = int(box.cls[0])
        name = model.names.get(cls, str(cls))
        dets.append({'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2,
                     'cx': cx, 'cy': cy, 'conf': conf, 'cls': cls, 'name': name})
    return dets

# Label left->right
def label_left_to_right(dets):
    dets_sorted = sorted(dets, key=lambda d: d['cx'])
    for i, d in enumerate(dets_sorted, 1):
        d['label'] = i
    return dets_sorted

# Draw boxes and center dot exactly from undistorted boxes
def draw_all(img, dets):
    out = img.copy()
    for d in dets:
        cv2.rectangle(out, (d['x1'], d['y1']), (d['x2'], d['y2']), (0,0,255), 2)
        cv2.circle(out, (d['cx'], d['cy']), 6, (0,255,0), -1)   # exact center dot
        txt = f"{d.get('label', '?')}. {d['name']} {d['conf']:.2f}"
        cv2.putText(out, txt, (d['x1'], max(d['y1']-8, 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,255,0), 2)
    return out

# Matching: if counts equal -> pair by index; else nearest-center (with class penalty)
def match_detections(detA, detB):
    matches = []
    if not detA or not detB:
        return matches
    if len(detA) == len(detB):
        matches = [(a,b) for a,b in zip(detA, detB)]
        return matches
    # cost matrix
    costs = np.zeros((len(detA), len(detB)), dtype=np.float64)
    for i,a in enumerate(detA):
        for j,b in enumerate(detB):
            class_penalty = 0.0 if a['cls'] == b['cls'] else 40.0
            costs[i,j] = math.hypot(a['cx'] - b['cx'], a['cy'] - b['cy']) + class_penalty
    usedB = set()
    for i in range(len(detA)):
        j = int(np.argmin(costs[i]))
        if j in usedB:
            # find next best
            sorted_idx = np.argsort(costs[i])
            found = False
            for idx in sorted_idx:
                if idx not in usedB and costs[i, idx] <= MATCH_DISTANCE_THRESH:
                    j = int(idx); found = True; break
            if not found:
                continue
        if costs[i,j] <= MATCH_DISTANCE_THRESH:
            matches.append((detA[i], detB[j]))
            usedB.add(j)
    return matches

# Triangulate pinhole (LEFT = Cam A, RIGHT = Cam B)
def triangulate_pinhole(dL, dR, newK_left, baseline_cm):
    # dL from left camera (A), dR from right camera (B)
    uL, vL = dL['cx'], dL['cy']
    uR, vR = dR['cx'], dR['cy']
    fx = newK_left[0,0]; fy = newK_left[1,1]; ox = newK_left[0,2]; oy = newK_left[1,2]
    disparity = abs(uL - uR)  # Use absolute value to avoid negative Z
    if disparity < 1e-6:
        return None
    Z = (baseline_cm * fx) / disparity
    X = Z * (uL - ox) / fx
    Y = Z * (vL - oy) / fy
    return X, Y, Z, disparity, fx

def pixel_to_quadrant(cx, cy):
    cx0 = cx - FRAME_W//2
    cy0 = (FRAME_H//2) - cy
    return cx0, cy0

# ----------------------------
# Main
# ----------------------------
def main():
    print("[INFO] Loading YOLO model...")
    try:
        model = YOLO(YOLO_MODEL_PATH)
    except Exception as e:
        print("[ERROR] YOLO failed to load:", e)
        return

    print("[INFO] Capturing a single frame from both cameras...")
    frameA = capture_single(CAM_A_ID)
    frameB = capture_single(CAM_B_ID)
    if frameA is None or frameB is None:
        print("[FATAL] Camera capture failed")
        return

    # Build undistort maps and undistort frames
    mapAx, mapAy, newK_A = build_undistort_maps(K_A, dist_A)
    mapBx, mapBy, newK_B = build_undistort_maps(K_B, dist_B)
    undA = cv2.remap(frameA, mapAx, mapAy, cv2.INTER_LINEAR)
    undB = cv2.remap(frameB, mapBx, mapBy, cv2.INTER_LINEAR)

    # Run YOLO on UNDISTORTED images (this is critical)
    detA_raw = detect_on_image(model, undA)
    detB_raw = detect_on_image(model, undB)

    # Hard dedupe (IoU)
    detA_clean = merge_overlapping(detA_raw, IOU_DEDUPE_THRESH)
    detB_clean = merge_overlapping(detB_raw, IOU_DEDUPE_THRESH)

    # Recompute centers from undistorted boxes (they are already computed in detect_on_image)
    # Label left->right (A then B)
    detA = label_left_to_right(detA_clean)
    detB = label_left_to_right(detB_clean)

    print("\n--- DET A ---")
    print(detA)
    print("\n--- DET B ---")
    print(detB)

    # Match
    matches = match_detections(detA, detB)
    print("\n--- MATCHES ---")
    for a,b in matches:
        print(f"L{a['label']} <-> R{b['label']}  centers: A({a['cx']},{a['cy']}) B({b['cx']},{b['cy']})")

    # Make combo image with A left, B right (consistent)
    combo = np.zeros((FRAME_H, FRAME_W*2, 3), dtype=np.uint8)
    combo[:, :FRAME_W] = undA
    combo[:, FRAME_W:] = undB

    # draw boxes on combo and separate windows
    undA_vis = draw_all(undA, detA)
    undB_vis = draw_all(undB, detB)

    # draw match lines and labels on combo
    for a,b in matches:
        x1, y1 = int(a['cx']), int(a['cy'])
        x2, y2 = int(b['cx']) + FRAME_W, int(b['cy'])
        cv2.line(combo, (x1, y1), (x2, y2), (255, 200, 0), 2)
        cv2.putText(combo, f"L{a['label']}", (x1-10, y1-12), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0),2)
        cv2.putText(combo, f"R{b['label']}", (x2-10, y2-12), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0),2)

    # Triangulate matched pairs (LEFT = Cam A, RIGHT = Cam B)
    print("\n--- 3D RESULTS ---")
    any_valid = False
    for a, b in matches:
        # a is from Cam A (left), b is from Cam B (right)
        tri = triangulate_pinhole(a, b, newK_A, BASELINE_CM)
        if tri is None:
            print(f"L{a['label']} -> invalid disparity (skip)")
            continue
        X, Y, Z, disp, fx = tri
        any_valid = True
        qAx, qAy = pixel_to_quadrant(a['cx'], a['cy'])
        qBx, qBy = pixel_to_quadrant(b['cx'], b['cy'])
        print(f"L{a['label']} ({a['name']}):")
        print(f"  pixel centers A: ({a['cx']},{a['cy']})  B: ({b['cx']},{b['cy']})")
        print(f"  quadrant A: ({qAx},{qAy})  B: ({qBx},{qBy})")
        print(f"  disparity = |uL-uR| = {disp:.3f} px, fx={fx:.3f}")
        print(f"  TRIANGULATED (cm) -> X={X:.2f}, Y={Y:.2f}, Z={Z:.2f}")

        # overlay Z near the box (left & right)
        cv2.putText(undA_vis, f"Z={Z:.1f}cm", (a['x1'], a['y2'] + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)
        cv2.putText(undB_vis, f"Z={Z:.1f}cm", (b['x1'], b['y2'] + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)
        # also on combo (offset right box coords by FRAME_W)
        cv2.putText(combo, f"Z={Z:.1f}cm", (a['x1'], a['y2'] + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)
        cv2.putText(combo, f"Z={Z:.1f}cm", (b['x1'] + FRAME_W, b['y2'] + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)

    if not any_valid:
        print("[WARN] No valid triangulations found. Check matches/disparity.")

    # Show images (boxes drawn from undistorted coordinates)
    cv2.imshow("Left undistorted (Cam A)", undA_vis)
    cv2.imshow("Right undistorted (Cam B)", undB_vis)
    cv2.imshow("Matches (A|B)", combo)

    print("[INFO] Press any key to exit...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
