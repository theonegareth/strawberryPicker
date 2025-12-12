# dual_camera_capture_yolo_calibrated_final.py
import cv2
import numpy as np
from ultralytics import YOLO
import sys

# -------------------------------------------------------------
# CAMERA SETTINGS
# -------------------------------------------------------------
CAM_A_ID = 1   # wide
CAM_B_ID = 2   # normal

FRAME_W = 640
FRAME_H = 408

YOLO_MODEL_PATH = "face.pt"

# -------------------------------------------------------------
# CAMERA A CALIBRATION (FINAL)
# -------------------------------------------------------------
K_A = np.array([
    [629.10808758,    0.0,         347.20913144],
    [0.0,             631.11321979, 277.5222819],
    [0.0,               0.0,         1.0]
], dtype=np.float32)

dist_A = np.array([
    -0.35469562,  0.10232556, -0.0005468, -0.00174671, 0.01546246
], dtype=np.float32)

# -------------------------------------------------------------
# CAMERA B CALIBRATION (FINAL)
# -------------------------------------------------------------
K_B = np.array([
    [1001.67997,    0.0,         367.736216],
    [0.0,           996.698369,  312.866527],
    [0.0,             0.0,         1.0]
], dtype=np.float32)

dist_B = np.array([
    -0.49543094,  0.82826695, -0.00180861, -0.00362202, -1.42667838
], dtype=np.float32)


# -------------------------------------------------------------
# CAPTURE SINGLE FRAME SAFELY
# -------------------------------------------------------------
def capture_single(cam_id):
    cap = cv2.VideoCapture(cam_id, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print(f"[ERROR] Camera {cam_id} failed to open.")
        return None

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)

    for _ in range(4):  # warm-up
        cap.read()

    ret, frame = cap.read()
    cap.release()

    if not ret:
        print(f"[ERROR] Camera {cam_id} failed to capture.")
        return None

    return frame


# -------------------------------------------------------------
# BUILD UNDISTORT RECTIFY MAPS
# -------------------------------------------------------------
def build_maps(K, dist):
    newK, _ = cv2.getOptimalNewCameraMatrix(K, dist, (FRAME_W, FRAME_H), 1.0)
    mapx, mapy = cv2.initUndistortRectifyMap(
        K, dist, None, newK, (FRAME_W, FRAME_H), cv2.CV_32FC1
    )
    return mapx, mapy, newK


# -------------------------------------------------------------
# YOLO DETECTION + DRAW
# -------------------------------------------------------------
def detect(model, img):
    r = model(img)[0]
    dets = []

    for box in r.boxes:
        x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
        cx = int((x1 + x2) / 2)
        cy = int((y1 + y2) / 2)
        conf = float(box.conf[0])
        cls = int(box.cls[0])
        name = model.names.get(cls, str(cls))

        # draw
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.circle(img, (cx, cy), 6, (0, 255, 0), -1)
        cv2.putText(img, f"{name} {conf:.2f}",
                    (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (0, 255, 0), 2)

        dets.append({"cx": cx, "cy": cy, "conf": conf, "class_name": name})

    return img, dets


# -------------------------------------------------------------
# MAIN
# -------------------------------------------------------------
def main():
    print("[INFO] Loading YOLO model...")
    try:
        model = YOLO(YOLO_MODEL_PATH)
    except Exception as e:
        print("[ERROR] YOLO failed to load:", e)
        sys.exit(1)

    print("[INFO] Building undistortion maps...")
    mapAx, mapAy, newK_A = build_maps(K_A, dist_A)
    mapBx, mapBy, newK_B = build_maps(K_B, dist_B)

    print("[INFO] Capturing from CAM A...")
    frameA = capture_single(CAM_A_ID)

    print("[INFO] Capturing from CAM B...")
    frameB = capture_single(CAM_B_ID)

    if frameA is None or frameB is None:
        print("[FATAL] One camera failed. Exiting.")
        return

    undA = cv2.remap(frameA, mapAx, mapAy, cv2.INTER_LINEAR)
    undB = cv2.remap(frameB, mapBx, mapBy, cv2.INTER_LINEAR)

    # visualize camera centers
    cv2.putText(undA, f"Center = ({int(newK_A[0,2])}, {int(newK_A[1,2])})",
                (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,0), 2)
    cv2.putText(undB, f"Center = ({int(newK_B[0,2])}, {int(newK_B[1,2])})",
                (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,0), 2)

    print("[INFO] Running YOLO detection...")
    undA, detA = detect(model, undA)
    undB, detB = detect(model, undB)

    print("\n--- CAM A detections ---")
    print(detA if detA else "No detections.")

    print("\n--- CAM B detections ---")
    print(detB if detB else "No detections.")

    cv2.imshow("CAM A (wide)", undA)
    cv2.imshow("CAM B (normal)", undB)

    print("[INFO] Press any key to exit...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
