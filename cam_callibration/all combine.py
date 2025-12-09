# dual_camera_capture_yolo_calibrated.py
import cv2
import numpy as np
from ultralytics import YOLO
import sys

# -----------------------------
# USER SETTINGS (change if needed)
# -----------------------------
CAM_A_ID = 0   # wide lens (you said Cam A = wide)
CAM_B_ID = 1   # normal lens

FRAME_W = 1280
FRAME_H = 720

# IMPORTANT: replace these with measured cm-per-pixel values.
# You can compute this by photographing a known length (e.g., 10 cm)
# and dividing real_cm / pixel_length_on_undistorted_image.
CM_PER_PIXEL_A = 0.05   # <-- replace for Cam A (wide)
CM_PER_PIXEL_B = 0.05   # <-- replace for Cam B (normal)

# Path to your YOLOv8 model
YOLO_MODEL_PATH = "face.pt"


# -----------------------------
# HARD-CODED CALIBRATION (from you)
# -----------------------------
# Cam A (wide)
K_A = np.array([
    [1902.95841,    0.0,       1383.58093],
    [0.0,           1905.27146, 832.29197],
    [0.0,           0.0,         1.0]
], dtype=np.float32)

dist_A = np.array([-0.35917977, 0.11238831, -0.00357951, -0.00304661, -0.01140789],
                  dtype=np.float32)

# Cam B (normal)
K_B = np.array([
    [2263.50596,   0.0,        964.761094],
    [0.0,          2275.52656, 698.162562],
    [0.0,          0.0,           1.0]
], dtype=np.float32)

dist_B = np.array([-0.40170703, 0.5627054, 0.00803042, -0.00481173, -0.58891806],
                  dtype=np.float32)


# -----------------------------
# UTIL: capture single frame, safe open
# -----------------------------
def capture_single_frame(cam_id, width=FRAME_W, height=FRAME_H):
    cap = cv2.VideoCapture(cam_id, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print(f"ERROR: Camera {cam_id} failed to open.")
        return None

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    # give the camera a short warmup
    for _ in range(3):
        ret, frame = cap.read()

    ret, frame = cap.read()
    cap.release()

    if not ret or frame is None:
        print(f"ERROR: Camera {cam_id} failed to capture an image.")
        return None

    return frame


# -----------------------------
# UTIL: undistort using maps (precompute)
# -----------------------------
def make_undistort_map(K, dist, size=(FRAME_W, FRAME_H)):
    newK, _ = cv2.getOptimalNewCameraMatrix(K, dist, size, 0)
    map_x, map_y = cv2.initUndistortRectifyMap(K, dist, None, newK, size, cv2.CV_32FC1)
    return map_x, map_y, newK


# -----------------------------
# YOLO helper: detect and draw
# -----------------------------
def run_yolo_and_annotate(model, img, cm_per_pixel, label_prefix="CAM"):
    """
    Runs YOLOv8 on `img`, draws boxes and centers, returns annotated image and list of detections.
    Detections list entries: dict { 'box':(x1,y1,x2,y2), 'conf':float, 'cls':int, 'name':str,
                                    'cx_px':int, 'cy_px':int, 'cx_cm':float, 'cy_cm':float,
                                    'coord_x_centered_px':int, 'coord_y_centered_px':int,
                                    'coord_x_centered_cm':float, 'coord_y_centered_cm':float }
    """
    results = model(img)[0]  # single image inference
    detections = []

    h, w = img.shape[:2]
    origin_x = w // 2
    origin_y = h // 2

    # iterate boxes
    for box in results.boxes:
        # xyxy may be a tensor; convert to python floats
        x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
        conf = float(box.conf[0])
        cls_id = int(box.cls[0])
        name = model.names.get(cls_id, str(cls_id))

        # center in pixel coords (image coordinates)
        cx = int((x1 + x2) / 2)
        cy = int((y1 + y2) / 2)

        # centered coordinate (origin in the image center)
        coord_x = cx - origin_x          # right = positive
        coord_y = origin_y - cy         # up = positive

        # convert to cm using provided scale
        cx_cm = cx * cm_per_pixel
        cy_cm = cy * cm_per_pixel
        coord_x_cm = coord_x * cm_per_pixel
        coord_y_cm = coord_y * cm_per_pixel

        # draw box and center + label
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.circle(img, (cx, cy), 6, (0, 255, 0), -1)
        cv2.putText(img, f"{name} {conf:.2f}", (x1, y1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)
        cv2.putText(img, f"px:({coord_x},{coord_y}) cm:({coord_x_cm:.2f},{coord_y_cm:.2f})",
                    (cx + 10, cy - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

        detections.append({
            'box': (x1, y1, x2, y2),
            'conf': conf,
            'cls': cls_id,
            'name': name,
            'cx_px': cx, 'cy_px': cy,
            'cx_cm': cx_cm, 'cy_cm': cy_cm,
            'coord_x_centered_px': coord_x, 'coord_y_centered_px': coord_y,
            'coord_x_centered_cm': coord_x_cm, 'coord_y_centered_cm': coord_y_cm
        })

    return img, detections


# -----------------------------
# MAIN
# -----------------------------
def main():
    # Load YOLOv8 model
    try:
        model = YOLO(YOLO_MODEL_PATH)
    except Exception as e:
        print("ERROR loading YOLO model:", e)
        sys.exit(1)

    # Precompute undistort maps for both cameras
    mapA_x, mapA_y, newK_A = make_undistort_map(K_A, dist_A, (FRAME_W, FRAME_H))
    mapB_x, mapB_y, newK_B = make_undistort_map(K_B, dist_B, (FRAME_W, FRAME_H))

    print("Capturing single frame from each camera...")
    frameA = capture_single_frame(CAM_A_ID)
    frameB = capture_single_frame(CAM_B_ID)

    if frameA is None or frameB is None:
        print("Capture failed for one or more cameras. Exiting.")
        return

    # Undistort
    undA = cv2.remap(frameA, mapA_x, mapA_y, cv2.INTER_LINEAR)
    undB = cv2.remap(frameB, mapB_x, mapB_y, cv2.INTER_LINEAR)

    # Put camera centers on images
    cxA, cyA = int(newK_A[0,2]), int(newK_A[1,2])
    cxB, cyB = int(newK_B[0,2]), int(newK_B[1,2])
    cv2.putText(undA, f"Center:({cxA},{cyA})", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0), 2)
    cv2.putText(undB, f"Center:({cxB},{cyB})", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0), 2)

    # Run YOLO and annotate
    annotatedA, detsA = run_yolo_and_annotate(model, undA, CM_PER_PIXEL_A, label_prefix="CAM A")
    annotatedB, detsB = run_yolo_and_annotate(model, undB, CM_PER_PIXEL_B, label_prefix="CAM B")

    # Print detections to console (most-important info)
    print("--- CAM A Detections ---")
    if detsA:
        for d in detsA:
            print(f"{d['name']} conf={d['conf']:.2f} center_px=({d['cx_px']},{d['cy_px']}) center_cm=({d['cx_cm']:.2f},{d['cy_cm']:.2f}) centered_px=({d['coord_x_centered_px']},{d['coord_y_centered_px']}) centered_cm=({d['coord_x_centered_cm']:.2f},{d['coord_y_centered_cm']:.2f})")
    else:
        print("No detections.")

    print("--- CAM B Detections ---")
    if detsB:
        for d in detsB:
            print(f"{d['name']} conf={d['conf']:.2f} center_px=({d['cx_px']},{d['cy_px']}) center_cm=({d['cx_cm']:.2f},{d['cy_cm']:.2f}) centered_px=({d['coord_x_centered_px']},{d['coord_y_centered_px']}) centered_cm=({d['coord_x_centered_cm']:.2f},{d['coord_y_centered_cm']:.2f})")
    else:
        print("No detections.")

    # Show windows (one per camera)
    cv2.imshow("CAM A (wide) - result", annotatedA)
    cv2.imshow("CAM B (normal) - result", annotatedB)

    print("Press any key in the image window to exit.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
