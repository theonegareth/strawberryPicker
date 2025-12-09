import cv2
import numpy as np


# ==========================================================
# CAMERA 1 — wide angle 1280 * 720
# (Your calibration pasted below)
# ==========================================================

camera1_matrix = np.array([
    [2263.50596,      0.0,        964.761094],
    [0.0,         2275.52656,     698.162562],
    [0.0,              0.0,           1.0    ]
], dtype=np.float32)

camera1_dist = np.array([
    -0.40170703, 0.5627054, 0.00803042, -0.00481173, -0.58891806
], dtype=np.float32)

# You must compute this correctly later.
cm_per_pixel_cam1 = 0.05     # TEMPORARY VALUE


# ==========================================================
# CAMERA 2 — normal 1280 * 720
# (Your calibration pasted below)
# ==========================================================

camera2_matrix = np.array([
    [1190.33622,   0.0,        1384.23659],
    [0.0,          1195.47438,  833.37699],
    [0.0,              0.0,         1.0  ]
], dtype=np.float32)

camera2_dist = np.array([
    -0.35899454, 0.11213625, -0.00363389, -0.00311402, -0.01137052
], dtype=np.float32)

cm_per_pixel_cam2 = 0.05     # TEMPORARY VALUE



# ==========================================================
# OBJECT DETECTION (replace with your Hailo/Yolo function)
# ==========================================================

def detect_object(frame):
    """
    Returns center (cx, cy) in pixel, or None.
    Replace this with your model output.
    """
    h, w = frame.shape[:2]
    return w // 2, h // 2  # placeholder



# ==========================================================
# DRAW AXIS OVERLAY (like the image you sent)
# ==========================================================

def draw_vectors(frame):
    h, w = frame.shape[:2]
    cx = w // 2
    cy = h // 2

    # X-axis (horizontal)
    cv2.line(frame, (0, cy), (w, cy), (255, 0, 0), 2)
    cv2.putText(frame, "X", (cx + 40, cy - 25),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    # Y-axis (vertical)
    cv2.line(frame, (cx, 0), (cx, h), (255, 0, 0), 2)
    cv2.putText(frame, "Y", (cx + 20, cy + 45),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)



# ==========================================================
# Pixel → CM converter
# ==========================================================
def px_to_cm(px, py, scale):
    return px * scale, py * scale



# ==========================================================
# OPEN CAMERAS
# ==========================================================

cam1 = cv2.VideoCapture(1)
cam2 = cv2.VideoCapture(2)

# 1080p
cam1.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cam1.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# 2K (2560 × 1440 typical)
cam2.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cam2.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

print("Both cameras started successfully.")


# ==========================================================
# MAIN LOOP
# ==========================================================

while True:

    ret1, frame1 = cam1.read()
    ret2, frame2 = cam2.read()

    if not ret1 or not ret2:
        print("ERROR: Cannot read camera feed.")
        break

    # Undistort with each camera's calibration
    frame1 = cv2.undistort(frame1, camera1_matrix, camera1_dist)
    frame2 = cv2.undistort(frame2, camera2_matrix, camera2_dist)

    # Draw axes
    draw_vectors(frame1)
    draw_vectors(frame2)

    # ---------------- CAMERA 1 DETECTION ----------------
    obj1 = detect_object(frame1)
    if obj1:
        cx, cy = obj1
        cmx, cmy = px_to_cm(cx, cy, cm_per_pixel_cam1)

        cv2.circle(frame1, (cx, cy), 10, (0, 255, 0), -1)
        cv2.putText(frame1,
                    f"{cx}px,{cy}px  ->  {cmx:.2f}cm,{cmy:.2f}cm",
                    (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

        print(f"CAM1: {cx}px {cy}px  |  {cmx:.2f}cm {cmy:.2f}cm")

    # ---------------- CAMERA 2 DETECTION ----------------
    obj2 = detect_object(frame2)
    if obj2:
        cx, cy = obj2
        cmx, cmy = px_to_cm(cx, cy, cm_per_pixel_cam2)

        cv2.circle(frame2, (cx, cy), 10, (0, 255, 0), -1)
        cv2.putText(frame2,
                    f"{cx}px,{cy}px  ->  {cmx:.2f}cm,{cmy:.2f}cm",
                    (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

        print(f"CAM2: {cx}px {cy}px  |  {cmx:.2f}cm {cmy:.2f}cm")


    # Show windows
    cv2.imshow("Camera 1 - 1080p", frame1)
    cv2.imshow("Camera 2 - 2K", frame2)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC
        break


cam1.release()
cam2.release()
cv2.destroyAllWindows()
