import cv2
import time

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Cannot open camera")
    exit()

while True:
    ret, frame = cap.read()
    if ret:
        cv2.imwrite('captured_frame.jpg', frame)
        print("Frame captured and saved as captured_frame.jpg")
    else:
        print("Can't receive frame")
        break
    time.sleep(1)

cap.release()