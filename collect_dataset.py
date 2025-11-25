import cv2
import os

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Cannot open camera")
    exit()

good_dir = 'dataset/good'
bad_dir = 'dataset/bad'
os.makedirs(good_dir, exist_ok=True)
os.makedirs(bad_dir, exist_ok=True)

good_count = len(os.listdir(good_dir))
bad_count = len(os.listdir(bad_dir))

print("Press 'g' to save as good, 'b' to save as bad, 'q' to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Can't receive frame")
        break
    cv2.imshow('Dataset Collection', frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('g'):
        filename = f'{good_dir}/good_{good_count:04d}.jpg'
        cv2.imwrite(filename, frame)
        good_count += 1
        print(f"Saved {filename}")
    elif key == ord('b'):
        filename = f'{bad_dir}/bad_{bad_count:04d}.jpg'
        cv2.imwrite(filename, frame)
        bad_count += 1
        print(f"Saved {filename}")
    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()