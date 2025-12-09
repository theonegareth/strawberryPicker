import os
import cv2
from ultralytics import YOLO
def main():
    # Load your strawberry model (.pt)
    model = YOLO("strawb.pt")

    # Open webcam (0 = default camera in Windows)
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    if not cap.isOpened():
        print("❌ Cannot access Windows camera.")
        return

    print("🍓 Strawberry detection started... Press Q to exit")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Failed to read frame.")
            break

        # Run YOLO model on frame
        results = model(frame, stream=True)

        for r in results:
            boxes = r.boxes

            for box in boxes:
                # Bounding box
                x1, y1, x2, y2 = box.xyxy[0]

                # Confidence
                conf = float(box.conf[0])

                # Class name
                cls = int(box.cls[0])
                name = model.names[cls]

                # Draw bounding box and label
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
                cv2.putText(
                    frame,
                    f"{name} {conf:.2f}",
                    (int(x1), int(y1) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 0, 255),
                    2
                )
        
        for r in results:
            for box in r.boxes:
        # get box coordinates
                x1, y1, x2, y2 = box.xyxy[0]

        # find bounding box center
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)

        # frame center
            frame_h, frame_w = frame.shape[:2]
            origin_x = frame_w // 2
            origin_y = frame_h // 2

        # convert to centered coordinate system
            coord_x = cx - origin_x       # right = positive, left = negative
            coord_y = origin_y - cy       # up = positive, down = negative

        # print to console
            print(f"Object center (pixels): ({coord_x}, {coord_y})")

        # draw a dot at the center
            cv2.circle(frame, (cx, cy), 6, (0, 255, 0), -1)

        # draw coordinate text on screen
            cv2.putText(frame, f"({coord_x}, {coord_y})",
                        (cx + 10, cy - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (0, 255, 0), 2)


        # Show frame
        cv2.imshow("Strawberry Detection (Windows)", frame)

        # Quit on Q key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
