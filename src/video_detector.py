from ultralytics import YOLO
import cv2
import time

# Load YOLO model
model = YOLO("yolo11n.pt")

# Open video
video_path = "videos/test.mp4"
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

previous_time = 0

while True:

    ret, frame = cap.read()

    if not ret:
        print("Video finished.")
        break

    # YOLO detection
    results = model(frame, verbose=False)

    result = results[0]

    # Draw detections
    output = result.plot()

    # Calculate FPS
    current_time = time.time()

    fps = 1 / (current_time - previous_time) if previous_time != 0 else 0

    previous_time = current_time

    # Display FPS
    cv2.putText(
        output,
        f"FPS: {fps:.1f}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    # Display
    cv2.imshow("AI Video Object Detection", output)

    # Q to quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()