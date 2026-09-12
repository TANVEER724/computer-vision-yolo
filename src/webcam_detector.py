from ultralytics import YOLO
import cv2
import time

# Load YOLO model
model = YOLO("yolo11n.pt")

# Open webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

previous_time = 0

while True:

    # Read webcam frame
    ret, frame = cap.read()

    if not ret:
        print("Error: Could not read webcam frame.")
        break

    # Run YOLO detection
    results = model(
    frame,
    imgsz=416,
    verbose=False)

    # Get first result
    result = results[0]

    # Draw detections
    output = result.plot()

    # Calculate FPS
    current_time = time.time()

    if previous_time != 0:
        fps = 1 / (current_time - previous_time)
    else:
        fps = 0

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

    # Display webcam
    cv2.imshow("AI Real-Time Object Detection", output)

    # Press Q to quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release webcam
cap.release()

# Close OpenCV windows
cv2.destroyAllWindows()