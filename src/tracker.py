from ultralytics import YOLO
import cv2

# Load YOLO model
model = YOLO("yolo11n.pt")

# Open webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

while True:

    # Read webcam frame
    ret, frame = cap.read()

    if not ret:
        print("Error: Could not read frame.")
        break

    # Run YOLO tracking
    results = model.track(
        frame,
        persist=True,
        verbose=False
    )

    result = results[0]

    # Draw tracking results
    output = result.plot()

    # Display
    cv2.imshow(
        "AI Object Tracking",
        output
    )

    # Press Q to quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()