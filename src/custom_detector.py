from ultralytics import YOLO
import cv2

# Load model
model = YOLO("yolo11n.pt")

# Load image
image = cv2.imread("images/test.jpg")

if image is None:
    print("Error: Could not load image.")
    exit()

# Run detection
results = model(image)

result = results[0]

# Class names
names = result.names

# Create a copy
output = image.copy()

# Process every detected object
for box in result.boxes:

    # Get class
    class_id = int(box.cls[0])

    # Get confidence
    confidence = float(box.conf[0])

    # Get coordinates
    x1, y1, x2, y2 = map(int, box.xyxy[0])

    # Get object name
    object_name = names[class_id]

    # Label
    label = f"{object_name} {confidence:.2f}"

    # Draw bounding box
    cv2.rectangle(
        output,
        (x1, y1),
        (x2, y2),
        (0, 255, 0),
        2
    )

    # Draw label
    cv2.putText(
        output,
        label,
        (x1, y1 - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 0),
        2
    )

# Show result
cv2.imshow("Custom Object Detector", output)

cv2.waitKey(0)
cv2.destroyAllWindows()