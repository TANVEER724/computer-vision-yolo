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

# Get first result
result = results[0]

# Get class names
names = result.names

# Get detected boxes
boxes = result.boxes

print("\nDetected Objects:")
print("------------------")

for box in boxes:

    # Class ID
    class_id = int(box.cls[0])

    # Confidence
    confidence = float(box.conf[0])

    # Bounding box coordinates
    x1, y1, x2, y2 = box.xyxy[0].tolist()

    object_name = names[class_id]

    print(
        f"Object: {object_name} | "
        f"Confidence: {confidence:.2f} | "
        f"Box: ({x1:.0f}, {y1:.0f}, {x2:.0f}, {y2:.0f})"
    )