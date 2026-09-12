from ultralytics import YOLO
import cv2

# Load YOLO model
model = YOLO("yolo11n.pt")

# Load image
image_path = "images/test.jpg"
image = cv2.imread(image_path)

if image is None:
    print("Error: Could not load image.")
    exit()

# Run object detection
results = model(image)

# Draw detection results
annotated_image = results[0].plot()

# Display result
cv2.imshow("YOLO Object Detection", annotated_image)

cv2.waitKey(0)
cv2.destroyAllWindows()