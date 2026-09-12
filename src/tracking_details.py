from ultralytics import YOLO
import cv2

model = YOLO("yolo11n.pt")

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

while True:

    ret, frame = cap.read()

    if not ret:
        break

    results = model.track(
        frame,
        persist=True,
        verbose=False
    )

    result = results[0]

    # Check whether tracking IDs exist
    if result.boxes.id is not None:

        boxes = result.boxes.xyxy.cpu().numpy()
        track_ids = result.boxes.id.int().cpu().tolist()
        class_ids = result.boxes.cls.int().cpu().tolist()
        confidences = result.boxes.conf.cpu().tolist()

        for box, track_id, class_id, confidence in zip(
            boxes,
            track_ids,
            class_ids,
            confidences
        ):

            x1, y1, x2, y2 = map(int, box)

            object_name = result.names[class_id]

            label = (
                f"ID: {track_id} "
                f"{object_name} "
                f"{confidence:.2f}"
            )

            # Draw box
            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )

            # Draw label
            cv2.putText(
                frame,
                label,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )

    cv2.imshow(
        "Tracking Details",
        frame
    )

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()