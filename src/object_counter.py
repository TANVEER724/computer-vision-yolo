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

    people_count = 0
    car_count = 0

    if result.boxes.id is not None:

        class_ids = result.boxes.cls.int().cpu().tolist()

        for class_id in class_ids:

            object_name = result.names[class_id]

            if object_name == "person":
                people_count += 1

            elif object_name == "car":
                car_count += 1

    # Display statistics
    cv2.putText(
        frame,
        f"People: {people_count}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2
    )

    cv2.putText(
        frame,
        f"Cars: {car_count}",
        (20, 75),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2
    )

    # Draw detections
    output = result.plot()

    # Put statistics over output
    cv2.putText(
        output,
        f"People: {people_count}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2
    )

    cv2.putText(
        output,
        f"Cars: {car_count}",
        (20, 75),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2
    )

    cv2.imshow(
        "AI Object Counter",
        output
    )

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()