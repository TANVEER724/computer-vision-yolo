from ultralytics import YOLO
import cv2

# ==============================
# 1. Load YOLO model
# ==============================

model = YOLO("yolo11n.pt")


# ==============================
# 2. Open webcam
# ==============================

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()


# ==============================
# 3. Variables
# ==============================

# Number of people entering
people_in = 0

# Number of people leaving
people_out = 0

# Store previous center positions
previous_positions = {}


# ==============================
# 4. Line position
# ==============================

line_y = 300


# ==============================
# 5. Main loop
# ==============================

while True:

    ret, frame = cap.read()

    if not ret:
        print("Error: Could not read frame.")
        break


    # ==============================
    # 6. Object tracking
    # ==============================

    results = model.track(
        frame,
        persist=True,
        verbose=False
    )

    result = results[0]


    # ==============================
    # 7. Draw counting line
    # ==============================

    cv2.line(
        frame,
        (0, line_y),
        (frame.shape[1], line_y),
        (0, 255, 255),
        3
    )


    # ==============================
    # 8. Check tracking IDs
    # ==============================

    if result.boxes.id is not None:

        boxes = result.boxes.xyxy.cpu().numpy()

        track_ids = (
            result.boxes.id
            .int()
            .cpu()
            .tolist()
        )

        class_ids = (
            result.boxes.cls
            .int()
            .cpu()
            .tolist()
        )


        # ==============================
        # 9. Process each object
        # ==============================

        for box, track_id, class_id in zip(
            boxes,
            track_ids,
            class_ids
        ):

            # Only count people
            object_name = result.names[class_id]

            if object_name != "person":
                continue


            # ==============================
            # 10. Bounding box
            # ==============================

            x1, y1, x2, y2 = map(
                int,
                box
            )


            # ==============================
            # 11. Calculate center
            # ==============================

            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2


            # ==============================
            # 12. Draw bounding box
            # ==============================

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )


            # Draw center point

            cv2.circle(
                frame,
                (center_x, center_y),
                5,
                (0, 0, 255),
                -1
            )


            # ==============================
            # 13. Display tracking ID
            # ==============================

            cv2.putText(
                frame,
                f"Person ID: {track_id}",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )


            # ==============================
            # 14. Check previous position
            # ==============================

            if track_id in previous_positions:

                previous_y = previous_positions[track_id]


                # ==============================
                # 15. Person moves DOWN
                # ==============================

                if (
                    previous_y < line_y
                    and center_y >= line_y
                ):

                    people_in += 1

                    print(
                        f"Person {track_id} entered."
                    )


                # ==============================
                # 16. Person moves UP
                # ==============================

                elif (
                    previous_y > line_y
                    and center_y <= line_y
                ):

                    people_out += 1

                    print(
                        f"Person {track_id} exited."
                    )


            # ==============================
            # 17. Save current position
            # ==============================

            previous_positions[track_id] = center_y


    # ==============================
    # 18. Display statistics
    # ==============================

    cv2.putText(
        frame,
        f"IN: {people_in}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (0, 255, 0),
        2
    )

    cv2.putText(
        frame,
        f"OUT: {people_out}",
        (20, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (0, 255, 0),
        2
    )


    # ==============================
    # 19. Display frame
    # ==============================

    cv2.imshow(
        "AI People Counter",
        frame
    )


    # ==============================
    # 20. Quit
    # ==============================

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


# ==============================
# 21. Cleanup
# ==============================

cap.release()
cv2.destroyAllWindows()