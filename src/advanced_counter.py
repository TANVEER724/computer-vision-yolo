from ultralytics import YOLO
import cv2
import time

# ==========================================
# Load YOLO model
# ==========================================

model = YOLO("yolo11n.pt")


# ==========================================
# Open webcam
# ==========================================

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()


# ==========================================
# Counters
# ==========================================

people_in = 0
people_out = 0


# ==========================================
# Track previous positions
# ==========================================

previous_positions = {}


# ==========================================
# Track which IDs have already crossed
# ==========================================

counted_in = set()
counted_out = set()


# ==========================================
# Counting line
# ==========================================

line_y = 300


# ==========================================
# FPS
# ==========================================

previous_time = 0


# ==========================================
# Main loop
# ==========================================

while True:

    ret, frame = cap.read()

    if not ret:
        print("Error reading webcam.")
        break


    # ======================================
    # YOLO tracking
    # ======================================

    results = model.track(
        frame,
        persist=True,
        verbose=False
    )

    result = results[0]


    # ======================================
    # Draw counting line
    # ======================================

    cv2.line(
        frame,
        (0, line_y),
        (frame.shape[1], line_y),
        (0, 255, 255),
        3
    )


    # ======================================
    # Process tracked objects
    # ======================================

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

        confidences = (
            result.boxes.conf
            .cpu()
            .tolist()
        )


        for box, track_id, class_id, confidence in zip(
            boxes,
            track_ids,
            class_ids,
            confidences
        ):

            # ==================================
            # Only detect people
            # ==================================

            object_name = result.names[class_id]

            if object_name != "person":
                continue


            # ==================================
            # Confidence filtering
            # ==================================

            if confidence < 0.50:
                continue


            # ==================================
            # Bounding box
            # ==================================

            x1, y1, x2, y2 = map(
                int,
                box
            )


            # ==================================
            # Center point
            # ==================================

            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2


            # ==================================
            # Draw bounding box
            # ==================================

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )


            # ==================================
            # Draw center
            # ==================================

            cv2.circle(
                frame,
                (center_x, center_y),
                5,
                (0, 0, 255),
                -1
            )


            # ==================================
            # Display ID
            # ==================================

            cv2.putText(
                frame,
                f"ID: {track_id}",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )


            # ==================================
            # Check previous position
            # ==================================

            if track_id in previous_positions:

                previous_y = previous_positions[track_id]


                # ==================================
                # Moving DOWN
                # ==================================

                if (
                    previous_y < line_y
                    and center_y >= line_y
                    and track_id not in counted_in
                ):

                    people_in += 1

                    counted_in.add(track_id)

                    print(
                        f"Person {track_id} ENTERED"
                    )


                # ==================================
                # Moving UP
                # ==================================

                elif (
                    previous_y > line_y
                    and center_y <= line_y
                    and track_id not in counted_out
                ):

                    people_out += 1

                    counted_out.add(track_id)

                    print(
                        f"Person {track_id} EXITED"
                    )


            # ==================================
            # Update position
            # ==================================

            previous_positions[track_id] = center_y


    # ======================================
    # FPS calculation
    # ======================================

    current_time = time.time()

    if previous_time != 0:

        fps = (
            1 /
            (current_time - previous_time)
        )

    else:

        fps = 0

    previous_time = current_time


    # ======================================
    # Display statistics
    # ======================================

    cv2.putText(
        frame,
        f"IN: {people_in}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2
    )

    cv2.putText(
        frame,
        f"OUT: {people_out}",
        (20, 75),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2
    )

    cv2.putText(
        frame,
        f"FPS: {fps:.1f}",
        (20, 110),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2
    )


    # ======================================
    # Display
    # ======================================

    cv2.imshow(
        "Advanced AI People Counter",
        frame
    )


    # ======================================
    # Quit
    # ======================================

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


# ==========================================
# Cleanup
# ==========================================

cap.release()
cv2.destroyAllWindows()