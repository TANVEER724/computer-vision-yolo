import cv2

video_path = "videos/test.mp4"

cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

while True:

    ret, frame = cap.read()

    if not ret:
        print("Video finished.")
        break

    cv2.imshow("Video", frame)

    # Press Q to quit
    if cv2.waitKey(25) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()