import cv2

image = cv2.imread("images/test.jpg")

if image is None:
    print("Error: Image not found.")
    exit()

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

blur = cv2.GaussianBlur(gray, (7, 7), 0)

edges = cv2.Canny(blur, 50, 150)

contours, hierarchy = cv2.findContours(
    edges,
    cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_SIMPLE
)

result = image.copy()

for contour in contours:

    # Ignore very small contours
    area = cv2.contourArea(contour)

    if area < 500:
        continue

    # Get bounding rectangle
    x, y, width, height = cv2.boundingRect(contour)

    # Draw rectangle
    cv2.rectangle(
        result,
        (x, y),
        (x + width, y + height),
        (0, 255, 0),
        2
    )

    # Display coordinates
    text = f"X:{x} Y:{y}"

    cv2.putText(
        result,
        text,
        (x, y - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 255, 0),
        2
    )

cv2.imshow("Object Bounding Boxes", result)

cv2.waitKey(0)
cv2.destroyAllWindows()