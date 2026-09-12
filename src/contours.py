import cv2

image = cv2.imread("images/test.jpg")

if image is None:
    print("Error: Image not found.")
    exit()

# Convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Blur
blur = cv2.GaussianBlur(gray, (7, 7), 0)

# Edge detection
edges = cv2.Canny(blur, 50, 150)

# Find contours
contours, hierarchy = cv2.findContours(
    edges,
    cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_SIMPLE
)

print("Number of contours:", len(contours))

# Draw contours
result = image.copy()

cv2.drawContours(
    result,
    contours,
    -1,
    (0, 255, 0),
    2
)

cv2.imshow("Original", image)
cv2.imshow("Contours", result)

cv2.waitKey(0)
cv2.destroyAllWindows()