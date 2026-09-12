import cv2

image = cv2.imread("images/test.jpg")

if image is None:
    print("Error: Image not found.")
    exit()

# Convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Blur image
blur = cv2.GaussianBlur(gray, (7, 7), 0)

# Detect edges
edges = cv2.Canny(blur, 50, 150)

cv2.imshow("Original", image)
cv2.imshow("Edges", edges)

cv2.waitKey(0)
cv2.destroyAllWindows()