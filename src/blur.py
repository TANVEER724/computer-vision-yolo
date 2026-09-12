import cv2

image = cv2.imread("images/test.jpg")

if image is None:
    print("Error: Image not found.")
    exit()

# Convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply Gaussian blur
blur = cv2.GaussianBlur(gray, (7, 7), 0)

cv2.imshow("Original", image)
cv2.imshow("Grayscale", gray)
cv2.imshow("Blurred", blur)

cv2.waitKey(0)
cv2.destroyAllWindows()