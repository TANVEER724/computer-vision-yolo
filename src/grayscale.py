import cv2

# Load image
image = cv2.imread("images/test.jpg")

if image is None:
    print("Error: Image not found.")
    exit()

# Convert BGR image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Display original image
cv2.imshow("Original Image", image)

# Display grayscale image
cv2.imshow("Grayscale Image", gray)

# Wait for keyboard input
cv2.waitKey(0)

# Close windows
cv2.destroyAllWindows()