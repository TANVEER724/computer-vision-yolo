import cv2

image_path = "images/test.jpg"

image = cv2.imread(image_path)

if image is None:
    print("Error: Could not load the image.")
else:
    height, width, channels = image.shape

    print("Image loaded successfully!")
    print("Width:", width)
    print("Height:", height)
    print("Channels:", channels)

    cv2.imshow("Computer Vision", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()