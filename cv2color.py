import numpy as np
import cv2

# Load the image
image = cv2.imread(r"C:\Users\panip\Downloads\car.jpg")

# Check if the image is loaded correctly
if image is None:
    print("Error: Image not found or cannot be loaded!")
    exit()  # Exit if image is not loaded
# Convert the image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)



    # Display the output image
cv2.imshow('colour Detection', gray)
cv2.waitKey(0)  # Wait for a key press to close the window

# Close all OpenCV windows
cv2.destroyAllWindows()
