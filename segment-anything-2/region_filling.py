import cv2
import numpy as np

# Load the image
image = cv2.imread('F:\RunningProjects\SAM2\segment-anything-2\\rendered_frames\\full_mask_00200.png')

# # Convert to grayscale if needed
# gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#
# # Threshold the image to create a binary mask
# _, binary_mask = cv2.threshold(gray, 0, 100, cv2.THRESH_BINARY_INV)
#
# # Define a seed point (x, y) for the flood fill
# seed_point = (50, 50)  # Change to your desired point
#
# # Flood fill
# cv2.floodFill(binary_mask, None, seed_point, 255)
#
# # Show the results
# cv2.imshow('Flood Fill Result', binary_mask)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
# image = cv2.imread('path/to/your/image.jpg')

# Convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Threshold the image to create a binary mask
_, binary_mask = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)

# Find contours
contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Create a blank mask for filling
filled_mask = np.zeros_like(binary_mask)

# Fill the contours
cv2.drawContours(filled_mask, contours, -1, 255, thickness=cv2.FILLED)

# Show the results
cv2.imshow('Filled Contours', filled_mask)
cv2.waitKey(0)
cv2.destroyAllWindows()
