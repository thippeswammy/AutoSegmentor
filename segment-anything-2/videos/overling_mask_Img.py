# import time
#
# import cv2
#
# for i in range(0, 1000):
#     # Load the original image and the mask
#     original = cv2.imread(f'road_imgs/road3_{i:05d}.png')
#     mask = cv2.imread(f'../rendered_frames/road3_{i:05d}.png', cv2.IMREAD_GRAYSCALE)  # Load as grayscale
#
#     # Ensure mask is binary (0 or 255)
#     _, binary_mask = cv2.threshold(mask, 128, 255, cv2.THRESH_BINARY)
#
#     # Combine the images using the mask
#     combined = cv2.bitwise_and(original, original, mask=binary_mask)
#
#     # Show the result
#     cv2.imshow('Combined Image', combined)
#     time.sleep(0.5)
# cv2.destroyAllWindows()
import cv2

cv2.namedWindow("Combined Image", cv2.WINDOW_NORMAL)
for i in range(0, 1000):
    # Load the original image and the color mask
    original = cv2.imread(f'road_imgs_1/road3_{i:05d}.png')
    mask = cv2.imread(f'../rendered_frames/road3_{i:05d}.png')  # Load as color

    # Ensure the mask is in the same size as the original image
    if original.shape[:2] != mask.shape[:2]:
        mask = cv2.resize(mask, (original.shape[1], original.shape[0]))

    # Combine the images using the mask
    # Here we use bitwise operations to keep the color from the mask
    combined = cv2.addWeighted(original, 0.5, mask, 0.5, 0)

    cv2.imshow('Combined Image', combined)
    cv2.waitKey(0)

cv2.destroyAllWindows()
