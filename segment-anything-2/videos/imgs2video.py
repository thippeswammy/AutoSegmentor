import os

import cv2

# Folder containing images
image_folder = '../rendered_frames'

# Output video file name
video_name = f'output1_video.mp4'

# Frames per second
fps = 30

# Get a list of all images in the folder and sort them numerically/alphabetically
# Ensure that files are sorted in a way that matches the order they should appear in the video
images = sorted([img for img in os.listdir(image_folder) if img.endswith(".jpg") or img.endswith(".png")])

# Read the first image to get dimensions (height, width)
if len(images) > 0:
    first_image = os.path.join(image_folder, images[0])
    frame = cv2.imread(first_image)
    height, width, layers = frame.shape

    # Define the codec and create a VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for mp4
    video = cv2.VideoWriter(video_name, fourcc, fps, (width, height))

    # Add each image to the video
    for image in images:
        img_path = os.path.join(image_folder, image)
        img = cv2.imread(img_path)
        video.write(img)

    # Release the VideoWriter object
    video.release()

    print("Video saved as", video_name)
else:
    print("No images found in the folder.")
