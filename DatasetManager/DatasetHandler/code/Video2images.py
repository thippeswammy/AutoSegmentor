import os

import cv2

# Path to the input video file
video_path = 'F:\RunningProjects\LaneLinesDetection\InputVideo\\video18.mp4'

# Directory where the frames will be saved
output_dir = 'F:\RunningProjects\segment-anything-2\\videos\Road'
os.makedirs(output_dir, exist_ok=True)

# Load the video
cap = cv2.VideoCapture(video_path)

frame_count = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break
    if frame_count > 100:
        break
    # Save each frame as a .jpeg file with a filename like 0000.jpeg, 0001.jpeg, etc.
    frame_filename = os.path.join(output_dir, f'{frame_count:04d}.jpeg')
    cv2.imwrite(frame_filename, frame)
    frame_count += 1

cap.release()
print(f"Extracted {frame_count} frames to '{output_dir}'")
