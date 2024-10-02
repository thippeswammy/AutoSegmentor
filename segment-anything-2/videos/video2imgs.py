# import os
#
# import cv2
#
# # Path to the input video file
# video_path = 'F:\RunningProjects\LaneLinesDetection\InputVideo\\video18.mp4'
#
# # Directory where the frames will be saved
# output_dir = 'road_imgs1'
# os.makedirs(output_dir, exist_ok=True)
#
# # Load the video
# cap = cv2.VideoCapture(video_path)
#
# existing_files = os.listdir(output_dir)
# frame_count = 3743
# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break
#     # if frame_count >= 50000:
#     #     break
#     # Save each frame as a .jpeg file with a filename like 0000.jpeg, 0001.jpeg, etc.
#     frame_filename = os.path.join(output_dir, f'{frame_count:04d}.jpeg')
#     # print(frame_filename)
#     cv2.imwrite(frame_filename, frame)
#     frame_count += 1
#
# cap.release()
# print(f"Extracted {frame_count} frames to '{output_dir}'")


import os
import cv2
from tqdm import tqdm

# Path to the input video file
video_path = 'F:\\RunningProjects\\LaneLinesDetection\\InputVideo\\video23.mp4'

# Directory where the frames will be saved
output_dir = 'road_imgs'
os.makedirs(output_dir, exist_ok=True)

# Load the video
cap = cv2.VideoCapture(video_path)

# Get the total number of frames in the video
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# Initialize the progress bar
with tqdm(total=total_frames, desc='Extracting Frames', unit='frame') as pbar:
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Save each frame as a .png file with a filename like road3_00000.png, etc.
        frame_filename = os.path.join(output_dir, f'road4_{frame_count:05d}.png')
        cv2.imwrite(frame_filename, frame)
        frame_count += 1

        # Update the progress bar
        pbar.update(1)

cap.release()
print(f"Extracted {frame_count} frames to '{output_dir}'")

'''
    video18=road1
    video19=road2
    video2 =road3
    video23=road4
'''
