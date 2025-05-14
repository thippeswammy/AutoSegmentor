import os
import threading

import cv2
from tqdm import tqdm

i = 78
image_folder = r'./verified/TempImg'
mask_folder = r'./verified/TempMasks'
image_folders = [image_folder, mask_folder]
video_names = [f'OrgVideo{i}.mp4', f'MaskVideo{i}.mp4']
fps = 30

# Shared progress counter
lock = threading.Lock()
total_images = 0  # total images to process
processed_images = 0  # shared progress counter

# Count total images
for folder in image_folders:
    total_images += len([img for img in os.listdir(folder) if img.endswith(".jpg") or img.endswith(".png")])


def create_video(image_folder, video_name, progress_bar):
    global processed_images
    images = sorted([img for img in os.listdir(image_folder) if img.endswith(".jpg") or img.endswith(".png")])

    if len(images) > 0:
        first_image = os.path.join(image_folder, images[0])
        frame = cv2.imread(first_image)
        height, width, layers = frame.shape

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video = cv2.VideoWriter(video_name, fourcc, fps, (width, height))

        for image in images:
            img_path = os.path.join(image_folder, image)
            img = cv2.imread(img_path)
            video.write(img)

            # Update shared progress bar
            with lock:
                processed_images += 1
                progress_bar.update(1)

        video.release()
        print(f"\nVideo saved as {video_name}")
    else:
        print(f"\nNo images found in {image_folder}.")


# Create a single shared progress bar
with tqdm(total=total_images, desc="Creating Videos", unit="frame", position=0) as pbar:
    threads = []
    for folder, name in zip(image_folders, video_names):
        thread = threading.Thread(target=create_video, args=(folder, name, pbar))
        threads.append(thread)
        thread.start()

    for thread in threads:
        thread.join()

print("All videos created.")
