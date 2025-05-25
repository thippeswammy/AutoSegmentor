import os
import threading

import cv2
from tqdm import tqdm

from logger_config import logger


class VideoCreator:
    def __init__(self, image_folders, video_names, fps=30):
        self.image_folders = image_folders
        self.video_names = video_names
        self.fps = fps
        self.valid_extensions = ('.png', '.jpg', '.jpeg')
        self.lock = threading.Lock()
        self.total_images = sum(len([img for img in os.listdir(folder) if img.endswith(self.valid_extensions)])
                                for folder in image_folders)
        self.processed_images = 0

    def create_video(self, image_folder, video_name, progress_bar):
        images = sorted([img for img in os.listdir(image_folder) if img.endswith(self.valid_extensions)])
        if not images:
            logger.warning(f"No images found in {image_folder}.")
            return
        first_image = os.path.join(image_folder, images[0])
        frame = cv2.imread(first_image)
        height, width, layers = frame.shape
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video = cv2.VideoWriter(video_name, fourcc, self.fps, (width, height))
        try:
            for image in images:
                img_path = os.path.join(image_folder, image)
                img = cv2.imread(img_path)
                video.write(img)
                with self.lock:
                    self.processed_images += 1
                    progress_bar.update(1)
        finally:
            video.release()
        cap = cv2.VideoCapture(video_name)
        if not cap.isOpened():
            logger.error(f"Failed to create valid video: {video_name}")
        else:
            logger.info(f"Video saved as {video_name}")
        cap.release()

    def run(self):
        with tqdm(total=self.total_images, desc="Creating Videos", unit="frame") as pbar:
            threads = []
            for folder, name in zip(self.image_folders, self.video_names):
                thread = threading.Thread(target=self.create_video, args=(folder, name, pbar))
                threads.append(thread)
                thread.start()
            for thread in threads:
                thread.join()
