import json
import os
import re
import shutil
import threading

import cv2
import numpy as np
import pygetwindow as gw
import torch

from sam2.build_sam import build_sam2_video_predictor
from videos import multithreaded_video_frame_extractor as video2imgs_


class VideoFrameProcessor:
    def __init__(self, video_number=None, batch_size=120, frames_directory="videos/road_imgs"):
        self.video_number = video_number
        self.batch_size = batch_size
        self.frames_directory = frames_directory
        self.device = self._select_device()
        self.sam2_checkpoint = "../checkpoints/sam2_hiera_large.pt"
        self.model_config = "sam2_hiera_l.yaml"
        self.sam2_predictor = build_sam2_video_predictor(self.model_config, self.sam2_checkpoint, device=self.device)
        self.current_class_label = 1
        self.selected_points = []
        self.selected_labels = []
        self.points_collection_list = []
        self.labels_collection_list = []
        self.label_colors = {1: (0, 0, 255), 2: (255, 0, 0), 3: (0, 255, 0)}
        self.image_counter = 0
        self.val = 0
        self.count = 0
        self.is_drawing = False
        self.current_frame = None

    def _select_device(self):
        device = torch.device(
            "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        )
        print(f"Using device: {device}")
        if device.type == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        return device

    def clear_directory(self, directory):
        if os.path.exists(directory):
            for filename in os.listdir(directory):
                file_path = os.path.join(directory, filename)
                try:
                    if os.path.isfile(file_path) or os.path.islink(file_path):
                        os.unlink(file_path)
                    elif os.path.isdir(file_path):
                        shutil.rmtree(file_path)
                except Exception as e:
                    print(f"Failed to delete {file_path}. Reason: {e}")

    def move_and_copy_frames(self, batch_index, frame_paths, target_directory="videos/Temp"):
        frames_to_copy = frame_paths[batch_index:batch_index + self.batch_size]
        self.clear_directory(target_directory)
        for frame_path in frames_to_copy:
            try:
                if not os.path.exists(target_directory):
                    os.makedirs(target_directory)
                shutil.copy(frame_path, target_directory)
            except Exception as e:
                print(f"Failed to copy {frame_path}. Reason: {e}")

    def show_zoom_view(self, frame, x, y, zoom_factor=4, zoom_size=200):
        height, width = frame.shape[:2]
        half_zoom = zoom_size // 2
        x_start = max(x - half_zoom // zoom_factor, 0)
        x_end = min(x + half_zoom // zoom_factor, width)
        y_start = max(y - half_zoom // zoom_factor, 0)
        y_end = min(y + half_zoom // zoom_factor, height)
        zoomed_area = frame[y_start:y_end, x_start:x_end]
        zoomed_area_resized = cv2.resize(zoomed_area, (zoom_size, zoom_size), interpolation=cv2.INTER_LINEAR)
        zoom_view = np.zeros((zoom_size, zoom_size, 3), dtype=np.uint8)
        zoom_view[:zoom_size, :zoom_size] = zoomed_area_resized
        scaled_x = zoom_size // 2
        scaled_y = zoom_size // 2
        cv2.circle(zoom_view, (scaled_x, scaled_y), 5, (0, 255, 0), -1)
        return zoom_view

    def click_event(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.selected_points.append([x, y])
            self.selected_labels.append(self.current_class_label)
            cv2.circle(self.current_frame, (x, y), 2, self.label_colors[self.current_class_label], -1)
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.is_drawing:
                self.selected_points.append([x, y])
                self.selected_labels.append(self.current_class_label)
                cv2.circle(self.current_frame, (x, y), 2, self.label_colors[self.current_class_label], -1)
            zoom_view = self.show_zoom_view(self.current_frame, x, y)
            cv2.imshow("Zoom View", zoom_view)
            try:
                zoom_window = gw.getWindowsWithTitle("Zoom View")[0]
                zoom_window.activate()
            except IndexError:
                pass
        elif event == cv2.EVENT_LBUTTONUP:
            self.is_drawing = False
        elif event == cv2.EVENT_RBUTTONDOWN:
            self.selected_points.append([x, y])
            self.selected_labels.append(-self.current_class_label)
            cv2.circle(self.current_frame, (x, y), 4, (0, 0, 255), -1)
        cv2.imshow(f"Frame{self.count}/{self.val}", self.current_frame)

    def collect_user_points(self):
        self.count = 1
        window_width = 200
        window_height = 200
        cv2.namedWindow("Zoom View", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Zoom View", window_width, window_height)
        for frame_path in self.frame_paths:
            self.current_frame = cv2.imread(frame_path)
            self.val = len(self.frame_paths)
            cv2.namedWindow(f"Frame{self.count}/{self.val}", cv2.WINDOW_NORMAL)
            cv2.imshow(f"Frame{self.count}/{self.val}", self.current_frame)
            cv2.setMouseCallback(f"Frame{self.count}/{self.val}", self.click_event)
            while True:
                key = cv2.waitKey(0)
                if key == 13:  # Enter key
                    self.points_collection_list.append(self.selected_points[:])
                    self.labels_collection_list.append(self.selected_labels[:])
                    self.selected_points.clear()
                    self.selected_labels.clear()
                    break
                if key == ord('q'):
                    return
                elif key in map(ord, '123456789'):
                    self.current_class_label = int(chr(key))
            cv2.destroyAllWindows()
        self.save_points_and_labels()

    def save_points_and_labels(self, filename=None):
        if filename is None:
            filename = f"points_labels_video{self.video_number}.json"
        with open(filename, 'w') as f:
            json.dump({"points": self.points_collection_list, "labels": self.labels_collection_list}, f)

    def run(self):
        # video2imgs_.VIDEO_NUMBER = self.video_number
        # video2imgs_.main(self.video_number)
        self.frame_paths = sorted(
            [os.path.join(self.frames_directory, p) for p in os.listdir(self.frames_directory)
             if os.path.splitext(p)[-1].lower() in [".jpg", ".jpeg", '.png']],
            key=lambda p: int(re.search(r'(\d+)', os.path.splitext(p)[0]).group()) if re.search(r'(\d+)',
                                                                                                os.path.splitext(p)[
                                                                                                    0]) else float(
                'inf')
        )
        threading.Thread(target=self.collect_user_points).start()


if __name__ == "__main__":
    processor = VideoFrameProcessor(video_number=4)
    processor.run()
