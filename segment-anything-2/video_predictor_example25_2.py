import argparse
import json
import logging
import os
import re
import shutil
import time

import GPUtil
import cv2
import numpy as np
import pygetwindow as gw
import torch

from sam2.build_sam import build_sam2_video_predictor
from videos.FrameExtractor import FrameExtractor

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


class VideoFrameProcessor:
    def __init__(self, video_number, batch_size=120, images_starting_count=0, images_ending_count=None,
                 prefixFileName="file", video_path_template=None, rendered_frames_dirs=None, is_drawing=False,
                 window_size=None, label_colors=None, extractorImages=True):
        if rendered_frames_dirs is None:
            rendered_frames_dirs = [f'./videos/outputs']
            # for i in range(1, 10):
            #     rendered_frames_dirs.append(f'./videos/outputs/rendered_frames_{i}')
        if window_size is None:
            window_size = [200, 200]
        if label_colors is None:
            # label_colors = {1: (0, 0, 255), 2: (255, 0, 0), 3: (0, 255, 0)}
            label_colors = {
                1: (0, 0, 255),
                2: (255, 0, 0),
                3: (0, 255, 0),
                4: (0, 255, 255),
                5: (255, 0, 255),
                6: (255, 255, 0),
                7: (128, 0, 128),
                8: (0, 165, 255),
                9: (255, 255, 255),
                10: (0, 0, 0),
            }
        if video_path_template is None:
            print("missing the video file paths or video")
            exit(100)
        self.device = self.get_device()
        self.gpus = GPUtil.getGPUs()
        self.batch_size = batch_size
        self.video_number = video_number
        self.prefixFileName = prefixFileName
        self.video_path_template = video_path_template
        self.rendered_frames_dirs = rendered_frames_dirs
        self.temp_directory = "videos/Temp"
        self.model_config = "sam2_hiera_l.yaml"
        self.frames_directory = "videos/Images"
        self.sam2_checkpoint = "../checkpoints/sam2_hiera_large.pt"
        self.sam2_predictor = self.build_predictor()
        extractor = FrameExtractor(self.video_number, prefixFileName=self.prefixFileName,
                                   limitedImages=images_ending_count, video_path_template=self.video_path_template)
        if extractorImages:
            extractor.run()
        self.is_drawing = is_drawing
        self.window_size = window_size
        self.label_colors = label_colors
        self.image_counter = images_starting_count
        self.current_class_label = 1
        self.current_frame = None
        self.selected_points = []
        self.selected_labels = []
        self.points_collection_list = []
        self.labels_collection_list = []
        self.color_map = self.get_color_map(9)
        self.frame_paths = self.get_frame_paths()

    def get_device(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logging.info(f"Using device: {device}")
        if device.type == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        return device

    def build_predictor(self):
        # logging.info("Building SAM2 video predictor.")
        return build_sam2_video_predictor(
            self.model_config, self.sam2_checkpoint, device=self.device
        )

    def save_points_and_labels(self, points_collection, labels_collection, filename=None):
        """Saves the collected points and labels to a JSON file."""
        if filename is None:
            filename = f"points_labels_{self.prefixFileName}{self.video_number}_{self.batch_size}.json"
        with open(filename, 'w') as f:
            json.dump({"points": points_collection, "labels": labels_collection}, f)
        logging.info(f"Saved points and labels to: {output_file}")

    def load_points_and_labels(self):
        filename = f"points_labels_{self.prefixFileName}{self.video_number}.json"
        with open(filename, 'r') as f:
            data = json.load(f)
        return data["points"], data["labels"]

    def load_user_points(self):
        self.points_collection_list, self.labels_collection_list = self.load_points_and_labels()

    def get_frame_paths(self):
        if not os.path.exists(self.frames_directory):
            logging.error(f"Frames directory '{self.frames_directory}' does not exist.")
            raise FileNotFoundError(f"Directory '{self.frames_directory}' not found.")

        paths = [os.path.join(self.frames_directory, p) for p in os.listdir(self.frames_directory)
                 if os.path.splitext(p)[-1].lower() in [".jpg", ".jpeg", '.png']]
        return sorted(paths, key=lambda p: int(re.search(r'(\d+)', os.path.splitext(p)[0]).group())
        if re.search(r'(\d+)', os.path.splitext(p)[0]) else float('inf'))

    def get_color_map(self, num_colors):
        colors = []
        for _ in range(num_colors):
            color = np.random.randint(0, 256, size=3).tolist()
            colors.append(color)
        return colors

    def gpu_memory_usage(self, ind=0):
        # for gpu in gpus[inf]:
        #     print(f"GPU Name: {gpu.name}")
        #     print(f"GPU Memory Total: {gpu.memoryTotal} MB")
        #     print(f"GPU Memory Used: {gpu.memoryUsed} MB")
        #     print(f"GPU Memory Free: {gpu.memoryFree} MB")
        #     print(f"GPU Memory Utilization: {gpu.memoryUtil * 100}%")
        return self.gpus[ind]

    def mask_2_color_mask_img(self, mask):
        colors = np.array([
            [0, 0, 0],  # Black (Background)
            [0, 0, 255],  # Red
            [0, 255, 0],  # Green
            [255, 0, 0],  # Blue
            [0, 255, 255],  # Yellow
            [255, 0, 255],  # Magenta
            [255, 255, 0],  # Cyan
            [128, 0, 128],  # Purple
            [0, 165, 255],  # Orange
            [255, 255, 255],  # White
        ], dtype=np.uint8)
        mask_image = colors[mask]
        return mask_image

    def move_and_copy_frames(self, batch_index):
        frames_to_copy = self.frame_paths[batch_index:batch_index + self.batch_size]
        self.clear_directory(self.temp_directory)
        for frame_path in frames_to_copy:
            os.makedirs(self.temp_directory, exist_ok=True)
            shutil.copy(frame_path, self.temp_directory)

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
        else:
            print(f"Directory {directory} does not exist.")

    def change_class_label(self, label):
        """Changes the current class label."""
        self.current_class_label = label
        # logging.info(f"Class label changed to: {self.current_class_label}")

    def collect_user_points(self):
        """Allows the user to interactively collect points on frames."""
        logging.info("Starting user point collection...")
        self.points_collection_list = []
        self.labels_collection_list = []
        cv2.namedWindow("Zoom View", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Zoom View", self.window_size[0], self.window_size[1])
        frames_batch_paths = [self.frame_paths[i] for i in range(0, len(self.frame_paths), self.batch_size)]
        for frame_path in frames_batch_paths:
            self.current_class_label = 1
            self.current_frame = cv2.imread(frame_path)
            cv2.namedWindow(f"Frame", cv2.WINDOW_NORMAL)
            cv2.imshow(f"Frame", self.current_frame)
            cv2.setMouseCallback(f"Frame", self.click_event)
            while True:
                key = cv2.waitKey(0)
                if key == 13:  # Enter key
                    self.points_collection_list.append(self.selected_points[:])
                    self.labels_collection_list.append(self.selected_labels[:])
                    self.selected_points.clear()
                    self.selected_labels.clear()
                    break
                if key == ord('q'):
                    logging.info("User exited point collection with 'q'.")
                    return
                elif key == ord('1'):
                    logging.warning("User reset points for the current frame.")
                    self.change_class_label(1)
                elif key == ord('2'):
                    self.change_class_label(2)
                elif key == ord('3'):
                    self.change_class_label(3)
                elif key == ord('4'):
                    self.change_class_label(4)
                elif key == ord('5'):
                    self.change_class_label(5)
                elif key == ord('6'):
                    self.change_class_label(6)
                elif key == ord('7'):
                    self.change_class_label(7)
                elif key == ord('8'):
                    self.change_class_label(8)
                elif key == ord('9'):
                    self.change_class_label(9)
                elif key == ord('r'):
                    self.selected_points = []
                    self.selected_labels = []
                    self.current_frame = cv2.imread(frame_path)
                    # current_frame = cv2.resize(current_frame, (window_width, window_height), interpolation=cv2.INTER_AREA)
        cv2.destroyAllWindows()
        logging.info("User point collection completed. Saving points and labels.")
        self.save_points_and_labels(self.points_collection_list, self.labels_collection_list)

    def click_event(self, event, x, y, flags, param):
        """Mouse callback to collect points on the frame."""
        if event == cv2.EVENT_LBUTTONDOWN:
            # logging.info(f"Point selected: ({x}, {y}), Label: {self.current_class_label}")
            self.is_drawing = True
            self.selected_points.append([x, y])
            self.selected_labels.append(self.current_class_label)  # Assign the current class label to the point
            cv2.circle(self.current_frame, (x, y), 2, self.label_colors[self.current_class_label],
                       -1)  # Draw color based on label
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.is_drawing:
                self.selected_points.append([x, y])
                self.selected_labels.append(self.current_class_label)
                cv2.circle(self.current_frame, (x, y), 2, self.label_colors[self.current_class_label], -1)
            zoom_view = self.show_zoom_view(self.current_frame, x, y)
            cv2.imshow("Zoom View", zoom_view)  # Update the zoom view
            # Keep the Zoom View window on top
            try:
                zoom_window = gw.getWindowsWithTitle("Zoom View")[0]
                zoom_window.activate()  # Bring it to the front
            except IndexError:
                pass  # Zoom window is not available
        elif event == cv2.EVENT_LBUTTONUP:
            self.is_drawing = False
        elif event == cv2.EVENT_RBUTTONDOWN:
            self.selected_points.append([x, y])
            self.selected_labels.append(self.current_class_label)  # 0 is for background points
            cv2.circle(self.current_frame, (x, y), 4, (0, 0, 255), -1)
        cv2.imshow(f"Frame", self.current_frame)

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

    def process_batch(self, batch_number):
        frame_filenames = sorted(
            [p for p in os.listdir(self.temp_directory) if
             os.path.splitext(p)[-1].lower() in [".jpg", ".jpeg", '.png']],
            key=lambda p: int(re.search(r'(\d+)', os.path.splitext(p)[0]).group())
            if re.search(r'(\d+)', os.path.splitext(p)[0]) else float('inf')
        )
        inference_state = self.sam2_predictor.init_state(video_path=self.temp_directory)
        self.sam2_predictor.reset_state(inference_state)
        points_np = np.array(self.points_collection_list[batch_number], dtype=np.float32)
        labels_np = np.array(self.labels_collection_list[batch_number], np.int32)
        # os.makedirs(rendered_frames_dir, exist_ok=True)
        ann_frame_idx = 0
        # Simulate model segmentation for each object (replace with real model interaction)
        for ann_obj_id in (set(abs(labels_np))):
            if not os.path.exists(self.rendered_frames_dirs[0]):
                os.makedirs(self.rendered_frames_dirs[0])
            present_count = self.image_counter + 0
            # Initialize and reset inference state for segmentation
            labels_np1 = labels_np.copy()
            labels_np1[labels_np1 == ann_obj_id] = labels_np1[labels_np1 == ann_obj_id] // ann_obj_id
            labels_np1[labels_np1 < 0] = 0
            labels_np1 = labels_np1[abs(labels_np) == ann_obj_id]
            points_np1 = points_np[abs(labels_np) == ann_obj_id]
            # inference_state = self.sam2_predictor.init_state(video_path=self.temp_directory)
            _, object_ids, mask_logits = self.sam2_predictor.add_new_points_or_box(
                inference_state=inference_state,
                frame_idx=ann_frame_idx,
                clear_old_points=False,
                obj_id=int(ann_obj_id),
                points=points_np1,
                labels=labels_np1
            )
        video_segments = {}
        for out_frame_idx, out_obj_ids, out_mask_logits in self.sam2_predictor.propagate_in_video(inference_state):
            video_segments[out_frame_idx] = {
                out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
                for i, out_obj_id in enumerate(out_obj_ids)
            }

        # Frame processing
        for out_frame_idx in range(len(frame_filenames)):
            frame_path = os.path.join(self.temp_directory, frame_filenames[out_frame_idx])
            frame = cv2.imread(frame_path)
            # Create a blank mask image with the same dimensions as the frame
            full_mask = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)

            # Iterate over masks and overlay them on the full mask
            for out_obj_id, out_mask in video_segments[out_frame_idx].items():
                # Convert mask to uint8 if it's boolean
                if out_mask.dtype == np.bool_:
                    out_mask = out_mask.astype(np.uint8)
                # Squeeze the mask to remove singleton dimension if necessary
                out_mask = out_mask.squeeze()

                # Resize the mask to fit the frame
                out_mask_resized = cv2.resize(out_mask, (frame.shape[1], frame.shape[0]))
                # Combine the mask into the full mask using bitwise OR
                out_mask_resized = out_mask_resized * out_obj_id
                full_mask = full_mask + out_mask_resized
            # Save the combined full mask image
            color_mask_image = self.mask_2_color_mask_img(full_mask)
            cv2.imwrite(
                os.path.join(
                    self.rendered_frames_dirs[
                        0] + f"/{self.prefixFileName}{self.video_number}_{present_count:05d}.png"),
                color_mask_image)
            present_count = present_count + 1
        self.image_counter = present_count

    def run(self):
        if os.path.exists(f"points_labels_{self.prefixFileName}{self.video_number}.json"):
            self.load_user_points()
        else:
            import threading
            thread = threading.Thread(target=self.collect_user_points)
            thread.start()
        batch_index = 0
        while batch_index < len(self.frame_paths):
            while len(self.points_collection_list) <= batch_index // self.batch_size:
                time.sleep(1)
            print(
                f"Processing batch {(batch_index + 1) // self.batch_size + 1}/{(len(self.frame_paths) // self.batch_size) + 1}")
            self.move_and_copy_frames(batch_index)
            self.process_batch(batch_index // self.batch_size)
            batch_index += self.batch_size
            print('-' * 28, "completed", '-' * 28)
        self.clear_directory(self.temp_directory)


if __name__ == "__main__":
    # Argument parser
    parser = argparse.ArgumentParser(description="Process video frames using VideoFrameProcessor.")

    # parser.add_argument("--video_number", type=int, default=100, help="Video number to process (default: 100).")
    parser.add_argument("--video_number", type=int, default=100, help="Video number to process (default: 100).")
    parser.add_argument("--batch_size", type=int, default=120, help="The batch size for processing frames.")
    parser.add_argument("--images_starting_count", type=int, default=0, help="Starting count for image numbering.")
    parser.add_argument("--images_ending_count", type=int, help="Ending count for limiting images.")
    parser.add_argument("--prefixFileName", type=str, default="bedroom", help="Prefix for output file names.")
    parser.add_argument("--video_path_template", type=str, default=r'D:\downloadFiles\front_3\video{}.mp4',
                        help="Template path for input videos (default: 'D:\\downloadFiles\\front_3\\video{}.mp4').")
    # parser.add_argument("--rendered_frames_dirs", type=str, nargs='*', default=None,
    #                     help="Directories to store rendered frames.")
    parser.add_argument("--rendered_frames_dirs", type=str, nargs='+', default=None,
                        help="Directories for rendered frames (default: None).")
    parser.add_argument("--is_drawing", action="store_true", default=False,
                        help="Enable interactive drawing for annotation.")
    parser.add_argument("--extractorImages", action="store_true", default=True,
                        help="The Extraction images for processing.")
    parser.add_argument("--window_size", type=int, nargs=2, default=[200, 200],
                        help="Window size for zoom view [width, height].")

    args = parser.parse_args()

    # Create processor instance with parsed arguments
    processor = VideoFrameProcessor(
        video_number=args.video_number,
        batch_size=args.batch_size,
        images_starting_count=args.images_starting_count,
        images_ending_count=args.images_ending_count,
        prefixFileName=args.prefixFileName,
        video_path_template=args.video_path_template,
        rendered_frames_dirs=args.rendered_frames_dirs,
        is_drawing=args.is_drawing,
        window_size=args.window_size,
        extractorImages=args.extractorImages
    )
    processor.run()

    # from videos.ImageOverlayProcessor import ImageOverlayProcessor

# python your_script.py --video_number 200 --prefixFileName "livingroom"
