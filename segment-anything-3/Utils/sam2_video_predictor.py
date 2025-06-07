import argparse
import json
import os
import re
import shutil
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

import GPUtil
import cv2
import numpy as np
import pygetwindow as gw
import torch

print(torch.cuda.get_device_name(0))
from Utils.FrameExtractor import FrameExtractor
from Utils.ImageCopier import ImageCopier
from Utils.ImageOverlayProcessor import ImageOverlayProcessor
from Utils.VideoCreator import VideoCreator
from Utils.logger_config import logger
from sam2.build_sam import build_sam2_video_predictor


# form some devices we need to set False
# torch.backends.cuda.enable_flash_sdp(False)


def get_resource_path(relative_path):
    """ Get absolute path to resource, works for dev and for PyInstaller """
    if hasattr(sys, '_MEIPASS'):
        return os.path.join(sys._MEIPASS, relative_path)
    return os.path.join(os.path.abspath(".."), relative_path)


def ensure_directory(path):
    """Create directory if it doesn't exist."""
    os.makedirs(path, exist_ok=True)
    return path


def resource_path(relative_path):
    """Get absolute path to resource, works for dev and for PyInstaller."""
    try:
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        base_path = sys._MEIPASS
    except AttributeError:
        # In development, use the current directory
        base_path = os.path.abspath("..")

    # Construct the full path and normalize it
    full_path = os.path.normpath(os.path.join(base_path, relative_path))
    return full_path


def clear_directory(directory):
    if os.path.exists(directory):
        for filename in os.listdir(directory):
            file_path = os.path.join(directory, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                logger.error(f"Failed to delete {file_path}. Reason: {e}")
    else:
        logger.debug(f"Directory {directory} does not exist.")


class sam2_video_predictor:
    def __init__(self, video_number, batch_size=120, images_starting_count=0, images_ending_count=None,
                 prefix="file", video_path_template=None, images_extract_dir=None, rendered_frames_dir=None,
                 temp_processing_dir=None, is_drawing=False, window_size=None, label_colors=None, memory_bank_size=5,
                 prompt_memory_size=5):
        self.isPrompted = False
        self.mask_box_points = {}
        self.box_points = None
        self.prefixFileName = prefix
        self.video_number = video_number
        self.frame_list = None
        self.current_frame_only_text = None
        self.rendered_frames_dir = rendered_frames_dir or './videos/outputs'
        self.frames_directory = images_extract_dir or './videos/images'
        self.temp_directory = temp_processing_dir or './videos/temp'
        ensure_directory(self.rendered_frames_dir)
        ensure_directory(self.frames_directory)
        ensure_directory(self.temp_directory)
        self.points_collection_list, self.labels_collection_list, self.frame_indices = self.load_points_and_labels()
        self.window_size = window_size or [200, 200]
        self.current_frame_only_with_points = None
        self.window_name = "SAM2 Annotation Tool"
        self.current_class_label = 1
        self.current_instance_id = 1
        self.display_text = f"In class ID {self.current_class_label}, instance ID: {self.current_instance_id}"
        self.label_colors = label_colors or {
            1: (0, 0, 255), 2: (255, 0, 0), 3: (0, 255, 0), 4: (0, 255, 255),
            5: (255, 0, 255), 6: (255, 255, 0), 7: (128, 0, 128), 8: (0, 165, 255),
            9: (255, 255, 255), 10: (0, 0, 0)
        }
        self.class_instance_counter = defaultdict(int)
        self.current_instance_id = 1
        if video_path_template is None:
            logger.error("Missing the video file paths or video")
            sys.exit(100)
        self.device = self.get_device()
        self.gpus = GPUtil.getGPUs()
        self.batch_size = batch_size
        self.video_path_template = video_path_template
        self.rendered_frames_dirs = rendered_frames_dir
        self.temp_directory = temp_processing_dir
        self.model_config = resource_path("sam2_configs/sam2_hiera_l.yaml")
        self.frames_directory = images_extract_dir
        self.sam2_checkpoint = resource_path("checkpoints/sam2_hiera_large.pt")
        self.memory_bank_size = memory_bank_size  # n: Store last n masks/encodings
        self.prompt_memory_size = prompt_memory_size
        self.last_mask = None,
        self.sam2_predictor = self.build_predictor()
        extractor = FrameExtractor(self.video_number, prefixFileName=self.prefixFileName,
                                   limitedImages=images_ending_count, video_path_template=self.video_path_template,
                                   output_dir=images_extract_dir)
        extractor.run()
        self.is_drawing = is_drawing
        self.image_counter = images_starting_count
        self.current_class_label = 1
        self.current_frame = None
        self.selected_points = []
        self.selected_labels = []
        self.points_collection_list = []
        self.labels_collection_list = []
        self.frame_paths = self.get_frame_paths()

    @staticmethod
    def get_device():
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")
        if device.type == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        return device

    def build_predictor(self):
        return build_sam2_video_predictor(
            self.model_config, self.sam2_checkpoint, device=self.device,
            memory_bank_size=self.memory_bank_size, prompt_memory_size=self.prompt_memory_size

        )

    def get_frame_paths(self):
        frame_paths = []
        for p in os.listdir(self.frames_directory):
            if os.path.splitext(p)[-1].lower() in [".jpg", ".jpeg", ".png"]:
                # Extract the numeric frame index after the underscore (e.g., '00000' from 'road4_00000.jpg')
                match = re.search(r'_(\d+)\.(?:jpg|jpeg|png)$', p, re.IGNORECASE)
                if match:
                    frame_paths.append(os.path.join(self.frames_directory, p))
                else:
                    logger.warning(f"Skipping file {p} due to invalid filename format")
        # Sort by the numeric frame index
        return sorted(frame_paths,
                      key=lambda p: int(re.search(r'_(\d+)\.(?:jpg|jpeg|png)$', p, re.IGNORECASE).group(1)))

    @staticmethod
    def mask2colorMaskImg(mask):
        colors = np.array([
            [0, 0, 0], [0, 0, 255], [0, 255, 0], [255, 0, 0], [0, 255, 255],
            [255, 0, 255], [255, 255, 0], [128, 0, 128], [0, 165, 255], [255, 255, 255]
        ], dtype=np.uint8)
        max_valid_id = len(colors) - 1
        mask = np.clip(mask, 0, max_valid_id)
        # logger.debug(f"Unique mask values: {np.unique(mask)}")
        return colors[mask]

    def gpu_memory_usage(self, ind=0):
        return self.gpus[ind]

    @staticmethod
    def encode_label(class_id, instance_id):
        return class_id * 1000 + instance_id

    def change_class_label(self, label):
        self.current_class_label = label
        self.current_instance_id = 1
        for i in self.selected_labels:
            if abs(i // 1000) == label:
                self.current_instance_id = max(abs(i) % 1000, self.current_instance_id)
        self.display_text = f"In class ID {self.current_class_label}, instance ID: {self.current_instance_id}"
        cv2.putText(self.current_frame, self.display_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (255, 255, 255), 2)
        self.draw_text_with_background(self.current_frame)
        cv2.imshow(self.window_name, self.current_frame)

    @staticmethod
    def show_zoom_view(frame, x, y, zoom_factor=4, zoom_size=200):
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

    def draw_text_with_background(self, image, position=(10, 30), font=cv2.FONT_HERSHEY_SIMPLEX,
                                  font_scale=1, text_color=(255, 255, 255), bg_color=(0, 0, 0),
                                  thickness=2, padding=5):
        text = self.display_text
        (text_width, text_height), _ = cv2.getTextSize(text, font, font_scale, thickness)
        x, y = position
        top_left = (x - padding, y - text_height - padding)
        bottom_right = (x + text_width + padding, y + padding)
        cv2.rectangle(image, top_left, bottom_right, bg_color, thickness=-1)
        cv2.putText(image, text, position, font, font_scale, text_color, thickness)
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)

    def load_points_and_labels(self):
        filename = f"./UserPrompts/points_labels_{self.prefixFileName}{self.video_number}.json"
        if not os.path.exists(filename):
            logger.warning(f"Points and labels file {filename} not found")
            return [], [], []
        try:
            with open(filename, 'r') as f:
                data = json.load(f)
            points_collection = [entry["points"] for entry in data]
            labels_collection = [entry["labels"] for entry in data]
            frame_indices = [entry["frame_idx"] for entry in data]
            return points_collection, labels_collection, frame_indices
        except Exception as e:
            logger.error(f"Error loading points and labels from {filename}: {e}")
            return [], [], []

    def check_data_sufficiency(self):
        total_batches = (len(self.frame_paths) + self.batch_size - 1) // self.batch_size
        if len(self.points_collection_list) >= total_batches:
            logger.info("Sufficient points and labels data for all batches")
            return 0  # Start from the first batch (all data is available)
        else:
            missing_batches = total_batches - len(self.points_collection_list)
            logger.info(f"Missing points and labels for {missing_batches} batches")
            return len(self.points_collection_list) * self.batch_size  # Start from the first missing batch

    def save_points_and_labels(self, points_collection, labels_collection, frame_indices, filename=None):
        filename = filename or f"./UserPrompts/points_labels_{self.prefixFileName}{self.video_number}.json"
        data = [
            {"frame_idx": frame_idx, "points": points, "labels": labels}
            for frame_idx, points, labels in zip(frame_indices, points_collection, labels_collection)
        ]
        try:
            # with open(filename, 'w') as f:
            #     json.dump(data, f, indent=2)
            # logger.info(f"Saved points, labels, and frame indices to {filename}")
            pass
        except Exception as e:
            logger.error(f"Error saving points and labels to {filename}: {e}")

    def move_and_copy_frames(self, batch_index):
        frames_to_copy = self.frame_paths[batch_index:batch_index + self.batch_size]
        clear_directory(self.temp_directory)
        ensure_directory(self.temp_directory)
        for i, frame_path in enumerate(frames_to_copy):
            # Extract the frame index from the filename (e.g., '00000' from 'road4_00000.jpg')
            match = re.search(r'_(\d+)\.(?:jpg|jpeg|png)$', frame_path, re.IGNORECASE)
            if not match:
                logger.warning(f"Skipping file {frame_path} due to invalid filename format")
                continue
            frame_index = match.group(1)
            # Create a new filename with only the frame index (e.g., '00000.jpg')
            new_filename = f"{frame_index}.jpg"
            dst_path = os.path.join(self.temp_directory, new_filename)
            shutil.copy(frame_path, dst_path)
            # logger.debug(f"Copied {frame_path} to {dst_path}")

    def mask_generator(self, batch_number):
        frame_file_names = sorted(
            [p for p in os.listdir(self.temp_directory) if
             os.path.splitext(p)[-1].lower() in [".jpg", ".jpeg", ".png"]],
            key=lambda p: int(re.search(r'_(\d+)\.(?:jpg|jpeg|png)$', p, re.IGNORECASE).group(1))
            if re.search(r'_(\d+)\.(?:jpg|jpeg|png)$', p, re.IGNORECASE) else float('inf')
        )
        inference_state = self.sam2_predictor.init_state(
            video_path=self.temp_directory,
            frame_paths=None)
        # self.sam2_predictor.reset_state(inference_state)
        if not (self.last_mask is None or isinstance(self.last_mask, (tuple, list)) and self.last_mask in [(None,),
                                                                                                           [None]]):
            self.box_points = None
            self.box_points = self.AutoPromptEncodingWithImageEncoding(inference_state, batch_number)
        else:
            self.box_points = None
        self.PromptEncodingWithImageEncoding(inference_state, batch_number)
        video_segments = {}
        for out_frame_idx, out_obj_ids, out_mask_logits in self.sam2_predictor.propagate_in_video(inference_state):
            video_segments[out_frame_idx] = {
                out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
                for i, out_obj_id in enumerate(out_obj_ids)
            }
        present_count = self.image_counter
        with ThreadPoolExecutor(max_workers=os.cpu_count() - 2) as executor:
            futures = [
                executor.submit(self.binary_mask_2_color_mask, out_frame_idx,
                                frame_file_names, video_segments, present_count + i)
                for i, out_frame_idx in enumerate(sorted(video_segments.keys()))]
            for future in futures:
                present_count = max(present_count, future.result())
        self.image_counter = present_count

    def collect_user_points(self, batch):
        cv2.namedWindow("Zoom View", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Zoom View", self.window_size[0], self.window_size[1])

        start_batch_idx = self.check_data_sufficiency()
        frames_batch_paths = [self.frame_paths[i] for i in
                              range(start_batch_idx, len(self.frame_paths), self.batch_size)]
        frames_batch_paths = [self.frame_paths[batch * self.batch_size]]
        print('frames_batch_paths==>', frames_batch_paths)
        for i, frame_path in enumerate(frames_batch_paths):
            print('frame_path==>', frame_path)
            batch_idx = (start_batch_idx // self.batch_size) + i
            frame_idx = batch_idx * self.batch_size
            inference_state_temp = self.sam2_predictor.init_state(
                video_path=None,
                frame_paths=[os.path.abspath(self.frame_paths[batch * self.batch_size])]
            )

            self.current_frame = self.current_frame_only_text = self.current_frame_only_with_points = cv2.imread(
                frame_path)
            self.current_class_label = self.current_instance_id = 1
            cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)

            parm = [inference_state_temp, frame_path]
            cv2.setMouseCallback(self.window_name, self.click_event, parm)
            if self.last_mask is None or isinstance(self.last_mask, (tuple, list)) and self.last_mask in [(None,),
                                                                                                          [None]]:
                pass
            else:
                self.userPromptAdder(inference_state_temp, frame_path, batch)
            while True:
                self.display_text = f"In class ID {self.current_class_label}, instance ID: {self.current_instance_id}"
                self.draw_text_with_background(self.current_frame)
                cv2.imshow(self.window_name, self.current_frame)
                key = cv2.waitKey(0)

                if key == 13:  # Enter key
                    if len(self.selected_points) > 0:
                        self.points_collection_list.append(self.selected_points[:])
                        self.labels_collection_list.append(self.selected_labels[:])
                        self.frame_indices.append(frame_idx)
                        self.save_points_and_labels(self.points_collection_list, self.labels_collection_list,
                                                    self.frame_indices)
                    self.selected_points.clear()
                    self.selected_labels.clear()
                    cv2.destroyAllWindows()
                    break
                elif key == ord('q'):
                    cv2.destroyAllWindows()
                    return
                elif key == 9:  # Tab
                    self.current_instance_id += 1
                    self.draw_text_with_background(self.current_frame)
                    cv2.imshow(self.window_name, self.current_frame)
                elif key == 353:  # Shift + Tab
                    if self.current_instance_id > 0:
                        self.current_instance_id -= 1
                        self.draw_text_with_background(self.current_frame)
                        cv2.imshow(self.window_name, self.current_frame)
                elif key == ord('u'):
                    if self.selected_points:
                        self.selected_points.pop()
                        self.selected_labels.pop()
                        self.current_frame = cv2.imread(frame_path)
                        self.draw_text_with_background(self.current_frame)
                        cv2.imshow(self.window_name, self.current_frame)
                        for pt, lbl in zip(self.selected_points, self.selected_labels):
                            cv2.circle(self.current_frame, (int(pt[0]), int(pt[1])), 2,
                                       self.label_colors[abs(lbl // 1000)], -1)
                            cv2.circle(self.current_frame_only_with_points, (int(pt[0]), int(pt[1])), 2,
                                       self.label_colors[abs(lbl // 1000)], -1)
                        if len(self.selected_points) > 0:
                            self.userPromptAdder(inference_state_temp, frame_path)
                elif key in [ord(str(i)) for i in range(1, 10)]:
                    self.change_class_label(int(chr(key)))
                elif key == ord('r'):
                    self.selected_points = []
                    self.selected_labels = []
                    self.current_frame = self.current_frame_only_text = (
                        self).current_frame_only_with_points = cv2.imread(
                        frame_path)
                elif key == ord('f'):
                    frame_idx_input = input("Enter frame index to annotate: ")
                    try:
                        new_frame_idx = int(frame_idx_input)
                        if 0 <= new_frame_idx < len(self.frame_paths):
                            frame_path = self.frame_paths[new_frame_idx]
                            inference_state_temp = self.sam2_predictor.init_state(
                                video_path=frame_path,
                                frame_paths=[os.path.abspath(frame_path)]
                            )
                            self.current_frame = self.current_frame_only_text = (
                                self).current_frame_only_with_points = cv2.imread(
                                frame_path)
                            self.current_class_label = self.current_instance_id = 1
                            parm = [inference_state_temp, frame_path]
                            cv2.setMouseCallback(self.window_name, self.click_event, parm)
                            if len(self.selected_points) > 0:
                                self.points_collection_list.append(self.selected_points[:])
                                self.labels_collection_list.append(self.selected_labels[:])
                                self.frame_indices.append(new_frame_idx)
                                self.save_points_and_labels(self.points_collection_list, self.labels_collection_list,
                                                            self.frame_indices)
                            self.selected_points.clear()
                            self.selected_labels.clear()
                        else:
                            logger.warning(f"Invalid frame index: {new_frame_idx}")
                    except ValueError:
                        logger.warning("Invalid input for frame index")
            self.save_points_and_labels(self.points_collection_list, self.labels_collection_list, self.frame_indices)
        cv2.destroyAllWindows()

    def click_event(self, event, x, y, flags, parm):
        inference_state_temp = parm[0]
        frame_path = parm[1]
        if event == cv2.EVENT_LBUTTONDOWN:
            self.selected_points.append([x, y])
            '''
            WE NEED TO CHECK THE POINTS POSITION IS PRESENT INSIDE THE self.box_points then full_label
            should the key of the box
            '''

            boxPrompt = self.mask_box_points
            full_label = self.encode_label(self.current_class_label, self.current_instance_id)
            if not (boxPrompt is None):
                points_list = []
                label_list = []
                for k, v in boxPrompt.items():
                    points_list.append(v)
                    label_list.append(k)
                matching_boxes = []
                for i in range(len(points_list)):
                    x0, y0, x1, y1 = points_list[i]
                    if x0 <= x <= x1 and y0 <= y <= y1:
                        area = (x1 - x0) * (y1 - y0)
                        matching_boxes.append((area, label_list[i]))
                if matching_boxes:
                    # Choose the box with the smallest area
                    matching_boxes.sort(key=lambda x: x[0])
                    full_label = matching_boxes[0][1]
            if not (self.last_mask is None or isinstance(self.last_mask, (tuple, list))
                    and self.last_mask in [(None,), [None]]):
                if self.last_mask[y][x] > 0:
                    full_label = self.last_mask[y][x]
            self.selected_labels.append(full_label)
            cv2.circle(self.current_frame, (x, y), 2, self.label_colors[self.current_class_label], -1)
            cv2.circle(self.current_frame_only_with_points, (x, y), 2, self.label_colors[self.current_class_label], -1)
            self.userPromptAdder(inference_state_temp, frame_path)
            print("Click =>", f'({x}, {y}) ,self.selected_labels => {self.selected_labels}')
            print("Click =>", f'({x}, {y}) ,self.labels_collection_list => {self.labels_collection_list}')
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.is_drawing:
                self.selected_points.append([x, y])

                full_label = self.encode_label(self.current_class_label, self.current_instance_id)
                self.selected_labels.append(full_label)

                cv2.circle(self.current_frame, (x, y), 2, self.label_colors[self.current_class_label], -1)
                cv2.circle(self.current_frame_only_with_points, (x, y), 2, self.label_colors[self.current_class_label],
                           -1)
            zoom_view = self.show_zoom_view(self.current_frame, x, y)
            cv2.imshow("Zoom View", zoom_view)
            try:
                zoom_window = gw.getWindowsWithTitle("Zoom View")[0]
                try:
                    zoom_window.activate()
                except gw.PyGetWindowException:
                    pass
            except IndexError:
                pass  # No Zoom View window found, skip
        elif event == cv2.EVENT_LBUTTONUP:
            self.is_drawing = False
        elif event == cv2.EVENT_RBUTTONDOWN:
            self.selected_points.append([x, y])

            boxPrompt = self.mask_box_points
            full_label = self.encode_label(self.current_class_label, self.current_instance_id) * -1
            if not (boxPrompt is None):
                points_list = []
                label_list = []
                for k, v in boxPrompt.items():
                    points_list.append(v)
                    label_list.append(k)
                matching_boxes = []
                for i in range(len(points_list)):
                    x0, y0, x1, y1 = points_list[i]
                    if x0 <= x <= x1 and y0 <= y <= y1:
                        area = (x1 - x0) * (y1 - y0)
                        matching_boxes.append((area, label_list[i]))
                if matching_boxes:
                    # Choose the box with the smallest area
                    matching_boxes.sort(key=lambda x: x[0])
                    full_label = matching_boxes[0][1] * -1
            if not (self.last_mask is None or isinstance(self.last_mask, (tuple, list))
                    and self.last_mask in [(None,), [None]]):
                if self.last_mask[y][x] > 0:
                    full_label = self.last_mask[y][x] * -1
            self.selected_labels.append(full_label)
            cv2.circle(self.current_frame, (x, y), 4, (0, 0, 255), -1)
            cv2.circle(self.current_frame_only_with_points, (x, y), 4, (0, 0, 255), -1)
            self.userPromptAdder(inference_state_temp, frame_path)
        self.draw_text_with_background(self.current_frame)
        cv2.imshow(self.window_name, self.current_frame)

    def userPromptAdder(self, inference_state_temp, frame_path, batch=-1):
        self.sam2_predictor.reset_state(inference_state_temp)
        if not (self.last_mask is None or isinstance(self.last_mask, (tuple, list)) and self.last_mask in [(None,),
                                                                                                           [None]]):
            self.box_points = None
            self.box_points = self.AutoPromptEncodingWithImageEncoding(inference_state_temp, batch)
        else:
            self.box_points = None
        self.PromptEncodingWithImageEncoding(inference_state_temp, -1)
        if self.isPrompted:
            video_segments = {}
            for out_frame_idx, out_obj_ids, out_mask_logits in self.sam2_predictor.propagate_in_video(
                    inference_state_temp,
                    isSingle=True):
                video_segments[out_frame_idx] = {
                    out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
                    for i, out_obj_id in enumerate(out_obj_ids)
                }
            mask = self.binary_mask_2_color_mask(out_frame_idx, frame_path, video_segments, 0, save=False)
            current_frame_org = self.current_frame_only_with_points.copy()
            # self.current_frame = cv2.addWeighted(current_frame_org, 0.5, mask, 0.5, 0)

            # Make 2D mask and broadcast to 3D
            non_zero_mask = np.any(mask > 0, axis=-1)  # (H, W)
            non_zero_mask_3d = np.stack([non_zero_mask] * 3, axis=-1)  # (H, W, 3)

            # Perform blending
            blended = cv2.addWeighted(current_frame_org, 0.5, mask, 0.5, 0)
            # cv2.imshow('blended', cv2.resize(blended, (640, 480)))
            # Only update non-zero regions
            current_frame_org[non_zero_mask_3d] = blended[non_zero_mask_3d]
            # current_frame_org =
            self.current_frame = self.show_box(self.box_points, current_frame_org)

    @staticmethod
    def show_box(boxs, img):
        if boxs is None: return img
        for box in boxs:
            x0, y0 = int(box[0]), int(box[1])
            x1, y1 = int(box[2]), int(box[3])
            # Draw rectangle on the image
            cv2.rectangle(img, (x0, y0), (x1, y1), color=(0, 255, 0), thickness=1)
        return img

    def PromptEncodingWithImageEncoding(self, inference_state, batch_number=-1):
        if batch_number == -1:
            points_list = self.selected_points
            label_list = self.selected_labels
            frame_idx = 0
        else:
            if len(self.points_collection_list) > batch_number:
                points_list = self.points_collection_list[batch_number]
                label_list = self.labels_collection_list[batch_number]
                frame_idx = self.frame_indices[batch_number]
            else:
                points_list = []
                label_list = []
                frame_idx = 0
        points_np = np.array(points_list, dtype=np.float32)
        labels_np = np.array(label_list, dtype=np.int32)

        # Validate labels
        unique_labels = np.unique(np.abs(labels_np))
        max_expected_label = 10
        if unique_labels is None:
            raise ValueError(
                f"Invalid labels in batch {batch_number}: {unique_labels}. Max expected: {max_expected_label}")
        ensure_directory(self.rendered_frames_dirs)
        labels_np_c = labels_np.copy()
        print('points_np, labels_np=>', points_np, labels_np)
        for label in unique_labels:
            self.isPrompted = True
            # class_id = label // 1000
            # instance_id = label % 1000
            obj_mask = np.abs(labels_np) == label
            points_np1 = points_np[obj_mask]
            raw_labels_np1 = labels_np[obj_mask]

            # Binary mask: foreground = 1, background = 0
            labels_np1 = (raw_labels_np1 > 0).astype(np.int32)
            # print('self.box_points=>', self.box_points, label)

            _, object_ids, mask_logits = self.sam2_predictor.add_new_points_or_box(
                inference_state=inference_state,
                frame_idx=(frame_idx % self.batch_size),
                clear_old_points=False,
                obj_id=int(label),
                points=points_np1,
                labels=labels_np1
            )
            # else:
            #     _, object_ids, mask_logits = self.sam2_predictor.add_new_points_or_box(
            #         inference_state=inference_state,
            #         frame_idx=(frame_idx % self.batch_size),
            #         clear_old_points=True,
            #         obj_id=int(label),  # unique for each class+instance
            #         points=points_np1,
            #         labels=labels_np1,
            #         box=self.box_points[0]
            #     )
        labels_np = labels_np_c.copy()

    def AutoPromptEncodingWithImageEncoding(self, inference_state, batch_number):
        # self.selected_labels = self.labels_collection_list[batch_number]
        # self.selected_points = self.points_collection_list[batch_number]
        points_list = []
        label_list = []
        boxPrompt = self.mask_to_boxes(self.last_mask)
        print('boxPrompt =>', boxPrompt)
        print('points_list =>', points_list)
        print('label_list =>', label_list)
        frame_idx = 0
        for k, v in boxPrompt.items():
            points_list.append(v)
            label_list.append(k)
        print('points_list =>', points_list)
        print('label_list =>', label_list)
        # self.points_collection_list.append(points_list)
        # self.labels_collection_list.append(label_list)
        points_np = []
        labels_np = []
        for i in points_list:
            points_np.append(np.array(i, dtype=np.float32))

        labels_np = label_list
        # points_np = np.array(points_list, dtype=np.float32)
        # labels_np = np.array(label_list, dtype=np.int32)
        # Validate labels
        ensure_directory(self.rendered_frames_dirs)
        for i in range(len(points_np)):
            self.isPrompted = True
            print('label, point =>', labels_np[i], points_np[i])
            self.sam2_predictor.add_new_points_or_box(
                inference_state=inference_state,
                frame_idx=0,
                obj_id=int(labels_np[i]),
                box=points_np[i]
            )
        return points_np

    def mask_to_boxes(self, mask: np.ndarray):
        boxes = {}
        object_ids = np.unique(mask)
        print("unique object_ids =>", object_ids)
        object_ids = object_ids[object_ids != 0]  # Ignore background (0)
        for obj_id in object_ids:
            # Create a binary mask for this object
            binary_mask = (mask == obj_id).astype(np.uint8)

            # Find contours
            contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                continue

            # Bounding box from largest contour
            largest_contour = max(contours, key=cv2.contourArea)
            if cv2.contourArea(largest_contour) < int(mask.shape[0] * mask.shape[1] * 0.00015):
                continue
            x, y, w, h = cv2.boundingRect(largest_contour)
            boxes[int(obj_id)] = [max(x, 0), max(y, 0), min(x + w, mask.shape[1]),
                                  min(y + h, mask.shape[0])]
        self.mask_box_points = boxes
        return boxes

    def binary_mask_2_color_mask(self, out_frame_idx, frame_filenames, video_segments, present_count, save=True):
        if save:
            frame_path = os.path.join(self.temp_directory, frame_filenames[out_frame_idx])
        else:
            frame_path = frame_filenames
        frame = cv2.imread(frame_path)
        full_mask = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint16)
        temp = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint16)
        for out_obj_id in sorted(video_segments[out_frame_idx].keys(), reverse=True):
            out_mask = video_segments[out_frame_idx][out_obj_id]
            # logger.debug(f"Processing object ID: {out_obj_id}")
            if out_mask.dtype == np.bool_:
                out_mask = out_mask.astype(np.uint8)
            out_mask = out_mask.squeeze()
            if out_mask.shape[:2] != (frame.shape[0], frame.shape[1]):
                out_mask_resized = cv2.resize(out_mask, (frame.shape[1], frame.shape[0]),
                                              interpolation=cv2.INTER_NEAREST_EXACT)
            else:
                out_mask_resized = out_mask
            mask_condition = (out_mask_resized > 0) & (full_mask == 0)
            full_mask[mask_condition] = abs(out_obj_id // 1000)
            temp[mask_condition] = abs(out_obj_id)
        if save and out_frame_idx == len(video_segments) - 1:
            self.last_mask = temp.copy()
        color_mask_image = self.mask2colorMaskImg(full_mask)
        if save:
            cv2.imwrite(
                os.path.join(self.rendered_frames_dirs,
                             f"{self.prefixFileName}{self.video_number}_{present_count:05d}.png"),
                color_mask_image
            )
        else:
            return color_mask_image
        return present_count + 1

    def run(self):
        self.points_collection_list, self.labels_collection_list, self.frame_indices = self.load_points_and_labels()
        start_batch_idx = self.check_data_sufficiency()
        # if start_batch_idx > 0:
        logger.info(f"Starting point collection from batch {start_batch_idx // self.batch_size}")
        # thread = threading.Thread(target=self.collect_user_points)
        # thread.start()

        batch_index = 0
        while batch_index < len(self.frame_paths):
            logger.info(
                f"Processing batch {((batch_index + 1) // self.batch_size) + 1}/"
                f"{(len(self.frame_paths) // self.batch_size)}")
            self.move_and_copy_frames(batch_index)
            # if batch_index // self.batch_size == 0:
            self.isPrompted = False
            self.collect_user_points(batch_index // self.batch_size)
            self.isPrompted = False
            self.mask_generator(batch_index // self.batch_size)
            batch_index += self.batch_size
            logger.info('-' * 28 + " completed " + '-' * 28)
        clear_directory(self.temp_directory)


def run_pipeline(video_number, video_path_template, images_extract_dir, rendered_dirs, overlap_dir, verified_img_dir,
                 verified_mask_dir, prefix, batch_size, fps, final_video_path, temp_processing_dir, delete,
                 images_ending_count):
    """Run the entire pipeline for a single video number."""
    print(f"Processing video {video_number}")

    processor = sam2_video_predictor(
        video_number=video_number,
        prefix=prefix,
        batch_size=batch_size,
        video_path_template=video_path_template,
        images_extract_dir=images_extract_dir,
        rendered_frames_dir=rendered_dirs,
        temp_processing_dir=temp_processing_dir,
        images_ending_count=images_ending_count
    )
    processor.run()

    overlay_processor = ImageOverlayProcessor(
        original_folder=images_extract_dir,
        mask_folder=rendered_dirs,
        output_folder=overlap_dir,
        all_consider=prefix,
        image_count=0
    )
    overlay_processor.process_all_images()
    while delete != 'yes':
        user_input = input(
            "Have you verified all the overlay masks on original images?"
            " Enter 'yes' to proceed or 'no' to exit: ").lower()
        if user_input == 'yes':
            break
        elif user_input == 'no':
            logger.info("Pipeline terminated: Verification not completed")
            sys.exit(0)

    logger.info(f"Copying verified images and masks by default input {delete == 'yes'}")

    copier = ImageCopier(
        original_folder=images_extract_dir,
        mask_folder=rendered_dirs,
        overlap_images_folder=overlap_dir,
        output_original_folder=verified_img_dir,
        output_mask_folder=verified_mask_dir
    )
    copier.copy_images()

    ensure_directory(final_video_path)
    video_names = [f"{final_video_path}/OrgVideo{video_number}.mp4", f"{final_video_path}/MaskVideo{video_number}.mp4",
                   f"{final_video_path}/OverlappedVideo{video_number}.mp4"]
    video_creator = VideoCreator(
        image_folders=[verified_img_dir, verified_mask_dir, overlap_dir],
        video_names=video_names,
        fps=fps
    )
    video_creator.run()


def main():
    parser = argparse.ArgumentParser(description="Automated video processing pipeline.")
    parser.add_argument('--video_start', type=int, default=1, help='Starting video number (inclusive)')
    parser.add_argument('--video_end', type=int, default=1, help='Ending video number (exclusive)')
    parser.add_argument('--prefix', type=str, default='Img', help='Prefix for output filenames')
    parser.add_argument('--batch_size', type=int, default=5, help='Batch size for processing frames')
    parser.add_argument('--fps', type=int, default=30, help='Frames per second for output videos')
    parser.add_argument('--delete', type=str, choices=['yes', 'no'], default='yes',
                        help='Delete working directory without verification prompt (yes/no)')
    parser.add_argument('--working_dir_name', type=str, default='working_dir',
                        help='Base directory name for working directories')
    parser.add_argument('--video_path_template', type=str, default='./VideoInputs/Video{}.mp4',
                        help='Template path for video files, e.g., ./VideoInputs/Video{}.mp4')
    parser.add_argument('--images_extract_dir', type=str, default='./working_dir/images',
                        help='Directory to extract images')
    parser.add_argument('--temp_processing_dir', type=str, default='./working_dir/temp',
                        help='Directory for temporary processing images')
    parser.add_argument('--rendered_dir', type=str, default='./working_dir/render',
                        help='Directory for rendered mask outputs')
    parser.add_argument('--overlap_dir', type=str, default='./working_dir/overlap',
                        help='Directory for overlapped images')
    parser.add_argument('--verified_img_dir', type=str, default='./working_dir/verified/images',
                        help='Directory for verified original images')
    parser.add_argument('--verified_mask_dir', type=str, default='./working_dir/verified/mask',
                        help='Directory for verified mask images')
    parser.add_argument('--final_video_path', type=str, default='./outputs',
                        help='Directory to save output videos')
    parser.add_argument('--images_ending_count', type=int, default=15,
                        help='Directory to save output videos')

    args = parser.parse_args()

    for i in range(args.video_start, args.video_start + args.video_end):
        if os.path.exists(args.working_dir_name):
            if args.delete.lower() == 'yes':
                shutil.rmtree(args.working_dir_name)
                pass
            else:
                confirm = input(
                    f"Do you want to clear prev working directory '{args.working_dir_name}'? (yes/no): "
                ).lower()
                if confirm == 'yes':
                    shutil.rmtree(args.working_dir_name)
                    # logger.info(f"Clearing prev working directory: {args.working_dir_name}")
                    pass
                else:
                    logger.info(f"Working directory '{args.working_dir_name}' not deleted")
                    logger.info('Stopping the process')
                    sys.exit(1000)

        run_pipeline(
            fps=args.fps,
            video_number=i,
            prefix=args.prefix,
            batch_size=args.batch_size,
            delete=args.delete.lower(),
            video_path_template=args.video_path_template.replace('working_dir', args.working_dir_name),
            images_extract_dir=args.images_extract_dir.replace('working_dir', args.working_dir_name),
            temp_processing_dir=args.temp_processing_dir.replace('working_dir', args.working_dir_name),
            rendered_dirs=args.rendered_dir.replace('working_dir', args.working_dir_name),
            overlap_dir=args.overlap_dir.replace('working_dir', args.working_dir_name),
            verified_img_dir=args.verified_img_dir.replace('working_dir', args.working_dir_name),
            verified_mask_dir=args.verified_mask_dir.replace('working_dir', args.working_dir_name),
            final_video_path=args.final_video_path,
            images_ending_count=args.images_ending_count
        )

        logger.info("Pipeline completed for all videos.")
        if os.path.exists(args.working_dir_name):
            if args.delete.lower() == 'yes':
                shutil.rmtree(args.working_dir_name)
                logger.info(f"Cleared working directory: {args.working_dir_name}")
            else:
                confirm = input(
                    f"Are you sure you want to delete the working directory '{args.working_dir_name}'? (yes/no): "
                ).lower()
                if confirm == 'yes':
                    shutil.rmtree(args.working_dir_name)
                    logger.info(f"Cleared working directory: {args.working_dir_name}")
                else:
                    logger.info(f"Working directory '{args.working_dir_name}' not deleted")


if __name__ == "__main__":
    main()

'''

pyinstaller --name sam2_video_predictor --add-data "F:\RunningProjects\SAM2\checkpoints\sam2_hiera_large.pt;checkpoints" --add-data "F:\RunningProjects\SAM2\sam2_configs\sam2_hiera_l.yaml;sam2_configs" --add-data "F:\RunningProjects\SAM2\sam2_configs\__init__.py;sam2_configs" --add-data "F:\RunningProjects\SAM2\sam2;sam2" --hidden-import torch --hidden-import cv2 --hidden-import numpy --hidden-import GPUtil --hidden-import sam2 --hidden-import sam2.build_sam --hidden-import sam2.sam2_configs --hidden-import hydra --hidden-import hydra.core --hidden-import hydra.experimental --collect-all sam2 --collect-all hydra --collect-all torch --collect-all numpy --collect-all cv2 --collect-all GPUtil --clean --onefile F:\RunningProjects\SAM2\segment-anything-3\sam2_video_predictor.py

'''
