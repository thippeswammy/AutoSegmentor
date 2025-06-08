import os
from collections import defaultdict

import cv2
import numpy as np
from .logger_config import logger


class UserInteractionHandler:
    """Handles user interface and interaction logic."""

    def __init__(self, config, annotation_manager, sam2_video_predictor):
        self.config = config
        self.annotation_manager = annotation_manager
        self.sam2_video_predictor = sam2_video_predictor
        self.window_name = "SAM2 Annotation Tool"
        self.current_class_label = 1
        self.current_instance_id = 1
        self.display_text = f"In class ID {self.current_class_label}, instance ID: {self.current_instance_id}"
        self.is_drawing = False
        self.selected_points = []
        self.selected_labels = []
        self.current_frame = None
        self.current_frame_only_text = None
        self.current_frame_only_with_points = None
        self.class_instance_counter = defaultdict(int)

    @staticmethod
    def encode_label(class_id, instance_id):
        """Encode class and instance IDs into a single label."""
        return class_id * 1000 + instance_id

    def change_class_label(self, label):
        """Change the current class label and update instance ID."""
        self.current_class_label = label
        self.current_instance_id = 1
        for i in self.selected_labels:
            if abs(i // 1000) == label:
                self.current_instance_id = max(abs(i) % 1000, self.current_instance_id)
        self.display_text = f"In class ID {self.current_class_label}, instance ID: {self.current_instance_id}"
        self.draw_text_with_background(self.current_frame)
        cv2.imshow(self.window_name, self.current_frame)

    @staticmethod
    def show_zoom_view(frame, x, y, zoom_factor=4, zoom_size=200):
        """Show a zoomed view of the frame at the cursor position."""
        height, width = frame.shape[:2]
        half_zoom = zoom_size // 2
        x_start = max(x - half_zoom // zoom_factor, 0)
        x_end = min(x + half_zoom // zoom_factor, width)
        y_start = max(y - half_zoom // zoom_factor, 0)
        y_end = min(y + half_zoom // zoom_factor, height)
        zoomed_area = frame[y_start:y_end, x_start:x_end]
        zoomed_area_resized = cv2.resize(zoomed_area, (zoom_size, zoom_size), interpolation=cv2.INTER_LINEAR)
        zoom_view = np.zeros((zoom_size, zoom_size, 3), dtype=np.uint8)
        zoom_view[:zoomed_area_resized.shape[0], :zoomed_area_resized.shape[1]] = zoomed_area_resized
        scaled_x = zoom_size // 2
        scaled_y = zoom_size // 2
        cv2.circle(zoom_view, (scaled_x, scaled_y), 5, (0, 255, 0), -1)
        return zoom_view

    def draw_text_with_background(self, frame, position=(10, 30), font=cv2.FONT_HERSHEY_SIMPLEX,
                                  font_scale=1, text_color=(255, 255, 255), bg_color=(0, 0, 0),
                                  thickness=2, padding=5):
        """Draw text with a background rectangle."""
        if frame is None:
            return
        text = self.display_text
        (text_width, text_height), _ = cv2.getTextSize(text, font, font_scale, thickness)
        x, y = position
        top_left = (x - padding, y - text_height - padding)
        bottom_right = (x + text_width + padding, y + padding)
        cv2.rectangle(frame, top_left, bottom_right, bg_color, thickness=-1)
        cv2.putText(frame, text, position, font, font_scale, text_color, thickness)
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)

    def collect_user_points(self, batch, frame_paths, sam2_predictor, click_event_callback, mask_processor):
        """Collect user points for annotation."""
        cv2.namedWindow("Zoom View", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Zoom View", self.config.window_size[0], self.config.window_size[1])
        start_batch_idx = self.annotation_manager.check_data_sufficiency()
        frame_path = frame_paths[batch * self.config.batch_size]
        batch_idx = (start_batch_idx // self.config.batch_size)
        frame_idx = batch_idx * self.config.batch_size
        inference_state_temp = sam2_predictor.init_state(
            video_path=None,
            frame_paths=[os.path.abspath(frame_path)]
        )
        self.current_frame = self.current_frame_only_text = self.current_frame_only_with_points = cv2.imread(
            frame_path)
        self.current_class_label = self.current_instance_id = 1
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        param = [inference_state_temp, frame_path]
        cv2.setMouseCallback(self.window_name, click_event_callback, param)
        self.sam2_video_predictor.user_prompt_adder(inference_state_temp, frame_path)
        while True:
            self.display_text = f"In class ID {self.current_class_label}, instance ID: {self.current_instance_id}"
            self.draw_text_with_background(self.current_frame)
            cv2.imshow(self.window_name, self.current_frame)
            key = cv2.waitKey(0)
            if key == 13:  # Enter key
                if self.selected_points:
                    self.annotation_manager.points_collection.append(self.selected_points[:])
                    self.annotation_manager.labels_collection.append(self.selected_labels[:])
                    self.annotation_manager.frame_indices.append(frame_idx)
                    self.annotation_manager.save_points_and_labels()
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
                    for pt, lbl in zip(self.selected_points, self.selected_labels):
                        cv2.circle(
                            self.current_frame,
                            (int(pt[0]), int(pt[1])), 2,
                            self.config.label_colors[abs(lbl // 1000)], -1
                        )
                        cv2.circle(
                            self.current_frame_only_with_points,
                            (int(pt[0]), int(pt[1])), 2,
                            self.config.label_colors[abs(lbl // 1000)], -1
                        )
                        self.sam2_video_predictor.user_prompt_adder(inference_state_temp, frame_path)
                    cv2.imshow(self.window_name, self.current_frame)
            elif key in [ord(str(i)) for i in range(1, 10)]:
                self.change_class_label(int(chr(key)))
            elif key == ord('r'):
                self.selected_points = []
                self.selected_labels = []
                self.current_frame = self.current_frame_only_text = self.current_frame_only_with_points = cv2.imread(
                    frame_path)
            elif key == ord('f'):
                frame_idx_input = input("Enter frame index to annotate: ")
                try:
                    new_frame_idx = int(frame_idx_input)
                    if 0 <= new_frame_idx < len(frame_paths):
                        frame_path = frame_paths[new_frame_idx]
                        inference_state_temp = sam2_predictor.init_state(
                            video_path=None,
                            frame_paths=[os.path.abspath(frame_path)]
                        )
                        self.current_frame = self.current_frame_only_text = self.current_frame_only_with_points = cv2.imread(
                            frame_path)
                        self.current_class_label = self.current_instance_id = 1
                        param = [inference_state_temp, frame_path]
                        cv2.setMouseCallback(self.window_name, click_event_callback, param)
                        if self.selected_points:
                            self.annotation_manager.points_collection.append(self.selected_points[:])
                            self.annotation_manager.labels_collection.append(self.selected_labels[:])
                            self.annotation_manager.frame_indices.append(new_frame_idx)
                            self.annotation_manager.save_points_and_labels()
                        self.selected_points.clear()
                        self.selected_labels.clear()
                    else:
                        logger.warning(f"Invalid frame index: {new_frame_idx}")
                except ValueError:
                    logger.warning("Invalid input for frame index")
        self.annotation_manager.save_points_and_labels()
        cv2.destroyAllWindows()
