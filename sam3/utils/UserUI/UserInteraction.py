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
        self.current_frame_only_with_points = None
        self.class_instance_counter = defaultdict(int)

        # Pose Estimation State
        self.pose_mode = False
        self.pose_keypoints = []
        self.pose_click_coords = []  # Stores [{"name": str, "point_id": int, "x": int, "y": int}]
        self.current_keypoint_index = 0
        self.pose_class_id = 1
        self.pose_object_id = 1
        if self.config.pose_config and self.config.pose_config.get('enabled'):
            self.pose_mode = True
            self.pose_keypoints = self.config.pose_config.get('keypoints', [])
            self.pose_class_id = self.config.pose_config.get('class_id', 1)
            self.pose_object_id = self.config.pose_config.get('object_id', 1)
            # Lock class_label and instance_id to the single pose object identity
            self.current_class_label = self.pose_class_id
            self.current_instance_id = self.pose_object_id
            self.display_text = f"Click: {self.pose_keypoints[0]}" if self.pose_keypoints else "Pose Mode: No keypoints defined"

    @staticmethod
    def encode_label(class_id, instance_id):
        """Encode class and instance IDs into a single label."""
        return class_id * 1000 + instance_id

    def change_class_label(self, label):
        """Change the current class label and update instance ID."""
        if self.pose_mode:
            logger.warning("Class selection disabled in Pose Mode")
            return
        self.current_class_label = label
        self.current_instance_id = 1
        for i in self.selected_labels:
            if abs(i // 1000) == label:
                self.current_instance_id = max(abs(i) % 1000, self.current_instance_id)
        self.display_text = f"In class ID {self.current_class_label}, instance ID: {self.current_instance_id}"
        self.draw_text_with_background(self.current_frame)
        cv2.imshow(self.window_name, self.current_frame)

    def record_pose_click(self, x, y):
        """Record a keypoint click coordinate in Pose Mode."""
        if not self.pose_mode or self.current_keypoint_index >= len(self.pose_keypoints):
            return
        kp_name = self.pose_keypoints[self.current_keypoint_index]
        self.pose_click_coords.append({
            "name": kp_name,
            "point_id": self.current_keypoint_index,
            "x": int(x),
            "y": int(y)
        })
        logger.debug(f"Pose keypoint recorded: {kp_name} at ({x}, {y})")

    def next_keypoint(self):
        """Advance to the next keypoint in Pose Mode (does NOT change class_label)."""
        if not self.pose_mode or not self.pose_keypoints:
            return

        self.current_keypoint_index += 1
        if self.current_keypoint_index < len(self.pose_keypoints):
             self.display_text = f"Click: {self.pose_keypoints[self.current_keypoint_index]}"
             # class_label stays fixed — all keypoints share the same SAM2 obj_id
        else:
             self.display_text = "All Keypoints Set. Press Enter."

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
        
        if self.pose_mode:
            self.current_keypoint_index = 0
            # Reset selected points/labels for this fresh start
            self.selected_points = []
            self.selected_labels = []
            self.pose_click_coords = []
            # Lock to the single pose object identity from config
            self.current_class_label = self.pose_class_id
            self.current_instance_id = self.pose_object_id

            # --- Carry forward keypoints from previous batch or stored annotations ---
            prev_kps = None
            # 1. Check if stored annotation exists for this batch
            if batch < len(self.annotation_manager.pose_keypoints_collection):
                stored = self.annotation_manager.pose_keypoints_collection[batch]
                if stored:
                    prev_kps = stored
            # 2. If no stored annotation, carry forward from previous batch
            if prev_kps is None and batch > 0:
                for b in range(batch - 1, -1, -1):
                    if b < len(self.annotation_manager.pose_keypoints_collection):
                        stored = self.annotation_manager.pose_keypoints_collection[b]
                        if stored:
                            prev_kps = stored
                            break

            if prev_kps and len(prev_kps) == len(self.pose_keypoints):
                # Pre-populate with previous keypoints
                full_label = self.encode_label(self.pose_class_id, self.pose_object_id)
                for kp in sorted(prev_kps, key=lambda k: k["point_id"]):
                    x, y = kp["x"], kp["y"]
                    self.selected_points.append([x, y])
                    self.selected_labels.append(full_label)
                    self.pose_click_coords.append({
                        "name": kp["name"],
                        "point_id": kp["point_id"],
                        "x": x,
                        "y": y
                    })
                    # Draw the pre-populated point
                    cv2.circle(self.current_frame, (x, y), 2,
                               self.config.label_colors[self.pose_class_id], -1)
                    cv2.circle(self.current_frame_only_with_points, (x, y), 2,
                               self.config.label_colors[self.pose_class_id], -1)
                self.current_keypoint_index = len(self.pose_keypoints)
                self.display_text = "Prev keypoints loaded. Enter=Accept, R=Re-click"
                logger.info(f"Batch {batch + 1}: Pre-populated {len(prev_kps)} keypoints from previous data")
            else:
                if self.pose_keypoints:
                    self.display_text = f"Click: {self.pose_keypoints[0]}"
                else:
                    self.display_text = "Pose Mode Error: No Keypoints"
        else:
            self.current_class_label = self.current_instance_id = 1
            self.display_text = f"In class ID {self.current_class_label}, instance ID: {self.current_instance_id}"
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        param = [inference_state_temp, frame_path]
        cv2.setMouseCallback(self.window_name, click_event_callback, param)
        self.sam2_video_predictor.user_prompt_adder(inference_state_temp, frame_path)
        while True:
            if not self.pose_mode:
                 self.display_text = f"In class ID {self.current_class_label}, instance ID: {self.current_instance_id}"
            
            # Update display text for Pose Mode during loop (e.g. if undo happens)
            if self.pose_mode and self.pose_keypoints:
                 if self.current_keypoint_index < len(self.pose_keypoints):
                     self.display_text = f"Click: {self.pose_keypoints[self.current_keypoint_index]}"
                 else:
                     self.display_text = "All Keypoints Set. Press Enter."

            self.draw_text_with_background(self.current_frame)
            cv2.imshow(self.window_name, self.current_frame)
            key = cv2.waitKey(0)
            if key == 13:  # Enter key
                if self.selected_points:
                    self.annotation_manager.points_collection.append(self.selected_points[:])
                    self.annotation_manager.labels_collection.append(self.selected_labels[:])
                    self.annotation_manager.frame_indices.append(frame_idx)
                    if self.pose_mode and self.pose_click_coords:
                        self.annotation_manager.pose_keypoints_collection.append(self.pose_click_coords[:])
                    else:
                        self.annotation_manager.pose_keypoints_collection.append([])
                    self.annotation_manager.save_points_and_labels()
                self.selected_points.clear()
                self.selected_labels.clear()
                self.pose_click_coords.clear()
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
                    
                    if self.pose_mode:
                        self.current_keypoint_index = max(0, self.current_keypoint_index - 1)
                        if self.pose_click_coords:
                            self.pose_click_coords.pop()
                        # class_label stays fixed — no re-calculation needed

                    cv2.imshow(self.window_name, self.current_frame)
            elif key in [ord(str(i)) for i in range(1, 10)]:
                self.change_class_label(int(chr(key)))
            elif key == ord('r'):
                self.selected_points = []
                self.selected_labels = []
                self.pose_click_coords = []
                if self.pose_mode:
                    self.current_keypoint_index = 0
                    self.current_class_label = self.pose_class_id
                    self.current_instance_id = self.pose_object_id
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
                        self.current_frame = self.current_frame_only_text = (
                            self).current_frame_only_with_points = cv2.imread(frame_path)
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
