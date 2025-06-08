import os
import sys

import cv2
import numpy as np
import pygetwindow as gw
import torch

from ..UserUI.AnnotationManager import AnnotationManager
from ..Model.SAM2Config import SAM2Config
from ..FileManagement.FileManager import clear_directory
from ..FileManagement.FrameExtractor import FrameExtractor
from ..FileManagement.FrameHandler import FrameHandler
from ..FileManagement.MaskProcessor import MaskProcessor
from ..Model.SAM2Model import SAM2Model
from ..UserUI.UserInteraction import UserInteractionHandler
from ..UserUI.logger_config import logger

print(torch.cuda.get_device_name(0))


class SAM2VideoProcessor(SAM2Model):
    """Main class for SAM2 video processing."""

    def __init__(self, video_number, batch_size=120, images_starting_count=0, images_ending_count=None,
                 prefix="file", video_path_template=None, images_extract_dir=None,
                 rendered_frames_dir=None, temp_processing_dir=None, is_drawing=False,
                 window_size=None, label_colors=None, memory_bank_size=5, prompt_memory_size=5):
        self.inference_state = None
        sam2Config = SAM2Config(
            video_number=video_number, batch_size=batch_size, images_starting_count=images_starting_count,
            images_ending_count=images_ending_count, prefix=prefix, video_path_template=video_path_template,
            images_extract_dir=images_extract_dir, rendered_frames_dir=rendered_frames_dir,
            temp_processing_dir=temp_processing_dir, window_size=window_size,
            label_colors=label_colors, memory_bank_size=memory_bank_size, prompt_memory_size=prompt_memory_size
        )
        super().__init__(sam2Config)
        if video_path_template is None:
            logger.error("Missing the video file paths or video")
            sys.exit(1)
        self.is_prompted = False
        self.is_drawing = is_drawing
        extractor = FrameExtractor(
            video_number, prefixFileName=prefix, limitedImages=images_ending_count,
            video_path_template=video_path_template, output_dir=images_extract_dir
        )
        extractor.run()
        self.frame_handler = FrameHandler(sam2Config.frames_directory, sam2Config.temp_directory)
        self.frame_paths = self.frame_handler.get_frame_files()
        self.annotation_manager = AnnotationManager(sam2Config, self.frame_paths)
        self.user_interaction = UserInteractionHandler(sam2Config, self.annotation_manager, self)
        self.mask_processor = MaskProcessor(sam2Config)

    def click_event(self, event, x, y, flags, param):
        """Handle mouse events for point selection."""
        inference_state_temp, frame_path = param
        if event == cv2.EVENT_LBUTTONDOWN:
            self.user_interaction.selected_points.append([x, y])
            full_label = self.user_interaction.encode_label(
                self.user_interaction.current_class_label, self.user_interaction.current_instance_id)
            if self.mask_processor.mask_box_points:
                points_list = list(self.mask_processor.mask_box_points.values())
                label_list = list(self.mask_processor.mask_box_points.keys())
                matching_boxes = [
                    ((points_list[i][2] - points_list[i][0]) * (points_list[i][3] - points_list[i][1]), label_list[i])
                    for i in range(len(points_list))
                    if points_list[i][0] <= x <= points_list[i][2] and points_list[i][1] <= y <= points_list[i][3]
                ]
                if matching_boxes:
                    matching_boxes.sort(key=lambda x: x[0])
                    full_label = matching_boxes[0][1]
            if not (self.mask_processor.last_mask is None or isinstance(self.mask_processor.last_mask,
                                                                        (tuple, list)) and
                    self.mask_processor.last_mask in [(None,), [None]]):
                if self.mask_processor.last_mask[y][x] > 0:
                    full_label = self.mask_processor.last_mask[y][x]
            cv2.circle(self.user_interaction.current_frame, (x, y), 2,
                       self.config.label_colors[self.user_interaction.current_class_label], -1)
            cv2.circle(self.user_interaction.current_frame_only_with_points, (x, y), 2,
                       self.config.label_colors[self.user_interaction.current_class_label], -1)
            self.user_interaction.selected_labels.append(full_label)
            self.user_prompt_adder(inference_state_temp, frame_path)
            self.user_interaction.draw_text_with_background(self.user_interaction.current_frame)
            logger.debug(f"Click: ({x}, {y}), Labels: {self.user_interaction.selected_labels}")
            cv2.imshow(self.user_interaction.window_name, self.user_interaction.current_frame)
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.is_drawing:
                self.user_interaction.selected_points.append([x, y])
                full_label = self.user_interaction.encode_label(
                    self.user_interaction.current_class_label, self.user_interaction.current_instance_id)
                self.user_interaction.selected_labels.append(full_label)
                cv2.circle(self.user_interaction.current_frame, (x, y), 2,
                           self.config.label_colors[self.user_interaction.current_class_label], -1)
                cv2.circle(self.user_interaction.current_frame_only_with_points, (x, y), 2,
                           self.config.label_colors[self.user_interaction.current_class_label], -1)
            zoom_view = self.user_interaction.show_zoom_view(self.user_interaction.current_frame, x, y)
            cv2.imshow("Zoom View", zoom_view)
            try:
                zoom_window = gw.getWindowsWithTitle("Zoom View")[0]
                zoom_window.activate()
            except Exception:
                pass
        elif event == cv2.EVENT_LBUTTONUP:
            self.is_drawing = False
        elif event == cv2.EVENT_RBUTTONDOWN:
            self.user_interaction.selected_points.append([x, y])
            full_label = self.user_interaction.encode_label(
                self.user_interaction.current_class_label, self.user_interaction.current_instance_id) * -1
            if self.mask_processor.mask_box_points:
                points_list = list(self.mask_processor.mask_box_points.values())
                label_list = list(self.mask_processor.mask_box_points.keys())
                matching_boxes = [
                    ((points_list[i][2] - points_list[i][0]) * (points_list[i][3] - points_list[i][1]), label_list[i])
                    for i in range(len(points_list))
                    if points_list[i][0] <= x <= points_list[i][2] and points_list[i][1] <= y <= points_list[i][3]
                ]
                if matching_boxes:
                    matching_boxes.sort(key=lambda x: x[0])
                    full_label = matching_boxes[0][1] * -1
            if not (self.mask_processor.last_mask is None or isinstance(self.mask_processor.last_mask,
                                                                        (tuple, list)) and
                    self.mask_processor.last_mask in [(None,), [None]]):
                if int(self.mask_processor.last_mask[y][x]) > 0:
                    full_label = int(self.mask_processor.last_mask[y][x]) * -1
            cv2.circle(self.user_interaction.current_frame, (x, y), 4, (0, 0, 255), -1)
            cv2.circle(self.user_interaction.current_frame_only_with_points, (x, y), 4, (0, 0, 255), -1)
            self.user_interaction.selected_labels.append(full_label)
            self.user_prompt_adder(inference_state_temp, frame_path)
            self.user_interaction.draw_text_with_background(self.user_interaction.current_frame)
            logger.debug(f"Click: ({x}, {y}), Labels: {self.user_interaction.selected_labels}")

    def user_prompt_adder(self, inference_state, frame_path):
        """Add user prompts and update the displayed frame."""
        self.sam2_predictor.reset_state(inference_state)
        box_points = None
        if not (self.mask_processor.last_mask is None or isinstance(self.mask_processor.last_mask, (
                tuple, list)) and self.mask_processor.last_mask in [(None,),
                                                                    [None]]):
            box_points = self.auto_prompt_encoding(inference_state)
        self.prompt_encoding(inference_state)
        if self.is_prompted:
            video_segments = {}
            for out_frame_idx, out_obj_ids, out_mask_logits in self.sam2_predictor.propagate_in_video(
                    inference_state, isSingle=True):
                video_segments[out_frame_idx] = {
                    out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
                    for i, out_obj_id in enumerate(out_obj_ids)
                }
            mask = self.mask_processor.binary_mask_2_color_mask(
                out_frame_idx, frame_path, video_segments, 0, self.config.temp_directory, False)
            current_frame_org = self.user_interaction.current_frame_only_with_points.copy()
            non_zero_mask = np.any(mask > 0, axis=-1)
            non_zero_mask_3d = np.stack([non_zero_mask] * 3, axis=-1)
            blended = cv2.addWeighted(current_frame_org, 0.5, mask, 0.5, 0)
            current_frame_org[non_zero_mask_3d] = blended[non_zero_mask_3d]
            self.user_interaction.current_frame = self.show_box(box_points, current_frame_org)
            cv2.imshow(self.user_interaction.window_name, self.user_interaction.current_frame)

    @staticmethod
    def show_box(boxes, img):
        """Draw boxes on image."""
        if boxes is None:
            return img
        for box in boxes:
            x0, y0 = int(box[0]), int(box[1])
            x1, y1 = int(box[2]), int(box[3])
            cv2.rectangle(img, (x0, y0), (x1, y1), (0, 255, 0), thickness=2)
        return img

    def prompt_encoding(self, inference_state, batch_number=-1):
        """Encode prompts for SAM2 model."""
        if batch_number == -1:
            points_list = self.user_interaction.selected_points
            label_list = self.user_interaction.selected_labels
            frame_idx = 0
        else:
            if len(self.annotation_manager.points_collection) > batch_number:
                points_list = self.annotation_manager.points_collection[batch_number]
                label_list = self.annotation_manager.labels_collection[batch_number]
                frame_idx = self.annotation_manager.frame_indices[batch_number]
            else:
                points_list = []
                label_list = []
                frame_idx = 0
        points_np = np.array(points_list, dtype=np.float32)
        labels_np = np.array(label_list, dtype=np.int32)
        unique_labels = np.unique(np.abs(labels_np))
        if len(unique_labels) == 0:
            return None
        for label in unique_labels:
            self.is_prompted = True
            obj_mask = np.abs(labels_np) == label
            points_np1 = points_np[obj_mask]
            raw_labels_np1 = labels_np[obj_mask]
            labels_np1 = (raw_labels_np1 > 0).astype(np.int32)
            self.sam2_predictor.add_new_points_or_box(
                inference_state=inference_state,
                frame_idx=(frame_idx % self.config.batch_size),
                clear_old_points=False,
                obj_id=int(label),
                points=points_np1,
                labels=labels_np1
            )
        return not None

    def auto_prompt_encoding(self, inference_state):
        """Encode automatic prompts from previous masks."""
        points_list = []
        label_list = []
        box_prompt = self.mask_processor.mask_to_boxes(self.mask_processor.last_mask)
        if box_prompt is None:
            return None
        for k, v in box_prompt.items():
            points_list.append(v)
            label_list.append(k)
        points_np = [np.array(points, dtype=np.float32) for points in points_list]
        labels_np = label_list
        for i in range(len(points_np)):
            self.is_prompted = True
            self.sam2_predictor.add_new_points_or_box(
                inference_state=inference_state,
                frame_idx=0,
                obj_id=int(labels_np[i]),
                box=points_np[i]
            )
        return points_np

    def run(self):
        """Run the SAM2 video predictor pipeline."""
        start_batch_idx = self.annotation_manager.check_data_sufficiency()
        batch_index = 0
        while batch_index < len(self.frame_paths):
            logger.info(
                f"Processing batch {(batch_index // self.config.batch_size) + 1}/"
                f"{(len(self.frame_paths) + self.config.batch_size - 1) // self.config.batch_size}")
            self.frame_handler.move_and_copy_frames(batch_index, self.frame_paths, self.config.batch_size)
            self.is_prompted = False
            if batch_index >= start_batch_idx:
                self.user_interaction.collect_user_points(
                    batch_index // self.config.batch_size,
                    self.frame_paths,
                    self.sam2_predictor,
                    self.click_event,
                    self.mask_processor
                )
            self.is_prompted = False
            self.mask_processor.generate_mask(
                batch_number=batch_index // self.config.batch_size,
                sam2_predictor=self.sam2_predictor,
                temp_directory=self.config.temp_directory,
                prompt_encoding=self.prompt_encoding,
                auto_prompt_encoding=self.auto_prompt_encoding
            )
            batch_index += self.config.batch_size
            logger.info('-' * 28 + " completed" + '-' * 28)
        clear_directory(self.config.temp_directory)
