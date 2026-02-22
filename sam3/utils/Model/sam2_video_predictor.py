import sys
import threading
import time

import cv2
import numpy as np
import pygetwindow as gw
import torch

from ..FileManagement.FileManager import clear_directory
from ..FileManagement.FrameExtractor import FrameExtractor
from ..FileManagement.FrameHandler import FrameHandler
from ..FileManagement.MaskProcessor import MaskProcessor
from ..Model.SAM2Config import SAM2Config
from ..Model.SAM2Model import SAM2Model
from ..UserUI.AnnotationManager import AnnotationManager
from ..UserUI.UserInteraction import UserInteractionHandler
from ..UserUI.logger_config import logger

print(torch.cuda.get_device_name(0))


class SAM2VideoProcessor(SAM2Model):
    """Main class for SAM2 video processing."""

    def __init__(self, video_number, batch_size=120, images_starting_count=0, images_ending_count=None,
                 prefix="file", video_path_template=None, images_extract_dir=None,
                 rendered_frames_dir=None, temp_processing_dir=None, is_drawing=False,
                 window_size=None, label_colors=None, memory_bank_size=5, prompt_memory_size=5, pose_config=None,
                 auto_prompt_encoding=True):
        self.inference_state = None
        sam2Config = SAM2Config(
            video_number=video_number, batch_size=batch_size, images_starting_count=images_starting_count,
            images_ending_count=images_ending_count, prefix=prefix, video_path_template=video_path_template,
            images_extract_dir=images_extract_dir, rendered_frames_dir=rendered_frames_dir,
            temp_processing_dir=temp_processing_dir, window_size=window_size,
            label_colors=label_colors, memory_bank_size=memory_bank_size, prompt_memory_size=prompt_memory_size,
            pose_config=pose_config, auto_prompt_encoding=auto_prompt_encoding
        )
        super().__init__(sam2Config)
        if video_path_template is None:
            logger.error("Missing the video file paths or video")
            sys.exit(1)
        self.is_prompted = False
        self.per_batch_tracked_data = []  # Accumulated per-batch CoTracker results
        self.is_drawing = is_drawing
        self._predictor_lock = threading.Lock()
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
            if self.user_interaction.pose_mode:
                self.user_interaction.record_pose_click(x, y)
                self.user_interaction.next_keypoint()

            self.user_prompt_adder(inference_state_temp, frame_path)
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
        with self._predictor_lock:
            self.sam2_predictor.reset_state(inference_state)
            self.is_prompted = False
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
            else:
                self.user_interaction.current_frame = self.user_interaction.current_frame_only_with_points.copy()


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
        if not self.config.auto_prompt_encoding:
            return None
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

    def _track_batch_inline(self, batch_number):
        """Run CoTracker on a single batch's frames and store the tracked data.
        
        This function processes a specified batch of frames by first checking if the
        pose  configuration is enabled. It retrieves keypoints for the current batch or
        carries  forward keypoints from the previous batch if necessary. The function
        then logs the  tracking process and initializes the CoTrackerKeypointTracker
        with the appropriate  parameters, including keypoints and frame paths. Finally,
        it attempts to track the  keypoints and store the results, handling any
        exceptions that may occur during the  tracking process.
        """
        pose_cfg = self.config.pose_config
        if not pose_cfg or not pose_cfg.get('enabled'):
            return

        # Determine keypoints for this batch
        pose_kps_collection = self.annotation_manager.pose_keypoints_collection
        batch_kps = None
        if batch_number < len(pose_kps_collection) and pose_kps_collection[batch_number]:
            batch_kps = pose_kps_collection[batch_number]
        elif self.per_batch_tracked_data:
            # Carry forward from the last frame of the previous batch
            prev_tracked = self.per_batch_tracked_data[-1]
            if prev_tracked:
                last_entry = prev_tracked[-1]  # Last frame's tracking data
                batch_kps = last_entry.get("keypoints", [])

        if not batch_kps:
            logger.info(f"[CoTracker] Batch {batch_number + 1}: No keypoints available, skipping inline tracking")
            self.per_batch_tracked_data.append([])
            return

        # Get this batch's frame paths
        batch_start = batch_number * self.config.batch_size
        batch_end = min(batch_start + self.config.batch_size, len(self.frame_paths))
        batch_frame_paths = self.frame_paths[batch_start:batch_end]

        logger.info(
            f"[CoTracker] Batch {batch_number + 1}: Tracking {len(batch_kps)} keypoints "
            f"across {len(batch_frame_paths)} frames"
        )

        try:
            from ..FileManagement.CoTrackerKeypointTracker import CoTrackerKeypointTracker
            ct_cfg = pose_cfg.get('cotracker', {})
            checkpoint = ct_cfg.get('checkpoint', '../co-tracker/checkpoints/scaled_offline.pth')
            import os as _os
            base_path = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), '..', '..'))
            checkpoint = _os.path.normpath(_os.path.join(base_path, checkpoint))
            window_len = ct_cfg.get('window_len', 60)

            tracker = CoTrackerKeypointTracker(
                keypoint_defs=pose_cfg.get('keypoints', []),
                initial_coords=batch_kps,
                frame_paths=batch_frame_paths,
                checkpoint=checkpoint,
                window_len=window_len,
            )
            tracked = tracker.get_all_tracked()
            self.per_batch_tracked_data.append(tracked)
            logger.info(f"[CoTracker] Batch {batch_number + 1}: Inline tracking complete ({len(tracked)} frames)")
        except Exception as e:
            logger.error(f"[CoTracker] Batch {batch_number + 1}: Inline tracking failed: {e}")
            self.per_batch_tracked_data.append([])

    def _mask_generation_consumer(self, total_batches):
        """Generates masks for batches as prompts become available."""
        for batch_num in range(total_batches):
            batch_index = batch_num * self.config.batch_size

            # Poll until prompt data is available for this batch
            while len(self.annotation_manager.points_collection) <= batch_num:
                target_file = f"./inputs/UserPrompts/points_labels_{self.config.prefix}{self.config.video_number}.json"
                logger.info(f"[MaskGen] Waiting for prompts for batch {batch_num + 1}/{total_batches} in {target_file}... (retrying in 5s)")
                time.sleep(5)
                self.annotation_manager.load_points_and_labels()

            logger.info(
                f"[MaskGen] Starting mask generation for batch {batch_num + 1}/{total_batches}")

            # Copy frames to the shared temp directory
            self.frame_handler.move_and_copy_frames(batch_index, self.frame_paths, self.config.batch_size)

            # Generate mask (MaskProcessor now handles granular locking internally)
            self.mask_processor.generate_mask(
                batch_number=batch_num,
                sam2_predictor=self.sam2_predictor,
                temp_directory=self.config.temp_directory,
                prompt_encoding=self.prompt_encoding,
                auto_prompt_encoding=self.auto_prompt_encoding,
                predictor_lock=self._predictor_lock
            )
            logger.info(f"[MaskGen] ══ Batch {batch_num + 1}/{total_batches} mask generation completed ══")

        logger.info("[MaskGen] All batches processed. Background mask generation finished.")

    def run(self):
        """Run the SAM2 video predictor pipeline.
        
        This function orchestrates the processing of video frames in either parallel or
        sequential mode based on the configuration. It manages batch processing, user
        interaction for prompt collection, and mask generation. In parallel mode, it
        spawns a consumer thread for mask generation while collecting user prompts as
        needed. In sequential mode, it processes each batch, collects user points, and
        generates masks accordingly, ensuring efficient handling of frames and
        resources.
        
        Args:
            self: The instance of the class containing the configuration and methods for
                processing.
        """
        total_batches = (len(self.frame_paths) + self.config.batch_size - 1) // self.config.batch_size
        logger.info(f"[Pipeline] {len(self.frame_paths)} frames, {total_batches} batches (batch_size={self.config.batch_size})")

        if not self.config.auto_prompt_encoding:
            # --- Parallel mode: producer (main thread) + consumer (background) ---
            consumer = threading.Thread(
                target=self._mask_generation_consumer,
                args=(total_batches,),
                daemon=True
            )
            consumer.start()

            # Check if all prompts are already available (pre-loaded from JSON)
            start_batch_idx = self.annotation_manager.check_data_sufficiency()
            if start_batch_idx >= len(self.frame_paths):
                logger.info(f"[Annotation] All {total_batches} batches already have prompts — skipping interactive collection")
            else:
                # Collect user prompts only for batches that need them
                for batch_num in range(total_batches):
                    if batch_num * self.config.batch_size >= start_batch_idx:
                        logger.info(f"[Annotation] Collecting prompts for batch {batch_num + 1}/{total_batches}")
                        self.user_interaction.collect_user_points(
                            batch_num,
                            self.frame_paths,
                            self.sam2_predictor,
                            self.click_event,
                            self.mask_processor
                        )

            # Wait for background mask generation to complete
            consumer.join()
            clear_directory(self.config.temp_directory)
            return

        # --- Sequential mode (auto_prompt_encoding=True): existing behavior ---
        batch_index = 0
        while batch_index < len(self.frame_paths):
            logger.info(
                f"Processing batch {(batch_index // self.config.batch_size) + 1}/"
                f"{total_batches}")
            self.is_prompted = False
            start_batch_idx = self.annotation_manager.check_data_sufficiency()
            if batch_index >= start_batch_idx:
                self.user_interaction.collect_user_points(
                    batch_index // self.config.batch_size,
                    self.frame_paths,
                    self.sam2_predictor,
                    self.click_event,
                    self.mask_processor
                )
            self.frame_handler.move_and_copy_frames(batch_index, self.frame_paths, self.config.batch_size)
            batch_number = batch_index // self.config.batch_size
            self.mask_processor.generate_mask(
                batch_number=batch_number,
                sam2_predictor=self.sam2_predictor,
                temp_directory=self.config.temp_directory,
                prompt_encoding=self.prompt_encoding,
                auto_prompt_encoding=self.auto_prompt_encoding,
                predictor_lock=self._predictor_lock
            )
            # Inline CoTracker tracking for this batch (mirrors auto_prompt pattern)
            if (self.config.pose_config and self.config.pose_config.get('enabled')
                    and self.config.pose_config.get('tracker', 'lk').lower() == 'cotracker'):
                self._track_batch_inline(batch_number)
            batch_index += self.config.batch_size
            logger.info(f"[Pipeline] ══ Batch {(batch_index // self.config.batch_size)}/{total_batches} completed ══")
        clear_directory(self.config.temp_directory)
