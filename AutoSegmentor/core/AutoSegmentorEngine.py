import sys
import threading
import time
import traceback

import cv2
import numpy as np
import pygetwindow as gw
import torch

from ..file_management.FileManager import clear_directory
from ..file_management.FrameExtractor import FrameExtractor
from ..file_management.FrameHandler import FrameHandler
from ..file_management.MaskProcessor import MaskProcessor
from ..models.SAM.AppConfig import AppConfig
from ..models.SAM.SAM2Model import SAM2Model
from ..ui.AnnotationManager import AnnotationManager
from ..ui.UserInteraction import UserInteractionHandler
from ..ui.logger_config import logger

print(torch.cuda.get_device_name(0))


class AutoSegmentorEngine(SAM2Model):
    """Main engine for AutoSegmentor video processing.

    Orchestrates SAM2 (mask generation) and CoTracker (pose/keypoint tracking)
    across batches of video frames via an interactive annotation UI.
    """

    def __init__(self, video_number, batch_size=120, images_starting_count=0, images_ending_count=None,
                 prefix="file", video_path_template=None, images_extract_dir=None,
                 rendered_frames_dir=None, temp_processing_dir=None, working_dir=None, is_drawing=False,
                 window_size=None, label_colors=None, memory_bank_size=5, prompt_memory_size=5,
                 pose_config=None, auto_prompt_encoding=True, sam_enabled=True, sam_config=None, review_from_start=False):
        self.inference_state = None
        self.review_from_start = review_from_start
        config = AppConfig(
            video_number=video_number, batch_size=batch_size, images_starting_count=images_starting_count,
            images_ending_count=images_ending_count, prefix=prefix, video_path_template=video_path_template,
            images_extract_dir=images_extract_dir, rendered_frames_dir=rendered_frames_dir,
            temp_processing_dir=temp_processing_dir, working_dir=working_dir, window_size=window_size,
            label_colors=label_colors, memory_bank_size=memory_bank_size, prompt_memory_size=prompt_memory_size,
            pose_config=pose_config, auto_prompt_encoding=auto_prompt_encoding, sam_enabled=sam_enabled,
            sam_config=sam_config
        )
        super().__init__(config)
        if video_path_template is None:
            logger.error("Missing the video file paths or video")
            sys.exit(1)
        self.is_prompted = False
        self.is_drawing = is_drawing
        # Use RLock so the same thread can re-enter the lock
        # (e.g. user_prompt_adder_pyqt -> user_prompt_adder both acquire it)
        self._predictor_lock = threading.RLock()
        extractor = FrameExtractor(
            video_number, prefixFileName=prefix, limitedImages=images_ending_count,
            video_path_template=video_path_template, output_dir=images_extract_dir
        )
        extractor.run()
        self.frame_handler = FrameHandler(config.frames_directory, config.temp_directory)
        self.frame_paths = self.frame_handler.get_frame_files()
        total_batches = (len(self.frame_paths) + config.batch_size - 1) // config.batch_size
        self.per_batch_tracked_data = [[] for _ in range(total_batches)]
        self.annotation_manager = AnnotationManager(config, self.frame_paths)
        self.user_interaction = UserInteractionHandler(config, self.annotation_manager, self)
        self.mask_processor = MaskProcessor(config)

    def click_event(self, event, x, y, flags, param):
        """Handle mouse events for point selection."""
        inference_state_temp, frame_path = param
        if event == cv2.EVENT_LBUTTONDOWN:
            logger.debug(f"[Engine] LButtonDown at ({x},{y})")
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
                    logger.debug(f"[Engine] LClick: overrode label from mask_box_points -> {full_label}")
            if not (self.mask_processor.last_mask is None or isinstance(self.mask_processor.last_mask,
                                                                        (tuple, list)) and
                    self.mask_processor.last_mask in [(None,), [None]]):
                if self.mask_processor.last_mask[y][x] > 0:
                    full_label = self.mask_processor.last_mask[y][x]
                    logger.debug(f"[Engine] LClick: overrode label from last_mask pixel -> {full_label}")
            cv2.circle(self.user_interaction.current_frame, (x, y), 2,
                       self.config.label_colors[self.user_interaction.current_class_label], -1)
            cv2.circle(self.user_interaction.current_frame_only_with_points, (x, y), 2,
                       self.config.label_colors[self.user_interaction.current_class_label], -1)
            self.user_interaction.selected_labels.append(full_label)
            logger.debug(f"[Engine] LClick: point=({x},{y})  label={full_label}  total_pts={len(self.user_interaction.selected_points)}")
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
            logger.debug(f"[Engine] RButtonDown at ({x},{y})")
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
                    logger.debug(f"[Engine] RClick: overrode label from mask_box_points -> {full_label}")
            if not (self.mask_processor.last_mask is None or isinstance(self.mask_processor.last_mask,
                                                                        (tuple, list)) and
                    self.mask_processor.last_mask in [(None,), [None]]):
                if int(self.mask_processor.last_mask[y][x]) > 0:
                    full_label = int(self.mask_processor.last_mask[y][x]) * -1
                    logger.debug(f"[Engine] RClick: overrode label from last_mask pixel -> {full_label}")
            cv2.circle(self.user_interaction.current_frame, (x, y), 4, (0, 0, 255), -1)
            cv2.circle(self.user_interaction.current_frame_only_with_points, (x, y), 4, (0, 0, 255), -1)
            self.user_interaction.selected_labels.append(full_label)
            self.user_prompt_adder(inference_state_temp, frame_path)
            self.user_interaction.draw_text_with_background(self.user_interaction.current_frame)
            logger.debug(f"[Engine] RClick: point=({x},{y})  label={full_label}  total_pts={len(self.user_interaction.selected_points)}")

    def user_prompt_adder(self, inference_state, frame_path):
        """Add user prompts for a single frame and update the displayed frame.

        This is the core single-frame SAM2 preview. It must be called while
        holding (or able to acquire) _predictor_lock. Since _predictor_lock is
        now an RLock, re-entrant calls from the same thread are safe.
        """
        logger.debug(
            f"[Engine] user_prompt_adder: frame_path={frame_path}  "
            f"pts={len(self.user_interaction.selected_points)}  "
            f"labels={self.user_interaction.selected_labels}"
        )
        if not self.sam2_predictor:
            logger.debug("[Engine] user_prompt_adder: sam2_predictor is None, skipping")
            self.user_interaction.current_frame = self.user_interaction.current_frame_only_with_points.copy()
            return

        try:
            with self._predictor_lock:
                logger.debug("[Engine] user_prompt_adder: acquired predictor_lock, resetting state")
                self.sam2_predictor.reset_state(inference_state)
                self.is_prompted = False
                box_points = None
                if not (self.mask_processor.last_mask is None or isinstance(self.mask_processor.last_mask, (
                        tuple, list)) and self.mask_processor.last_mask in [(None,), [None]]):
                    logger.debug("[Engine] user_prompt_adder: running auto_prompt_encoding from last_mask")
                    box_points = self.auto_prompt_encoding(inference_state)
                    logger.debug(f"[Engine] user_prompt_adder: auto_prompt_encoding done  box_points={box_points is not None}")
                logger.debug("[Engine] user_prompt_adder: running prompt_encoding")
                self.prompt_encoding(inference_state)  # batch_number=-1 → single-frame mode
                logger.debug(f"[Engine] user_prompt_adder: is_prompted={self.is_prompted}")
                if self.is_prompted:
                    video_segments = {}
                    logger.debug("[Engine] user_prompt_adder: starting propagate_in_video")
                    for out_frame_idx, out_obj_ids, out_mask_logits in self.sam2_predictor.propagate_in_video(
                            inference_state, isSingle=True):
                        video_segments[out_frame_idx] = {
                            out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
                            for i, out_obj_id in enumerate(out_obj_ids)
                        }
                    logger.debug(f"[Engine] user_prompt_adder: propagate done  segments={list(video_segments.keys())}")
                    mask = self.mask_processor.binary_mask_2_color_mask(
                        out_frame_idx, frame_path, video_segments, 0, self.config.temp_directory, False)
                    if mask is None:
                        logger.warning("[Prompt] binary_mask_2_color_mask returned None — skipping overlay")
                        self.user_interaction.current_frame = self.user_interaction.current_frame_only_with_points.copy()
                        return
                    # Use the clean raw frame (no disk mask) as the base.
                    # current_frame_only_with_points has the disk mask baked in,
                    # which would cause a double-red-mask when the new SAM mask
                    # is overlaid on top. _raw_frame is always clean.
                    raw = getattr(self.user_interaction, '_raw_frame', None)
                    current_frame_org = (raw if raw is not None
                                         else self.user_interaction.current_frame_only_with_points).copy()
                    non_zero_mask = np.any(mask > 0, axis=-1)
                    non_zero_mask_3d = np.stack([non_zero_mask] * 3, axis=-1)
                    blended = cv2.addWeighted(current_frame_org, 0.5, mask, 0.5, 0)
                    current_frame_org[non_zero_mask_3d] = blended[non_zero_mask_3d]
                    self.user_interaction.current_frame = self.show_box(box_points, current_frame_org)
                    logger.debug("[Engine] user_prompt_adder: frame overlay applied successfully")
                else:
                    logger.debug("[Engine] user_prompt_adder: not prompted — showing points-only frame")
                    self.user_interaction.current_frame = self.user_interaction.current_frame_only_with_points.copy()
        except Exception:
            logger.error(f"[Prompt] user_prompt_adder failed:\n{traceback.format_exc()}")
            self.user_interaction.current_frame = self.user_interaction.current_frame_only_with_points.copy()

    @staticmethod
    def show_box(boxes, img):
        """Draw bounding boxes on image."""
        if boxes is None:
            return img
        for box in boxes:
            x0, y0 = int(box[0]), int(box[1])
            x1, y1 = int(box[2]), int(box[3])
            cv2.rectangle(img, (x0, y0), (x1, y1), (0, 255, 0), thickness=2)
        return img

    def prompt_encoding(self, inference_state, batch_number=-1):
        """Encode prompts for SAM2 model. Handles multiple prompt frames per batch."""
        logger.debug(f"[Engine] prompt_encoding: batch_number={batch_number}")
        if batch_number == -1:
            points_list = self.user_interaction.selected_points
            label_list = self.user_interaction.selected_labels
            frame_idx = 0
            prompts = [{"frame_idx": 0, "points": points_list, "labels": label_list}]
        else:
            if not self.sam2_predictor:
                return None
            
            # Fetch all prompts for the batch from AnnotationManager
            prompts_raw = self.annotation_manager.get_batch_prompts(batch_number, self.config.batch_size)
            prompts = []
            for p in prompts_raw:
                f_idx = p["frame_idx"]
                filtered = self.annotation_manager.get_points_for_model(f_idx, "sam")
                if filtered and len(filtered["points"]) > 0:
                    prompts.append({
                        "frame_idx": f_idx,
                        "points": filtered["points"],
                        "labels": filtered["labels"]
                    })

        if not prompts:
            logger.debug(f"[Engine] prompt_encoding: no prompts for batch={batch_number}, returning None")
            return None

        logger.debug(f"[Engine] prompt_encoding: encoding {len(prompts)} prompt frame(s)")
        for p_data in prompts:
            f_idx = p_data["frame_idx"]

            # Additional safety: ensure f_idx is within video range
            if f_idx >= len(self.frame_paths):
                logger.debug(f"[Engine] prompt_encoding: f_idx={f_idx} out of range, skipping")
                continue

            points_np = np.array(p_data["points"], dtype=np.float32)
            labels_np = np.array(p_data["labels"], dtype=np.int32)

            unique_labels = np.unique(np.abs(labels_np))
            if len(unique_labels) == 0:
                logger.debug(f"[Engine] prompt_encoding: f_idx={f_idx} has no unique labels, skipping")
                continue

            logger.debug(f"[Engine] prompt_encoding: f_idx={f_idx}  unique_labels={unique_labels.tolist()}  pts={len(points_np)}")
            for label in unique_labels:
                self.is_prompted = True
                obj_mask = np.abs(labels_np) == label
                points_np1 = points_np[obj_mask]
                raw_labels_np1 = labels_np[obj_mask]
                labels_np1 = (raw_labels_np1 > 0).astype(np.int32)
                logger.debug(f"[Engine] prompt_encoding: adding obj_id={label}  frame_idx={f_idx % self.config.batch_size}  pts_for_obj={len(points_np1)}")

                self.sam2_predictor.add_new_points_or_box(
                    inference_state=inference_state,
                    frame_idx=(f_idx % self.config.batch_size),
                    clear_old_points=False,
                    obj_id=int(label),
                    points=points_np1,
                    labels=labels_np1
                )
        logger.debug(f"[Engine] prompt_encoding: done  is_prompted={self.is_prompted}")
        return True

    def auto_prompt_encoding(self, inference_state):
        """Encode automatic prompts from previous masks."""
        logger.debug("[Engine] auto_prompt_encoding called")
        if not self.config.auto_prompt_encoding:
            logger.debug("[Engine] auto_prompt_encoding: disabled in config, skipping")
            return None
        points_list = []
        label_list = []
        box_prompt = self.mask_processor.mask_to_boxes(self.mask_processor.last_mask)
        if box_prompt is None:
            logger.debug("[Engine] auto_prompt_encoding: mask_to_boxes returned None, skipping")
            return None
        logger.debug(f"[Engine] auto_prompt_encoding: {len(box_prompt)} box(es) found")
        for k, v in box_prompt.items():
            points_list.append(v)
            label_list.append(k)
        points_np = [np.array(points, dtype=np.float32) for points in points_list]
        labels_np = label_list
        for i in range(len(points_np)):
            logger.debug(f"[Engine] auto_prompt_encoding: adding box obj_id={labels_np[i]}  box={points_np[i].tolist()}")
            self.is_prompted = True
            self.sam2_predictor.add_new_points_or_box(
                inference_state=inference_state,
                frame_idx=0,
                obj_id=int(labels_np[i]),
                box=points_np[i]
            )
        logger.debug(f"[Engine] auto_prompt_encoding: done  {len(points_np)} box(es) added")
        return points_np

    def _track_batch_cotracker(self, batch_number, query_frame_idx=None, backward_tracking=True):
        """Run CoTracker on a single batch from a specific query frame."""
        pose_cfg = self.config.pose_config
        if not pose_cfg or not pose_cfg.get('enabled'):
            return

        batch_prompts = self.annotation_manager.get_batch_prompts(batch_number, self.config.batch_size)

        batch_kps = None
        chosen_query_rel_idx = 0

        if query_frame_idx is not None:
            for p in batch_prompts:
                if p["frame_idx"] == query_frame_idx:
                    batch_kps = p["pose_keypoints"]
                    chosen_query_rel_idx = query_frame_idx % self.config.batch_size
                    break

        if not batch_kps:
            if batch_prompts:
                latest_p = batch_prompts[-1]
                batch_kps = latest_p["pose_keypoints"]
                chosen_query_rel_idx = latest_p["frame_idx"] % self.config.batch_size
            elif hasattr(self, 'per_batch_tracked_data'):
                for b in range(batch_number - 1, -1, -1):
                    if b < len(self.per_batch_tracked_data) and self.per_batch_tracked_data[b]:
                        last_entry = self.per_batch_tracked_data[b][-1]
                        kps_prev = last_entry.get("keypoints", [])
                        if kps_prev:
                            prev_idx = (b + 1) * self.config.batch_size - 1
                            curr_idx = batch_number * self.config.batch_size

                            try:
                                from ..models.Tracking.CoTrackerPredictor import track_between_frames
                                ct_cfg = pose_cfg.get('cotracker', {})
                                checkpoint_raw = ct_cfg.get('checkpoint') or 'external/co-tracker/checkpoints/scaled_offline.pth'
                                import os as _os
                                base_path = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), '..', '..'))
                                checkpoint = _os.path.normpath(_os.path.join(base_path, checkpoint_raw))
                                
                                # Fallback to external/segment_anything_2/checkpoints if not found in co-tracker
                                if not _os.path.exists(checkpoint):
                                    alt_path = _os.path.normpath(_os.path.join(base_path, 'external', 'segment_anything_2', 'checkpoints', 'scaled_offline.pth'))
                                    if _os.path.exists(alt_path):
                                        checkpoint = alt_path
                                window_len = ct_cfg.get('window_len', 60)

                                tracked_gap = track_between_frames(
                                    kps_prev, self.frame_paths[prev_idx], self.frame_paths[curr_idx],
                                    checkpoint, window_len
                                )
                                if tracked_gap:
                                    batch_kps = tracked_gap
                                    chosen_query_rel_idx = 0
                                    break
                            except Exception as e:
                                logger.warning(f"Failed to track inter-batch gap: {e}")
                                batch_kps = kps_prev
                                chosen_query_rel_idx = 0
                                break

        if not batch_kps:
            logger.info(f"[CoTracker] Batch {batch_number + 1}: No keypoints available, skipping")
            if batch_number < len(self.per_batch_tracked_data):
                self.per_batch_tracked_data[batch_number] = []
            return

        batch_start = batch_number * self.config.batch_size
        batch_end = min(batch_start + self.config.batch_size + 1, len(self.frame_paths))
        
        # If backward_tracking is False and we have a query_frame_idx, we only need frames from there onwards
        if not backward_tracking and chosen_query_rel_idx > 0:
             batch_start_sliced = batch_start + chosen_query_rel_idx
             batch_frame_paths = self.frame_paths[batch_start_sliced:batch_end]
             query_idx_to_pass = 0
             base_frame_idx = chosen_query_rel_idx
        else:
             batch_frame_paths = self.frame_paths[batch_start:batch_end]
             query_idx_to_pass = chosen_query_rel_idx
             base_frame_idx = 0

        logger.info(
            f"[CoTracker] Batch {batch_number + 1}: Tracking up to {len(batch_frame_paths)} frames "
            f"from {batch_start + chosen_query_rel_idx} ({len(batch_kps)} keypoints) "
            f"backward_tracking={backward_tracking}"
        )

        try:
            from ..models.Tracking.CoTrackerPredictor import CoTrackerPredictor
            ct_cfg = pose_cfg.get('cotracker', {})
            checkpoint_raw = ct_cfg.get('checkpoint') or 'external/co-tracker/checkpoints/scaled_offline.pth'
            import os as _os
            base_path = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), '..', '..'))
            checkpoint = _os.path.normpath(_os.path.join(base_path, checkpoint_raw))

            # Fallback to external/segment_anything_2/checkpoints if not found in co-tracker
            if not _os.path.exists(checkpoint):
                alt_path = _os.path.normpath(_os.path.join(base_path, 'external', 'segment_anything_2', 'checkpoints', 'scaled_offline.pth'))
                if _os.path.exists(alt_path):
                    checkpoint = alt_path
            window_len = ct_cfg.get('window_len', 60)

            classes = pose_cfg.get('classes', [])
            # Merge keypoint definitions from ALL classes into one list for tracking
            keypoints = []
            for cls in classes:
                for kp in cls.get('keypoints', []):
                    qualified = f"cls{cls.get('class_id', 1)}_{kp}"
                    if qualified not in keypoints:
                        keypoints.append(qualified)
            if not keypoints:
                keypoints = pose_cfg.get('keypoints', [])
            
            tracker = CoTrackerPredictor(
                keypoint_defs=keypoints,
                initial_coords=batch_kps,
                frame_paths=batch_frame_paths,
                checkpoint=checkpoint,
                window_len=window_len,
                query_frame_idx=query_idx_to_pass,
                backward_tracking=backward_tracking,
                base_frame_idx=base_frame_idx
            )
            tracked = tracker.get_all_tracked()
            if batch_number < len(self.per_batch_tracked_data):
                if not backward_tracking and base_frame_idx > 0:
                    # Merge: keep old data for frames before base_frame_idx
                    old_data = self.per_batch_tracked_data[batch_number]
                    if old_data and len(old_data) >= base_frame_idx:
                        # Ensure we don't have duplicates or gaps
                        # tracked has indices starting from base_frame_idx
                        self.per_batch_tracked_data[batch_number] = old_data[:base_frame_idx] + tracked
                    else:
                        self.per_batch_tracked_data[batch_number] = tracked
                else:
                    self.per_batch_tracked_data[batch_number] = tracked
            logger.info(f"[CoTracker] Batch {batch_number + 1}: Tracking complete")
        except Exception as e:
            logger.error(f"[CoTracker] Batch {batch_number + 1}: Failed: {e}")
            if batch_number < len(self.per_batch_tracked_data):
                self.per_batch_tracked_data[batch_number] = []

    # Keep the old name as an alias for backward compatibility with MainWindow.py
    def _track_batch_inline(self, batch_number, query_frame_idx=None, backward_tracking=True):
        """Alias for _track_batch_cotracker for backward compatibility."""
        return self._track_batch_cotracker(batch_number, query_frame_idx=query_frame_idx, backward_tracking=backward_tracking)

    def _mask_generation_consumer(self, total_batches):
        """Generates masks for batches as prompts become available."""
        if not self.config.sam_enabled:
            logger.info("[MaskGen] SAM is disabled, skipping mask generation consumer.")
            return

        for batch_num in range(total_batches):
            batch_index = batch_num * self.config.batch_size

            while len(self.annotation_manager.points_collection) <= batch_num:
                target_file = f"./workspace/inputs/UserPrompts/points_labels_{self.config.prefix}{self.config.video_number}.json"
                logger.info(f"[MaskGen] Waiting for prompts for batch {batch_num + 1}/{total_batches} in {target_file}... (retrying in 5s)")
                time.sleep(5)
                self.annotation_manager.load_points_and_labels()

            logger.info(
                f"[MaskGen] Starting mask generation for batch {batch_num + 1}/{total_batches}")

            self.frame_handler.move_and_copy_frames(batch_index, self.frame_paths, self.config.batch_size)

            self.mask_processor.generate_mask(
                batch_number=batch_num,
                sam2_predictor=self.sam2_predictor,
                temp_directory=self.config.temp_directory,
                prompt_encoding=self.prompt_encoding,
                auto_prompt_encoding=self.auto_prompt_encoding,
                predictor_lock=self._predictor_lock,
                starting_frame_idx=batch_index
            )
            logger.info(f"[MaskGen] ══ Batch {batch_num + 1}/{total_batches} mask generation completed ══")

        logger.info("[MaskGen] All batches processed. Background mask generation finished.")

    def run(self):
        """Run the AutoSegmentor video processing pipeline.

        Delegates flow control to the UI. The UI calls processing methods
        in a background thread when the user accepts annotations.
        """
        total_batches = (len(self.frame_paths) + self.config.batch_size - 1) // self.config.batch_size
        logger.info(f"[Engine] {len(self.frame_paths)} frames, {total_batches} batches (batch_size={self.config.batch_size})")

        self.per_batch_tracked_data = [[] for _ in range(total_batches)]

        if self.review_from_start:
            logger.info("[Annotation] Review Mode enabled: starting at Frame 0.")
            start_frame_idx = 0
            initial_batch = 0
        else:
            start_frame_idx = self.annotation_manager.check_data_sufficiency()
            initial_batch = start_frame_idx // self.config.batch_size

            # Lazy-tracking replaces auto-tracking inside engine at startup.
            # Tracking will only be triggered when UserInteraction loads a batch!
            if start_frame_idx >= len(self.frame_paths):
                logger.info(f"[Annotation] All {total_batches} batches already have prompts, starting UI at the end.")

        # In case the check pushes us past the end, bring it back
        start_frame_idx = min(start_frame_idx, len(self.frame_paths) - 1)
        
        result = self.user_interaction.start_ui_loop(self.frame_paths, start_frame_idx)

        clear_directory(self.config.temp_directory)
        return result
