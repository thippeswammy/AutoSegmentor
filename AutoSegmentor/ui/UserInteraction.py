import os
import traceback
from collections import defaultdict

import cv2
import numpy as np

from .logger_config import logger


class UserInteractionHandler:
    """Handles user interface, state, and interaction logic."""

    def __init__(self, config, annotation_manager, pipeline_engine):
        self.config = config
        self.annotation_manager = annotation_manager
        self.pipeline_engine = pipeline_engine
        self.pipeline_processor = pipeline_engine
        self.window_name = "AutoSegmentor Annotation Tool"
        
        self.current_class_label = 1
        self.current_instance_id = 1
        self.display_text = f"In class ID {self.current_class_label}, instance ID: {self.current_instance_id}"
        self.is_drawing = False
        
        self.selected_points = []
        self.selected_labels = []
        self.current_frame = None
        self.current_frame_only_text = None
        self.current_frame_only_with_points = None
        self.frame_paths = []
        self.current_frame_idx = 0
        self.inference_state_temp = None
        self.window = None

        # Pose Estimation State
        self.pose_mode = False
        self.pose_keypoints = []
        self.pose_click_coords = []
        self.current_keypoint_index = 0
        self.pose_class_id = 1
        self.pose_object_id = 1
        if self.config.pose_config and self.config.pose_config.get('enabled'):
            self.pose_mode = True
            classes = self.config.pose_config.get('classes', [{}])
            primary_class = classes[0] if classes else {}
            
            self.pose_keypoints = primary_class.get('keypoints', [])
            self.pose_class_id = primary_class.get('class_id', 1)
            self.pose_object_id = primary_class.get('object_id', 1)
            self.current_class_label = self.pose_class_id
            self.current_instance_id = self.pose_object_id
            self.display_text = f"Click: {self.pose_keypoints[0]}" if self.pose_keypoints else "Pose Mode: No keypoints defined"

    @staticmethod
    def encode_label(class_id, instance_id):
        return class_id * 1000 + instance_id

    def get_instance_keypoint_count(self, class_id=None, instance_id=None):
        """Count positive keypoints for a specific (class, instance) combo.
        
        If class_id/instance_id are None, uses the current active values.
        """
        if class_id is None:
            class_id = self.current_class_label
        if instance_id is None:
            instance_id = self.current_instance_id
        target_label = self.encode_label(class_id, instance_id)
        return sum(1 for pc in self.pose_click_coords
                   if pc.get('name') != 'Negative_Point'
                   and abs(pc.get('label', 0)) == target_label)

    def _recalc_keypoint_index(self):
        """Recalculate current_keypoint_index for the current class+instance."""
        self.current_keypoint_index = self.get_instance_keypoint_count()

    def change_class_label_pyqt(self, label):
        self.current_class_label = label
        self.current_instance_id = 1
        for i in self.selected_labels:
            if abs(i // 1000) == label:
                self.current_instance_id = max(abs(i) % 1000, self.current_instance_id)
                
        # Update pose_keypoints for the new class_id if available
        if self.pose_mode and self.config.pose_config:
            classes = self.config.pose_config.get('classes', [])
            for cls in classes:
                if cls.get('class_id') == label:
                    self.pose_keypoints = cls.get('keypoints', [])
                    self.pose_class_id = label
                    self.pose_object_id = cls.get('object_id', 1)
                    break
        
        self._recalc_keypoint_index()

    def user_prompt_adder_pyqt(self):
        """Trigger single-frame SAM2 mask preview.

        inference_state_temp is initialized HERE lazily on first call per frame,
        so that scrolling through frames (hold D) never triggers GPU init_state.
        """
        logger.debug(f"[UI] user_prompt_adder_pyqt called: frame={self.current_frame_idx}  points={len(self.selected_points)}")
        if self.pipeline_engine.sam2_predictor is None:
            logger.debug("[UI] user_prompt_adder_pyqt: sam2_predictor is None, skipping")
            return
        if not self.selected_points:  # No points yet — nothing to preview
            logger.debug("[UI] user_prompt_adder_pyqt: no selected_points, skipping")
            return

        # Lazy init: only call init_state when the user actually needs a preview
        if self.inference_state_temp is None:
            frame_path = getattr(self, '_current_frame_path', None)
            if frame_path is None:
                logger.debug("[UI] user_prompt_adder_pyqt: _current_frame_path is None, cannot init_state")
                return
            logger.debug(f"[UI] Lazy-init inference_state_temp for frame {self.current_frame_idx}: {frame_path}")
            try:
                self.inference_state_temp = self.pipeline_engine.sam2_predictor.init_state(
                    video_path=None, frame_paths=[os.path.abspath(frame_path)]
                )
                logger.debug(f"[UI] inference_state_temp lazy-init OK for frame {self.current_frame_idx}")
            except Exception:
                logger.error(f"[UI] lazy init_state failed:\n{traceback.format_exc()}")
                return
        else:
            logger.debug(f"[UI] Reusing existing inference_state_temp for frame {self.current_frame_idx}")

        try:
            logger.debug(f"[UI] Calling user_prompt_adder: frame={self.current_frame_idx}  pts={self.selected_points}  labels={self.selected_labels}")
            self.pipeline_engine.user_prompt_adder(
                self.inference_state_temp,
                self.frame_paths[self.current_frame_idx]
            )
            logger.debug(f"[UI] user_prompt_adder returned OK for frame {self.current_frame_idx}")
        except Exception:
            logger.error(f"[UI] user_prompt_adder_pyqt failed:\n{traceback.format_exc()}")
            self.inference_state_temp = None  # Reset so next click retries


    def start_ui_loop(self, frame_paths, start_frame_idx=None):
        self.frame_paths = frame_paths
        
        if start_frame_idx is None:
            start_frame_idx = self.annotation_manager.check_data_sufficiency()
            
        initial_batch = start_frame_idx // self.config.batch_size
        if initial_batch * self.config.batch_size >= len(frame_paths):
            initial_batch = max(0, (len(frame_paths) - 1) // self.config.batch_size)
            
        from .MainWindow import AnnotationWindow
        self.window = AnnotationWindow(self, self.config)
        
        # Load the actual start_frame_idx instead of strictly the start of the batch
        # so if the user resumes at frame 45, they see frame 45 instead of 30.
        target_frame = min(start_frame_idx, len(frame_paths) - 1)
        self.load_frame_for_ui(target_frame)
        logger.info(f"Opening UI window at frame {target_frame}.")
        try:
            result = self.window.exec_()
            return result == 1  # 1 = QDialog.Accepted
        finally:
            # Final cleanup: ensure the window object is cleared
            # and any lingering state is reset if needed.
            if self.window:
                self.window.deleteLater()
                self.window = None
            logger.info("UI window closed. Manual phase ended.")
        
    def save_current_annotation(self):
        """Save current annotation state to disk."""
        logger.debug(
            f"[UI] save_current_annotation: frame={self.current_frame_idx}  "
            f"points={len(self.selected_points)}  labels={self.selected_labels}  "
            f"pose_kps={len(self.pose_click_coords) if self.pose_mode else 'N/A'}"
        )
        # Save at current frame index, supporting multiple corrections per batch
        self.annotation_manager.save_points_and_labels(
            frame_idx=self.current_frame_idx,
            points=self.selected_points,
            labels=self.selected_labels,
            pose_keypoints=self.pose_click_coords if self.pose_mode else None
        )

    def load_frame_for_ui(self, frame_idx):
        if frame_idx >= len(self.frame_paths) or frame_idx < 0:
            logger.debug(f"[UI] load_frame_for_ui({frame_idx}): out of range (total={len(self.frame_paths)}), skipping")
            return

        logger.debug(f"[UI] load_frame_for_ui: frame={frame_idx}  prev_frame={self.current_frame_idx}")

        # Clear the old inference state before loading a new frame
        self.inference_state_temp = None

        self.current_frame_idx = frame_idx
        frame_path = self.frame_paths[frame_idx]
        batch = frame_idx // self.config.batch_size
        logger.debug(f"[UI] load_frame_for_ui: batch={batch}  path={frame_path}")

        self.current_frame_only_with_points = cv2.imread(frame_path)
        if self.current_frame_only_with_points is None:
            logger.error(f"[UI] load_frame_for_ui: cv2.imread returned None for {frame_path}")
            return
        # Keep a clean raw copy (no disk mask, no overlays).
        # Used as the base for SAM preview (avoids double-mask) and by Reset.
        self._raw_frame = self.current_frame_only_with_points.copy()

        # Mix mask if exists
        mask_filename = f"{self.config.prefix}{self.config.video_number}_{frame_idx:05d}.png"
        mask_path = os.path.join(self.config.rendered_frames_dir, mask_filename)
        if os.path.exists(mask_path):
            logger.debug(f"[UI] load_frame_for_ui: disk mask found -> {mask_path}")
            mask = cv2.imread(mask_path)
            non_zero_mask = np.any(mask > 0, axis=-1)
            non_zero_mask_3d = np.stack([non_zero_mask] * 3, axis=-1)
            blended = cv2.addWeighted(self.current_frame_only_with_points, 0.5, mask, 0.5, 0)
            np.copyto(self.current_frame_only_with_points, blended, where=non_zero_mask_3d)
        else:
            logger.debug(f"[UI] load_frame_for_ui: no disk mask for frame {frame_idx}")

        self.current_frame = self.current_frame_only_with_points.copy()

        self.selected_points = []
        self.selected_labels = []
        self.pose_click_coords = []
        self.current_keypoint_index = 0

        # ── Store current frame path — inference_state_temp is initialized LAZILY ─
        # init_state is a GPU call. We must NOT call it on every frame navigation
        # (e.g. when the user holds D to scroll through frames).
        # It will be initialized on-demand in user_prompt_adder_pyqt on first click.
        self._current_frame_path = frame_path
        self.inference_state_temp = None  # Will be created lazily on first click

        # ── PRIORITIZED LOADING LOGIC ─────────────────────────────────────────
        # 1. Try to load manual prompt for this specific frame (highest priority)
        manual_prompt = self.annotation_manager.get_prompt_for_frame(frame_idx)

        if manual_prompt:
            self.selected_points = [list(p) for p in manual_prompt["points"]]
            self.selected_labels = [int(l) for l in manual_prompt["labels"]]
            logger.debug(
                f"[UI] load_frame_for_ui: manual prompt loaded for frame {frame_idx}  "
                f"pts={len(self.selected_points)}  labels={self.selected_labels}"
            )
            if self.pose_mode:
                self.pose_click_coords = manual_prompt["pose_keypoints"]
                self.current_keypoint_index = len(self.pose_click_coords)
                logger.debug(f"[UI] load_frame_for_ui: pose_click_coords loaded: {len(self.pose_click_coords)} kp(s)")
            # SAM preview triggered async via debounced PreviewThread

        else:
            logger.debug(f"[UI] load_frame_for_ui: no manual prompt for frame {frame_idx}, checking tracked data")
            # 2. Try to load existing tracking results (Current Batch or Prev Batch Preview)
            tracked_entry = None
            if self.pose_mode and hasattr(self.pipeline_processor, 'per_batch_tracked_data'):
                # Check current batch
                if batch < len(self.pipeline_processor.per_batch_tracked_data):
                    tracked_batch = self.pipeline_processor.per_batch_tracked_data[batch]
                    local_idx = frame_idx % self.config.batch_size
                    logger.debug(f"[UI] Checking tracked_batch[{batch}] local_idx={local_idx}  len={len(tracked_batch)}")
                    if tracked_batch and local_idx < len(tracked_batch):
                        tracked_entry = tracked_batch[local_idx]
                        logger.debug(f"[UI] Tracked entry found in current batch {batch}")

                # Check previous batch's overflow (Plus-One Preview)
                if not tracked_entry and batch > 0 and batch - 1 < len(self.pipeline_processor.per_batch_tracked_data):
                    prev_batch_tracked = self.pipeline_processor.per_batch_tracked_data[batch-1]
                    if prev_batch_tracked and len(prev_batch_tracked) > self.config.batch_size:
                        tracked_entry = prev_batch_tracked[self.config.batch_size]
                        logger.debug(f"[UI] Tracked entry from prev batch {batch-1} overflow")

            if tracked_entry:
                kps = tracked_entry["keypoints"]
                logger.debug(f"[UI] Loading {len(kps)} tracked keypoints for frame {frame_idx}")
                for kp in kps:
                    if "label" in kp:
                        full_label = kp["label"]
                    else:
                        full_label = self.encode_label(self.pose_class_id, self.pose_object_id)
                    
                    if kp.get("visible", 2) > 0:
                        self.selected_points.append([kp["x"], kp["y"]])
                        self.selected_labels.append(full_label)
                    self.pose_click_coords.append(kp)
                self.current_keypoint_index = len(self.pose_click_coords)
                logger.debug(f"[UI] Tracked data loaded: selected_points={len(self.selected_points)}")
                # SAM preview triggered async via debounced PreviewThread

            elif frame_idx % self.config.batch_size == 0:
                # 3. If no data exists yet, we NO LONGER do auto carry-forward during navigation.
                # This prevents UI freezes when scrolling. User can process manually.
                logger.debug(f"[UI] load_frame_for_ui: frame {frame_idx} is batch boundary with no data — skipping carry-forward")


        # NOTE: inference_state_temp is intentionally kept alive here.
        # It will be used by user_prompt_adder_pyqt() on every click until
        # the user navigates to a different frame (cleared at the top of this function).

        total_batches = (len(self.frame_paths) + self.config.batch_size - 1) // self.config.batch_size
        if self.window:
            self.window.undo_stack.clear()
            self.window.set_batch_info(batch, total_batches, frame_idx, len(self.frame_paths))
            self.window.refresh_display()
            self.window._update_sidebar()
        
    def has_data_for_frame(self, frame_idx):
        """Check if a frame has manual prompts, tracking results, or a rendered mask."""
        # 1. Manual prompt check
        if self.annotation_manager.get_prompt_for_frame(frame_idx):
            logger.debug(f"[UI] has_data_for_frame({frame_idx}): manual prompt found")
            return True

        # 2. Rendered mask file check (SAM has already processed this frame)
        mask_filename = f"{self.config.prefix}{self.config.video_number}_{frame_idx:05d}.png"
        mask_path = os.path.join(self.config.rendered_frames_dir, mask_filename)
        if os.path.exists(mask_path):
            logger.debug(f"[UI] has_data_for_frame({frame_idx}): disk mask found at {mask_path}")
            return True

        # 3. Tracking data check (CoTracker results)
        batch = frame_idx // self.config.batch_size
        if hasattr(self.pipeline_processor, 'per_batch_tracked_data'):
            # Current batch data
            if batch < len(self.pipeline_processor.per_batch_tracked_data):
                tracked_batch = self.pipeline_processor.per_batch_tracked_data[batch]
                local_idx = frame_idx % self.config.batch_size
                if tracked_batch and local_idx < len(tracked_batch):
                    logger.debug(f"[UI] has_data_for_frame({frame_idx}): tracked data found in batch {batch} local_idx={local_idx}")
                    return True

            # Previous batch overflow (Plus-One Preview)
            if batch > 0 and batch - 1 < len(self.pipeline_processor.per_batch_tracked_data):
                prev_batch_tracked = self.pipeline_processor.per_batch_tracked_data[batch-1]
                if prev_batch_tracked and len(prev_batch_tracked) > self.config.batch_size:
                    if frame_idx == batch * self.config.batch_size:
                        logger.debug(f"[UI] has_data_for_frame({frame_idx}): prev batch overflow found")
                        return True
        logger.debug(f"[UI] has_data_for_frame({frame_idx}): no data found")
        return False
    def prepare_batch_for_annotation(self, batch):
        """Prepare the first frame of a batch for annotation.

        Carry-forward CoTracker tracking is ONLY run when:
        1. The batch has no rendered mask files (SAM hasn't processed it yet), AND
        2. The batch has no tracked keypoint data yet

        Skipping this when data already exists prevents CoTracker from firing
        on every navigation keystroke through already-processed batches.
        """
        frame_idx = batch * self.config.batch_size
        frame_path = self.frame_paths[frame_idx]

        # ── Guard: skip if this batch already has data ────────────────────────
        # Check rendered mask for the first frame of this batch
        mask_filename = f"{self.config.prefix}{self.config.video_number}_{frame_idx:05d}.png"
        mask_path = os.path.join(self.config.rendered_frames_dir, mask_filename)
        batch_already_processed = os.path.exists(mask_path)

        # Check if tracked data already exists for this batch
        batch_has_tracked = (
            hasattr(self.pipeline_processor, 'per_batch_tracked_data')
            and batch < len(self.pipeline_processor.per_batch_tracked_data)
            and bool(self.pipeline_processor.per_batch_tracked_data[batch])
        )

        if batch_already_processed or batch_has_tracked:
            logger.debug(f"[UI] Batch {batch} already processed — skipping carry-forward")
            return

        # ── Not yet processed: attempt carry-forward tracking ─────────────────
        if self.pose_mode:
            self.current_keypoint_index = 0
            self.current_class_label = self.pose_class_id
            self.current_instance_id = self.pose_object_id

            prev_kps = None
            prev_frame_idx = None

            # 1. Try to get from tracked data of PREVIOUS batch
            if batch > 0:
                if hasattr(self.pipeline_processor, 'per_batch_tracked_data'):
                    if batch - 1 < len(self.pipeline_processor.per_batch_tracked_data):
                        prev_batch_tracked = self.pipeline_processor.per_batch_tracked_data[batch-1]
                        if prev_batch_tracked:
                            last_entry = prev_batch_tracked[-1]
                            prev_kps = last_entry.get("keypoints", [])
                            prev_frame_idx = batch * self.config.batch_size - 1

            # 2. If no tracked data, try to find the LATEST manual prompt before this frame
            if not prev_kps:
                latest_manual = self.annotation_manager.get_latest_prompt_before(frame_idx)
                if latest_manual:
                    prev_kps = latest_manual["pose_keypoints"]
                    prev_frame_idx = latest_manual["frame_idx"]

            if prev_kps and len(prev_kps) == len(self.pose_keypoints):
                tracked_kps = prev_kps
                tracker_type = self.config.pose_config.get('tracker', 'lk').lower() if self.config.pose_config else 'lk'
                if tracker_type == 'cotracker' and batch > 0 and prev_frame_idx is not None:
                    try:
                        from ..models.Tracking.CoTrackerPredictor import track_between_frames
                        if 0 <= prev_frame_idx < len(self.frame_paths):
                            prev_frame_path = self.frame_paths[prev_frame_idx]
                            ct_cfg = self.config.pose_config.get('cotracker', {})
                            checkpoint = ct_cfg.get('checkpoint', 'external/co-tracker/checkpoints/scaled_offline.pth')
                            import os as _os
                            base_path = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), '..', '..'))
                            checkpoint = _os.path.normpath(_os.path.join(base_path, checkpoint))
                            window_len = ct_cfg.get('window_len', 60)
                            result = track_between_frames(prev_kps, prev_frame_path, frame_path, checkpoint, window_len)
                            if result: tracked_kps = result
                    except Exception as e:
                        logger.warning(f"CoTracker carry-forward failed: {e}")

                for kp in sorted(tracked_kps, key=lambda k: k["point_id"]):
                    x, y = kp["x"], kp["y"]
                    
                    if "label" in kp:
                        full_label = kp["label"]
                    else:
                        full_label = self.encode_label(self.pose_class_id, self.pose_object_id)
                        
                    if kp.get("visible", 2) > 0:
                        self.selected_points.append([x, y])
                        self.selected_labels.append(full_label)
                    self.pose_click_coords.append({
                        "name": kp["name"], "point_id": kp["point_id"],
                        "x": x, "y": y, "visible": kp.get("visible", True),
                        "label": full_label
                    })
                self.current_keypoint_index = len(self.pose_click_coords)
                self.user_prompt_adder_pyqt()

