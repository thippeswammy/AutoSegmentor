import os
from collections import defaultdict

import cv2
import numpy as np

from .logger_config import logger


class UserInteractionHandler:
    """Handles user interface, state, and interaction logic."""

    def __init__(self, config, annotation_manager, sam2_video_predictor):
        self.config = config
        self.annotation_manager = annotation_manager
        self.sam2_video_predictor = sam2_video_predictor
        self.pipeline_processor = sam2_video_predictor
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
            self.pose_keypoints = self.config.pose_config.get('keypoints', [])
            self.pose_class_id = self.config.pose_config.get('class_id', 1)
            self.pose_object_id = self.config.pose_config.get('object_id', 1)
            self.current_class_label = self.pose_class_id
            self.current_instance_id = self.pose_object_id
            self.display_text = f"Click: {self.pose_keypoints[0]}" if self.pose_keypoints else "Pose Mode: No keypoints defined"

    @staticmethod
    def encode_label(class_id, instance_id):
        return class_id * 1000 + instance_id

    def change_class_label_pyqt(self, label):
        if self.pose_mode:
            logger.warning("Class selection disabled in Pose Mode")
            return
        self.current_class_label = label
        self.current_instance_id = 1
        for i in self.selected_labels:
            if abs(i // 1000) == label:
                self.current_instance_id = max(abs(i) % 1000, self.current_instance_id)

    def user_prompt_adder_pyqt(self):
        # We need the predictor lock if it's running
        with self.pipeline_processor._predictor_lock:
            if self.inference_state_temp is not None:
                self.sam2_video_predictor.user_prompt_adder(self.inference_state_temp, self.frame_paths[self.current_frame_idx])

    def start_ui_loop(self, frame_paths):
        self.frame_paths = frame_paths
        
        start_batch_idx = self.annotation_manager.check_data_sufficiency()
        initial_batch = start_batch_idx // self.config.batch_size
        if initial_batch * self.config.batch_size >= len(frame_paths):
            initial_batch = max(0, (len(frame_paths) - 1) // self.config.batch_size)
            
        from .MainWindow import AnnotationWindow
        self.window = AnnotationWindow(self, self.config)
        
        self.load_frame_for_ui(initial_batch * self.config.batch_size)
        logger.info("Opening UI window. Proceed with annotations.")
        self.window.exec_()
        
    def save_current_annotation(self):
        batch = self.current_frame_idx // self.config.batch_size
        frame_idx = batch * self.config.batch_size
        
        # Replace or append
        while len(self.annotation_manager.points_collection) <= batch:
            self.annotation_manager.points_collection.append([])
            self.annotation_manager.labels_collection.append([])
            self.annotation_manager.frame_indices.append(0)
            self.annotation_manager.pose_keypoints_collection.append([])
            
        self.annotation_manager.points_collection[batch] = self.selected_points[:]
        self.annotation_manager.labels_collection[batch] = self.selected_labels[:]
        self.annotation_manager.frame_indices[batch] = frame_idx
        
        if self.pose_mode and self.pose_click_coords:
            self.annotation_manager.pose_keypoints_collection[batch] = self.pose_click_coords[:]
            
        self.annotation_manager.save_points_and_labels()

    def load_frame_for_ui(self, frame_idx):
        if frame_idx >= len(self.frame_paths) or frame_idx < 0:
            return
            
        self.current_frame_idx = frame_idx
        frame_path = self.frame_paths[frame_idx]
        batch = frame_idx // self.config.batch_size
        
        self.current_frame_only_with_points = cv2.imread(frame_path)
        
        # Mix mask if exists
        mask_filename = f"{self.config.prefix}{self.config.video_number}_{frame_idx:05d}.png"
        mask_path = os.path.join(self.config.rendered_frames_dir, mask_filename)
        if os.path.exists(mask_path):
            mask = cv2.imread(mask_path)
            non_zero_mask = np.any(mask > 0, axis=-1)
            non_zero_mask_3d = np.stack([non_zero_mask] * 3, axis=-1)
            blended = cv2.addWeighted(self.current_frame_only_with_points, 0.5, mask, 0.5, 0)
            np.copyto(self.current_frame_only_with_points, blended, where=non_zero_mask_3d)
            
        self.current_frame = self.current_frame_only_with_points.copy()
        
        self.selected_points = []
        self.selected_labels = []
        self.pose_click_coords = []
        self.current_keypoint_index = 0
        
        # If it's a key frame (batch start)
        if frame_idx % self.config.batch_size == 0:
            if batch < len(self.annotation_manager.points_collection) and len(self.annotation_manager.points_collection[batch]) > 0:
                self.selected_points = [list(p) for p in self.annotation_manager.points_collection[batch]]
                self.selected_labels = [int(l) for l in self.annotation_manager.labels_collection[batch]]
                if self.pose_mode:
                    if batch < len(self.annotation_manager.pose_keypoints_collection):
                        self.pose_click_coords = self.annotation_manager.pose_keypoints_collection[batch]
                        self.current_keypoint_index = len(self.pose_keypoints)
                
                # init state
                with self.pipeline_processor._predictor_lock:
                    if self.sam2_video_predictor.sam2_predictor is not None:
                        self.inference_state_temp = self.sam2_video_predictor.sam2_predictor.init_state(video_path=None, frame_paths=[os.path.abspath(frame_path)])
                    else:
                        self.inference_state_temp = None
                self.user_prompt_adder_pyqt()
            else:
                self.prepare_batch_for_annotation(batch)
        else:
            # Maybe load inline tracking result to display
            if self.pose_mode and hasattr(self.pipeline_processor, 'per_batch_tracked_data'):
                if batch < len(self.pipeline_processor.per_batch_tracked_data):
                    tracked_batch = self.pipeline_processor.per_batch_tracked_data[batch]
                    local_idx = frame_idx % self.config.batch_size
                    if tracked_batch and local_idx < len(tracked_batch):
                        entry = tracked_batch[local_idx]
                        kps = entry["keypoints"]
                        full_label = self.encode_label(self.pose_class_id, self.pose_object_id)
                        for kp in kps:
                            if kp.get("visible", 2) > 0:
                                self.selected_points.append([kp["x"], kp["y"]])
                                self.selected_labels.append(full_label)
                            self.pose_click_coords.append(kp)
                        self.current_keypoint_index = len(self.pose_keypoints)
            self.inference_state_temp = None

        total_batches = (len(self.frame_paths) + self.config.batch_size - 1) // self.config.batch_size
        if self.window:
            self.window.set_batch_info(batch, total_batches, frame_idx, len(self.frame_paths))
            self.window.refresh_display()
            self.window._update_sidebar()
        
    def prepare_batch_for_annotation(self, batch):
        frame_idx = batch * self.config.batch_size
        frame_path = self.frame_paths[frame_idx]
        
        with self.pipeline_processor._predictor_lock:
            if self.sam2_video_predictor.sam2_predictor is not None:
                self.inference_state_temp = self.sam2_video_predictor.sam2_predictor.init_state(video_path=None, frame_paths=[os.path.abspath(frame_path)])
            else:
                self.inference_state_temp = None
            
        if self.pose_mode:
            self.current_keypoint_index = 0
            self.current_class_label = self.pose_class_id
            self.current_instance_id = self.pose_object_id
            
            prev_kps = None
            prev_frame_idx = None
            if batch > 0:
                # carry forward logic
                if hasattr(self.pipeline_processor, 'per_batch_tracked_data'):
                    for b in range(batch - 1, -1, -1):
                        if b < len(self.pipeline_processor.per_batch_tracked_data):
                            prev_tracked = self.pipeline_processor.per_batch_tracked_data[b]
                            if prev_tracked:
                                last_entry = prev_tracked[-1]
                                prev_kps = last_entry.get("keypoints", [])
                                if prev_kps: 
                                    prev_frame_idx = (b + 1) * self.config.batch_size - 1
                                    break
                if prev_kps is None:
                    for b in range(batch - 1, -1, -1):
                        if b < len(self.annotation_manager.pose_keypoints_collection):
                            stored = self.annotation_manager.pose_keypoints_collection[b]
                            if len(stored) > 0:
                                prev_kps = stored
                                prev_frame_idx = b * self.config.batch_size
                                break
                                
            if prev_kps and len(prev_kps) == len(self.pose_keypoints):
                tracked_kps = prev_kps
                tracker_type = self.config.pose_config.get('tracker', 'lk').lower() if self.config.pose_config else 'lk'
                if tracker_type == 'cotracker' and batch > 0 and prev_frame_idx is not None:
                    try:
                        from ..FileManagement.CoTrackerKeypointTracker import track_between_frames
                        if 0 <= prev_frame_idx < len(self.frame_paths):
                            prev_frame_path = self.frame_paths[prev_frame_idx]
                            ct_cfg = self.config.pose_config.get('cotracker', {})
                            checkpoint = ct_cfg.get('checkpoint', '../co-tracker/checkpoints/scaled_offline.pth')
                            import os as _os
                            base_path = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), '..', '..'))
                            checkpoint = _os.path.normpath(_os.path.join(base_path, checkpoint))
                            window_len = ct_cfg.get('window_len', 60)
                            result = track_between_frames(prev_kps, prev_frame_path, frame_path, checkpoint, window_len)
                            if result: tracked_kps = result
                    except Exception as e:
                        logger.warning(f"CoTracker carry-forward failed: {e}")
                        
                full_label = self.encode_label(self.pose_class_id, self.pose_object_id)
                for kp in sorted(tracked_kps, key=lambda k: k["point_id"]):
                    x, y = kp["x"], kp["y"]
                    if kp.get("visible", 2) > 0:
                        self.selected_points.append([x, y])
                        self.selected_labels.append(full_label)
                    self.pose_click_coords.append({
                        "name": kp["name"], "point_id": kp["point_id"],
                        "x": x, "y": y, "visible": kp.get("visible", True)
                    })
                self.current_keypoint_index = len(self.pose_keypoints)
                self.user_prompt_adder_pyqt()
