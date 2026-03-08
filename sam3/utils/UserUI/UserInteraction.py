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

    def change_class_label_pyqt(self, label):
        """Change the current class label and update instance ID."""
        if self.pose_mode:
            logger.warning("Class selection disabled in Pose Mode")
            return
        self.current_class_label = label
        self.current_instance_id = 1
        for i in self.selected_labels:
            if abs(i // 1000) == label:
                self.current_instance_id = max(abs(i) % 1000, self.current_instance_id)

    def user_prompt_adder_pyqt(self):
        """Called by MainWindow to update drawing without showing cv2 window."""
        self.sam2_video_predictor.user_prompt_adder(self.inference_state_temp, self.frame_path)


    def collect_user_points(self, batch, frame_paths, sam2_predictor, click_event_callback, mask_processor):
        """Collect user points for annotation in a video frame.
        
        This function manages the user interaction for annotating keypoints in a video
        frame. It initializes the display, checks for existing annotations, and allows
        the user to select points either by clicking or through keyboard inputs. The
        function also handles the carry-forward of keypoints from previous batches,
        integrates with a tracking system if available, and updates the annotation
        manager with the selected points and labels.
        
        Args:
            self: The instance of the class.
            batch (int): The current batch index for processing frames.
            frame_paths (list): A list of paths to the video frames.
            sam2_predictor: An object responsible for making predictions on the frames.
            click_event_callback: A callback function for handling mouse click events.
            mask_processor: An object for processing masks related to the annotations.
        """
        start_batch_idx = self.annotation_manager.check_data_sufficiency()
        self.frame_path = frame_paths[batch * self.config.batch_size]
        batch_idx = (start_batch_idx // self.config.batch_size)
        frame_idx = batch_idx * self.config.batch_size
        self.inference_state_temp = None
        if sam2_predictor:
            self.inference_state_temp = sam2_predictor.init_state(
                video_path=None,
                frame_paths=[os.path.abspath(self.frame_path)]
            )
        self.current_frame = self.current_frame_only_text = self.current_frame_only_with_points = cv2.imread(
            self.frame_path)
            
        if self.pose_mode:
            self.current_keypoint_index = 0
            self.selected_points = []
            self.selected_labels = []
            self.pose_click_coords = []
            self.current_class_label = self.pose_class_id
            self.current_instance_id = self.pose_object_id

            prev_kps = None
            if batch < len(self.annotation_manager.pose_keypoints_collection):
                stored = self.annotation_manager.pose_keypoints_collection[batch]
                if stored:
                    prev_kps = stored
            if prev_kps is None and batch > 0:
                if hasattr(self.sam2_video_predictor, 'per_batch_tracked_data'):
                    for b in range(batch - 1, -1, -1):
                        if b < len(self.sam2_video_predictor.per_batch_tracked_data):
                            prev_tracked = self.sam2_video_predictor.per_batch_tracked_data[b]
                            if prev_tracked:
                                last_entry = prev_tracked[-1]
                                prev_kps = last_entry.get("keypoints", [])
                                if prev_kps:
                                    logger.info(f"[Annotation] Batch {batch + 1}: Carry-forward using inline-tracked last-frame positions from batch {b + 1}")
                                    break
                if prev_kps is None:
                    for b in range(batch - 1, -1, -1):
                        if b < len(self.annotation_manager.pose_keypoints_collection):
                            stored = self.annotation_manager.pose_keypoints_collection[b]
                            if stored:
                                prev_kps = stored
                                break

            if prev_kps and len(prev_kps) == len(self.pose_keypoints):
                tracked_kps = prev_kps
                tracker_type = (
                    self.config.pose_config.get('tracker', 'lk').lower()
                    if self.config.pose_config else 'lk'
                )
                if tracker_type == 'cotracker' and batch > 0:
                    try:
                        from ..FileManagement.CoTrackerKeypointTracker import track_between_frames
                        prev_batch_end_idx = batch * self.config.batch_size - 1
                        if 0 <= prev_batch_end_idx < len(frame_paths):
                            prev_frame_path = frame_paths[prev_batch_end_idx]
                            curr_frame_path = self.frame_path 
                            ct_cfg = self.config.pose_config.get('cotracker', {})
                            checkpoint = ct_cfg.get('checkpoint', '../co-tracker/checkpoints/scaled_offline.pth')
                            import os as _os
                            base_path = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), '..', '..'))
                            checkpoint = _os.path.normpath(_os.path.join(base_path, checkpoint))
                            window_len = ct_cfg.get('window_len', 60)
                            result = track_between_frames(
                                prev_kps, prev_frame_path, curr_frame_path,
                                checkpoint=checkpoint, window_len=window_len
                            )
                            if result:
                                tracked_kps = result
                                logger.info(f"[Annotation] Batch {batch + 1}: CoTracker carry-forward tracking applied")
                    except Exception as e:
                        logger.warning(f"CoTracker carry-forward failed, using static copy: {e}")

                full_label = self.encode_label(self.pose_class_id, self.pose_object_id)
                for kp in sorted(tracked_kps, key=lambda k: k["point_id"]):
                    x, y = kp["x"], kp["y"]
                    self.selected_points.append([x, y])
                    self.selected_labels.append(full_label)
                    self.pose_click_coords.append({
                        "name": kp["name"],
                        "point_id": kp["point_id"],
                        "x": x,
                        "y": y,
                        "visible": kp.get("visible", True)
                    })
                self.current_keypoint_index = len(self.pose_keypoints)
                logger.info(f"[Annotation] Batch {batch + 1}: Reviewing carry-forwarded points. Press Accept to confirm.")
        else:
            self.current_class_label = self.current_instance_id = 1

        # --- Launch PyQt MainWindow ---
        from .MainWindow import AnnotationWindow
        window = AnnotationWindow(self, self.config)
        total_batches = (len(frame_paths) + self.config.batch_size - 1) // self.config.batch_size
        window.set_batch_info(batch, total_batches, frame_idx, len(frame_paths))
        
        # Populate initial mask if pre-populated keypoints exist
        if self.selected_points:
            self.user_prompt_adder_pyqt()
            window.refresh_display()
            window._update_sidebar()

        # Block the pipeline until user hits Accept (which closes the dialog)
        window.exec_()
        
        # Save points back
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
