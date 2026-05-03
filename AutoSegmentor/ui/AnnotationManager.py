import json
import os
from os.path import exists

import numpy as np

from ..file_management.FileManager import ensure_directory
from ..ui.logger_config import logger


class AnnotationManager:
    """Manages annotation data (points, labels, frame indices)."""

    def __init__(self, config, frame_paths, save_dir=None):
        self.config = config
        self.frame_paths = frame_paths
        self.save_dir = save_dir
        self.points_collection = []
        self.labels_collection = []
        self.targets_collection = []  # List of lists of target model strings per frame
        self.frame_indices = []
        self.pose_keypoints_collection = []  # Per-batch pose keypoint coords
        self._all_batches_logged = False  # Dedup flag for check_data_sufficiency
        self.load_points_and_labels()

    def load_points_and_labels(self):
        """Load points and labels from JSON file."""
        if self.save_dir:
             save_dir = self.save_dir
        else:
             base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
             save_dir = os.path.join(base_path, "workspace", "inputs", "UserPrompts")
        
        ensure_directory(save_dir)
        filename = os.path.join(save_dir, f"points_labels_{self.config.prefix}{self.config.video_number}.json")
        logger.debug(f"[AnnotMgr] Loading annotations from: {filename}")

        if not exists(filename):
            logger.warning(f"Points and labels file {filename} not found")
            return

        try:
            with open(filename, 'r', encoding="utf-8") as f:
                data = json.load(f)

            # Convert lists back to numpy arrays (if needed)
            self.points_collection = [np.array(entry["points"], dtype=np.float32) for entry in data]
            self.labels_collection = [np.array(entry["labels"], dtype=np.int32) for entry in data]
            # Handle legacy data that might not have target_models
            self.targets_collection = [entry.get("target_models", [["sam", "pose"]] * len(entry["points"])) for entry in data]
            self.frame_indices = [int(entry["frame_idx"]) for entry in data]
            self.pose_keypoints_collection = [entry.get("pose_keypoints", []) for entry in data]

            logger.debug(f"[AnnotMgr] Loaded {len(self.points_collection)} annotations from {filename}")
            logger.debug(f"[AnnotMgr] Frame indices present: {self.frame_indices}")
        except Exception as e:
            logger.error(f"Error loading points and labels from {filename}: {e}")

    def get_batch_prompts(self, batch_number, batch_size):
        """Get all prompts that fall within the specified batch range.

        Returns:
            List of dicts: [{"frame_idx": int, "points": np.array, "labels": np.array, "pose_keypoints": [...]}, ...]
        """
        start_frame = batch_number * batch_size
        end_frame = start_frame + batch_size
        logger.debug(f"[AnnotMgr] get_batch_prompts: batch={batch_number}  frames=[{start_frame}, {end_frame})")
        results = []

        for i in range(len(self.frame_indices)):
            f_idx = self.frame_indices[i]
            # Strict safety: only return prompts for frames that exist in the current video
            if start_frame <= f_idx < end_frame and f_idx < len(self.frame_paths):
                results.append({
                    "frame_idx": f_idx,
                    "points": self.points_collection[i],
                    "labels": self.labels_collection[i],
                    "target_models": self.targets_collection[i],
                    "pose_keypoints": self.pose_keypoints_collection[i] if i < len(self.pose_keypoints_collection) else []
                })
        logger.debug(f"[AnnotMgr] get_batch_prompts: found {len(results)} prompt(s) for batch {batch_number}")
        return results

    def get_points_for_model(self, frame_idx, model_id):
        """Get filtered points and labels for a specific model (e.g. 'sam' or 'pose')."""
        prompt = self.get_prompt_for_frame(frame_idx)
        if not prompt:
            return None

        points = prompt["points"]
        labels = prompt["labels"]
        targets = prompt["target_models"]

        filtered_pts = []
        filtered_lbs = []

        for pt, lb, tg in zip(points, labels, targets):
            if model_id in tg:
                filtered_pts.append(pt)
                filtered_lbs.append(lb)

        return {
            "points": np.array(filtered_pts, dtype=np.float32) if filtered_pts else np.empty((0, 2), dtype=np.float32),
            "labels": np.array(filtered_lbs, dtype=np.int32) if filtered_lbs else np.empty((0,), dtype=np.int32)
        }

    def get_prompt_for_frame(self, frame_idx):
        """Get the prompt for a specific frame, if any."""
        for i, f_idx in enumerate(self.frame_indices):
            if f_idx == frame_idx:
                pts = self.points_collection[i]
                lbs = self.labels_collection[i]
                logger.debug(f"[AnnotMgr] get_prompt_for_frame({frame_idx}): found {len(pts)} points")
                return {
                    "points": pts,
                    "labels": lbs,
                    "target_models": self.targets_collection[i],
                    "pose_keypoints": self.pose_keypoints_collection[i] if i < len(self.pose_keypoints_collection) else []
                }
        logger.debug(f"[AnnotMgr] get_prompt_for_frame({frame_idx}): no prompt found")
        return None

    def get_latest_prompt_before(self, frame_idx):
        """Find the most recent prompt that occurs before the given frame index."""
        logger.debug(f"[AnnotMgr] get_latest_prompt_before({frame_idx})")
        latest_idx = -1
        latest_data = None
        for i, f_idx in enumerate(self.frame_indices):
            if f_idx < frame_idx:
                if f_idx > latest_idx:
                    latest_idx = f_idx
                    latest_data = {
                        "frame_idx": f_idx,
                        "points": self.points_collection[i],
                        "labels": self.labels_collection[i],
                        "target_models": self.targets_collection[i],
                        "pose_keypoints": self.pose_keypoints_collection[i] if i < len(self.pose_keypoints_collection) else []
                    }
        if latest_data:
            logger.debug(f"[AnnotMgr] get_latest_prompt_before({frame_idx}): found at frame {latest_idx}")
        else:
            logger.debug(f"[AnnotMgr] get_latest_prompt_before({frame_idx}): none found")
        return latest_data

    def save_points_and_labels(self, frame_idx=None, points=None, labels=None, target_models=None, pose_keypoints=None):
        """Save/Update points and labels. If frame_idx is provided, it updates or appends that specific frame."""
        if frame_idx is not None:
            # Update existing or append new
            found = False
            for i in range(len(self.frame_indices)):
                if self.frame_indices[i] == frame_idx:
                    self.points_collection[i] = np.array(points, dtype=np.float32) if points is not None else self.points_collection[i]
                    self.labels_collection[i] = np.array(labels, dtype=np.int32) if labels is not None else self.labels_collection[i]
                    self.targets_collection[i] = target_models if target_models is not None else self.targets_collection[i]
                    if pose_keypoints is not None:
                        while len(self.pose_keypoints_collection) <= i:
                            self.pose_keypoints_collection.append([])
                        self.pose_keypoints_collection[i] = pose_keypoints
                    found = True
                    break
            if not found:
                self.frame_indices.append(frame_idx)
                self.points_collection.append(np.array(points, dtype=np.float32) if points is not None else np.array([]))
                self.labels_collection.append(np.array(labels, dtype=np.int32) if labels is not None else np.array([]))
                self.targets_collection.append(target_models if target_models is not None else [["sam", "pose"]] * len(self.points_collection[-1]))
                self.pose_keypoints_collection.append(pose_keypoints if pose_keypoints is not None else [])
            
            # Sort by frame_idx to keep file clean
            combined = sorted(zip(self.frame_indices, self.points_collection, self.labels_collection, self.targets_collection, self.pose_keypoints_collection), key=lambda x: x[0])
            self.frame_indices, self.points_collection, self.labels_collection, self.targets_collection, self.pose_keypoints_collection = map(list, zip(*combined))

        if self.save_dir:
             save_dir = self.save_dir
        else:
             base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
             save_dir = os.path.join(base_path, "workspace", "inputs", "UserPrompts")
        
        ensure_directory(save_dir)
        filename = os.path.join(save_dir, f"points_labels_{self.config.prefix}{self.config.video_number}.json")

        # Safely convert numpy types to native Python types
        def safe_convert(item):
            if isinstance(item, np.ndarray):
                return item.tolist()
            elif isinstance(item, (np.integer, np.floating)):
                return item.item()
            elif isinstance(item, list):
                return [safe_convert(i) for i in item]
            else:
                return item

        data = []
        for i in range(len(self.frame_indices)):
            entry = {
                "frame_idx": int(self.frame_indices[i]),
                "points": safe_convert(self.points_collection[i]),
                "labels": safe_convert(self.labels_collection[i]),
                "target_models": self.targets_collection[i],
            }
            if i < len(self.pose_keypoints_collection) and self.pose_keypoints_collection[i]:
                entry["pose_keypoints"] = self.pose_keypoints_collection[i]
            data.append(entry)

        try:
            with open(filename, 'w', encoding="utf-8") as f:
                json.dump(data, f, indent=2)
            logger.debug(f"[AnnotMgr] Saved {len(data)} annotations to {filename}")
            logger.debug(f"[AnnotMgr] Saved frame indices: {[e['frame_idx'] for e in data]}")
        except Exception as e:
            logger.error(f"Error saving points and labels to {filename}: {e}")

    def save_tracked_batch(self, tracked_dataset, batch_number):
        """Batch update tracked data into internal collections and save to JSON.
        
        Args:
            tracked_dataset: List of dicts from CoTracker containing frames and their keypoints.
                             [{"frame_index": int, "keypoints": [...]}, ...]
            batch_number: Integer batch number (0-indexed) to calculate absolute frame indices.
        """
        if not tracked_dataset:
            return

        batch_start = batch_number * self.config.batch_size

        for entry in tracked_dataset:
            # CoTrackerPredictor uses "frame_index" (relative), fallback to "frame_idx"
            rel_idx = entry.get("frame_index", entry.get("frame_idx", 0))
            f_idx = batch_start + rel_idx
            kps = entry["keypoints"]
            
            # Extract points and labels from the keypoints list
            pts = []
            lbls = []
            for kp in kps:
                # Use visible=2 or visible=True to determine if it's a point to show
                if kp.get("visible", 2) > 0:
                    pts.append([kp["x"], kp["y"]])
                    lbls.append(kp.get("label", 1001)) # Default to class 1 inst 1
            
            # Update/Overwrite existing or append new
            found = False
            for i in range(len(self.frame_indices)):
                if self.frame_indices[i] == f_idx:
                    self.points_collection[i] = np.array(pts, dtype=np.float32)
                    self.labels_collection[i] = np.array(lbls, dtype=np.int32)
                    self.pose_keypoints_collection[i] = kps
                    found = True
                    break
            
            if not found:
                self.frame_indices.append(f_idx)
                self.points_collection.append(np.array(pts, dtype=np.float32))
                self.labels_collection.append(np.array(lbls, dtype=np.int32))
                self.targets_collection.append([["sam", "pose"]] * len(pts))
                self.pose_keypoints_collection.append(kps)

        # Sort and Save to disk
        logger.debug(f"[AnnotMgr] save_tracked_batch: batch={batch_number}  entries={len(tracked_dataset)}")
        self.save_points_and_labels()

    def check_data_sufficiency(self):
        """Check if enough points and labels are available (at least one per batch)."""
        total_batches = (len(self.frame_paths) + self.config.batch_size - 1) // self.config.batch_size
        batches_with_data = set()
        # check_data_sufficiency: only count frames that exist in the current video
        for f_idx in self.frame_indices:
            if f_idx < len(self.frame_paths):
                batches_with_data.add(f_idx // self.config.batch_size)
        
        if len(batches_with_data) >= total_batches:
            if not self._all_batches_logged:
                logger.info(f"[DataCheck] All {total_batches} batches have at least one prompt set ready")
                self._all_batches_logged = True
            return len(self.frame_paths) # Signal all done
            
        self._all_batches_logged = False
        available = len(batches_with_data)
        logger.info(f"[DataCheck] Prompts available for {available}/{total_batches} batches")
        # Return the first missing batch's start frame
        for b in range(total_batches):
            if b not in batches_with_data:
                return b * self.config.batch_size
        return 0
