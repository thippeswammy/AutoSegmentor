import json
import os
from os.path import exists

import numpy as np

from ..FileManagement.FileManager import ensure_directory
from ..UserUI.logger_config import logger


class AnnotationManager:
    """Manages annotation data (points, labels, frame indices)."""

    def __init__(self, config, frame_paths):
        self.config = config
        self.frame_paths = frame_paths
        self.points_collection = []
        self.labels_collection = []
        self.frame_indices = []
        self.pose_keypoints_collection = []  # Per-batch pose keypoint coords
        self._all_batches_logged = False  # Dedup flag for check_data_sufficiency
        self.load_points_and_labels()

    def load_points_and_labels(self):
        """Load points and labels from JSON file."""
        ensure_directory("./inputs/UserPrompts")
        filename = f"./inputs/UserPrompts/points_labels_{self.config.prefix}{self.config.video_number}.json"

        if not exists(filename):
            logger.warning(f"Points and labels file {filename} not found")
            return

        try:
            with open(filename, 'r', encoding="utf-8") as f:
                data = json.load(f)

            # Convert lists back to numpy arrays (if needed)
            self.points_collection = [np.array(entry["points"], dtype=np.float32) for entry in data]
            self.labels_collection = [np.array(entry["labels"], dtype=np.int32) for entry in data]
            self.frame_indices = [int(entry["frame_idx"]) for entry in data]
            self.pose_keypoints_collection = [entry.get("pose_keypoints", []) for entry in data]

            logger.debug(f"Loaded {len(self.points_collection)} annotations from {filename}")
        except Exception as e:
            logger.error(f"Error loading points and labels from {filename}: {e}")

    def get_batch_prompts(self, batch_number, batch_size):
        """Get all prompts that fall within the specified batch range.

        Returns:
            List of dicts: [{"frame_idx": int, "points": np.array, "labels": np.array, "pose_keypoints": [...]}, ...]
        """
        start_frame = batch_number * batch_size
        end_frame = start_frame + batch_size
        results = []

        for i in range(len(self.frame_indices)):
            f_idx = self.frame_indices[i]
            if start_frame <= f_idx < end_frame:
                results.append({
                    "frame_idx": f_idx,
                    "points": self.points_collection[i],
                    "labels": self.labels_collection[i],
                    "pose_keypoints": self.pose_keypoints_collection[i] if i < len(self.pose_keypoints_collection) else []
                })
        return results

    def get_prompt_for_frame(self, frame_idx):
        """Get the prompt for a specific frame, if any."""
        for i, f_idx in enumerate(self.frame_indices):
            if f_idx == frame_idx:
                return {
                    "points": self.points_collection[i],
                    "labels": self.labels_collection[i],
                    "pose_keypoints": self.pose_keypoints_collection[i] if i < len(self.pose_keypoints_collection) else []
                }
        return None

    def get_latest_prompt_before(self, frame_idx):
        """Find the most recent prompt that occurs before the given frame index."""
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
                        "pose_keypoints": self.pose_keypoints_collection[i] if i < len(self.pose_keypoints_collection) else []
                    }
        return latest_data

    def save_points_and_labels(self, frame_idx=None, points=None, labels=None, pose_keypoints=None):
        """Save/Update points and labels. If frame_idx is provided, it updates or appends that specific frame."""
        if frame_idx is not None:
            # Update existing or append new
            found = False
            for i in range(len(self.frame_indices)):
                if self.frame_indices[i] == frame_idx:
                    self.points_collection[i] = np.array(points, dtype=np.float32) if points is not None else self.points_collection[i]
                    self.labels_collection[i] = np.array(labels, dtype=np.int32) if labels is not None else self.labels_collection[i]
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
                self.pose_keypoints_collection.append(pose_keypoints if pose_keypoints is not None else [])
            
            # Sort by frame_idx to keep file clean
            combined = sorted(zip(self.frame_indices, self.points_collection, self.labels_collection, self.pose_keypoints_collection), key=lambda x: x[0])
            self.frame_indices, self.points_collection, self.labels_collection, self.pose_keypoints_collection = map(list, zip(*combined))

        base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
        save_dir = os.path.join(base_path, "inputs/UserPrompts")
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
            }
            if i < len(self.pose_keypoints_collection) and self.pose_keypoints_collection[i]:
                entry["pose_keypoints"] = self.pose_keypoints_collection[i]
            data.append(entry)

        try:
            with open(filename, 'w', encoding="utf-8") as f:
                json.dump(data, f, indent=2)
            logger.debug(f"Saved {len(data)} annotations to {filename}")
        except Exception as e:
            logger.error(f"Error saving points and labels to {filename}: {e}")

    def check_data_sufficiency(self):
        """Check if enough points and labels are available (at least one per batch)."""
        total_batches = (len(self.frame_paths) + self.config.batch_size - 1) // self.config.batch_size
        batches_with_data = set()
        for f_idx in self.frame_indices:
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
