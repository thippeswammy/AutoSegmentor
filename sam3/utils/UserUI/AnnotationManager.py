import json
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

            logger.debug(f"Loaded {len(self.points_collection)} annotations from {filename}")
        except Exception as e:
            logger.error(f"Error loading points and labels from {filename}: {e}")

    def save_points_and_labels(self, points_collection=None, labels_collection=None, frame_indices=None):
        """Save points and labels to JSON file."""
        ensure_directory("./inputs/UserPrompts")
        filename = f"./inputs/UserPrompts/points_labels_{self.config.prefix}{self.config.video_number}.json"
        points_collection = points_collection or self.points_collection
        labels_collection = labels_collection or self.labels_collection
        frame_indices = frame_indices or self.frame_indices

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

        data = [
            {
                "frame_idx": int(frame_idx),
                "points": safe_convert(points),
                "labels": safe_convert(labels),
            }
            for frame_idx, points, labels in zip(frame_indices, points_collection, labels_collection)
        ]

        try:
            with open(filename, 'w', encoding="utf-8") as f:
                json.dump(data, f, indent=2)
            logger.debug(f"Saved {len(data)} annotations to {filename}")
        except Exception as e:
            logger.error(f"Error saving points and labels to {filename}: {e}")

    def check_data_sufficiency(self):
        """Check if enough points and labels are available."""
        total_batches = (len(self.frame_paths) + self.config.batch_size - 1) // self.config.batch_size
        if len(self.points_collection) >= total_batches:
            logger.info("Sufficient points and labels data for all batches")
            return len(self.points_collection) * self.config.batch_size
        missing_batches = total_batches - len(self.points_collection)
        logger.info(
            f"Missing points and labels for {missing_batches} batches from {total_batches - missing_batches + 1}")
        return len(self.points_collection) * self.config.batch_size
