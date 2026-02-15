import json
import os
import cv2
from ..UserUI.logger_config import logger
from ..FileManagement.FileManager import ensure_directory
from .KeypointTracker import KeypointTracker


class PoseExporter:
    """Exports keypoint coordinates tracked across video frames to JSON.

    For each batch:
      - Uses that batch's annotated keypoint coordinates as anchor points.
      - Tracks them through all frames in the batch using the configured tracker.
    Supports two tracker backends:
      - "cotracker": CoTracker3 (learned point tracker, processes entire batch at once)
      - "lk": Lucas-Kanade sparse optical flow (frame-by-frame)
    Concatenates results across all batches.
    """

    def __init__(self, config, verified_mask_dir, annotation_manager):
        self.config = config
        self.verified_mask_dir = verified_mask_dir
        self.annotation_manager = annotation_manager
        self.output_file = os.path.join(
            self.verified_mask_dir.replace('mask', ''),
            'pose_labels.json'
        )
        ensure_directory(os.path.dirname(self.output_file))

        self.keypoints_def = (
            self.config.pose_config.get('keypoints', [])
            if self.config.pose_config else []
        )

        # Determine tracker type
        self.tracker_type = "lk"  # default fallback
        if self.config.pose_config:
            self.tracker_type = self.config.pose_config.get('tracker', 'lk').lower()

    def _get_cotracker_config(self):
        """Extract CoTracker config from pose_config."""
        ct_cfg = self.config.pose_config.get('cotracker', {})
        # Resolve checkpoint path relative to sam3 directory
        checkpoint = ct_cfg.get('checkpoint', '../co-tracker/checkpoints/scaled_offline.pth')
        base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
        checkpoint = os.path.normpath(os.path.join(base_path, checkpoint))
        window_len = ct_cfg.get('window_len', 60)
        return checkpoint, window_len

    def _track_batch_cotracker(self, batch_kps, batch_frame_paths):
        """Track keypoints through a batch using CoTracker."""
        from .CoTrackerKeypointTracker import CoTrackerKeypointTracker
        checkpoint, window_len = self._get_cotracker_config()
        tracker = CoTrackerKeypointTracker(
            keypoint_defs=self.keypoints_def,
            initial_coords=batch_kps,
            frame_paths=batch_frame_paths,
            checkpoint=checkpoint,
            window_len=window_len,
        )
        return tracker.get_all_tracked()

    def _track_batch_lk(self, batch_kps, batch_frame_paths):
        """Track keypoints through a batch using Lucas-Kanade optical flow."""
        first_frame = cv2.imread(batch_frame_paths[0])
        if first_frame is None:
            logger.error(f"  Failed to read: {batch_frame_paths[0]}")
            return []

        tracker = KeypointTracker(
            keypoint_defs=self.keypoints_def,
            initial_coords=batch_kps,
            initial_frame=first_frame
        )

        for i in range(1, len(batch_frame_paths)):
            frame = cv2.imread(batch_frame_paths[i])
            if frame is None:
                logger.warning(f"  Failed to read: {batch_frame_paths[i]}")
                continue
            tracker.track(frame, frame_index=i)

        return tracker.get_all_tracked()

    def process_masks(self):
        """Run per-batch keypoint tracking and export to JSON."""
        if not self.config.pose_config or not self.config.pose_config.get('enabled'):
            logger.info("Pose estimation disabled. Skipping export.")
            return

        # Get the verified images directory (parallel to mask dir)
        verified_img_dir = self.verified_mask_dir.replace('mask', 'images')
        if not os.path.exists(verified_img_dir):
            logger.error(f"Verified images directory not found: {verified_img_dir}")
            return

        logger.info(f"Starting Pose Export with per-batch keypoint tracking")
        logger.info(f"  Tracker: {self.tracker_type}")
        logger.info(f"  Images: {verified_img_dir}")
        logger.info(f"  Output: {self.output_file}")

        # Get sorted frame files
        frame_files = sorted([
            f for f in os.listdir(verified_img_dir)
            if f.lower().endswith(('.png', '.jpg', '.jpeg'))
        ])

        if not frame_files:
            logger.warning(f"No frames found in {verified_img_dir}")
            return

        total_frames = len(frame_files)
        batch_size = self.config.batch_size
        total_batches = (total_frames + batch_size - 1) // batch_size
        pose_kps_collection = self.annotation_manager.pose_keypoints_collection

        logger.info(f"  Total frames: {total_frames}, Batch size: {batch_size}, Batches: {total_batches}")
        logger.info(f"  Available keypoint sets: {len(pose_kps_collection)}")

        pose_data = []
        last_known_kps = None

        for batch_idx in range(total_batches):
            batch_start = batch_idx * batch_size
            batch_end = min(batch_start + batch_size, total_frames)
            batch_frames = frame_files[batch_start:batch_end]

            # Determine keypoints for this batch
            batch_kps = None
            if batch_idx < len(pose_kps_collection) and pose_kps_collection[batch_idx]:
                batch_kps = pose_kps_collection[batch_idx]
            elif last_known_kps:
                batch_kps = last_known_kps
                logger.info(f"  Batch {batch_idx + 1}: Using carried-forward keypoints")

            if not batch_kps:
                logger.warning(f"  Batch {batch_idx + 1}: No keypoints available, skipping")
                # Fill with empty keypoints
                for i, fname in enumerate(batch_frames):
                    frame_idx = batch_start + i
                    try:
                        fid = int(os.path.splitext(fname)[0].split('_')[-1])
                    except ValueError:
                        fid = frame_idx
                    pose_data.append({
                        "image_id": fname,
                        "frame_index": fid,
                        "keypoints": [{"name": kp, "point_id": j, "x": -1, "y": -1, "visible": 0}
                                       for j, kp in enumerate(self.keypoints_def)]
                    })
                continue

            last_known_kps = batch_kps
            logger.info(f"  Batch {batch_idx + 1}: Tracking {len(batch_kps)} keypoints across {len(batch_frames)} frames")

            # Build full paths for batch frames
            batch_frame_paths = [os.path.join(verified_img_dir, f) for f in batch_frames]

            # Track using configured backend
            if self.tracker_type == "cotracker":
                tracked_data = self._track_batch_cotracker(batch_kps, batch_frame_paths)
            else:
                tracked_data = self._track_batch_lk(batch_kps, batch_frame_paths)

            # Collect tracked data for this batch
            for entry in tracked_data:
                local_idx = entry["frame_index"]
                global_idx = batch_start + local_idx
                fname = batch_frames[local_idx] if local_idx < len(batch_frames) else f"frame_{global_idx}"

                try:
                    fid = int(os.path.splitext(fname)[0].split('_')[-1])
                except ValueError:
                    fid = global_idx

                pose_data.append({
                    "image_id": fname,
                    "frame_index": fid,
                    "keypoints": entry["keypoints"]
                })

        # Save to JSON
        try:
            with open(self.output_file, 'w') as f:
                json.dump(pose_data, f, indent=4)
            logger.info(f"Pose data exported: {len(pose_data)} frames, {len(self.keypoints_def)} keypoints each")
            logger.info(f"  Output file: {self.output_file}")
        except Exception as e:
            logger.error(f"Failed to export pose data: {e}")
