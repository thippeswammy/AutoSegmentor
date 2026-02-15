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
      - Tracks them through all frames in the batch using optical flow.
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

            # Read first frame of this batch
            first_frame_path = os.path.join(verified_img_dir, batch_frames[0])
            first_frame = cv2.imread(first_frame_path)
            if first_frame is None:
                logger.error(f"  Failed to read: {first_frame_path}")
                continue

            # Initialize tracker for this batch
            tracker = KeypointTracker(
                keypoint_defs=self.keypoints_def,
                initial_coords=batch_kps,
                initial_frame=first_frame
            )

            # Track through remaining frames in this batch
            for i in range(1, len(batch_frames)):
                frame_path = os.path.join(verified_img_dir, batch_frames[i])
                frame = cv2.imread(frame_path)
                if frame is None:
                    logger.warning(f"  Failed to read: {frame_path}")
                    continue
                tracker.track(frame, frame_index=i)

            # Collect tracked data for this batch
            tracked_data = tracker.get_all_tracked()
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
