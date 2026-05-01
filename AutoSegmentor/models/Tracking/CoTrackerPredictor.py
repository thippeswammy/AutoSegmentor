import os
import sys

import cv2
import numpy as np
import torch

from ...ui.logger_config import logger

# Add co-tracker to the path so we can import from it
_COTRACKER_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', 'external', 'co-tracker')
)
if _COTRACKER_ROOT not in sys.path:
    sys.path.insert(0, _COTRACKER_ROOT)

DEFAULT_DEVICE = (
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)

# Singleton: reuse the same model across batches
_cotracker_model = None
_cotracker_checkpoint = None


def _get_cotracker_model(checkpoint, window_len=60):
    """Load or return cached CoTracker model (singleton)."""
    global _cotracker_model, _cotracker_checkpoint

    if _cotracker_model is not None and _cotracker_checkpoint == checkpoint:
        return _cotracker_model

    from cotracker.predictor import CoTrackerPredictor

    abs_checkpoint = os.path.abspath(checkpoint)
    if not os.path.exists(abs_checkpoint):
        logger.error(f"CoTracker checkpoint not found: {abs_checkpoint}")
        raise FileNotFoundError(f"CoTracker checkpoint not found: {abs_checkpoint}")

    logger.info(f"Loading CoTracker model from {abs_checkpoint} (window_len={window_len})")
    model = CoTrackerPredictor(
        checkpoint=abs_checkpoint,
        offline=True,
        v2=False,
        window_len=window_len,
    )
    model = model.to(DEFAULT_DEVICE)
    _cotracker_model = model
    _cotracker_checkpoint = checkpoint
    logger.info(f"CoTracker model loaded on {DEFAULT_DEVICE}")
    return model


class CoTrackerPredictor:
    """Tracks keypoint coordinates across video frames using CoTracker3.

    Given initial keypoint positions and a batch of frame paths, this class
    uses the CoTracker3 offline model to track all keypoints simultaneously
    across all frames in the batch.

    Attributes:
        keypoint_defs: List of keypoint name definitions.
        model: CoTrackerPredictor instance (singleton).
        tracked_frames: List of per-frame keypoint dicts (same format as LKKeypointTracker).
    """

    def __init__(self, keypoint_defs, initial_coords, frame_paths, checkpoint, window_len=60,
                 query_frame_idx=0, backward_tracking=False, base_frame_idx=0):
        """Initialize and run CoTracker tracking on the batch.

        Args:
            keypoint_defs: List of keypoint name strings.
            initial_coords: List of dicts [{"name": str, "point_id": int, "x": int, "y": int}, ...].
            frame_paths: List of absolute paths to all frames in this batch.
            checkpoint: Path to the CoTracker model checkpoint (.pth).
            window_len: Window length for the offline model (default: 60).
            query_frame_idx: Frame index within this batch to use as the query (default: 0).
            backward_tracking: Whether to run backward tracking (default: False).
            base_frame_idx: Base frame index to add to frame results (default: 0).
        """
        self.keypoint_defs = keypoint_defs
        self.tracked_frames = []

        # Sort initial coords by point_id
        sorted_coords = sorted(initial_coords, key=lambda c: c["point_id"])
        num_keypoints = len(sorted_coords)

        logger.info(
            f"CoTrackerPredictor: {num_keypoints} keypoints, "
            f"{len(frame_paths)} frames, query_frame={query_frame_idx}"
        )

        # Load the model (singleton, only loaded once)
        model = _get_cotracker_model(checkpoint, window_len)

        # Load all frames into a video tensor: (1, T, 3, H, W)
        frames = []
        for fp in frame_paths:
            img = cv2.imread(fp)
            if img is None:
                logger.warning(f"Failed to read frame: {fp}")
                if frames:
                    frames.append(frames[-1].copy())
                else:
                    frames.append(np.zeros((480, 640, 3), dtype=np.uint8))
                continue
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            frames.append(img_rgb)

        video_np = np.stack(frames, axis=0)  # (T, H, W, 3)
        video = torch.from_numpy(video_np).permute(0, 3, 1, 2)[None].float()  # (1, T, 3, H, W)
        video = video.to(DEFAULT_DEVICE)

        # Build queries: (1, N, 3) in format (t, x, y)
        queries = torch.zeros((1, num_keypoints, 3), dtype=torch.float32, device=DEFAULT_DEVICE)
        for i, coord in enumerate(sorted_coords):
            queries[0, i, 0] = float(query_frame_idx)
            queries[0, i, 1] = float(coord["x"])
            queries[0, i, 2] = float(coord["y"])

        # Run CoTracker inference
        logger.info(f"Running CoTracker inference (backward_tracking={backward_tracking})...")
        with torch.no_grad():
            pred_tracks, pred_visibility = model(
                video,
                queries=queries,
                backward_tracking=backward_tracking,
            )
        # pred_tracks: (1, T, N, 2) — (x, y) pixel coordinates
        # pred_visibility: (1, T, N) — boolean visibility

        tracks_np = pred_tracks[0].cpu().numpy()       # (T, N, 2)
        visibility_np = pred_visibility[0].cpu().numpy()  # (T, N)

        logger.info(
            f"CoTracker inference complete: tracks shape {tracks_np.shape}, "
            f"visibility shape {visibility_np.shape}"
        )

        # Convert to the same output format as LKKeypointTracker
        T = tracks_np.shape[0]
        for t in range(T):
            frame_kps = []
            for i, coord in enumerate(sorted_coords):
                is_visible = bool(visibility_np[t, i])
                x = int(round(tracks_np[t, i, 0])) if is_visible else -1
                y = int(round(tracks_np[t, i, 1])) if is_visible else -1

                kp_data = {
                    "name": coord["name"],
                    "point_id": coord["point_id"],
                    "x": x,
                    "y": y,
                    "visible": 2 if is_visible else 0  # COCO: 2=visible, 0=not labeled
                }
                if "label" in coord:
                    kp_data["label"] = coord["label"]
                frame_kps.append(kp_data)

            self.tracked_frames.append({
                "frame_index": t + base_frame_idx,
                "keypoints": frame_kps
            })

    def get_all_tracked(self):
        """Return all tracked frame data.

        Returns:
            List of dicts: [{"frame_index": int, "keypoints": [...]}, ...]
        """
        return self.tracked_frames


def track_between_frames(keypoint_coords, frame_from_path, frame_to_path, checkpoint, window_len=60):
    """Track keypoints from one frame to another using CoTracker.

    Used for batch-to-batch carry-forward: tracks keypoints from the last
    frame of the previous batch to the first frame of the next batch.

    Args:
        keypoint_coords: List of dicts [{"name": str, "point_id": int, "x": int, "y": int}, ...].
        frame_from_path: Path to the source frame.
        frame_to_path: Path to the destination frame.
        checkpoint: Path to CoTracker checkpoint.
        window_len: Window length for offline model.

    Returns:
        List of dicts with updated (x, y) positions, or None on failure.
    """
    model = _get_cotracker_model(checkpoint, window_len)

    sorted_coords = sorted(keypoint_coords, key=lambda c: c["point_id"])
    num_kps = len(sorted_coords)

    img_from = cv2.imread(frame_from_path)
    img_to = cv2.imread(frame_to_path)
    if img_from is None or img_to is None:
        logger.warning("Failed to read frames for carry-forward tracking")
        return None

    img_from_rgb = cv2.cvtColor(img_from, cv2.COLOR_BGR2RGB)
    img_to_rgb = cv2.cvtColor(img_to, cv2.COLOR_BGR2RGB)

    video_np = np.stack([img_from_rgb, img_to_rgb], axis=0)  # (2, H, W, 3)
    video = torch.from_numpy(video_np).permute(0, 3, 1, 2)[None].float()  # (1, 2, 3, H, W)
    video = video.to(DEFAULT_DEVICE)

    queries = torch.zeros((1, num_kps, 3), dtype=torch.float32, device=DEFAULT_DEVICE)
    for i, coord in enumerate(sorted_coords):
        queries[0, i, 0] = 0
        queries[0, i, 1] = float(coord["x"])
        queries[0, i, 2] = float(coord["y"])

    with torch.no_grad():
        pred_tracks, pred_visibility = model(video, queries=queries, backward_tracking=False)

    tracks_np = pred_tracks[0].cpu().numpy()       # (2, N, 2)
    visibility_np = pred_visibility[0].cpu().numpy()  # (2, N)

    result = []
    for i, coord in enumerate(sorted_coords):
        is_visible = bool(visibility_np[1, i])
        kp_data = {
            "name": coord["name"],
            "point_id": coord["point_id"],
            "x": int(round(tracks_np[1, i, 0])) if is_visible else coord["x"],
            "y": int(round(tracks_np[1, i, 1])) if is_visible else coord["y"],
        }
        if "label" in coord:
            kp_data["label"] = coord["label"]
        result.append(kp_data)

    logger.info(f"Carry-forward tracking: {num_kps} keypoints tracked between frames")
    return result
