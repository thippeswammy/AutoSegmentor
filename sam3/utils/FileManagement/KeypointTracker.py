import cv2
import numpy as np

from ..UserUI.logger_config import logger


class KeypointTracker:
    """Tracks keypoint coordinates across video frames using Lucas-Kanade sparse optical flow.

    Given initial keypoint positions on an annotated frame, this class uses
    cv2.calcOpticalFlowPyrLK to propagate those positions to subsequent frames.

    Attributes:
        keypoint_defs: List of keypoint name definitions (e.g., ["top_left_box1", ...]).
        lk_params: Parameters for Lucas-Kanade optical flow.
        prev_gray: Grayscale image of the previous frame.
        prev_points: Nx1x2 array of keypoint positions in the previous frame.
        tracked_frames: List of per-frame keypoint dicts.
    """

    def __init__(self, keypoint_defs, initial_coords, initial_frame):
        """Initialize tracker with keypoint definitions and initial positions.

        Args:
            keypoint_defs: List of keypoint name strings.
            initial_coords: List of dicts [{"name": str, "point_id": int, "x": int, "y": int}, ...].
            initial_frame: BGR image (numpy array) of the annotated frame.
        """
        self.keypoint_defs = keypoint_defs
        self.lk_params = dict(
            winSize=(21, 21),
            maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
        )

        self.prev_gray = cv2.cvtColor(initial_frame, cv2.COLOR_BGR2GRAY)

        # Build the point array from initial_coords (ordered by point_id)
        sorted_coords = sorted(initial_coords, key=lambda c: c["point_id"])
        self.prev_points = np.array(
            [[c["x"], c["y"]] for c in sorted_coords], dtype=np.float32
        ).reshape(-1, 1, 2)

        # Store the initial frame's keypoints
        self.tracked_frames = [self._build_frame_data(sorted_coords, frame_index=0)]

        logger.info(f"KeypointTracker initialized with {len(sorted_coords)} keypoints")

    def track(self, next_frame, frame_index):
        """Track keypoints to the next frame.

        Args:
            next_frame: BGR image (numpy array) of the next frame.
            frame_index: Index of this frame in the sequence.

        Returns:
            List of keypoint dicts for this frame, or None if tracking failed.
        """
        next_gray = cv2.cvtColor(next_frame, cv2.COLOR_BGR2GRAY)

        # Forward optical flow
        new_points, status, err = cv2.calcOpticalFlowPyrLK(
            self.prev_gray, next_gray, self.prev_points, None, **self.lk_params
        )

        if new_points is None:
            logger.warning(f"Optical flow failed at frame {frame_index}")
            return None

        # Optional: backward check for robustness
        back_points, back_status, _ = cv2.calcOpticalFlowPyrLK(
            next_gray, self.prev_gray, new_points, None, **self.lk_params
        )

        # Build keypoint data for this frame
        frame_kps = []
        for i, kp_name in enumerate(self.keypoint_defs):
            if i >= len(new_points):
                break

            # Check forward-backward consistency (threshold: 2 pixels)
            fb_ok = True
            if back_points is not None and i < len(back_points):
                fb_dist = np.linalg.norm(self.prev_points[i] - back_points[i])
                fb_ok = fb_dist < 2.0

            is_tracked = int(status[i][0]) == 1 and fb_ok

            frame_kps.append({
                "name": kp_name,
                "point_id": i,
                "x": int(round(new_points[i][0][0])) if is_tracked else -1,
                "y": int(round(new_points[i][0][1])) if is_tracked else -1,
                "visible": 2 if is_tracked else 0  # COCO: 2=visible, 0=not labeled
            })

        self.tracked_frames.append({
            "frame_index": frame_index,
            "keypoints": frame_kps
        })

        # Update state for next iteration
        self.prev_gray = next_gray
        # Only update successfully tracked points
        good_mask = (status.flatten() == 1)
        if good_mask.any():
            self.prev_points = new_points.copy()
            # For failed points, keep previous position (they'll be marked invisible)
            for i in range(len(good_mask)):
                if not good_mask[i]:
                    self.prev_points[i] = self.prev_points[i]  # no-op, but explicit

        return frame_kps

    def _build_frame_data(self, coords, frame_index):
        """Build a frame keypoint dict from coordinate list."""
        kps = []
        for coord in coords:
            kps.append({
                "name": coord["name"],
                "point_id": coord["point_id"],
                "x": coord["x"],
                "y": coord["y"],
                "visible": 2  # Original annotation is always visible
            })
        return {
            "frame_index": frame_index,
            "keypoints": kps
        }

    def get_all_tracked(self):
        """Return all tracked frame data.

        Returns:
            List of dicts: [{"frame_index": int, "keypoints": [...]}, ...]
        """
        return self.tracked_frames
