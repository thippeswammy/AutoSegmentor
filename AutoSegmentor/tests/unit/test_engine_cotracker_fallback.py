"""
Unit Tests: AutoSegmentorEngine._track_batch_cotracker backward fallback.

When a batch has no manual prompt of its own, _track_batch_cotracker walks
backward through per_batch_tracked_data for the nearest earlier batch with
keypoints and bridges the gap with CoTracker's track_between_frames. These
tests exercise that fallback directly against a lightweight fake `self`
(the real class pulls in torch/cv2/SAM2/pygetwindow via __init__, which is
unnecessary just to test this index arithmetic).
"""
import os
import sys
from types import SimpleNamespace
from unittest.mock import patch

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
# AutoSegmentorEngine imports SAM2Model, which needs the sam2 package — the
# same external libs run_main.py registers on sys.path at startup.
for _lib in ("external/segment_anything_2", "external/segment_anything_2/sam2", "external/co-tracker"):
    _lib_path = os.path.join(_ROOT, *_lib.split("/"))
    if _lib_path not in sys.path:
        sys.path.insert(0, _lib_path)

from autosegmentor.core.AutoSegmentorEngine import AutoSegmentorEngine


def _make_fake_engine(batch_size, per_batch_tracked_data, frame_paths):
    return SimpleNamespace(
        config=SimpleNamespace(
            batch_size=batch_size,
            pose_config={"enabled": True, "cotracker": {}, "classes": []},
        ),
        annotation_manager=SimpleNamespace(get_batch_prompts=lambda batch, size: []),
        per_batch_tracked_data=per_batch_tracked_data,
        frame_paths=frame_paths,
    )


class TestCoTrackerBackwardFallback:
    """batch_size=1: batch 0 was processed with its usual "+1 lookahead", so
    per_batch_tracked_data[0] holds keypoints for frames 0 AND 1 — 2 entries
    for a batch_size of 1. Batch 1 was never processed on its own (its slot
    stays empty), so processing frame 2 must fall back past it to batch 0's
    data — and must pair frame 1's keypoints (the actual last entry) with
    frame 1's own image, not frame 0's.
    """

    def test_prev_idx_uses_actual_last_tracked_frame_not_batch_boundary(self):
        frame_paths = [f"frame_{i:05d}.jpg" for i in range(5)]
        per_batch_tracked_data = [
            [{"keypoints": [{"x": 1, "y": 1}]}, {"keypoints": [{"x": 2, "y": 2}]}],  # batch 0: frames 0,1
            [],  # batch 1: never processed on its own
        ]
        engine = _make_fake_engine(batch_size=1, per_batch_tracked_data=per_batch_tracked_data, frame_paths=frame_paths)

        captured = {}

        def fake_track_between_frames(kps_prev, prev_path, curr_path, checkpoint, window_len):
            captured["kps_prev"] = kps_prev
            captured["prev_path"] = prev_path
            captured["curr_path"] = curr_path
            return [{"keypoints": [{"x": 3, "y": 3}]}]

        with patch(
            "autosegmentor.models.Tracking.CoTrackerPredictor.track_between_frames",
            side_effect=fake_track_between_frames,
        ):
            AutoSegmentorEngine._track_batch_cotracker(engine, batch_number=2)

        # The last tracked entry in batch 0's data is frame 1's (index 1 in
        # that batch's own list), not frame 0's — the gap-tracking call must
        # use frame 1's image and frame 1's keypoints together.
        assert captured["prev_path"] == frame_paths[1]
        assert captured["kps_prev"] == [{"x": 2, "y": 2}]
        assert captured["curr_path"] == frame_paths[2]

    def test_no_overflow_falls_back_to_batch_boundary_frame(self):
        """If a batch's tracked data was clipped with no "+1" overflow (e.g.
        batch_size > 1 and the batch has exactly batch_size entries), the
        last tracked frame IS the batch's own last frame — must still resolve
        correctly."""
        frame_paths = [f"frame_{i:05d}.jpg" for i in range(10)]
        per_batch_tracked_data = [
            [{"keypoints": [{"x": 1, "y": 1}]}, {"keypoints": [{"x": 2, "y": 2}]}],  # batch 0: frames 0,1 (no overflow, batch_size=2)
            [],  # batch 1: never processed
        ]
        engine = _make_fake_engine(batch_size=2, per_batch_tracked_data=per_batch_tracked_data, frame_paths=frame_paths)

        captured = {}

        def fake_track_between_frames(kps_prev, prev_path, curr_path, checkpoint, window_len):
            captured["prev_path"] = prev_path
            captured["curr_path"] = curr_path
            return [{"keypoints": [{"x": 3, "y": 3}]}]

        with patch(
            "autosegmentor.models.Tracking.CoTrackerPredictor.track_between_frames",
            side_effect=fake_track_between_frames,
        ):
            AutoSegmentorEngine._track_batch_cotracker(engine, batch_number=2)

        assert captured["prev_path"] == frame_paths[1]  # last frame of batch 0 (frames 0,1)
        assert captured["curr_path"] == frame_paths[4]  # batch 2 starts at frame 4


class TestCoTrackerBatchEndOverflow:
    """The normal (non-fallback) path slices batch_frame_paths using
    batch_end = batch_start + batch_size (+1 lookahead, only when
    batch_size > 1). With batch_size=1 that lookahead must be dropped so
    every frame gets tracked by its own explicit CoTracker call instead of
    silently covering the next frame too.
    """

    def _run_with_prompt(self, batch_size, frame_paths, batch_number=0):
        engine = SimpleNamespace(
            config=SimpleNamespace(
                batch_size=batch_size,
                pose_config={"enabled": True, "cotracker": {}, "classes": []},
            ),
            annotation_manager=SimpleNamespace(
                get_batch_prompts=lambda batch, size: [
                    {"frame_idx": batch_number * batch_size, "pose_keypoints": [{"x": 1, "y": 1}]}
                ]
            ),
            per_batch_tracked_data=[[] for _ in frame_paths],
            frame_paths=frame_paths,
        )
        captured = {}

        class FakeCoTrackerPredictor:
            def __init__(self, keypoint_defs, initial_coords, frame_paths, **kwargs):
                captured["frame_paths"] = frame_paths

            def get_all_tracked(self):
                return []

        with patch(
            "autosegmentor.models.Tracking.CoTrackerPredictor.CoTrackerPredictor",
            FakeCoTrackerPredictor,
        ):
            AutoSegmentorEngine._track_batch_cotracker(engine, batch_number=batch_number)
        return captured["frame_paths"]

    def test_batch_size_one_tracks_exactly_its_own_frame(self):
        frame_paths = [f"frame_{i:05d}.jpg" for i in range(5)]
        tracked = self._run_with_prompt(batch_size=1, frame_paths=frame_paths)
        assert tracked == [frame_paths[0]]

    def test_batch_size_greater_than_one_keeps_lookahead_frame(self):
        frame_paths = [f"frame_{i:05d}.jpg" for i in range(5)]
        tracked = self._run_with_prompt(batch_size=2, frame_paths=frame_paths)
        assert tracked == frame_paths[0:3]  # batch_size (2) + 1 lookahead
