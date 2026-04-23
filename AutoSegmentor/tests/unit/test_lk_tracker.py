"""
Unit Tests: LKKeypointTracker
Tests the Lucas-Kanade optical flow tracker with synthetic frames.
"""
import os
import sys
import numpy as np
import cv2
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from autosegmentor.models.Tracking.LKKeypointTracker import LKKeypointTracker


def _make_gray_frame(h=480, w=640, value=100):
    """Create a simple synthetic BGR frame."""
    img = np.full((h, w, 3), value, dtype=np.uint8)
    return img


def _make_keypoints(n=3):
    """Create n simple keypoint dicts."""
    defs = [f"kp{i}" for i in range(n)]
    coords = [{"name": f"kp{i}", "point_id": i, "x": 100 + i * 50, "y": 200} for i in range(n)]
    return defs, coords


class TestLKKeypointTrackerInit:

    def test_initializes_with_frame(self):
        defs, coords = _make_keypoints(3)
        frame = _make_gray_frame()
        tracker = LKKeypointTracker(keypoint_defs=defs, initial_coords=coords, initial_frame=frame)
        assert len(tracker.tracked_frames) == 1  # One initial frame
        assert tracker.tracked_frames[0]["frame_index"] == 0
        assert len(tracker.tracked_frames[0]["keypoints"]) == 3

    def test_initial_keypoints_are_visible(self):
        defs, coords = _make_keypoints(2)
        frame = _make_gray_frame()
        tracker = LKKeypointTracker(keypoint_defs=defs, initial_coords=coords, initial_frame=frame)
        for kp in tracker.tracked_frames[0]["keypoints"]:
            assert kp["visible"] == 2  # COCO visible

    def test_initial_coords_match(self):
        defs, coords = _make_keypoints(2)
        frame = _make_gray_frame()
        tracker = LKKeypointTracker(keypoint_defs=defs, initial_coords=coords, initial_frame=frame)
        for i, kp in enumerate(tracker.tracked_frames[0]["keypoints"]):
            assert kp["x"] == coords[i]["x"]
            assert kp["y"] == coords[i]["y"]


class TestLKKeypointTrackerTracking:

    def test_track_adds_frame(self):
        defs, coords = _make_keypoints(2)
        frame1 = _make_gray_frame(value=100)
        tracker = LKKeypointTracker(keypoint_defs=defs, initial_coords=coords, initial_frame=frame1)
        frame2 = _make_gray_frame(value=100)  # Same frame → tracking should succeed
        result = tracker.track(frame2, frame_index=1)
        assert len(tracker.tracked_frames) == 2
        assert tracker.tracked_frames[1]["frame_index"] == 1

    def test_get_all_tracked_returns_all_frames(self):
        defs, coords = _make_keypoints(2)
        frame = _make_gray_frame()
        tracker = LKKeypointTracker(keypoint_defs=defs, initial_coords=coords, initial_frame=frame)
        tracker.track(_make_gray_frame(value=100), frame_index=1)
        tracker.track(_make_gray_frame(value=100), frame_index=2)
        all_frames = tracker.get_all_tracked()
        assert len(all_frames) == 3  # initial + 2 tracked

    def test_keypoint_names_preserved(self):
        defs, coords = _make_keypoints(3)
        frame = _make_gray_frame()
        tracker = LKKeypointTracker(keypoint_defs=defs, initial_coords=coords, initial_frame=frame)
        tracker.track(_make_gray_frame(), frame_index=1)
        for kp, name in zip(tracker.tracked_frames[1]["keypoints"], defs):
            assert kp["name"] == name
