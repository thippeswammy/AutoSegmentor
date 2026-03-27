"""
Backward-compatibility shim for old import path:
  from utils.FileManagement.CoTrackerKeypointTracker import CoTrackerKeypointTracker
"""
from ..Models.Tracking.CoTrackerPredictor import CoTrackerPredictor as CoTrackerKeypointTracker  # noqa: F401
from ..Models.Tracking.CoTrackerPredictor import track_between_frames  # noqa: F401

__all__ = ["CoTrackerKeypointTracker", "track_between_frames"]
