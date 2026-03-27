"""
Backward-compatibility shim for old import path:
  from utils.FileManagement.KeypointTracker import KeypointTracker
"""
from ..Models.Tracking.LKKeypointTracker import LKKeypointTracker as KeypointTracker  # noqa: F401

__all__ = ["KeypointTracker"]
