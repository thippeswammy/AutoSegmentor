"""
Backward-compatibility shim for old import path:
  from autosegmentor.file_management.KeypointTracker import KeypointTracker
"""
from ..models.Tracking.LKKeypointTracker import LKKeypointTracker as KeypointTracker  # noqa: F401

__all__ = ["KeypointTracker"]
