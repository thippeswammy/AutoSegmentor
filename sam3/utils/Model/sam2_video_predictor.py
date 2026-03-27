"""
Backward-compatibility shim for old import path:
  from utils.Model.sam2_video_predictor import SAM2VideoProcessor
"""
from ..Core.AutoSegmentorEngine import AutoSegmentorEngine as SAM2VideoProcessor  # noqa: F401

__all__ = ["SAM2VideoProcessor"]
