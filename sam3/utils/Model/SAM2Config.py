"""
Backward-compatibility shim for old import path:
  from utils.Model.SAM2Config import SAM2Config
"""
from ..Models.AppConfig import AppConfig as SAM2Config  # noqa: F401

__all__ = ["SAM2Config"]
