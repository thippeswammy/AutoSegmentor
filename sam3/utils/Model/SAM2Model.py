"""
Backward-compatibility shim for old import path:
  from utils.Model.SAM2Model import SAM2Model
"""
from ..Models.SAM.SAM2Model import SAM2Model  # noqa: F401

__all__ = ["SAM2Model"]