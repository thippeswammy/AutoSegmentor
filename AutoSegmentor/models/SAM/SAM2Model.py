import os
import torch
import GPUtil
from ...ui.logger_config import logger
from ..model_info import missing_model_message
from sam2.build_sam import build_sam2_video_predictor


class SAM2Model:
    """Handles SAM2 model loading, device selection, and GPU management."""

    def __init__(self, config):
        self.config = config
        self.device = self.get_device()
        self.gpus = GPUtil.getGPUs()
        self.sam2_predictor = self.build_predictor() if getattr(self.config, 'sam_enabled', True) else None

    def get_device(self):
        """Determine available device (CUDA or CPU)."""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {device}")
        if device.type == "cuda":
            logger.debug("[SAM2Model] Enabling CUDA TF32 and CuDNN TF32")
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        return device

    def build_predictor(self):
        """Build and return the SAM2 video predictor."""
        logger.debug(f"[SAM2Model] build_predictor: cfg={self.config.model_config_path}  ckpt={self.config.checkpoint_path}")
        logger.debug(f"[SAM2Model] build_predictor: memory_bank_size={self.config.memory_bank_size}  prompt_memory_size={self.config.prompt_memory_size}")
        checkpoint_path = self.config.checkpoint_path
        if not os.path.exists(checkpoint_path):
            msg = missing_model_message("SAM2", [checkpoint_path])
            logger.error(msg)
            raise FileNotFoundError(msg)
        predictor = build_sam2_video_predictor(
            self.config.model_config_path,
            checkpoint_path,
            device=self.device,
            memory_bank_size=self.config.memory_bank_size,
            prompt_memory_size=self.config.prompt_memory_size
        )
        # Correction clicks (esp. negative points) added on a frame after tracking
        # has begun leave stale pre-correction memory on neighboring frames unless
        # this is enabled — see sam2_video_predictor.py's own docstring on the flag.
        predictor.clear_non_cond_mem_around_input = True
        predictor.clear_non_cond_mem_for_multi_obj = True  # AutoSegmentor tracks multiple obj_ids per video
        return predictor

    def gpu_memory_usage(self, ind=0):
        """Get GPU memory usage for the specified GPU index."""
        return self.gpus[ind]
