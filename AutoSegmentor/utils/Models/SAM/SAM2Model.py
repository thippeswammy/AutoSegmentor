import torch
import GPUtil
from ...UserUI.logger_config import logger
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
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        return device

    def build_predictor(self):
        """Build and return the SAM2 video predictor."""
        return build_sam2_video_predictor(
            self.config.model_config_path,
            self.config.checkpoint_path,
            device=self.device,
            memory_bank_size=self.config.memory_bank_size,
            prompt_memory_size=self.config.prompt_memory_size
        )

    def gpu_memory_usage(self, ind=0):
        """Get GPU memory usage for the specified GPU index."""
        return self.gpus[ind]
