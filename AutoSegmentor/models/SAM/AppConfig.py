import os
from ...file_management.FileManager import get_resource_path, ensure_directory
from ...ui.logger_config import logger
from ..model_info import missing_model_message


class AppConfig:
    """Global configuration for the AutoSegmentor pipeline.

    Holds all directories, model paths, batch settings, and pose configuration
    for a single video processing run. Replaces the SAM2-specific SAM2Config.
    """

    def __init__(self, video_number, batch_size=120, images_starting_count=0, images_ending_count=None,
                 prefix="file", video_path_template=None, images_extract_dir=None,
                 rendered_frames_dir=None, temp_processing_dir=None, working_dir=None,
                 window_size=None, label_colors=None, memory_bank_size=5, prompt_memory_size=5,
                 sam_enabled=True, **kwargs):
        self.video_number = video_number
        self.batch_size = batch_size
        self.images_starting_count = images_starting_count
        self.images_ending_count = images_ending_count
        self.prefix = prefix
        self.video_path_template = video_path_template or './workspace/VideoInputs/Video{}.mp4'
        self.working_dir = working_dir or './workspace/working_dir'
        self.frames_directory = images_extract_dir or os.path.join(self.working_dir, 'images')
        self.rendered_frames_dir = rendered_frames_dir or os.path.join(self.working_dir, 'outputs')
        self.temp_directory = temp_processing_dir or os.path.join(self.working_dir, 'temp')
        ensure_directory(self.frames_directory)
        ensure_directory(self.rendered_frames_dir)
        ensure_directory(self.temp_directory)
        self.window_size = window_size or [200, 200]
        self.label_colors = label_colors or {
            1: (0, 0, 255), 2: (255, 0, 0), 3: (0, 255, 0), 4: (0, 255, 255),
            5: (255, 0, 255), 6: (255, 255, 0), 7: (128, 0, 128), 8: (0, 165, 255),
            9: (255, 255, 255), 10: (0, 0, 0)
        }
        self.memory_bank_size = memory_bank_size
        self.prompt_memory_size = prompt_memory_size
        self.pose_config = kwargs.get('pose_config', {})
        self.sam_config = kwargs.get('sam_config', {})
        self.auto_prompt_encoding = kwargs.get('auto_prompt_encoding', True)
        self.ui_show_crosshair = kwargs.get('ui_show_crosshair', True)
        self.ui_show_grid = kwargs.get('ui_show_grid', False)
        self.sam_enabled = sam_enabled

        # Model Routing Preferences (Persistent)
        interaction = kwargs.get('interaction', {})
        self.active_target_models = interaction.get('active_target_models', ['sam', 'pose'])
        self.auto_shift_enabled = interaction.get('auto_shift_enabled', True)

        # Calculate base path up to project root (AutoSegmentor)
        # __file__ is AutoSegmentor/autosegmentor/models/SAM/AppConfig.py
        # root is 3 levels up
        base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))

        self.model_config_path = self.sam_config.get("model_config") or "sam2_hiera_l.yaml"
        checkpoint_raw = self.sam_config.get("checkpoint") or "external/segment_anything_2/checkpoints/sam2_hiera_large.pt"
        self.checkpoint_path = os.path.normpath(os.path.join(base_path, checkpoint_raw))
        if not os.path.exists(self.checkpoint_path):
            fallback = get_resource_path(checkpoint_raw)
            logger.warning(f"SAM2 checkpoint not found at {self.checkpoint_path}; trying fallback {fallback}")
            self.checkpoint_path = fallback
            if not os.path.exists(self.checkpoint_path):
                logger.error(missing_model_message(
                    "SAM2",
                    [os.path.normpath(os.path.join(base_path, checkpoint_raw)), fallback],
                ))
