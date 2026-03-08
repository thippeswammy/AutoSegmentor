import os
from ..FileManagement.FileManager import get_resource_path, ensure_directory


class SAM2Config:
    """Configuration for SAM2 video predictor."""

    def __init__(self, video_number, batch_size=120, images_starting_count=0, images_ending_count=None,
                 prefix="file", video_path_template=None, images_extract_dir=None,
                 rendered_frames_dir=None, temp_processing_dir=None, window_size=None,
                 label_colors=None, memory_bank_size=5, prompt_memory_size=5, sam_enabled=True, **kwargs):
        self.video_number = video_number
        self.batch_size = batch_size
        self.images_starting_count = images_starting_count
        self.images_ending_count = images_ending_count
        self.prefix = prefix
        self.video_path_template = video_path_template or './VideoInputs/Video{}.mp4'
        self.frames_directory = images_extract_dir or './videos/images'
        self.rendered_frames_dir = rendered_frames_dir or './videos/outputs'
        self.temp_directory = temp_processing_dir or './videos/temp'
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
        self.pose_config = kwargs.get('pose_config', None)
        self.auto_prompt_encoding = kwargs.get('auto_prompt_encoding', True)
        self.ui_show_crosshair = kwargs.get('ui_show_crosshair', True)
        self.ui_show_grid = kwargs.get('ui_show_grid', False)
        self.sam_enabled = sam_enabled
        
        # Calculate base path up to project root (AutoSegmentor)
        # __file__ is sam3/utils/Model/SAM2Config.py
        # root is 3 levels up: sam3/utils/Model -> sam3/utils -> sam3 -> AutoSegmentor
        base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
        
        self.model_config_path = os.path.join(base_path, "sam2_configs/sam2_hiera_l.yaml")
        # Ensure imports work for sam2 hydra config
        if not os.path.exists(self.model_config_path):
             # Fallback or try to use get_resource_path if packaged
             self.model_config_path = get_resource_path("./sam2_configs/sam2_hiera_l.yaml")
             
        self.checkpoint_path = os.path.join(base_path, "checkpoints/sam2_hiera_large.pt")
        if not os.path.exists(self.checkpoint_path):
             self.checkpoint_path = get_resource_path("./checkpoints/sam2_hiera_large.pt")
