import os
import re
import shutil

from .FileManager import ensure_directory, clear_directory
from ..UserUI.logger_config import logger


class FrameHandler:
    """Manages frame paths and batch copying."""

    def __init__(self, frames_directory, temp_directory):
        self.frames_directory = frames_directory
        self.temp_directory = temp_directory
        ensure_directory(self.frames_directory)
        ensure_directory(self.temp_directory)

    def get_frame_files(self):
        """Get sorted list of frame paths."""
        frame_paths = []
        print()
        for p in os.listdir(os.path.abspath(os.path.join(self.frames_directory))):
            if os.path.splitext(p)[-1].lower() in [".jpg", ".jpeg", ".png"]:
                match = re.search(r'_(\d+)\.(?:jpg|jpeg|png)$', p, re.IGNORECASE)
                if match:
                    frame_paths.append(os.path.join(self.frames_directory, p))
                else:
                    logger.warning(f"Skipping file {p} due to invalid filename format")
        return sorted(frame_paths,
                      key=lambda p: int(re.search(r'_(\d+)\.(?:jpg|jpeg|png)$', p, re.IGNORECASE).group(1)))

    def move_and_copy_frames(self, batch_index, frame_paths, batch_size):
        """Copy frames for the current batch to temp directory."""
        frames_to_copy = frame_paths[batch_index:batch_index + batch_size]
        clear_directory(self.temp_directory)
        ensure_directory(self.temp_directory)
        for frame_path in frames_to_copy:
            match = re.search(r'_(\d+)\.(?:jpg|jpeg|png)$', frame_path, re.IGNORECASE)
            if not match:
                logger.warning(f"Skipping file {frame_path} due to invalid filename format")
                continue
            frame_index = match.group(1)
            new_filename = f"{frame_index}.jpg"
            dst_path = os.path.join(self.temp_directory, new_filename)
            shutil.copy2(frame_path, dst_path)
            logger.debug(f"Copied {frame_path} to {dst_path}")
