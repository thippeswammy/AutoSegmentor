import os
import re
import shutil
import sys

from utils.UserUI.logger_config import logger


def get_resource_path(relative_path):
    """Get absolute path to resource, works for dev and for PyInstaller."""
    if hasattr(sys, '_MEIPASS'):
        return os.path.join(sys._MEIPASS, relative_path)
    return os.path.join(os.path.abspath(".."), relative_path)


def ensure_directory(path):
    """Create directory if it doesn't exist."""
    os.makedirs(path, exist_ok=True)
    return path


def clear_directory(directory):
    """Clear all contents of a directory."""
    if os.path.exists(directory):
        for filename in os.listdir(directory):
            file_path = os.path.join(directory, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                logger.error(f"Failed to delete {file_path}. Reason: {e}")
    else:
        logger.debug(f"Directory {directory} does not exist.")


def get_frame_paths(directory):
    """Get sorted list of frame paths from a directory."""
    frame_paths = []
    for p in os.listdir(directory):
        if os.path.splitext(p)[-1].lower() in [".jpg", ".jpeg", ".png"]:
            match = re.search(r'_(\d+)\.(?:jpg|jpeg|png)$', p, re.IGNORECASE)
            if match:
                frame_paths.append(os.path.join(directory, p))
            else:
                logger.warning(f"Skipping file {p} due to invalid filename format")
    return sorted(frame_paths, key=lambda p: int(re.search(r'_(\d+)\.(?:jpg|jpeg|png)$', p, re.IGNORECASE).group(1)))
