"""
main_app.py — Core application logic for AutoSegmentor.
"""

import os
import shutil
import sys
import time
import signal

# Allow terminal interrupts (Ctrl+C) to terminate the PyQt application safely.
signal.signal(signal.SIGINT, signal.SIG_DFL)

# Adjust path to include the project root (3 levels up from utils/Tools/main_app.py)
# Or let the root-level run_demo.py handle the path.
# For direct execution of this script:
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from PyQt5.QtWidgets import QApplication

from autosegmentor.ui.logger_config import logger
from autosegmentor.ui.SetupDialog import SetupDialog
from autosegmentor.pipeline import run_pipeline


def _handle_working_dir(working_dir_name: str, delete: str, prompt_msg: str) -> bool:
    """Handle working directory cleanup."""
    if not os.path.exists(working_dir_name):
        return True

    if delete == "yes":
        shutil.rmtree(working_dir_name)
        logger.info(f"Cleared working directory: {working_dir_name}")
        return True

    confirm = input(prompt_msg).strip().lower()
    if confirm == "yes":
        shutil.rmtree(working_dir_name)
        logger.info(f"Cleared working directory: {working_dir_name}")
        return True

    logger.info(f"Working directory '{working_dir_name}' kept.")
    return False


def start_application():
    app = QApplication(sys.argv)

    # ── Show the Setup Dialog ────────────────────────────────────────────────
    dialog = SetupDialog()
    if dialog.exec_() != SetupDialog.Accepted:
        logger.info("Setup cancelled by user. Exiting.")
        sys.exit(0)

    cfg = dialog.get_config()

    # ── Extract params ───────────────────────────────────────────────────────
    video_start          = cfg["video_start"]
    video_end            = cfg["video_end"]
    prefix               = cfg["prefix"]
    batch_size           = cfg["batch_size"]
    fps                  = cfg["fps"]
    delete               = cfg["delete"]
    working_dir_name     = cfg["working_dir_name"]
    video_path_template  = cfg["video_path_template"]
    images_extract_dir   = cfg["images_extract_dir"]
    temp_processing_dir  = cfg["temp_processing_dir"]
    rendered_dir         = cfg["rendered_dir"]
    overlap_dir          = cfg["overlap_dir"]
    verified_img_dir     = cfg["verified_img_dir"]
    verified_mask_dir    = cfg["verified_mask_dir"]
    final_video_path     = cfg["final_video_path"]
    images_ending_count  = cfg["images_ending_count"]
    pose_config          = cfg["pose_estimation"]
    run_mode             = cfg["run_mode"]
    auto_prompt_encoding = cfg["auto_prompt_encoding"]
    sam_enabled          = cfg["sam_enabled"]
    sam_config           = cfg.get("sam_config", {})
    review_from_start    = cfg.get("review_from_start", False)

    total_videos  = video_end
    overall_start = time.time()

    for idx, i in enumerate(range(video_start, video_start + video_end), start=1):
        logger.info(f"{'═' * 20} Video {i} ({idx}/{total_videos}) {'═' * 20}")

        if run_mode != "pose_only":
            cleared = _handle_working_dir(
                working_dir_name,
                delete,
                f"Do you want to clear prev working directory '{working_dir_name}'? (yes/no): ",
            )
            if not cleared:
                sys.exit(1000)

        run_pipeline(
            fps=fps,
            video_number=i,
            prefix=prefix,
            batch_size=batch_size,
            delete=delete,
            video_path_template=video_path_template,
            images_extract_dir=images_extract_dir,
            temp_processing_dir=temp_processing_dir,
            working_dir=working_dir_name,
            rendered_dirs=rendered_dir,
            overlap_dir=overlap_dir,
            verified_img_dir=verified_img_dir,
            verified_mask_dir=verified_mask_dir,
            final_video_path=final_video_path,
            images_ending_count=images_ending_count,
            pose_config=pose_config,
            run_mode=run_mode,
            auto_prompt_encoding=auto_prompt_encoding,
            sam_enabled=sam_enabled,
            sam_config=sam_config,
            review_from_start=review_from_start,
        )

        _handle_working_dir(
            working_dir_name,
            delete,
            f"Delete working directory '{working_dir_name}'? (yes/no): ",
        )

        logger.info("═" * 60)

    elapsed = time.time() - overall_start
    logger.info(f"Pipeline completed for all {total_videos} video(s) in {elapsed:.1f}s")


if __name__ == "__main__":
    start_application()
