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


def start_application(config_override: dict = None):
    app = QApplication(sys.argv)

    # ── Configuration Loading ────────────────────────────────────────────────
    if config_override:
        cfg = config_override
        logger.info("Running in Automated Mode with config override.")
    else:
        # Show the Setup Dialog for standard interactive mode
        dialog = SetupDialog()
        if dialog.exec_() != SetupDialog.Accepted:
            logger.info("Setup cancelled by user. Exiting.")
            sys.exit(0)
        cfg = dialog.get_config()

    # ── Extract params ───────────────────────────────────────────────────────
    vi = cfg["video_inputs"]
    vo = cfg["video_outputs"]
    pl = cfg["pipeline"]
    mods = cfg["models"]

    video_start          = vi["start"]
    video_end            = vi["end"]
    images_ending_count  = vi.get("max_frames", 0)
    video_path_template  = vi["template"]

    final_video_path     = vo["final_path"]
    base_working_dir     = vo["working_dir"]
    prefix               = vo["prefix"]
    delete               = vo["delete_after"]
    
    run_mode             = pl["run_mode"]
    batch_size           = pl["batch_size"]
    fps                  = pl["fps"]
    auto_prompt_encoding = pl["auto_prompt"]
    review_from_start    = pl.get("review_from_start", False)

    sam_enabled          = mods["sam"]["enabled"]
    sam_config           = mods["sam"]
    pose_config          = mods["pose"]

    total_videos  = video_end
    overall_start = time.time()

    is_demo = (config_override is not None)

    for idx, i in enumerate(range(video_start, video_start + video_end), start=1):
        logger.info(f"{'═' * 20} Video {i} ({idx}/{total_videos}) {'═' * 20}")

        # Dynamic video-specific working directory
        # If template is a direct file (like in demo), we use the filename for the working dir
        if "{}" not in video_path_template:
            video_name = os.path.splitext(os.path.basename(video_path_template))[0]
            working_dir_name = os.path.join(base_working_dir, video_name).replace("\\", "/")
        else:
            working_dir_name = os.path.join(base_working_dir, f"video{i}").replace("\\", "/")
        
        images_extract_dir   = os.path.join(working_dir_name, "images").replace("\\", "/")
        temp_processing_dir  = os.path.join(working_dir_name, "temp").replace("\\", "/")
        rendered_dir         = os.path.join(working_dir_name, "render").replace("\\", "/")
        overlap_dir          = os.path.join(working_dir_name, "overlap").replace("\\", "/")
        verified_img_dir     = os.path.join(working_dir_name, "verified", "images").replace("\\", "/")
        verified_mask_dir    = os.path.join(working_dir_name, "verified", "mask").replace("\\", "/")

        if is_demo:
            # Clean up existing prompt file for this video in demo mode to ensure it starts fresh
            prompt_file = os.path.join("workspace", "inputs", "UserPrompts", f"points_labels_{prefix}{i}.json")
            if os.path.exists(prompt_file):
                try:
                    os.remove(prompt_file)
                    logger.info(f"Demo Mode: Removed existing prompt file {prompt_file} to ensure Batch 0 start.")
                except Exception as e:
                    logger.warning(f"Demo Mode: Failed to remove {prompt_file}: {e}")

        if run_mode != "pose_only":
            # In demo mode, we always clear the working directory without asking
            delete_mode = "yes" if is_demo else delete
            cleared = _handle_working_dir(
                working_dir_name,
                delete_mode,
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
            review_from_start=review_from_start if not is_demo else True, # Demo always runs from start
        )

        if not is_demo:
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
