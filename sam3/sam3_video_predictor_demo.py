import os
import shutil
import sys
import time
sys.path.append(os.path.dirname(__file__))
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import yaml

from utils.UserUI.logger_config import logger
from utils.pipeline import run_pipeline


def load_config(config_path=os.path.join(os.path.dirname(__file__), "inputs/config/default_config.yaml")):
    """
    Load configuration from a YAML file and validate required keys.

    Args:
        config_path (str): Path to the YAML configuration file (default: config.yml).

    Returns:
        dict: Configuration parameters.

    Raises:
        FileNotFoundError: If the config file does not exist.
        Yaml.YAMLError: If the YAML file is invalid.
        KeyError: If required, configuration keys are missing.
        ValueError: If the 'delete' option is invalid.
    """
    try:
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)

        # Required configuration keys
        required_keys = [
            'video_start', 'video_end', 'prefix', 'batch_size', 'fps', 'delete',
            'working_dir_name', 'video_path_template', 'images_extract_dir',
            'temp_processing_dir', 'rendered_dir', 'overlap_dir',
            'verified_img_dir', 'verified_mask_dir', 'final_video_path',
            'images_ending_count', 'pose_estimation'
        ]

        # Check for missing keys
        missing_keys = [key for key in required_keys if key not in config]
        if missing_keys:
            logger.error(f"Missing required configuration keys in {config_path}: {', '.join(missing_keys)}")
            sys.exit(1)

        # Handle delete option (string or boolean)
        if isinstance(config['delete'], bool):
            config['delete'] = 'yes' if config['delete'] else 'no'
        elif isinstance(config['delete'], str):
            config['delete'] = config['delete'].lower()
        else:
            logger.error("Invalid 'delete' option in config. Must be 'yes', 'no', true, or false.")
            sys.exit(1)

        # Validate delete option
        if config['delete'] not in ['yes', 'no']:
            logger.error("Invalid 'delete' option in config. Must be 'yes', 'no', true, or false.")
            sys.exit(1)

        return config
    except FileNotFoundError:
        logger.error(f"Configuration file '{config_path}' not found.")
        sys.exit(1)
    except yaml.YAMLError as e:
        logger.error(f"Error parsing YAML file: {e}")
        sys.exit(1)


def _handle_working_dir(working_dir_name, delete, prompt_msg):
    """Handle working directory cleanup (auto-delete or prompt user).

    Returns True if directory was cleared or didn't exist, False if user declined.
    """
    if not os.path.exists(working_dir_name):
        return True

    if delete == 'yes':
        shutil.rmtree(working_dir_name)
        logger.info(f"Cleared working directory: {working_dir_name}")
        return True

    confirm = input(prompt_msg).lower()
    if confirm == 'yes':
        shutil.rmtree(working_dir_name)
        logger.info(f"Cleared working directory: {working_dir_name}")
        return True
    else:
        logger.info(f"Working directory '{working_dir_name}' not deleted")
        return False


def main():
    # Load configuration from YAML file
    config = load_config()

    # Extract configuration parameters
    video_start = config['video_start']
    video_end = config['video_end']
    prefix = config['prefix']
    batch_size = config['batch_size']
    fps = config['fps']
    delete = config['delete']  # Already lowercase string ('yes' or 'no')
    working_dir_name = config['working_dir_name']
    video_path_template = config['video_path_template']
    images_extract_dir = config['images_extract_dir']
    temp_processing_dir = config['temp_processing_dir']
    rendered_dir = config['rendered_dir']
    overlap_dir = config['overlap_dir']
    verified_img_dir = config['verified_img_dir']
    verified_mask_dir = config['verified_mask_dir']
    final_video_path = config['final_video_path']
    images_ending_count = config['images_ending_count']
    pose_config = config.get('pose_estimation', None)
    run_mode = config.get('run_mode', 'all').lower()
    auto_prompt_encoding = config.get('auto_prompt_encoding', True)

    total_videos = video_end
    overall_start = time.time()

    for idx, i in enumerate(range(video_start, video_start + video_end), start=1):
        logger.info(f"{'═' * 20} Video {i} ({idx}/{total_videos}) {'═' * 20}")

        if run_mode != 'pose_only':
            cleared = _handle_working_dir(
                working_dir_name, delete,
                f"Do you want to clear prev working directory '{working_dir_name}'? (yes/no): "
            )
            if not cleared:
                sys.exit(1000)

        run_pipeline(
            fps=fps,
            video_number=i,
            prefix=prefix,
            batch_size=batch_size,
            delete=delete,
            video_path_template=video_path_template.replace('working_dir', working_dir_name),
            images_extract_dir=images_extract_dir.replace('working_dir', working_dir_name),
            temp_processing_dir=temp_processing_dir.replace('working_dir', working_dir_name),
            rendered_dirs=rendered_dir.replace('working_dir', working_dir_name),
            overlap_dir=overlap_dir.replace('working_dir', working_dir_name),
            verified_img_dir=verified_img_dir.replace('working_dir', working_dir_name),
            verified_mask_dir=verified_mask_dir.replace('working_dir', working_dir_name),
            final_video_path=final_video_path,
            images_ending_count=images_ending_count,
            pose_config=pose_config,
            run_mode=run_mode,
            auto_prompt_encoding=auto_prompt_encoding
        )

        _handle_working_dir(
            working_dir_name, delete,
            f"Are you sure you want to delete the working directory '{working_dir_name}'? (yes/no): "
        )

        logger.info('═' * 60)

    elapsed = time.time() - overall_start
    logger.info(f"Pipeline completed for all {total_videos} video(s) in {elapsed:.1f}s")


if __name__ == "__main__":
    main()

