import os
import shutil
import sys
import time

from .file_management.FileManager import ensure_directory
from .file_management.ImageCopier import ImageCopier
from .file_management.ImageOverlayProcessor import ImageOverlayProcessor
from .file_management.VideoCreator import VideoCreator
from .file_management.PoseExporter import PoseExporter
from .ui.logger_config import logger


def run_pipeline(video_number, video_path_template, images_extract_dir, rendered_dirs, overlap_dir,
                 verified_img_dir, verified_mask_dir, prefix, batch_size, fps, final_video_path,
                 temp_processing_dir, delete, images_ending_count, working_dir=None, pose_config=None,
                 run_mode="all", auto_prompt_encoding=True, sam_enabled=True, sam_config=None, review_from_start=False):
    """Run the pipeline for processing a video with various modes.
    
    This function orchestrates the entire video processing pipeline, allowing for
    different modes of operation such as full processing, pose export only, or mask
    generation. It initializes the necessary components, processes the video
    frames, and manages the output, including copying verified images and
    generating final video files. The function also includes user verification
    steps to ensure the quality of the overlays before finalizing the output.
    
    Args:
        video_number (int): The identifier for the video being processed.
        video_path_template (str): The template for the video file path.
        images_extract_dir (str): Directory where extracted images are stored.
        rendered_dirs (str): Directory containing rendered frames.
        overlap_dir (str): Directory for storing overlay images.
        verified_img_dir (str): Directory for verified images.
        verified_mask_dir (str): Directory for verified masks.
        prefix (str): Prefix for naming output files.
        batch_size (int): Number of images to process in a batch.
        fps (int): Frames per second for the output video.
        final_video_path (str): Path for saving the final video.
        temp_processing_dir (str): Temporary directory for processing files.
        delete (str): Flag indicating whether to delete original images after processing.
        images_ending_count (int): Count of images to consider for processing.
        pose_config (dict?): Configuration for pose processing.
        run_mode (str?): Mode of operation ("all", "mask_only", "pose_only").
        auto_prompt_encoding (bool?): Flag for automatic prompt encoding.
        sam_config (dict?): Configuration for SAM models.
        review_from_start (bool?): Flag to force UI to start at Frame 0 bypassing pre-processing.
    """
    pipeline_start = time.time()
    logger.info(f"Processing video {video_number} (mode: {run_mode})")

    if run_mode == "pose_only":
        # Skip SAM2 — run only pose export using existing verified data
        _run_pose_only(
            video_number=video_number,
            prefix=prefix,
            batch_size=batch_size,
            verified_mask_dir=verified_mask_dir,
            pose_config=pose_config,
            images_ending_count=images_ending_count,
            video_path_template=video_path_template,
            images_extract_dir=images_extract_dir,
            rendered_dirs=rendered_dirs,
            temp_processing_dir=temp_processing_dir,
            working_dir=working_dir,
            final_video_path=final_video_path,
        )
        logger.info(f"[Pipeline] Total elapsed: {time.time() - pipeline_start:.1f}s")
        return

    # Full pipeline (mode: "all" or "mask_only")
    from .ui.AnnotationManager import AnnotationManager
    from .ui.UITheme import ANNOTATION_COLORS_BGR

    t0 = time.time()
    # In demo mode, save prompts to the final_video_path (demo/) instead of workspace
    save_dir = final_video_path if (working_dir and "demo" in working_dir) else None
    
    processor = AutoSegmentorEngine(
        video_number=video_number,
        prefix=prefix,
        batch_size=batch_size,
        video_path_template=video_path_template,
        images_extract_dir=images_extract_dir,
        rendered_frames_dir=rendered_dirs,
        temp_processing_dir=temp_processing_dir,
        working_dir=working_dir,
        images_ending_count=images_ending_count,
        pose_config=pose_config,
        auto_prompt_encoding=auto_prompt_encoding,
        sam_enabled=sam_enabled,
        sam_config=sam_config,
        review_from_start=review_from_start,
        prompt_save_dir=save_dir # Pass the custom save dir
    )
    if not processor.run():
        logger.info("Pipeline terminated: Manual annotation session was discarded or cancelled.")
        return
    logger.info(f"[Pipeline] Engine processing completed in {time.time() - t0:.1f}s")

    if run_mode != "mask_only" and pose_config and pose_config.get('enabled'):
        t0 = time.time()
        exporter = PoseExporter(processor.config, verified_mask_dir, processor.annotation_manager)
        precomputed = getattr(processor, 'per_batch_tracked_data', None)
        exporter.process_masks(precomputed_tracking=precomputed if precomputed else None)
        logger.info(f"[Pipeline] Pose export completed in {time.time() - t0:.1f}s")

        # Copy pose labels to final output directory
        ensure_directory(final_video_path)
        dest_pose_file = os.path.join(final_video_path, f"pose_label_video{video_number}.json")
        try:
            shutil.copy2(exporter.output_file, dest_pose_file)
            logger.info(f"[Pipeline] Pose labels copied to: {dest_pose_file}")
        except Exception as e:
            logger.error(f"[Pipeline] Failed to copy pose labels: {e}")

    if sam_enabled:
        t0 = time.time()
        overlay_processor = ImageOverlayProcessor(
            original_folder=images_extract_dir,
            mask_folder=rendered_dirs,
            output_folder=overlap_dir,
            all_consider=prefix,
            image_count=0
        )
        overlay_processor.process_all_images()
        logger.info(f"[Pipeline] Overlay generation completed in {time.time() - t0:.1f}s")
    else:
        logger.info("[Pipeline] SAM disabled: skipping overlay generation")

    logger.info('═' * 60)
    if delete != 'yes':
        while True:
            user_input = input(
                "Have you verified all the overlay masks on original images? (yes/no): ").lower()
            if user_input == 'yes':
                break
            elif user_input == 'no':
                logger.info("Pipeline terminated: Verification not completed")
                sys.exit(0)

    t0 = time.time()
    logger.info(f"Copying verified images and masks (delete={delete})")
    copier = ImageCopier(
        original_folder=images_extract_dir,
        mask_folder=rendered_dirs,
        overlap_images_folder=overlap_dir,
        output_original_folder=verified_img_dir,
        output_mask_folder=verified_mask_dir
    )
    copier.copy_images(filter_by_overlap=sam_enabled)
    logger.info(f"[Pipeline] Image copy completed in {time.time() - t0:.1f}s")

    logger.info('═' * 60)
    t0 = time.time()
    ensure_directory(final_video_path)
    image_folders = [verified_img_dir]
    video_names = [f"{final_video_path}/OrgVideo{video_number}.mp4"]
    
    if sam_enabled:
        image_folders.extend([verified_mask_dir, overlap_dir])
        video_names.extend([
            f"{final_video_path}/MaskVideo{video_number}.mp4",
            f"{final_video_path}/OverlappedVideo{video_number}.mp4"
        ])
        
    video_creator = VideoCreator(
        image_folders=image_folders,
        video_names=video_names,
        fps=fps
    )
    video_creator.run()
    logger.info(f"[Pipeline] Video creation completed in {time.time() - t0:.1f}s")

    logger.info(f"[Pipeline] Total elapsed: {time.time() - pipeline_start:.1f}s")


def _run_pose_only(video_number, prefix, batch_size, verified_mask_dir,
                   pose_config, images_ending_count, video_path_template,
                   images_extract_dir, rendered_dirs, temp_processing_dir,
                   working_dir, final_video_path):
    """Run only the pose export step using existing verified data."""
    if not pose_config or not pose_config.get('enabled'):
        logger.error("Pose estimation is not enabled in config. Cannot run pose_only mode.")
        return

    from .models.SAM.AppConfig import AppConfig
    from .ui.AnnotationManager import AnnotationManager

    # Build a lightweight config (no SAM2 model needed)
    config = AppConfig(
        video_number=video_number,
        batch_size=batch_size,
        prefix=prefix,
        video_path_template=video_path_template,
        images_extract_dir=images_extract_dir,
        rendered_frames_dir=rendered_dirs,
        temp_processing_dir=temp_processing_dir,
        working_dir=working_dir,
        images_ending_count=images_ending_count,
        pose_config=pose_config,
    )

    # Load existing annotations
    verified_img_dir = verified_mask_dir.replace('mask', 'images')
    frame_paths = sorted([
        os.path.join(verified_img_dir, f)
        for f in os.listdir(verified_img_dir)
        if f.lower().endswith(('.png', '.jpg', '.jpeg'))
    ]) if os.path.exists(verified_img_dir) else []

    if not frame_paths:
        logger.error(f"No verified images found in {verified_img_dir}. Run full pipeline first.")
        return

    annotation_manager = AnnotationManager(config, frame_paths)
    logger.info(f"Pose-only mode: {len(frame_paths)} verified frames, "
                f"{len(annotation_manager.pose_keypoints_collection)} keypoint sets")

    exporter = PoseExporter(config, verified_mask_dir, annotation_manager)
    exporter.process_masks()

    # Copy pose labels to final output directory
    ensure_directory(final_video_path)
    dest_pose_file = os.path.join(final_video_path, f"pose_label_video{video_number}.json")
    try:
        shutil.copy2(exporter.output_file, dest_pose_file)
        logger.info(f"[Pipeline] Pose labels copied to: {dest_pose_file}")
    except Exception as e:
        logger.error(f"[Pipeline] Failed to copy pose labels: {e}")


