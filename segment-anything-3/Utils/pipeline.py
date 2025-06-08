import sys

from Utils.FileManager import ensure_directory
from Utils.ImageCopier import ImageCopier
from Utils.ImageOverlayProcessor import ImageOverlayProcessor
from Utils.VideoCreator import VideoCreator
from Utils.logger_config import logger
from Utils.sam2_video_predictor import SAM2VideoProcessor


def run_pipeline(video_number, video_path_template, images_extract_dir, rendered_dirs, overlap_dir,
                 verified_img_dir, verified_mask_dir, prefix, batch_size, fps, final_video_path,
                 temp_processing_dir, delete, images_ending_count):
    """Run the entire pipeline for a single video number."""
    logger.info(f"Processing video {video_number}")

    processor = SAM2VideoProcessor(
        video_number=video_number,
        prefix=prefix,
        batch_size=batch_size,
        video_path_template=video_path_template,
        images_extract_dir=images_extract_dir,
        rendered_frames_dir=rendered_dirs,
        temp_processing_dir=temp_processing_dir,
        images_ending_count=images_ending_count
    )
    processor.run()

    overlay_processor = ImageOverlayProcessor(
        original_folder=images_extract_dir,
        mask_folder=rendered_dirs,
        output_folder=overlap_dir,
        all_consider=prefix,
        image_count=0
    )
    overlay_processor.process_all_images()

    if delete != 'yes':
        while True:
            user_input = input(
                "Have you verified all the overlay masks on original images? (yes/no): ").lower()
            if user_input == 'yes':
                break
            elif user_input == 'no':
                logger.info("Pipeline terminated: Verification not completed")
                sys.exit(0)

    logger.info(f"Copying verified images and masks (delete={delete})")
    copier = ImageCopier(
        original_folder=images_extract_dir,
        mask_folder=rendered_dirs,
        overlap_images_folder=overlap_dir,
        output_original_folder=verified_img_dir,
        output_mask_folder=verified_mask_dir
    )
    copier.copy_images()

    ensure_directory(final_video_path)
    video_names = [
        f"{final_video_path}/OrgVideo{video_number}.mp4",
        f"{final_video_path}/MaskVideo{video_number}.mp4",
        f"{final_video_path}/OverlappedVideo{video_number}.mp4"
    ]
    video_creator = VideoCreator(
        image_folders=[verified_img_dir, verified_mask_dir, overlap_dir],
        video_names=video_names,
        fps=fps
    )
    video_creator.run()
