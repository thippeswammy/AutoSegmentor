import argparse
import os
import shutil
import sys

from Utils.logger_config import logger
from Utils.pipeline import run_pipeline


def main():
    parser = argparse.ArgumentParser(description="Automated video processing pipeline.")
    parser.add_argument('--video_start', type=int, default=1, help='Starting video number (inclusive)')
    parser.add_argument('--video_end', type=int, default=1, help='Ending video number (exclusive)')
    parser.add_argument('--prefix', type=str, default='Img', help='Prefix for output filenames')
    parser.add_argument('--batch_size', type=int, default=5, help='Batch size for processing frames')
    parser.add_argument('--fps', type=int, default=30, help='Frames per second for output videos')
    parser.add_argument('--delete', type=str, choices=['yes', 'no'], default='yes',
                        help='Delete working directory without verification prompt (yes/no)')
    parser.add_argument('--working_dir_name', type=str, default='working_dir',
                        help='Base directory name for working directories')
    parser.add_argument('--video_path_template', type=str, default='./VideoInputs/Video{}.mp4',
                        help='Template path for video files, e.g., ./VideoInputs/Video{}.mp4')
    parser.add_argument('--images_extract_dir', type=str, default='./working_dir/images',
                        help='Directory to extract images')
    parser.add_argument('--temp_processing_dir', type=str, default='./working_dir/temp',
                        help='Directory for temporary processing images')
    parser.add_argument('--rendered_dir', type=str, default='./working_dir/render',
                        help='Directory for rendered mask outputs')
    parser.add_argument('--overlap_dir', type=str, default='./working_dir/overlap',
                        help='Directory for overlapped images')
    parser.add_argument('--verified_img_dir', type=str, default='./working_dir/verified/images',
                        help='Directory for verified original images')
    parser.add_argument('--verified_mask_dir', type=str, default='./working_dir/verified/mask',
                        help='Directory for verified mask images')
    parser.add_argument('--final_video_path', type=str, default='./outputs',
                        help='Directory to save output videos')
    parser.add_argument('--images_ending_count', type=int, default=15,
                        help='Number of images to process')

    args = parser.parse_args()

    for i in range(args.video_start, args.video_start + args.video_end):
        if os.path.exists(args.working_dir_name):
            if args.delete.lower() == 'yes':
                shutil.rmtree(args.working_dir_name)
                logger.info(f"Cleared working directory: {args.working_dir_name}")
            else:
                confirm = input(
                    f"Do you want to clear prev working directory '{args.working_dir_name}'? (yes/no): "
                ).lower()
                if confirm == 'yes':
                    shutil.rmtree(args.working_dir_name)
                    logger.info(f"Cleared working directory: {args.working_dir_name}")
                else:
                    logger.info(f"Working directory '{args.working_dir_name}' not deleted")
                    sys.exit(1000)

        run_pipeline(
            fps=args.fps,
            video_number=i,
            prefix=args.prefix,
            batch_size=args.batch_size,
            delete=args.delete.lower(),
            video_path_template=args.video_path_template.replace('working_dir', args.working_dir_name),
            images_extract_dir=args.images_extract_dir.replace('working_dir', args.working_dir_name),
            temp_processing_dir=args.temp_processing_dir.replace('working_dir', args.working_dir_name),
            rendered_dirs=args.rendered_dir.replace('working_dir', args.working_dir_name),
            overlap_dir=args.overlap_dir.replace('working_dir', args.working_dir_name),
            verified_img_dir=args.verified_img_dir.replace('working_dir', args.working_dir_name),
            verified_mask_dir=args.verified_mask_dir.replace('working_dir', args.working_dir_name),
            final_video_path=args.final_video_path,
            images_ending_count=args.images_ending_count
        )

        if os.path.exists(args.working_dir_name):
            if args.delete.lower() == 'yes':
                shutil.rmtree(args.working_dir_name)
                logger.info(f"Cleared working directory: {args.working_dir_name}")
            else:
                confirm = input(
                    f"Are you sure you want to delete the working directory '{args.working_dir_name}'? (yes/no): "
                ).lower()
                if confirm == 'yes':
                    shutil.rmtree(args.working_dir_name)
                    logger.info(f"Cleared working directory: {args.working_dir_name}")
                else:
                    logger.info(f"Working directory '{args.working_dir_name}' not deleted")

    logger.info("Pipeline completed for all videos.")


if __name__ == "__main__":
    main()
