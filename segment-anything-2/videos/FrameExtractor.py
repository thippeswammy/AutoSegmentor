import os
import shutil
from concurrent.futures import ThreadPoolExecutor

import cv2
from tqdm import tqdm


class FrameExtractor:
    def __init__(self, video_number, prefixFileName='files', num_threads=os.cpu_count() - 1, limitedImages=None,
                 video_path_template=r'D:\downloadFiles\front_3\video{}.mp4',
                 output_dir=r'F:\RunningProjects\SAM2\segment-anything-2\videos\Images'):
        self.video_number = video_number
        self.output_dir = output_dir
        self.limitedImages = limitedImages
        self.num_threads = num_threads
        self.prefixFileName = prefixFileName
        self.video_path_template = video_path_template
        self.video_path = self.video_path_template.format(video_number)

    def clear_output_directory(self):
        """Clear the output directory if it exists."""
        if os.path.exists(self.output_dir):
            for filename in os.listdir(self.output_dir):
                file_path = os.path.join(self.output_dir, filename)
                try:
                    if os.path.isfile(file_path) or os.path.islink(file_path):
                        os.unlink(file_path)
                    elif os.path.isdir(file_path):
                        shutil.rmtree(file_path)
                except Exception as e:
                    print(f"Failed to delete {file_path}. Reason: {e}")

    def save_frame(self, frame, frame_count):
        """Save the individual frame as a .png file."""
        frame_filename = os.path.join(self.output_dir,
                                      f'{self.prefixFileName}{self.video_number}_{frame_count:05d}.png')
        cv2.imwrite(frame_filename, frame)

    def extract_frames_in_range(self, start_frame, end_frame, progress_bar):
        """Extract and save frames for a given range of frame numbers."""
        cap = cv2.VideoCapture(self.video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        for frame_count in range(start_frame, end_frame):
            ret, frame = cap.read()
            if not ret:
                break
            self.save_frame(frame, frame_count)
            progress_bar.update(1)

        cap.release()

    def process_video_in_parallel(self):
        """Process the video by extracting frames in parallel using multiple threads."""
        cap = cv2.VideoCapture(self.video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        # if not self.limitedImages:
        #     total_frames = min(total_frames, self.limitedImages)
        cap.release()
        frames_per_thread = int(total_frames // self.num_threads)
        frame_ranges = [
            (i * frames_per_thread,
             (i + 1) * frames_per_thread if i != self.num_threads - 1 else total_frames)
            for i in range(self.num_threads)
        ]
        with tqdm(total=total_frames, desc='Extracting Frames', unit='frame') as progress_bar:
            with ThreadPoolExecutor(max_workers=self.num_threads) as executor:
                for start_frame, end_frame in frame_ranges:
                    executor.submit(self.extract_frames_in_range, start_frame, end_frame, progress_bar)

        print(f"Extracted frames to '{self.output_dir}'")

    def run(self):
        """Main method to manage the video processing."""
        os.makedirs(self.output_dir, exist_ok=True)
        self.clear_output_directory()
        self.process_video_in_parallel()


if __name__ == "__main__":
    # Configuration
    VIDEO_NUMBER = 4
    VIDEO_PATH_TEMPLATE = r'D:\downloadFiles\front_3\video{}.mp4'
    OUTPUT_DIR = r'F:\RunningProjects\SAM2\segment-anything-2\videos\road_imgs'

    # Create an instance of FrameExtractor and run it
    extractor = FrameExtractor(VIDEO_NUMBER, VIDEO_PATH_TEMPLATE, OUTPUT_DIR)
    extractor.run()
