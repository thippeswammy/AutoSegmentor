import os

import cv2
from tqdm import tqdm


class FrameExtractor:
    def __init__(self, video_number, prefixFileName="file", limitedImages=None, video_path_template=None,
                 output_dir=None):
        self.video_path = None
        self.video_number = video_number
        self.prefixFileName = prefixFileName
        self.limitedImages = limitedImages
        self.video_path_template = video_path_template
        self.output_dir = output_dir
        self.valid_extensions = (".jpg", ".jpeg", ".png")

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

    def run(self):
        """Extract frames from the video to the output directory."""
        if not self.video_path_template or not self.output_dir:
            raise ValueError("Video path template and output directory must be specified.")

        video_path = self.video_path_template.format(self.video_number)
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")

        os.makedirs(self.output_dir, exist_ok=True)
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video: {video_path}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_limit = self.limitedImages if self.limitedImages is not None else total_frames

        with tqdm(total=min(frame_limit, total_frames), desc="Extracting Frames") as pbar:
            frame_count = 0
            while cap.isOpened() and frame_count < frame_limit:
                ret, frame = cap.read()
                if not ret:
                    break
                output_path = os.path.join(self.output_dir,
                                           f"{self.prefixFileName}{self.video_number}_{frame_count:05d}.jpeg")
                cv2.imwrite(output_path, frame)
                frame_count += 1
                pbar.update(1)

        cap.release()
