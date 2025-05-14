import json
import os
import re
import shutil
import time
import cv2
import numpy as np
import torch
from sam2.build_sam import build_sam2_video_predictor
from videos.FrameExtractor import FrameExtractor


class VideoFrameProcessor:
    def __init__(self, video_number, batch_size=80, images_starting_count=0, images_ending_count=None,
                 prefixFileName="file", video_path_template=None, images_extract_dir=None,
                 road_mask_dir=None, lane_mask_dir=None):
        if video_path_template is None:
            print("Missing video file paths or video")
            exit(100)
        self.device = self.get_device()
        self.batch_size = batch_size
        self.video_number = video_number
        self.prefixFileName = prefixFileName
        self.video_path_template = video_path_template
        self.road_mask_dir = road_mask_dir or f'./videos/outputs/road_masks'
        self.lane_mask_dir = lane_mask_dir or f'./videos/outputs/lane_masks'
        self.temp_directory = "videos/Temp"
        self.model_config = "sam2_hiera_l.yaml"
        self.frames_directory = "videos/Images"
        self.sam2_checkpoint = "../checkpoints/sam2_hiera_large.pt"
        self.sam2_predictor = self.build_predictor()
        extractor = FrameExtractor(self.video_number, prefixFileName=self.prefixFileName,
                                   limitedImages=images_ending_count, video_path_template=self.video_path_template,
                                   output_dir=images_extract_dir)
        extractor.run()
        self.image_counter = images_starting_count
        self.frame_paths = self.get_frame_paths()
        self.label_colors = {
            0: (0, 0, 0),  # Background
            1: (0, 0, 255),  # Road
            2: (0, 255, 0)  # Lane
        }

    def get_device(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")
        if device.type == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        return device

    def build_predictor(self):
        return build_sam2_video_predictor(
            self.model_config, self.sam2_checkpoint, device=self.device
        )

    def get_frame_paths(self):
        return sorted(
            [os.path.join(self.frames_directory, p) for p in os.listdir(self.frames_directory)
             if os.path.splitext(p)[-1].lower() in [".jpg", ".jpeg", ".png"]],
            key=lambda p: int(re.search(r'(\d+)', os.path.splitext(p)[0]).group())
            if re.search(r'(\d+)', os.path.splitext(p)[0]) else float('inf')
        )

    def clear_directory(self, directory):
        if os.path.exists(directory):
            for filename in os.listdir(directory):
                file_path = os.path.join(directory, filename)
                try:
                    if os.path.isfile(file_path) or os.path.islink(file_path):
                        os.unlink(file_path)
                    elif os.path.isdir(file_path):
                        shutil.rmtree(file_path)
                except Exception as e:
                    print(f"Failed to delete {file_path}. Reason: {e}")
        else:
            os.makedirs(directory, exist_ok=True)

    def move_and_copy_frames(self, batch_index):
        frames_to_copy = self.frame_paths[batch_index:batch_index + self.batch_size]
        self.clear_directory(self.temp_directory)
        for frame_path in frames_to_copy:
            if not os.path.exists(frame_path):
                print(f"Frame not found: {frame_path}")
                continue
            os.makedirs(self.temp_directory, exist_ok=True)
            dest_path = os.path.join(self.temp_directory, os.path.basename(frame_path))
            shutil.copy(frame_path, dest_path)  # Use copy instead of symlink for reliability

    def mask2colorMaskImg(self, mask):
        colors = np.array([
            [0, 0, 0],  # Background
            [0, 0, 255],  # Road
            [0, 255, 0]  # Lane
        ], dtype=np.uint8)
        return colors[mask]

    def auto_select_points(self, frame, prev_road_mask=None):
        """Automatically select points for road and lane using image processing."""
        points = []
        labels = []

        # Convert to grayscale and apply edge detection for lane detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 100, 200)

        # Detect lines (lanes) using Hough Transform
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=100, minLineLength=100, maxLineGap=10)
        if lines is not None:
            for line in lines[:2]:  # Limit to 2 lanes
                x1, y1, x2, y2 = line[0]
                # Sample points along the line
                points.append([(x1 + x2) / 2, (y1 + y2) / 2])
                labels.append(2)  # Lane label

        # For road: Use previous road mask or sample center of frame
        if prev_road_mask is not None:
            # Find centroid of road mask
            moments = cv2.moments(prev_road_mask.astype(np.uint8))
            if moments['m00'] != 0:
                cx = int(moments['m10'] / moments['m00'])
                cy = int(moments['m01'] / moments['m00'])
                points.append([cx, cy])
                labels.append(1)  # Road label
        else:
            # Default to frame center
            h, w = frame.shape[:2]
            points.append([w // 2, h // 2])
            labels.append(1)  # Road label

        return np.array(points, dtype=np.float32), np.array(labels, np.int32)

    def process_batch(self, batch_number, points_list, labels_list):
        frame_filenames = sorted(
            [p for p in os.listdir(self.temp_directory) if
             os.path.splitext(p)[-1].lower() in [".jpg", ".jpeg", ".png"]],
            key=lambda p: int(re.search(r"(\d+)", os.path.splitext(p)[0]).group())
            if re.search(r"(\d+)", os.path.splitext(p)[0]) else float("inf")
        )
        if not frame_filenames:
            print(f"No frames found in {self.temp_directory}")
            return
        inference_state = self.sam2_predictor.init_state(video_path=self.temp_directory)
        self.sam2_predictor.reset_state(inference_state)
        points_np = np.array(points_list, dtype=np.float32)
        labels_np = np.array(labels_list, np.int32)

        os.makedirs(self.road_mask_dir, exist_ok=True)
        os.makedirs(self.lane_mask_dir, exist_ok=True)
        ann_frame_idx = 0
        for ann_obj_id in set(abs(labels_np)):
            labels_np1 = labels_np.copy()
            labels_np1[labels_np1 == ann_obj_id] = labels_np1[labels_np1 == ann_obj_id] // ann_obj_id
            labels_np1[labels_np1 < 0] = 0
            labels_np1 = labels_np1[abs(labels_np) == ann_obj_id]
            points_np1 = points_np[abs(labels_np) == ann_obj_id]
            _, object_ids, mask_logits = self.sam2_predictor.add_new_points_or_box(
                inference_state=inference_state,
                frame_idx=ann_frame_idx,
                clear_old_points=False,
                obj_id=int(ann_obj_id),
                points=points_np1,
                labels=labels_np1
            )

        video_segments = {}
        for out_frame_idx, out_obj_ids, out_mask_logits in self.sam2_predictor.propagate_in_video(inference_state):
            video_segments[out_frame_idx] = {
                out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
                for i, out_obj_id in enumerate(out_obj_ids)
            }

        present_count = self.image_counter
        kernel = np.ones((5, 5), np.uint8)
        for out_frame_idx in range(len(frame_filenames)):
            frame_path = os.path.join(self.temp_directory, frame_filenames[out_frame_idx])
            frame = cv2.imread(frame_path)
            road_mask = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)
            lane_mask = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)

            for out_obj_id, out_mask in video_segments[out_frame_idx].items():
                out_mask_resized = cv2.resize(
                    out_mask.squeeze().astype(np.uint8), (frame.shape[1], frame.shape[0])
                )
                if out_obj_id == 1:
                    road_mask[out_mask_resized > 0] = 1
                elif out_obj_id == 2:
                    lane_mask[out_mask_resized > 0] = 2
                    road_mask[out_mask_resized > 0] = 0  # Ensure lanes override road pixels

            # Post-process lane mask
            lane_mask = cv2.morphologyEx(lane_mask, cv2.MORPH_CLOSE, kernel)

            # Save masks
            road_color_mask = self.mask2colorMaskImg(road_mask)
            lane_color_mask = self.mask2colorMaskImg(lane_mask)
            cv2.imwrite(
                os.path.join(self.road_mask_dir, f"{self.prefixFileName}{self.video_number}_{present_count:05d}.png"),
                road_color_mask
            )
            cv2.imwrite(
                os.path.join(self.lane_mask_dir, f"{self.prefixFileName}{self.video_number}_{present_count:05d}.png"),
                lane_color_mask
            )
            present_count += 1
        self.image_counter = present_count

    def run(self):
        batch_index = 0
        prev_road_mask = None
        while batch_index < len(self.frame_paths):
            print(
                f"Processing batch {(batch_index // self.batch_size) + 1}/{(len(self.frame_paths) // self.batch_size) + 1}")
            self.move_and_copy_frames(batch_index)

            # Load first frame of batch to select points
            first_frame_path = self.frame_paths[batch_index]
            first_frame = cv2.imread(first_frame_path)
            points, labels = self.auto_select_points(first_frame, prev_road_mask)

            # Process batch with selected points
            self.process_batch(batch_index // self.batch_size, points, labels)

            # Update previous road mask for next batch
            prev_road_mask_path = os.path.join(
                self.road_mask_dir, f"{self.prefixFileName}{self.video_number}_{(self.image_counter - 1):05d}.png"
            )
            if os.path.exists(prev_road_mask_path):
                prev_road_mask = cv2.imread(prev_road_mask_path, cv2.IMREAD_GRAYSCALE)

            batch_index += self.batch_size
            print('-' * 28, "completed", '-' * 28)
        self.clear_directory(self.temp_directory)


if __name__ == "__main__":
    for i in range(6, 7):
        video_path_template = r'D:\downloadFiles\front_3\Video{}.mp4'
        images_extract_dir = r'/segment-anything-3/videos/Images'
        processor = VideoFrameProcessor(
            video_number=i,
            prefixFileName='roadW',
            batch_size=80,
            video_path_template=video_path_template,
            images_extract_dir=images_extract_dir,
            road_mask_dir='./videos/outputs/road_masks',
            lane_mask_dir='./videos/outputs/lane_masks'
        )
        processor.run()