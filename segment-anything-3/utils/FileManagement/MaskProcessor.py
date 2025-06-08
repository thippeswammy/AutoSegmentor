import os
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np


class MaskProcessor:
    """Processes masks and bounding boxes."""

    def __init__(self, config):
        self.config = config
        self.last_mask = None
        self.mask_box_points = {}
        self.image_counter = self.config.images_starting_count

    @staticmethod
    def mask2colorMaskImg(mask):
        """Convert mask to color image."""
        colors = np.array([
            [0, 0, 0], [0, 0, 255], [0, 255, 0], [255, 0, 0], [0, 255, 255],
            [255, 0, 255], [255, 255, 0], [128, 0, 128], [0, 165, 255], [255, 255, 255]
        ], dtype=np.uint8)
        max_valid_id = len(colors) - 1
        mask = np.clip(mask, 0, max_valid_id)
        return colors[mask]

    def mask_to_boxes(self, mask):
        """Convert mask to bounding boxes."""
        if mask is None or isinstance(mask, (tuple, list)) and mask in [(None,), [None]]:
            self.mask_box_points = None
            return None
        boxes = {}
        object_ids = np.unique(mask)
        object_ids = object_ids[object_ids != 0]
        for obj_id in object_ids:
            binary_mask = (mask == obj_id).astype(np.uint8)
            contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                continue
            largest_contour = max(contours, key=cv2.contourArea)
            if cv2.contourArea(largest_contour) < int(mask.shape[0] * mask.shape[1] * 0.00015):
                continue
            x, y, w, h = cv2.boundingRect(largest_contour)
            boxes[int(obj_id)] = [max(x, 0), max(y, 0), min(x + w, mask.shape[1]), min(y + h, mask.shape[0])]
        self.mask_box_points = boxes
        return boxes

    def binary_mask_2_color_mask(self, out_frame_idx, frame_filenames, video_segments, present_count, temp_directory,
                                 save=True):
        """Convert binary mask to color mask."""
        if save:
            frame_path = os.path.join(temp_directory, frame_filenames[out_frame_idx])
        else:
            frame_path = frame_filenames
        frame = cv2.imread(frame_path)
        full_mask = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint16)
        temp = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint16)
        for out_obj_id in sorted(video_segments[out_frame_idx].keys(), reverse=True):
            out_mask = video_segments[out_frame_idx][out_obj_id]
            if out_mask.dtype == np.bool_:
                out_mask = out_mask.astype(np.uint8)
            out_mask = out_mask.squeeze()
            if out_mask.shape[:2] != (frame.shape[0], frame.shape[1]):
                out_mask_resized = cv2.resize(out_mask, (frame.shape[1], frame.shape[0]),
                                              interpolation=cv2.INTER_NEAREST_EXACT)
            else:
                out_mask_resized = out_mask
            mask_condition = (out_mask_resized > 0) & (full_mask == 0)
            full_mask[mask_condition] = abs(out_obj_id // 1000)
            temp[mask_condition] = abs(out_obj_id)
        if save and out_frame_idx == len(video_segments) - 1:
            self.last_mask = temp.copy()
        color_mask_image = self.mask2colorMaskImg(full_mask)
        if save:
            cv2.imwrite(
                os.path.join(self.config.rendered_frames_dir,
                             f"{self.config.prefix}{self.config.video_number}_{present_count:05d}.png"),
                color_mask_image
            )
        else:
            return color_mask_image
        return present_count + 1

    def generate_mask(self, batch_number, sam2_predictor, temp_directory, prompt_encoding, auto_prompt_encoding):
        """Generate masks for a batch of frames."""
        frame_file_names = sorted(
            [p for p in os.listdir(temp_directory) if os.path.splitext(p)[-1].lower() in [".jpg", ".jpeg", ".png"]],
            key=lambda p: int(os.path.splitext(p)[0]) if p[:-4].isdigit() else float('inf')
        )
        inference_state = sam2_predictor.init_state(video_path=temp_directory, frame_paths=None)
        is_prompted = False
        if self.last_mask is None or isinstance(self.last_mask, (tuple, list)) and self.last_mask in [(None,), [None]]:
            pass
        else:
            is_prompted = auto_prompt_encoding(inference_state) is not None
        is_prompted = (prompt_encoding(inference_state, batch_number) is not None) or is_prompted
        if is_prompted:
            video_segments = {}
            for out_frame_idx, out_obj_ids, out_mask_logits in sam2_predictor.propagate_in_video(inference_state):
                video_segments[out_frame_idx] = {
                    out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
                    for i, out_obj_id in enumerate(out_obj_ids)
                }
            present_count = self.image_counter
            with ThreadPoolExecutor(max_workers=os.cpu_count() - 2) as executor:
                futures = [
                    executor.submit(self.binary_mask_2_color_mask, out_frame_idx, frame_file_names,
                                    video_segments, present_count + i, temp_directory)
                    for i, out_frame_idx in enumerate(sorted(video_segments.keys()))
                ]
                for future in futures:
                    present_count = max(present_count, future.result())
            self.image_counter = present_count
