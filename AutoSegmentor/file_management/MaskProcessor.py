import os
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np

from ..ui.logger_config import logger


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
        """Convert mask to bounding boxes. Handles numpy arrays and CUDA tensors."""
        if mask is None or isinstance(mask, (tuple, list)) and mask in [(None,), [None]]:
            self.mask_box_points = None
            logger.debug("[MaskProc] mask_to_boxes: mask is None, skipping")
            return None
        # Guard: if mask is a CUDA tensor, move to CPU before numpy conversion
        try:
            import torch
            if isinstance(mask, torch.Tensor):
                mask = mask.cpu().numpy()
        except ImportError:
            pass
        boxes = {}
        object_ids = np.unique(mask)
        object_ids = object_ids[object_ids != 0]
        logger.debug(f"[MaskProc] mask_to_boxes: object_ids={object_ids.tolist()}")
        for obj_id in object_ids:
            binary_mask = (mask == obj_id).astype(np.uint8)
            contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                logger.debug(f"[MaskProc] mask_to_boxes: obj_id={obj_id} has no contours, skipping")
                continue
            largest_contour = max(contours, key=cv2.contourArea)
            if cv2.contourArea(largest_contour) < int(mask.shape[0] * mask.shape[1] * 0.00015):
                logger.debug(f"[MaskProc] mask_to_boxes: obj_id={obj_id} contour too small, skipping")
                continue
            x, y, w, h = cv2.boundingRect(largest_contour)
            boxes[int(obj_id)] = [max(x, 0), max(y, 0), min(x + w, mask.shape[1]), min(y + h, mask.shape[0])]
            logger.debug(f"[MaskProc] mask_to_boxes: obj_id={obj_id}  box={boxes[int(obj_id)]}")
        self.mask_box_points = boxes
        logger.debug(f"[MaskProc] mask_to_boxes: returning {len(boxes)} box(es)")
        return boxes

    def binary_mask_2_color_mask(self, out_frame_idx, frame_filenames, video_segments, present_count, temp_directory,
                                 save=True, last_frame_idx=None):
        """Convert binary mask to color mask.

        last_frame_idx, when given, is the batch's true last frame index
        (len(frame_file_names) - 1) — used to decide whether THIS frame is the
        one whose mask should carry forward as next batch's auto-prompt seed.
        Defaults to len(video_segments) - 1 for backward compatibility, which
        is only correct when video_segments covers the whole batch contiguously
        from frame 0 (a mid-batch refinement's video_segments does not).
        """
        if save:
            frame_path = os.path.join(temp_directory, frame_filenames[out_frame_idx])
        else:
            frame_path = frame_filenames
        logger.debug(f"[MaskProc] binary_mask_2_color_mask: frame_idx={out_frame_idx}  save={save}  path={frame_path}")
        frame = cv2.imread(frame_path)
        if frame is None:
            logger.error(f"Failed to read frame: {frame_path}")
            return (present_count + 1) if save else None
        logger.debug(f"[MaskProc] Frame read OK: shape={frame.shape}")
        
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
        true_last_idx = last_frame_idx if last_frame_idx is not None else len(video_segments) - 1
        if save and out_frame_idx == true_last_idx:
            self.last_mask = temp.copy()
        color_mask_image = self.mask2colorMaskImg(full_mask)
        if save:
            out_path = os.path.join(
                self.config.rendered_frames_dir,
                f"{self.config.prefix}{self.config.video_number}_{present_count:05d}.png"
            )
            logger.debug(f"[MaskProc] Saving color mask -> {out_path}")
            cv2.imwrite(out_path, color_mask_image)
        else:
            logger.debug(f"[MaskProc] Returning in-memory color mask (present_count={present_count})")
            return color_mask_image
        return present_count + 1

    def generate_mask(self, batch_number, sam2_predictor, temp_directory, prompt_encoding, auto_prompt_encoding, predictor_lock=None, starting_frame_idx=None, on_frame_done=None, query_frame_idx=None, backward_tracking=False):
        """Generate masks for a batch of frames.

        on_frame_done, if given, is called as on_frame_done(frames_done, total_frames)
        once per frame as SAM2's propagate_in_video loop below yields — used by the
        UI to show real per-frame progress for this stage.

        query_frame_idx/backward_tracking mirror the same params CoTracker's
        batch tracking already takes (AutoSegmentorEngine._track_batch_cotracker):
        when reprocessing starting mid-batch (not the batch's first frame), only
        propagate from that anchor frame onward (or backward), instead of always
        recomputing the whole batch from a fresh frame-0 box prompt — otherwise
        SAM2 silently redoes frames CoTracker considers already-settled on every
        mid-batch refinement, and the two trackers drift out of step over time.
        """
        logger.debug(f"[MaskProc] generate_mask: batch={batch_number}  temp_dir={temp_directory}  starting_frame_idx={starting_frame_idx}  query_frame_idx={query_frame_idx}  backward_tracking={backward_tracking}")
        frame_file_names = sorted(
            [p for p in os.listdir(temp_directory) if os.path.splitext(p)[-1].lower() in [".jpg", ".jpeg", ".png"]],
            key=lambda p: int(os.path.splitext(p)[0]) if p[:-4].isdigit() else float('inf')
        )
        logger.debug(f"[MaskProc] generate_mask: {len(frame_file_names)} frame(s) found in temp dir")
        
        # Helper to safely acquire lock if provided
        class DummyLock:
            def __enter__(self): pass
            def __exit__(self, *args): pass
        
        lock = predictor_lock if predictor_lock else DummyLock()

        total_frames = len(frame_file_names)
        batch_start = starting_frame_idx if starting_frame_idx is not None else 0
        anchor_rel = (query_frame_idx - batch_start) if query_frame_idx is not None else None
        is_refinement = anchor_rel is not None and 0 < anchor_rel < total_frames
        logger.debug(f"[MaskProc] generate_mask: anchor_rel={anchor_rel}  is_refinement={is_refinement}")

        with lock:
            if not frame_file_names:
                logger.error(f"[MaskGen] No frames found in {temp_directory}. Skipping batch {batch_number}.")
                return

            try:
                logger.debug(f"[MaskProc] Calling sam2_predictor.init_state for batch {batch_number}")
                inference_state = sam2_predictor.init_state(video_path=temp_directory, frame_paths=None)
                logger.debug(f"[MaskProc] init_state OK for batch {batch_number}")
            except Exception as e:
                logger.error(f"[MaskGen] Failed to initialize inference state: {e}")
                return

            is_prompted = False
            if is_refinement:
                # Refining from a mid-batch anchor: don't reseed frame 0 with a
                # fresh box prompt — propagation below never revisits frames
                # before the anchor anyway, so that box would just be wasted
                # work (and would wrongly become the auto_prompt_encoding basis
                # for a run that isn't actually reprocessing the whole batch).
                logger.debug("[MaskProc] is_refinement — skipping auto_prompt_encoding")
            elif self.last_mask is None or isinstance(self.last_mask, (tuple, list)) and self.last_mask in [(None,), [None]]:
                logger.debug(f"[MaskProc] last_mask is None — skipping auto_prompt_encoding")
            else:
                result = auto_prompt_encoding(inference_state)
                is_prompted = result is not None
                logger.debug(f"[MaskProc] auto_prompt_encoding result: is_prompted={is_prompted}")
            manual_result = prompt_encoding(inference_state, batch_number)
            is_prompted = (manual_result is not None) or is_prompted
            logger.debug(f"[MaskProc] After prompt_encoding: is_prompted={is_prompted}")

        if is_prompted:
            logger.debug(f"[MaskProc] Starting propagate_in_video for batch {batch_number}")
            video_segments = {}
            # Granular propagation: propagate one frame at a time if possible, or release lock between batches
            # SAM2 propagate_in_video is a generator, we can wrap each step
            if is_refinement:
                propagate_kwargs = {"start_frame_idx": anchor_rel, "reverse": backward_tracking}
                progress_total = (anchor_rel + 1) if backward_tracking else (total_frames - anchor_rel)
            else:
                propagate_kwargs = {}
                progress_total = total_frames
            for out_frame_idx, out_obj_ids, out_mask_logits in sam2_predictor.propagate_in_video(inference_state, **propagate_kwargs):
                with lock:
                    video_segments[out_frame_idx] = {
                        out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
                        for i, out_obj_id in enumerate(out_obj_ids)
                    }
                if on_frame_done:
                    on_frame_done(len(video_segments), progress_total)
            logger.debug(f"[MaskProc] propagate_in_video done: {len(video_segments)} segment(s) produced")

            with ThreadPoolExecutor(max_workers=os.cpu_count() - 2) as executor:
                futures = [
                    executor.submit(self.binary_mask_2_color_mask, out_frame_idx, frame_file_names,
                                    video_segments, batch_start + out_frame_idx, temp_directory,
                                    True, total_frames - 1)
                    for out_frame_idx in sorted(video_segments.keys())
                ]
                present_count = batch_start
                for future in futures:
                    present_count = max(present_count, future.result())
            self.image_counter = present_count
            logger.debug(f"[MaskProc] generate_mask complete for batch {batch_number}  image_counter={self.image_counter}")
        else:
            logger.debug(f"[MaskProc] generate_mask: is_prompted=False — no masks written for batch {batch_number}")
