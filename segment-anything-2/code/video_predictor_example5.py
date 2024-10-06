import os
from contextlib import nullcontext

import cv2
import numpy as np
import torch
from PIL import Image

ImageCount = 1
batch_size = 10  # Number of images to load at a time

# Device selection and setup
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

if device.type == "cuda":
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    autocast_context = torch.autocast(device_type="cuda", dtype=torch.bfloat16)
else:
    autocast_context = nullcontext()

# Load SAM 2 predictor
from sam2.build_sam import build_sam2_video_predictor

sam2_checkpoint = "../checkpoints/sam2_hiera_large.pt"
model_cfg = "sam2_hiera_l.yaml"
predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint, device=device)

# Initialization and state setup
video_dir = "../videos/Temp"
frame_names = sorted(
    [p for p in os.listdir(video_dir) if os.path.splitext(p)[-1].lower() in [".jpg", ".jpeg"]],
    key=lambda p: int(os.path.splitext(p)[0])
)
for RoundImg in range(0, len(frame_names), batch_size):
    inference_state = predictor.init_state(video_path=video_dir[RoundImg:RoundImg + batch_size])
    predictor.reset_state(inference_state)

    # Global variables for mouse events
    points = []
    labels = []


    def collect_points(event, x, y, flags, param):
        """
        Callback function to capture points and assign labels based on mouse clicks.
        """
        if event == cv2.EVENT_LBUTTONDOWN:
            points.append([x, y])
            labels.append(1)  # Positive click
            print(f"Positive point: ({x}, {y})")

        elif event == cv2.EVENT_RBUTTONDOWN:
            points.append([x, y])
            labels.append(0)  # Negative click
            print(f"Negative point: ({x}, {y})")


    # Setup OpenCV window and mouse callback for point collection
    cv2.namedWindow("Frame", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("Frame", collect_points)


    def process_batch(frame_batch):
        """
        Process a batch of frames by applying segmentation based on points collected from the first image.
        """
        global ImageCount
        video_segments = {}

        # Show the first image and collect points interactively
        frame_image = cv2.imread(frame_batch[0])
        print(f"Click on the image to mark positive (left-click) or negative (right-click) points for the batch.")
        points.clear()
        labels.clear()

        while True:
            cv2.imshow("Frame", frame_image)
            key = cv2.waitKey(1) & 0xFF
            if key == 13:  # Press Enter to confirm points
                break
            elif key == 27:  # Press Esc to exit without confirming
                print("Exiting without saving points.")
                points.clear()
                labels.clear()
                return  # Exit the batch processing

        # Convert points and labels to numpy arrays and add them to the state
        if points and labels:
            points_np = np.array(points, dtype=np.float32)
            labels_np = np.array(labels, np.int32)
            ann_obj_id = 1

            # Update inference state with points for the first frame
            _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
                inference_state=inference_state,
                frame_idx=0,
                obj_id=ann_obj_id,
                points=points_np,
                labels=labels_np,
            )

        points.clear()
        labels.clear()

        # Propagate the segmentation results across the batch
        for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
            video_segments[out_frame_idx] = {
                out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
                for i, out_obj_id in enumerate(out_obj_ids)
            }

        # Save the rendered frames
        rendered_dir = "../rendered_frames"
        os.makedirs(rendered_dir, exist_ok=True)

        with autocast_context:
            for idx, frame_path in enumerate(frame_batch):
                image = Image.open(frame_path)

                # Add segmentation mask to the frame
                if idx in video_segments:
                    mask = video_segments[idx][ann_obj_id]

                    mask = np.squeeze(mask)

                    if mask.ndim == 2:
                        mask = np.stack([mask] * 3, axis=-1)

                    mask = mask * 255

                    mask_image = Image.fromarray(mask.astype(np.uint8))

                    blended_image = Image.blend(image, mask_image, alpha=0.5)

                    blended_image.save(os.path.join(rendered_dir, f"rendered_frame_{ImageCount:05d}.png"))
                    ImageCount += 1
                else:
                    image.save(os.path.join(rendered_dir, f"rendered_frame_{ImageCount:05d}.png"))
                    ImageCount += 1

        print(f"Batch of {len(frame_batch)} frames processed and saved.")


    def process_frames_in_batches():
        """
        Load and process the frames in batches of the specified size.
        """
        total_frames = len(frame_names)

        for start_idx in range(0, total_frames, batch_size):
            # Get the next batch of frames
            end_idx = min(start_idx + batch_size, total_frames)
            frame_batch = [os.path.join(video_dir, frame_names[i]) for i in range(start_idx, end_idx)]

            print(f"Processing frames {start_idx + 1} to {end_idx}...")
            process_batch(frame_batch)

        print("All batches processed.")


    # Start batch processing
    process_frames_in_batches()
