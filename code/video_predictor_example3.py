import os
import re
from contextlib import nullcontext

import cv2
import numpy as np
import torch
from PIL import Image

ImageCount = 1
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
video_dir = "../segment-anything-3/videos/Temp"
frame_names = sorted(
    [p for p in os.listdir(video_dir) if
     os.path.splitext(p)[-1].lower() in [".jpg", ".jpeg", '.png']],
    key=lambda p: int(re.search(r'(\d+)', os.path.splitext(p)[0]).group())
    if re.search(r'(\d+)', os.path.splitext(p)[0]) else float('inf')
)
print(video_dir)
inference_state = predictor.init_state(video_path=video_dir)
predictor.reset_state(inference_state)

# Global variables for mouse events
points = []
labels = []


def collect_points(event, x, y, flags, param):
    """
    Callback function to capture points and assign labels based on mouse clicks.
    """
    if event == cv2.EVENT_LBUTTONDOWN:
        # Left-click -> Positive point
        points.append([x, y])
        labels.append(1)  # Positive click
        print(f"Positive point: ({x}, {y})")

    elif event == cv2.EVENT_RBUTTONDOWN:
        # Right-click -> Negative point
        points.append([x, y])
        labels.append(0)  # Negative click
        print(f"Negative point: ({x}, {y})")


# Setup OpenCV window and mouse callback for point collection
cv2.namedWindow("Frame")
cv2.setMouseCallback("Frame", collect_points)

# Load the first frame for point collection
ann_frame_idx = 0

frame_path = os.path.join(video_dir, frame_names[ann_frame_idx])
frame_image = cv2.imread(frame_path)

# Show the image and collect points interactively
print("Click on the image to mark positive (left-click) or negative (right-click) points.")
while True:
    cv2.imshow("Frame", frame_image)
    key = cv2.waitKey(1) & 0xFF
    if key == 13:  # Press Enter to confirm points
        break
    elif key == 27:  # Press Esc to exit without confirming
        print("Exiting without saving points.")
        points.clear()
        labels.clear()
        break

cv2.destroyAllWindows()

# Convert points and labels to numpy arrays
if points and labels:
    points = np.array(points, dtype=np.float32)
    labels = np.array(labels, np.int32)
    ann_obj_id = 1

    _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
        inference_state=inference_state,
        frame_idx=ann_frame_idx,
        obj_id=ann_obj_id,
        points=points[:3],
        labels=labels[:3],
    )
    _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
        inference_state=inference_state,
        frame_idx=ann_frame_idx,
        obj_id=2,
        clear_old_points=False,
        points=points[3:],
        labels=labels[3:],
    )
    print(out_obj_ids, out_mask_logits)
    # Run propagation and save frames
    video_segments = {}
    rendered_dir = "rendered_frames"
    os.makedirs(rendered_dir, exist_ok=True)

    for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
        video_segments[out_frame_idx] = {
            out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
            for i, out_obj_id in enumerate(out_obj_ids)
        }

    # Efficient frame saving using PIL
    with autocast_context:
        for out_frame_idx in range(0, len(frame_names)):
            # Load the original frame
            image = Image.open(os.path.join(video_dir, frame_names[out_frame_idx]))

            # Add segmentation mask to the frame
            if out_frame_idx in video_segments:

                mask = video_segments[out_frame_idx][ann_obj_id]

                # Remove redundant dimensions (squeeze the mask)
                mask = np.squeeze(mask)

                # Ensure the mask is 2D (if single channel) or convert it appropriately for RGB
                if mask.ndim == 2:
                    mask = np.stack([mask] * 3, axis=-1)  # Convert to 3-channel format if it's grayscale

                mask = mask * 255  # Scale the mask to 0-255 for display

                # Convert mask to an Image
                mask_image = Image.fromarray(mask.astype(np.uint8))

                # Blend the original image with the mask
                blended_image = Image.blend(image, mask_image, alpha=0.5)

                # Save the blended image
                blended_image.save(os.path.join(rendered_dir, f"rendered_frame_{ImageCount:05d}.png"))
                ImageCount = +1
            else:
                # If no mask is available for this frame, save the original frame
                image.save(os.path.join(rendered_dir, f"rendered_frame_{ImageCount:05d}.png"))
                ImageCount = +1
    print("Rendering completed!")
else:
    print("No points were selected. Exiting.")
