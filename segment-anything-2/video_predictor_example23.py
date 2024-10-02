# import os
# import shutil
# from contextlib import nullcontext
#
# import cv2
# import numpy as np
# import torch
# from PIL import Image
#
# # Initialize global variables
# ImageCount = 1
# batch_size = 30
# points = []  # To store clicked points
# labels = []  # To store corresponding labels (positive/negative)
#
#
# # Video frames directory
# video_dir = "videos/road_imgs"
#
# # Device selection and setup
# device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
# print(f"Using device: {device}")
#
# if device.type == "cuda":
#     torch.backends.cuda.matmul.allow_tf32 = True
#     torch.backends.cudnn.allow_tf32 = True
#     autocast_context = torch.autocast(device_type="cuda", dtype=torch.bfloat16)
# else:
#     autocast_context = nullcontext()
#
# # Load SAM 2 predictor
# from sam2.build_sam import build_sam2_video_predictor
#
# sam2_checkpoint = "../checkpoints/sam2_hiera_large.pt"
# model_cfg = "sam2_hiera_l.yaml"
# predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint, device=device)
#
#
# # Mouse callback function for interactive point collection
# def click_event(event, x, y, flags, param):
#     global points, labels
#     # Left-click adds a positive point
#     if event == cv2.EVENT_LBUTTONDOWN:
#         points.append([x, y])
#         labels.append(1)
#         print(f"Positive point: ({x}, {y})")
#     # Right-click adds a negative point
#     elif event == cv2.EVENT_RBUTTONDOWN:
#         points.append([x, y])
#         labels.append(0)
#         print(f"Negative point: ({x}, {y})")
#
#
# # Clear the directory before saving new frames
# def delete_all_files_in_dir(directory):
#     if os.path.exists(directory):
#         for filename in os.listdir(directory):
#             file_path = os.path.join(directory, filename)
#             try:
#                 if os.path.isfile(file_path) or os.path.islink(file_path):
#                     os.unlink(file_path)
#                 elif os.path.isdir(file_path):
#                     shutil.rmtree(file_path)
#             except Exception as e:
#                 print(f"Failed to delete {file_path}. Reason: {e}")
#         print(f"All files in {directory} have been deleted.")
#     else:
#         print(f"The directory {directory} does not exist.")
#
#
# # Function to move and copy frames
# def image_moves_copy(rep, frame_names_1, batch_size, video_dir="videos/Temp"):
#     images_pat = frame_names_1[rep:rep + batch_size]
#     delete_all_files_in_dir(video_dir)
#     for image_path in images_pat:
#         try:
#             if not os.path.exists(video_dir):
#                 os.makedirs(video_dir)
#             shutil.copy(image_path, video_dir)
#             # print(f"Copied {image_path} to {video_dir}")
#         except Exception as e:
#             print(f"Failed to copy {image_path}. Reason: {e}")
#
#
# # Main process function with interactive point collection
# def repeated(rep):
#     global batch_size, ImageCount, points, labels
#     video_dir = "videos/Temp"
#
#     # Prepare video frame names
#     frame_names = sorted(
#         [p for p in os.listdir(video_dir) if os.path.splitext(p)[-1].lower() in [".jpg", ".jpeg"]],
#         key=lambda p: int(os.path.splitext(p)[0])
#     )
#
#     # Initialize and reset inference state for segmentation
#     inference_state = predictor.init_state(video_path=video_dir)
#     predictor.reset_state(inference_state)
#
#     # Interactive point collection using OpenCV
#     for frame_name in frame_names[:1]:
#         image_path = os.path.join(video_dir, frame_name)
#         image = cv2.imread(image_path)
#
#         # Display the image and wait for user interaction
#         cv2.imshow(f"Frame {ImageCount}", image)
#         cv2.setMouseCallback(f"Frame {ImageCount}", click_event)
#
#         # Wait for the user to input points or press a key
#         # print("Press any key after selecting points (Left-click for positive, Right-click for negative).")
#         cv2.waitKey(0)  # Wait indefinitely until the user presses a key
#         cv2.destroyAllWindows()
#
#     # Convert collected points and labels to numpy arrays
#     points_np = np.array(points, dtype=np.float32)
#     labels_np = np.array(labels, np.int32)
#
#     ann_frame_idx = 0
#     ann_obj_id = 1
#
#     # Run the segmentation model
#     _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
#         inference_state=inference_state,
#         frame_idx=ann_frame_idx,
#         obj_id=ann_obj_id,
#         points=points_np,
#         labels=labels_np,
#     )
#
#     # Store the output segmentation results
#     video_segments = {}
#     rendered_dir = "./rendered_frames"
#     os.makedirs(rendered_dir, exist_ok=True)
#
#     for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
#         video_segments[out_frame_idx] = {
#             out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
#             for i, out_obj_id in enumerate(out_obj_ids)
#         }
#
#     # Save the rendered frames
#     with autocast_context:
#         for out_frame_idx in range(0, len(frame_names)):
#             image = Image.open(os.path.join(video_dir, frame_names[out_frame_idx]))
#             if out_frame_idx in video_segments:
#                 mask = video_segments[out_frame_idx][ann_obj_id]
#                 mask = np.squeeze(mask)
#                 if mask.ndim == 2:
#                     mask = np.stack([mask] * 3, axis=-1)
#                 mask = mask * 255
#                 mask_image = Image.fromarray(mask.astype(np.uint8))
#                 blended_image = Image.blend(image, mask_image, alpha=0.5)
#                 blended_image.save(os.path.join(rendered_dir, f"rendered_frame_{ImageCount:05d}.png"))
#             else:
#                 image.save(os.path.join(rendered_dir, f"rendered_frame_{ImageCount:05d}.png"))
#             ImageCount += 1
#     print(f"Rendering for batch {rep} completed!")
#
#
# frame_names_1 = sorted(
#     [os.path.join(video_dir, p) for p in os.listdir(video_dir) if
#      os.path.splitext(p)[-1].lower() in [".jpg", ".jpeg"]],
#     key=lambda p: int(os.path.splitext(os.path.basename(p))[0])
# )
#
# # Loop over the images in batches and process them
# for rep in range(0, len(frame_names_1), batch_size):
#     points = []
#     labels = []
#     image_moves_copy(rep, frame_names_1, batch_size)
#     repeated(rep)  # Interactive point collection and processing

import os
import shutil
from contextlib import nullcontext

import cv2
import numpy as np
import torch
from PIL import Image

# Initialize global variables
ImageCount = 1
batch_size = 30
points = []  # To store clicked points
labels = []  # To store corresponding labels (positive/negative)
current_image = None  # To hold the current image for drawing

# Video frames directory
video_dir = "videos/road_imgs"

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


# Mouse callback function for interactive point collection
def click_event(event, x, y, flags, param):
    global points, labels, current_image
    # Left-click adds a positive point
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append([x, y])
        labels.append(1)
        print(f"Positive point: ({x}, {y})")
        cv2.circle(current_image, (x, y), 5, (0, 255, 0), -1)  # Draw green circle

    # Right-click adds a negative point
    elif event == cv2.EVENT_RBUTTONDOWN:
        points.append([x, y])
        labels.append(0)
        print(f"Negative point: ({x}, {y})")
        cv2.circle(current_image, (x, y), 5, (0, 0, 255), -1)  # Draw red circle

    # Update the image display
    cv2.imshow(f"Frame {ImageCount}", current_image)


# Clear the directory before saving new frames
def delete_all_files_in_dir(directory):
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
        print(f"All files in {directory} have been deleted.")
    else:
        print(f"The directory {directory} does not exist.")


# Function to move and copy frames
def image_moves_copy(rep, frame_names_1, batch_size, video_dir="videos/Temp"):
    images_pat = frame_names_1[rep:rep + batch_size]
    delete_all_files_in_dir(video_dir)
    for image_path in images_pat:
        try:
            if not os.path.exists(video_dir):
                os.makedirs(video_dir)
            shutil.copy(image_path, video_dir)
        except Exception as e:
            print(f"Failed to copy {image_path}. Reason: {e}")


# Main process function with interactive point collection
def repeated(rep):
    global batch_size, ImageCount, points, labels, current_image
    video_dir = "videos/Temp"

    # Prepare video frame names
    frame_names = sorted(
        [p for p in os.listdir(video_dir) if os.path.splitext(p)[-1].lower() in [".jpg", ".jpeg"]],
        key=lambda p: int(os.path.splitext(p)[0])
    )

    # Initialize and reset inference state for segmentation
    inference_state = predictor.init_state(video_path=video_dir)
    predictor.reset_state(inference_state)

    # Interactive point collection using OpenCV
    for frame_name in frame_names[:1]:
        image_path = os.path.join(video_dir, frame_name)
        current_image = cv2.imread(image_path)  # Store the current image

        # Display the image and wait for user interaction
        cv2.imshow(f"Frame {ImageCount}", current_image)
        cv2.setMouseCallback(f"Frame {ImageCount}", click_event)

        # Wait for the user to input points or press the Enter key
        while True:
            key = cv2.waitKey(0)
            if key == 13:  # ASCII value of Enter key
                break

        cv2.destroyAllWindows()

    # Convert collected points and labels to numpy arrays
    points_np = np.array(points, dtype=np.float32)
    labels_np = np.array(labels, np.int32)

    ann_frame_idx = 0
    ann_obj_id = 1

    # Run the segmentation model
    _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
        inference_state=inference_state,
        frame_idx=ann_frame_idx,
        obj_id=ann_obj_id,
        points=points_np,
        labels=labels_np,
    )

    # Store the output segmentation results
    video_segments = {}
    rendered_dir = "./rendered_frames"
    os.makedirs(rendered_dir, exist_ok=True)

    for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
        video_segments[out_frame_idx] = {
            out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
            for i, out_obj_id in enumerate(out_obj_ids)
        }

    # Save the rendered frames
    with autocast_context:
        for out_frame_idx in range(0, len(frame_names)):
            image = Image.open(os.path.join(video_dir, frame_names[out_frame_idx]))
            if out_frame_idx in video_segments:
                mask = video_segments[out_frame_idx][ann_obj_id]
                mask = np.squeeze(mask)
                if mask.ndim == 2:
                    mask = np.stack([mask] * 3, axis=-1)
                mask = mask * 255
                mask_image = Image.fromarray(mask.astype(np.uint8))
                blended_image = Image.blend(image, mask_image, alpha=0.5)
                blended_image.save(os.path.join(rendered_dir, f"rendered_frame_{ImageCount:05d}.png"))
            else:
                image.save(os.path.join(rendered_dir, f"rendered_frame_{ImageCount:05d}.png"))
            ImageCount += 1
    print(f"Rendering for batch {rep} completed!")


# Get frame names and start processing in batches
frame_names_1 = sorted(
    [os.path.join(video_dir, p) for p in os.listdir(video_dir) if
     os.path.splitext(p)[-1].lower() in [".jpg", ".jpeg"]],
    key=lambda p: int(os.path.splitext(os.path.basename(p))[0])
)

# Loop over the images in batches and process them
for rep in range(0, len(frame_names_1), batch_size):
    points = []
    labels = []
    image_moves_copy(rep, frame_names_1, batch_size)
    repeated(rep)  # Interactive point collection and processing
