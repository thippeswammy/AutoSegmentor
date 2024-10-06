# import os
# from contextlib import nullcontext
#
# import numpy as np
# import torch
# from PIL import Image
#
# ImageCount = 1
# batch_size = 10
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
# def delete_all_files_in_dir(directory):
#     """
#     Deletes all files and subdirectories in the specified directory.
#
#     Args:
#     directory (str): Path to the directory where files and subdirectories should be deleted.
#     """
#     # Ensure the directory exists
#     if os.path.exists(directory):
#         # Loop through all files and directories in the directory
#         for filename in os.listdir(directory):
#             file_path = os.path.join(directory, filename)
#
#             try:
#                 if os.path.isfile(file_path) or os.path.islink(file_path):
#                     os.unlink(file_path)  # Remove file or symbolic link
#                 elif os.path.isdir(file_path):
#                     shutil.rmtree(file_path)  # Remove directories and their contents
#             except Exception as e:
#                 print(f"Failed to delete {file_path}. Reason: {e}")
#         print(f"All files and subdirectories in {directory} have been deleted.")
#     else:
#         print(f"The directory {directory} does not exist.")
#
#
# def image_moves_copy(rep):
#     images_pat = frame_names_1[rep:rep + batch_size]
#     video_dir = "videos/bedroom"
#     delete_all_files_in_dir(video_dir)
#
#     pass
#
#
# def repeated():
#     global batch_size, ImageCount
#     # Initialization and state setup
#     video_dir = "videos/bedroom"
#     frame_names = sorted(
#         [p for p in os.listdir(video_dir) if os.path.splitext(p)[-1].lower() in [".jpg", ".jpeg"]],
#         key=lambda p: int(os.path.splitext(p)[0])
#     )
#     inference_state = predictor.init_state(video_path=video_dir)
#     predictor.reset_state(inference_state)
#
#     # Example segmentation points
#     points = np.array([[210, 350], [250, 220]], dtype=np.float32)
#     # for labels, `1` means positive click and `0` means negative click
#     labels = np.array([1, 1], np.int32)
#     ann_frame_idx = 0
#     ann_obj_id = 1
#
#     _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
#         inference_state=inference_state,
#         frame_idx=ann_frame_idx,
#         obj_id=ann_obj_id,
#         points=points,
#         labels=labels,
#     )
#
#     # Run propagation and save frames
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
#     # Efficient frame saving using PIL
#     with autocast_context:
#         for out_frame_idx in range(0, len(frame_names)):
#             # Load the original frame
#             image = Image.open(os.path.join(video_dir, frame_names[out_frame_idx]))
#
#             # Add segmentation mask to the frame
#             if out_frame_idx in video_segments:
#                 mask = video_segments[out_frame_idx][ann_obj_id]
#
#                 # Remove redundant dimensions (squeeze the mask)
#                 mask = np.squeeze(mask)
#
#                 # Ensure the mask is 2D (if single channel) or convert it appropriately for RGB
#                 if mask.ndim == 2:
#                     mask = np.stack([mask] * 3, axis=-1)  # Convert to 3-channel format if it's grayscale
#
#                 mask = mask * 255  # Scale the mask to 0-255 for display
#
#                 # Convert mask to an Image
#                 mask_image = Image.fromarray(mask.astype(np.uint8))
#
#                 # Blend the original image with the mask
#                 blended_image = Image.blend(image, mask_image, alpha=0.5)
#
#                 # Save the blended image
#                 blended_image.save(os.path.join(rendered_dir, f"rendered_frame_{ImageCount:05d}.png"))
#                 ImageCount = +1
#             else:
#                 # If no mask is available for this frame, save the original frame
#                 image.save(os.path.join(rendered_dir, f"rendered_frame_{ImageCount:05d}.png"))
#                 ImageCount = +1
#
#     print("Rendering completed!")
#
#
# frame_names_1 = sorted(
#     [p for p in os.listdir(video_dir) if os.path.splitext(p)[-1].lower() in [".jpg", ".jpeg"]],
#     key=lambda p: int(os.path.splitext(p)[0])
# )
#
# for rep in range(0, len(frame_names_1), batch_size):
#     repeated()
#     image_moves_copy(rep)

import os
import shutil
from contextlib import nullcontext

import numpy as np
import torch
from PIL import Image

ImageCount = 1
batch_size = 10
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


def delete_all_files_in_dir(directory):
    """
    Deletes all files and subdirectories in the specified directory.
    """
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


def image_moves_copy(rep, frame_names_1, batch_size, video_dir="videos/bedroom"):
    """
    Copies a batch of images from `frame_names_1` to `video_dir` after clearing `video_dir`.
    """
    # Get the batch of image paths
    images_pat = frame_names_1[rep:rep + batch_size]

    # Clear the target directory
    delete_all_files_in_dir(video_dir)

    # Copy each image in the batch to the destination directory
    for image_path in images_pat:
        try:
            # Ensure destination directory exists
            if not os.path.exists(video_dir):
                os.makedirs(video_dir)

            shutil.copy(image_path, video_dir)
            print(f"Copied {image_path} to {video_dir}")
        except Exception as e:
            print(f"Failed to copy {image_path}. Reason: {e}")


def repeated(rep):
    global batch_size, ImageCount
    video_dir = "../videos/Temp"

    # Prepare video frame names in sorted order
    frame_names = sorted(
        [p for p in os.listdir(video_dir) if os.path.splitext(p)[-1].lower() in [".jpg", ".jpeg"]],
        key=lambda p: int(os.path.splitext(p)[0])
    )

    # Initialize and reset inference state for segmentation
    inference_state = predictor.init_state(video_path=video_dir)
    predictor.reset_state(inference_state)

    points = np.array([[210, 350], [250, 220]], dtype=np.float32)
    labels = np.array([1, 1], np.int32)
    ann_frame_idx = 0
    ann_obj_id = 1

    _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
        inference_state=inference_state,
        frame_idx=ann_frame_idx,
        obj_id=ann_obj_id,
        points=points,
        labels=labels,
    )

    video_segments = {}
    rendered_dir = "../rendered_frames"
    os.makedirs(rendered_dir, exist_ok=True)

    # Process video frames
    for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
        video_segments[out_frame_idx] = {
            out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
            for i, out_obj_id in enumerate(out_obj_ids)
        }

    # Save rendered frames
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


video_dir = "videos/full_set_images"
frame_names_1 = sorted(
    [os.path.join(video_dir, p) for p in os.listdir(video_dir) if
     os.path.splitext(p)[-1].lower() in [".jpg", ".jpeg"]],
    key=lambda p: int(os.path.splitext(os.path.basename(p))[0])  # Use os.path.basename to extract filename for sorting
)

# Loop over the images in batches
for rep in range(0, len(frame_names_1), batch_size):
    image_moves_copy(rep, frame_names_1, batch_size)  # Move images to video_dir
    repeated(rep)  # Process and render images in batches
