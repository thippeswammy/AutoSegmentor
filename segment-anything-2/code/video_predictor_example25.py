import os
import shutil
import threading
import time
from contextlib import nullcontext

import cv2
import numpy as np
import torch
from PIL import Image

# Initialize global variables
image_counter = 1
batch_size = 50

points_collection_list = []
labels_collection_list = []
selected_points = []  # To store clicked points
selected_labels = []  # To store corresponding labels (positive/negative)
current_frame = None  # To hold the current image for drawing

# Video frames directory
frames_directory = "videos/road_imgs1"

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
model_config = "sam2_hiera_l.yaml"
sam2_predictor = build_sam2_video_predictor(model_config, sam2_checkpoint, device=device)


# Mouse callback function for interactive point collection
def click_event(event, x, y, flags, param):
    global selected_points, selected_labels, current_frame
    # Left-click adds a positive point
    if event == cv2.EVENT_LBUTTONDOWN:
        selected_points.append([x, y])
        selected_labels.append(1)
        # print(f"Positive point: ({x}, {y})")
        cv2.circle(current_frame, (x, y), 5, (0, 255, 0), -1)  # Draw green circle

    # Right-click adds a negative point
    elif event == cv2.EVENT_RBUTTONDOWN:
        selected_points.append([x, y])
        selected_labels.append(0)
        # print(f"Negative point: ({x}, {y})")
        cv2.circle(current_frame, (x, y), 5, (0, 0, 255), -1)  # Draw red circle
    # Update the image display
    cv2.imshow(f"Frame", current_frame)


# Clear the directory before saving new frames
def clear_directory(directory):
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
        # print(f"All files in {directory} have been deleted.")
    else:
        print(f"The directory {directory} does not exist.")


# Function to move and copy frames
def move_and_copy_frames(batch_index, frame_paths, batch_size, target_directory="videos/Temp"):
    frames_to_copy = frame_paths[batch_index:batch_index + batch_size]
    clear_directory(target_directory)
    for frame_path in frames_to_copy:
        try:
            if not os.path.exists(target_directory):
                os.makedirs(target_directory)
            shutil.copy(frame_path, target_directory)
        except Exception as e:
            print(f"Failed to copy {frame_path}. Reason: {e}")


# Main process function with interactive point collection
def process_batch(batch_index, batch_number):
    global batch_size, image_counter, points_collection_list, labels_collection_list
    temp_directory = "videos/Temp"

    # Prepare video frame names
    frame_filenames = sorted(
        [p for p in os.listdir(temp_directory) if os.path.splitext(p)[-1].lower() in [".jpg", ".jpeg"]],
        key=lambda p: int(os.path.splitext(p)[0])
    )

    # Initialize and reset inference state for segmentation
    inference_state = sam2_predictor.init_state(video_path=temp_directory)
    sam2_predictor.reset_state(inference_state)

    # Convert collected points and labels to numpy arrays
    points_np = np.array(points_collection_list[batch_number], dtype=np.float32)
    labels_np = np.array(labels_collection_list[batch_number], np.int32)
    # print("points_collection_list =", points_collection_list)
    # print("points_np =", points_np, "labels_np =", labels_np)
    ann_frame_idx = 0
    ann_obj_id = 1

    # Run the segmentation model
    _, object_ids, mask_logits = sam2_predictor.add_new_points_or_box(
        inference_state=inference_state,
        frame_idx=ann_frame_idx,
        obj_id=ann_obj_id,
        points=points_np,
        labels=labels_np,
    )

    # Store the output segmentation results
    video_segments = {}
    rendered_frames_dir = "../rendered_frames"
    os.makedirs(rendered_frames_dir, exist_ok=True)

    for frame_idx, object_ids, mask_logits in sam2_predictor.propagate_in_video(inference_state):
        video_segments[frame_idx] = {
            obj_id: (mask_logits[i] > 0.0).cpu().numpy()
            for i, obj_id in enumerate(object_ids)
        }

    # Save the rendered frames
    with autocast_context:
        for frame_idx in range(0, len(frame_filenames)):
            image = Image.open(os.path.join(temp_directory, frame_filenames[frame_idx]))
            if frame_idx in video_segments:
                mask = video_segments[frame_idx][ann_obj_id]
                mask = np.squeeze(mask)
                if mask.ndim == 2:
                    mask = np.stack([mask] * 3, axis=-1)
                    # image.save(os.path.join(rendered_frames_dir, f"rendered_frame_mask{image_counter:05d}.png"))
                    # mask = Image.fromarray(mask.astype(np.uint8))
                    mask = mask.astype(np.uint8)
                    mask = mask * 255
                    cv2.imwrite(os.path.join(rendered_frames_dir, f"rendered_frame_mask{image_counter:05d}.png"),
                                mask)
                # mask = mask * 255
                mask_image = Image.fromarray(mask.astype(np.uint8))
                blended_image = Image.blend(image, mask_image, alpha=0.5)
                # blended_image.save(os.path.join(rendered_frames_dir, f"rendered_frame_{image_counter:05d}.png"))
            else:
                # image.save(os.path.join(rendered_frames_dir, f"rendered_frame_{image_counter:05d}.png"))
                pass
            image_counter += 1
    # print(f"Rendering for batch {batch_index} completed!")


# Get frame names and start processing in batches
frame_paths = sorted(
    [os.path.join(frames_directory, p) for p in os.listdir(frames_directory) if
     os.path.splitext(p)[-1].lower() in [".jpg", ".jpeg"]],
    key=lambda p: int(os.path.splitext(os.path.basename(p))[0])
)


def collect_user_points():
    global points_collection_list, labels_collection_list, selected_points, selected_labels, current_frame
    selected_points = []
    selected_labels = []
    # Interactive point collection using OpenCV
    batch_frames = [frame_paths[i] for i in range(0, len(frame_paths), batch_size)]
    for frame_path in batch_frames:
        current_frame = cv2.imread(frame_path)  # Store the current image

        # Display the image and wait for user interaction
        cv2.imshow(f"Frame", current_frame)
        cv2.setMouseCallback(f"Frame", click_event)

        # Wait for the user to input points or press the Enter key
        while True:
            key = cv2.waitKey(0)
            if key == 13:  # ASCII value of Enter key
                points_collection_list.append(selected_points[:]), labels_collection_list.append(selected_labels[:])
                selected_points.clear()
                selected_labels.clear()
                break
        cv2.destroyAllWindows()


# Create and start a thread for collecting points
collect_user_points_thread = threading.Thread(target=collect_user_points)
collect_user_points_thread.start()

# Loop over the images in batches and process them
for batch_index in range(0, len(frame_paths), batch_size):
    # Check the condition: wait if points_collection_list is not long enough
    while len(points_collection_list) <= batch_index // batch_size:
        # print(f"Waiting... (batch_index: {batch_index}, batch_number: {batch_index // batch_size})")
        time.sleep(1)  # Wait for 1 second before checking again
    print(f"The current bach size ={batch_index // batch_size + 1}/{len(frame_paths) // batch_size + 1}")
    # Once the condition is met, proceed with the operations
    move_and_copy_frames(batch_index, frame_paths, batch_size)
    process_batch(batch_index, batch_index // batch_size)

collect_user_points_thread.join()
