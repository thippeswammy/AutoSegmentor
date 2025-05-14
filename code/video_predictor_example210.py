import json
import os
import re
import shutil
import threading
import time
from contextlib import nullcontext

import cv2
import numpy as np
import pygetwindow as gw
import torch

# Load SAM 2 predictor
from sam2.build_sam import build_sam2_video_predictor

# Initialize variables
batch_size = 40
image_counter = 0
is_drawing = False
current_frame = None
selected_points = []
selected_labels = []
points_collection_list = []
labels_collection_list = []
current_class_label = 1  # Default class label
label_colors = {1: (0, 0, 255), 2: (255, 0, 0), 3: (0, 255, 0)}  # Colors for classes

# Video frames directory
frames_directory = "videos/road_imgs"

# Device selection and setup
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

if device.type == "cuda":
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    autocast_context = torch.autocast(device_type="cuda", dtype=torch.bfloat16)
else:
    autocast_context = nullcontext()

sam2_checkpoint = "../checkpoints/sam2_hiera_large.pt"
model_config = "sam2_hiera_l.yaml"
sam2_predictor = build_sam2_video_predictor(model_config, sam2_checkpoint, device=device)


def show_zoom_view(frame, x, y, zoom_factor=4, zoom_size=200):
    height, width = frame.shape[:2]
    half_zoom = zoom_size // 2

    # Calculate the zoomed area with the point centered
    x_start = max(x - half_zoom // zoom_factor, 0)
    x_end = min(x + half_zoom // zoom_factor, width)
    y_start = max(y - half_zoom // zoom_factor, 0)
    y_end = min(y + half_zoom // zoom_factor, height)

    # Extract the zoomed area
    zoomed_area = frame[y_start:y_end, x_start:x_end]

    # Resize the zoomed area for display
    zoomed_area_resized = cv2.resize(zoomed_area, (zoom_size, zoom_size), interpolation=cv2.INTER_LINEAR)

    # Create a black canvas for the zoom view
    zoom_view = np.zeros((zoom_size, zoom_size, 3), dtype=np.uint8)
    zoom_view[:zoom_size, :zoom_size] = zoomed_area_resized

    # Draw the selected point at the center of the zoom view
    scaled_x = zoom_size // 2
    scaled_y = zoom_size // 2
    cv2.circle(zoom_view, (scaled_x, scaled_y), 5, (0, 255, 0), -1)  # Draw point in green

    return zoom_view


# Mouse callback function for interactive point collection
def click_event(event, x, y, flags, param):
    global selected_points, selected_labels, current_frame, is_drawing, current_class_label
    if event == cv2.EVENT_LBUTTONDOWN:
        is_drawing = True
        # print([x, y], current_class_label)
        selected_points.append([x, y])
        selected_labels.append(current_class_label)  # Assign the current class label to the point
        cv2.circle(current_frame, (x, y), 2, label_colors[current_class_label], -1)  # Draw color based on label
    elif event == cv2.EVENT_MOUSEMOVE:
        if is_drawing:
            selected_points.append([x, y])
            selected_labels.append(current_class_label)
            cv2.circle(current_frame, (x, y), 2, label_colors[current_class_label], -1)
        # Show the zoom view while moving the mouse
        zoom_view = show_zoom_view(current_frame, x, y)
        cv2.imshow("Zoom View", zoom_view)  # Update the zoom view

        # Keep the Zoom View window on top
        try:
            zoom_window = gw.getWindowsWithTitle("Zoom View")[0]
            zoom_window.activate()  # Bring it to the front
        except IndexError:
            pass  # Zoom window is not available

    elif event == cv2.EVENT_LBUTTONUP:
        is_drawing = False
    elif event == cv2.EVENT_RBUTTONDOWN:
        selected_points.append([x, y])
        selected_labels.append(-current_class_label)  # 0 is for background points
        cv2.circle(current_frame, (x, y), 4, (0, 0, 255), -1)
    cv2.imshow("Frame", current_frame)


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


# Function to change the class label
def change_class_label(label):
    global current_class_label
    current_class_label = label
    # print(f"Class label changed to: {label}")


# Function to process batches of frames
def process_batch(batch_number):
    global batch_size, image_counter, points_collection_list, labels_collection_list
    temp_directory = "videos/Temp"

    frame_filenames = sorted(
        [p for p in os.listdir(temp_directory) if os.path.splitext(p)[-1].lower() in [".jpg", ".jpeg", '.png']],
        key=lambda p: int(re.search(r'_(\d+)', os.path.splitext(p)[0]).group(1)) if re.search(r'_(\d+)',
                                                                                              os.path.splitext(p)[
                                                                                                  0]) else float('inf')
    )

    inference_state = sam2_predictor.init_state(video_path=temp_directory)
    sam2_predictor.reset_state(inference_state)

    # Convert points and labels to numpy arrays
    points_np = np.array(points_collection_list[batch_number], dtype=np.float32)
    labels_np = np.array(labels_collection_list[batch_number], np.int32)

    rendered_frames_dir = "../rendered_frames"
    os.makedirs(rendered_frames_dir, exist_ok=True)
    ann_frame_idx = 0

    # Simulate model segmentation for each object (replace with real model interaction)
    for ann_obj_id in set(abs(labels_np)):
        # Initialize and reset inference state for segmentation
        labels_np1 = labels_np.copy()
        labels_np1[labels_np1 == ann_obj_id] = labels_np1[labels_np1 == ann_obj_id] // ann_obj_id
        labels_np1[labels_np1 < 0] = 0
        labels_np1 = labels_np1[abs(labels_np) == ann_obj_id]
        points_np1 = points_np[abs(labels_np) == ann_obj_id]

        _, object_ids, mask_logits = sam2_predictor.add_new_points_or_box(
            inference_state=inference_state,
            frame_idx=ann_frame_idx,
            clear_old_points=False,
            obj_id=int(ann_obj_id),
            points=points_np1,
            labels=labels_np1
        )

    video_segments = {}
    for out_frame_idx, out_obj_ids, out_mask_logits in sam2_predictor.propagate_in_video(inference_state):
        video_segments[out_frame_idx] = {
            out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
            for i, out_obj_id in enumerate(out_obj_ids)
        }

    # Frame processing
    for out_frame_idx in range(len(frame_filenames)):
        frame_path = os.path.join(temp_directory, frame_filenames[out_frame_idx])
        frame = cv2.imread(frame_path)

        # Create a blank mask image with the same dimensions as the frame
        full_mask = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)

        # Iterate over masks and overlay them on the full mask
        for out_obj_id, out_mask in video_segments[out_frame_idx].items():
            # Convert mask to uint8 if it's boolean
            if out_mask.dtype == np.bool_:
                out_mask = out_mask.astype(np.uint8)

            # Squeeze the mask to remove singleton dimension if necessary
            out_mask = out_mask.squeeze()
            # Resize the mask to fit the frame
            out_mask_resized = cv2.resize(out_mask, (frame.shape[1], frame.shape[0]))

            # Combine the mask into the full mask using bitwise OR
            full_mask = full_mask + out_mask_resized

        # Save the combined full mask image
        color_mask_image = mask2colorMaskImg(full_mask)
        cv2.imwrite(os.path.join(rendered_frames_dir, f"full_mask_2_{image_counter:05d}.png"), color_mask_image)
        image_counter = image_counter + 1

        # Break the loop on key press (e.g., 'q' to quit)
    # for out_frame_idx in range(0, len(frame_filenames), 1):
    #     frame_path = os.path.join(temp_directory, frame_filenames[out_frame_idx])
    #     frame = cv2.imread(frame_path)
    #     frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert to RGB if needed
    #     # Iterate over masks and overlay them
    #     for out_obj_id, out_mask in video_segments[out_frame_idx].items():
    #         mask_image = show_mask(out_mask, obj_id=out_obj_id)
    #
    #         # Ensure the mask is properly resized to fit the frame if needed
    #         mask_image = cv2.resize(mask_image, (frame.shape[1], frame.shape[0]))
    #
    #         # Combine frame and mask
    #         # Create a transparency mask
    #         alpha = 0.6  # You can adjust transparency
    #         frame = cv2.addWeighted(frame, 1 - alpha, mask_image, alpha, 0)
    #
    #     # Display the frame with masks
    #     # cv2.imshow(f"Frame", frame)
    #     cv2.imwrite(os.path.join(rendered_frames_dir, f"rendered_frame_mask{image_counter:05d}.png"),
    #                 frame)
    #     # time.sleep(3)
    #     image_counter = image_counter + 1
    #     # Break the loop on key press (e.g., 'q' to quit)
    #     if cv2.waitKey(0) & 0xFF == ord('q'):
    #         break


# Get frame names and start processing in batches
frame_paths = sorted(
    [os.path.join(frames_directory, p) for p in os.listdir(frames_directory) if
     os.path.splitext(p)[-1].lower() in [".jpg", ".jpeg", '.png']],
    key=lambda p: int(os.path.splitext(os.path.basename(p))[0].split('_')[-1]) if p.split('_')[-1].isdigit() else float(
        'inf')
)


def save_points_and_labels(points_collection, labels_collection, filename="points_labels.json"):
    with open(filename, 'w') as f:
        json.dump({"points": points_collection, "labels": labels_collection}, f)


def mask2colorMaskImg(mask):
    # Define a basic color map as a NumPy array (BGR)
    colors = np.array([
        [0, 0, 0],  # 0: Black (background)
        [255, 255, 255],  # 1: White
        [0, 0, 255],  # 2: Red
        [0, 255, 0],  # 3: Green
        [0, 0, 255],  # 4: Blue
        [255, 255, 0],  # 5: Yellow
        [0, 255, 255],  # 6: Cyan
        [255, 0, 255]  # 7: Magenta
    ], dtype=np.uint8)

    # Ensure the mask is of type int and map it to colors
    mask_image = colors[mask]

    return mask_image


def get_color_map(num_colors):
    """ Generate a color map with unique colors for each mask. """
    colors = []
    for i in range(num_colors):
        color = np.random.randint(0, 256, size=3).tolist()  # Random color
        colors.append(color)
    return colors


color_map = get_color_map(9)


# Function to collect user points for multiple objects
def collect_user_points():
    global points_collection_list, labels_collection_list, selected_points
    global selected_labels, current_frame, current_class_label
    selected_points = []
    selected_labels = []
    current_class_label = 1
    cv2.namedWindow("Zoom View")
    batch_frames = [frame_paths[i] for i in range(0, len(frame_paths), batch_size)]
    for frame_path in batch_frames:
        current_class_label = 1
        current_frame = cv2.imread(frame_path)

        cv2.imshow(f"Frame", current_frame)
        cv2.setMouseCallback(f"Frame", click_event)

        while True:
            key = cv2.waitKey(0)
            if key == 13:  # Enter key
                points_collection_list.append(selected_points[:])
                labels_collection_list.append(selected_labels[:])
                selected_points.clear()
                selected_labels.clear()
                break
            if key == ord('q'):
                return
            elif key == ord('1'):
                change_class_label(1)
            elif key == ord('2'):
                change_class_label(2)
            elif key == ord('3'):
                change_class_label(3)
            elif key == ord('4'):
                change_class_label(4)
            elif key == ord('5'):
                change_class_label(5)
            elif key == ord('6'):
                change_class_label(6)
            elif key == ord('7'):
                change_class_label(7)
            elif key == ord('8'):
                change_class_label(8)
            elif key == ord('9'):
                change_class_label(9)
            elif key == ord('r'):
                selected_points = []
                selected_labels = []
                current_frame = cv2.imread(frame_path)

        cv2.destroyAllWindows()
    save_points_and_labels(points_collection_list, labels_collection_list)


# Thread to collect user points while processing
collect_user_points_thread = threading.Thread(target=collect_user_points)
collect_user_points_thread.start()

# Set up frame_paths (images already present in 'videos/Temp')
temp_directory = "videos/Temp"
batch_index = 0
# Loop through frames in batches and process them
while batch_index < len(frame_paths):
    while len(points_collection_list) <= batch_index // batch_size:
        time.sleep(1)
    print(f"Processing batch {batch_index // batch_size + 1}/{max(len(frame_paths) // batch_size, 1)}")
    move_and_copy_frames(batch_index, frame_paths, batch_size)
    process_batch(batch_index // batch_size)
    # user_input = input("is it correct=")
    # if user_input != "":
    #     batch_index = int(user_input)
    # else:
    batch_index = batch_index + batch_size
    print('-' * 10, "completed", '-' * 10)

collect_user_points_thread.join()
