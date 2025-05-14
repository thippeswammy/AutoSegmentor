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
from videos import multithreaded_video_frame_extractor as video2imgs_

val = 0
count = 0
batch_size = 10
videoNumber = 1
image_counter = 0
video2imgs_.VIDEO_NUMBER = videoNumber
video2imgs_.main(videoNumber)
is_drawing = False
current_frame = None
selected_points = []
selected_labels = []
current_class_label = 1
points_collection_list = []
labels_collection_list = []
label_colors = {1: (0, 0, 255), 2: (255, 0, 0), 3: (0, 255, 0)}

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
    x_start = max(x - half_zoom // zoom_factor, 0)
    x_end = min(x + half_zoom // zoom_factor, width)
    y_start = max(y - half_zoom // zoom_factor, 0)
    y_end = min(y + half_zoom // zoom_factor, height)
    zoomed_area = frame[y_start:y_end, x_start:x_end]
    zoomed_area_resized = cv2.resize(zoomed_area, (zoom_size, zoom_size), interpolation=cv2.INTER_LINEAR)
    zoom_view = np.zeros((zoom_size, zoom_size, 3), dtype=np.uint8)
    zoom_view[:zoom_size, :zoom_size] = zoomed_area_resized
    scaled_x = zoom_size // 2
    scaled_y = zoom_size // 2
    cv2.circle(zoom_view, (scaled_x, scaled_y), 5, (0, 255, 0), -1)
    return zoom_view


def click_event(event, x, y, flags, param):
    global selected_points, selected_labels, current_frame, is_drawing, current_class_label, val, count
    if event == cv2.EVENT_LBUTTONDOWN:
        is_drawing = True
        selected_points.append([x, y])
        selected_labels.append(current_class_label)  # Assign the current class label to the point
        cv2.circle(current_frame, (x, y), 2, label_colors[current_class_label], -1)  # Draw color based on label
    elif event == cv2.EVENT_MOUSEMOVE:
        if is_drawing:
            selected_points.append([x, y])
            selected_labels.append(current_class_label)
            cv2.circle(current_frame, (x, y), 2, label_colors[current_class_label], -1)
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
    cv2.imshow(f"Frame{count}/{val}", current_frame)


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


def analyze_mask(mask, min_area=300):
    """ Perform connected component analysis on the mask.
        Only areas with an area greater than or equal to min_area will be colored white;
        all other areas will remain black.
    """
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    output_image = np.zeros_like(mask)  # Create a blank output image

    for label in range(1, num_labels):  # Skip the background label
        area = stats[label, cv2.CC_STAT_AREA]
        if area >= min_area:
            # Color the region in the output image white for valid areas
            output_image[labels == label] = 255  # White for valid areas

    return output_image  # Return the modified image


def change_class_label(label):
    global current_class_label
    current_class_label = label


# Function to process batches of frames
def process_batch(batch_number):
    global batch_size, image_counter, points_collection_list, labels_collection_list
    temp_directory = "videos/Temp"

    frame_filenames = sorted(
        [p for p in os.listdir(temp_directory) if os.path.splitext(p)[-1].lower() in [".jpg", ".jpeg", '.png']],
        key=lambda p: int(re.search(r'(\d+)', os.path.splitext(p)[0]).group())
        if re.search(r'(\d+)', os.path.splitext(p)[0]) else float('inf')
    )
    inference_state = sam2_predictor.init_state(video_path=temp_directory)
    sam2_predictor.reset_state(inference_state)

    # Convert points and labels to numpy arrays
    points_np = np.array(points_collection_list[batch_number], dtype=np.float32)
    labels_np = np.array(labels_collection_list[batch_number], np.int32)

    rendered_frames_dir = "./rendered_frames"
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
        cv2.imwrite(os.path.join(rendered_frames_dir, f"road{videoNumber}_{image_counter:05d}.png"), color_mask_image)
        image_counter = image_counter + 1


def save_points_and_labels(points_collection, labels_collection, filename=f"points_labels_video{videoNumber}.json"):
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


# Function to collect user points for multiple objects
def load_points_and_labels(filename=f"points_labels_video{videoNumber}.json"):
    with open(filename, 'r') as f:
        data = json.load(f)
    return data["points"], data["labels"]


def load_user_points():
    global points_collection_list, labels_collection_list
    points_collection_list, labels_collection_list = load_points_and_labels()


def collect_user_points():
    global points_collection_list, labels_collection_list, selected_points
    global selected_labels, current_frame, current_class_label, val, count
    count = 1
    window_width = 200
    window_height = 200
    selected_points = []
    selected_labels = []
    current_class_label = 1
    cv2.namedWindow("Zoom View", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Zoom View", window_width, window_height)
    batch_frames = [frame_paths[i] for i in range(0, len(frame_paths), batch_size)]
    for frame_path in batch_frames:
        current_class_label = 1
        current_frame = cv2.imread(frame_path)
        val = len(batch_frames)
        cv2.namedWindow(f"Frame{count}/{len(batch_frames)}", cv2.WINDOW_NORMAL)
        cv2.imshow(f"Frame{count}/{len(batch_frames)}", current_frame)
        cv2.setMouseCallback(f"Frame{count}/{len(batch_frames)}", click_event)
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
                # current_frame = cv2.resize(current_frame, (window_width, window_height), interpolation=cv2.INTER_AREA)
        cv2.destroyAllWindows()
    save_points_and_labels(points_collection_list, labels_collection_list, f"points_labels_video{videoNumber}.json")


# Get frame names and start processing in batches

color_map = get_color_map(9)
frame_paths = sorted(
    [os.path.join(frames_directory, p) for p in os.listdir(frames_directory) if
     os.path.splitext(p)[-1].lower() in [".jpg", ".jpeg", '.png']],
    key=lambda p: int(os.path.splitext(os.path.basename(p))[0].split('_')[-1]) if p.split('_')[-1].isdigit() else float(
        'inf')
)

if os.path.exists(f"points_labels_video{videoNumber}.json") and True:
    load_user_points()
else:
    collect_user_points_thread = threading.Thread(target=collect_user_points)
    collect_user_points_thread.start()
batch_index = 0
temp_directory = "videos/Temp"
while batch_index < len(frame_paths) and True:
    while len(points_collection_list) <= batch_index // batch_size:
        time.sleep(1)
    print(f"Processing batch {(batch_index + 1) // batch_size + 1}/{(len(frame_paths) // batch_size) + 1}")
    move_and_copy_frames(batch_index, frame_paths, batch_size, target_directory=temp_directory)
    process_batch(batch_index // batch_size)
    batch_index = batch_index + batch_size
    print('-' * 28, "completed", '-' * 28)
# collect_user_points_thread.join()






