import os
import shutil
import threading
import time
from contextlib import nullcontext

import cv2
import numpy as np
import torch

# Initialize variables
selected_points = []
selected_labels = []
points_collection_list = []
labels_collection_list = []
current_frame = None
is_drawing = False
current_class_label = 1  # Default class label
label_colors = {1: (0, 255, 0), 2: (255, 0, 0), 3: (0, 0, 255)}  # Colors for classes
batch_size = 50
image_counter = 0

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
    global selected_points, selected_labels, current_frame, is_drawing, current_class_label
    if event == cv2.EVENT_LBUTTONDOWN:
        is_drawing = True
        # print([x, y], current_class_label)
        selected_points.append([x, y])
        selected_labels.append(current_class_label)  # Assign the current class label to the point
        cv2.circle(current_frame, (x, y), 5, label_colors[current_class_label], -1)  # Draw color based on label

    elif event == cv2.EVENT_MOUSEMOVE and is_drawing:
        selected_points.append([x, y])
        selected_labels.append(current_class_label)
        cv2.circle(current_frame, (x, y), 2, label_colors[current_class_label], -1)

    elif event == cv2.EVENT_LBUTTONUP:
        is_drawing = False

    elif event == cv2.EVENT_RBUTTONDOWN:
        selected_points.append([x, y])
        selected_labels.append(-current_class_label)  # 0 is for background points
        cv2.circle(current_frame, (x, y), 5, (0, 0, 255), -1)

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
def process_batch(batch_index, batch_number):
    global batch_size, image_counter, points_collection_list, labels_collection_list
    temp_directory = "videos/Temp"

    frame_filenames = sorted(
        [p for p in os.listdir(temp_directory) if os.path.splitext(p)[-1].lower() in [".jpg", ".jpeg"]],
        key=lambda p: int(os.path.splitext(p)[0])
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

        print("\npoints=", points_np1,
              "\nlabels=", labels_np,
              "\nlabels_np1=", labels_np1)

        _, object_ids, mask_logits = sam2_predictor.add_new_points_or_box(
            inference_state=inference_state,
            frame_idx=ann_frame_idx,
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

    for out_frame_idx in range(0, len(frame_filenames), 1):
        frame_path = os.path.join(temp_directory, frame_filenames[out_frame_idx])
        frame = cv2.imread(frame_path)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert to RGB if needed

        # Iterate over masks and overlay them
        for out_obj_id, out_mask in video_segments[out_frame_idx].items():
            mask_image = show_mask(out_mask, obj_id=out_obj_id)

            # Ensure the mask is properly resized to fit the frame if needed
            mask_image = cv2.resize(mask_image, (frame.shape[1], frame.shape[0]))

            # Combine frame and mask
            # Create a transparency mask
            alpha = 0.6  # You can adjust transparency
            frame = cv2.addWeighted(frame, 1 - alpha, mask_image, alpha, 0)

        # Display the frame with masks
        # cv2.imshow(f"Frame", frame)
        cv2.imwrite(os.path.join(rendered_frames_dir, f"rendered_frame_mask{image_counter:05d}.png"),
                    frame)
        image_counter = image_counter + 1
        # Break the loop on key press (e.g., 'q' to quit)
        if cv2.waitKey(0) & 0xFF == ord('q'):
            break


# Get frame names and start processing in batches
frame_paths = sorted(
    [os.path.join(frames_directory, p) for p in os.listdir(frames_directory) if
     os.path.splitext(p)[-1].lower() in [".jpg", ".jpeg"]],
    key=lambda p: int(os.path.splitext(os.path.basename(p))[0])
)


def show_mask(mask, obj_id=None, random_color=False):
    # Define colors directly, avoiding plt.get_cmap
    if random_color:
        color = np.random.rand(3)  # Random color
    else:
        # Define a basic color map as a list of BGR colors
        colors = [
            [255, 0, 0],  # Red
            [0, 255, 0],  # Green
            [0, 0, 255],  # Blue
            [255, 255, 0],  # Yellow
            [0, 255, 255],  # Cyan
            [255, 0, 255]  # Magenta
        ]
        color = colors[obj_id % len(colors)]  # Cycle through colors

    # Create a mask image
    h, w = mask.shape[-2:]
    mask_image = (mask.reshape(h, w, 1) * np.array(color).reshape(1, 1, -1)).astype(np.uint8)

    return mask_image  # Return the mask image


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
    global points_collection_list, labels_collection_list, selected_points, selected_labels, current_frame
    selected_points = []
    selected_labels = []

    batch_frames = [frame_paths[i] for i in range(0, len(frame_paths), batch_size)]
    for frame_path in batch_frames:
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
        cv2.destroyAllWindows()


# Thread to collect user points while processing
collect_user_points_thread = threading.Thread(target=collect_user_points)
collect_user_points_thread.start()

# Set up frame_paths (images already present in 'videos/Temp')
temp_directory = "videos/Temp"

# Loop through frames in batches and process them
for batch_index in range(0, len(frame_paths), batch_size):
    while len(points_collection_list) <= batch_index // batch_size:
        time.sleep(1)

    print(f"Processing batch {batch_index // batch_size + 1}/{max(len(frame_paths) // batch_size, 1)}")
    move_and_copy_frames(batch_index, frame_paths, batch_size)
    process_batch(batch_index, batch_index // batch_size)

collect_user_points_thread.join()
