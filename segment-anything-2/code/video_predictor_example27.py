import os
import threading
import time
from contextlib import nullcontext

import cv2
import numpy as np
import torch
from PIL import Image

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
        selected_labels.append(0)  # 0 is for background points
        cv2.circle(current_frame, (x, y), 5, (0, 0, 255), -1)

    cv2.imshow("Frame", current_frame)


# Function to change the class label
def change_class_label(label):
    global current_class_label
    current_class_label = label
    print(f"Class label changed to: {label}")


# Function to process batches of frames
def process_batch(batch_index, batch_number):
    global batch_size, image_counter, points_collection_list, labels_collection_list
    temp_directory = "videos/Temp"

    frame_filenames = sorted(
        [p for p in os.listdir(temp_directory) if os.path.splitext(p)[-1].lower() in [".jpg", ".jpeg"]],
        key=lambda p: int(os.path.splitext(p)[0])
    )

    # Initialize and reset inference state for segmentation (placeholder for actual model interaction)
    inference_state = sam2_predictor.init_state(video_path=temp_directory)
    sam2_predictor.reset_state(inference_state)

    # Convert points and labels to numpy arrays
    points_np = np.array(points_collection_list[batch_number], dtype=np.float32)
    labels_np = np.array(labels_collection_list[batch_number], np.int32)

    video_segments = {}
    rendered_frames_dir = "../rendered_frames"
    os.makedirs(rendered_frames_dir, exist_ok=True)
    ann_frame_idx = 0
    ann_obj_id = 1
    print(labels_np)
    # Simulate model segmentation for each object (replace with real model interaction)
    for ann_obj_id in tuple(labels_np):
        # _, object_ids, mask_logits = sam2_predictor.add_new_points_or_box(
        #     inference_state=inference_state,
        #     frame_idx=ann_frame_idx,
        #     obj_id=labels_np[labels_np == ann_obj_id],
        #     points=points_np[labels_np == ann_obj_id],
        #     labels=labels_np[labels_np == ann_obj_id],
        # )
        _, object_ids, mask_logits = sam2_predictor.add_new_points_or_box(
            inference_state=inference_state,
            frame_idx=ann_frame_idx,
            obj_id=int(ann_obj_id),  # Ensure obj_id is a scalar (convert to int if necessary)
            points=points_np[labels_np == ann_obj_id],
            labels=labels_np[labels_np == ann_obj_id],
        )

        for frame_idx, object_ids, mask_logits in sam2_predictor.propagate_in_video(inference_state):
            if frame_idx not in video_segments:
                video_segments[frame_idx] = {}
            video_segments[frame_idx][ann_obj_id] = (mask_logits[ann_obj_id - 1] > 0.0).cpu().numpy()

    for frame_idx in range(len(frame_filenames)):
        image = Image.open(os.path.join(temp_directory, frame_filenames[frame_idx]))

        composite_mask = np.zeros_like(np.array(image))

        if frame_idx in video_segments:
            for ann_obj_id in video_segments[frame_idx]:
                mask = video_segments[frame_idx][ann_obj_id]
                mask = np.squeeze(mask)
                if mask.ndim == 2:
                    mask = np.stack([mask] * 3, axis=-1)
                    mask = mask.astype(np.uint8) * (255 // 2)

                    composite_mask = np.maximum(composite_mask, mask)

                    cv2.imwrite(os.path.join(rendered_frames_dir,
                                             f"rendered_frame_mask_obj{ann_obj_id}_{image_counter:05d}.png"), mask)

        composite_mask_image = Image.fromarray(composite_mask.astype(np.uint8))
        blended_image = Image.blend(image, composite_mask_image, alpha=0.5)
        blended_image.save(os.path.join(rendered_frames_dir, f"rendered_frame_{image_counter:05d}.png"))
        image_counter += 1


# Get frame names and start processing in batches
frame_paths = sorted(
    [os.path.join(frames_directory, p) for p in os.listdir(frames_directory) if
     os.path.splitext(p)[-1].lower() in [".jpg", ".jpeg"]],
    key=lambda p: int(os.path.splitext(os.path.basename(p))[0])
)


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

    print(f"Processing batch {batch_index // batch_size + 1}/{len(frame_paths) // batch_size + 1}")
    process_batch(batch_index, batch_index // batch_size)
    time.sleep(1)

collect_user_points_thread.join()
