"""
visualize_pose_images.py
========================
Loads pose_labels.json, verified/original images, and masks to visualize all
keypoints on each frame. Draws keypoints as labeled circles and connects them
with lines to show box1 and box2 outlines.
"""

import argparse
import json
import os
import sys
import cv2
import numpy as np

# Colors for each keypoint (BGR)
KP_COLORS = [
    (0, 0, 255),      # 0: top_left_box1     - Red
    (0, 128, 255),    # 1: top_right_box1    - Orange
    (0, 255, 255),    # 2: bottom_right_box1 - Yellow
    (0, 255, 0),      # 3: bottom_left_box1  - Green
    (255, 0, 0),      # 4: top_left_box2     - Blue
    (255, 0, 128),    # 5: top_right_box2    - Purple
    (255, 128, 0),    # 6: bottom_right_box2 - Teal
    (255, 255, 0),    # 7: bottom_left_box2  - Cyan
]

# Box edge connections (keypoint indices)
BOX1_EDGES = [(0, 1), (1, 2), (2, 3), (3, 0)]
BOX2_EDGES = [(4, 5), (5, 6), (6, 7), (7, 4)]

# Short labels for display
SHORT_LABELS = ["TL1", "TR1", "BR1", "BL1", "TL2", "TR2", "BR2", "BL2"]

RADIUS = 5
LINE_THICKNESS = 2
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.4
FONT_THICKNESS = 1


def find_data_paths(base_dir):
    """Auto-discover pose_labels.json, images dir, and mask dir."""
    candidates = [
        # Verified output (after pipeline completion)
        {
            "pose": os.path.join(base_dir, "verified", "pose_labels.json"),
            "images": os.path.join(base_dir, "verified", "images"),
            "masks": os.path.join(base_dir, "verified", "mask"),
        },
        # Direct path (if --dir points to verified/)
        {
            "pose": os.path.join(base_dir, "pose_labels.json"),
            "images": os.path.join(base_dir, "images"),
            "masks": os.path.join(base_dir, "mask"),
        },
        # Working directory before verification (render + images)
        {
            "pose": os.path.join(base_dir, "verified", "pose_labels.json"),
            "images": os.path.join(base_dir, "images"),
            "masks": os.path.join(base_dir, "render"),
        },
        {
            "pose": os.path.join(base_dir, "pose_labels.json"),
            "images": os.path.join(base_dir, "images"),
            "masks": os.path.join(base_dir, "render"),
        },
    ]

    for c in candidates:
        if os.path.exists(c["pose"]) and os.path.isdir(c["images"]):
            return c

    return None


def load_data(pose_path):
    """Load pose JSON."""
    with open(pose_path, 'r') as f:
        pose_data = json.load(f)
    print(f"Loaded {len(pose_data)} frames from {pose_path}")
    return pose_data


def draw_keypoints(frame, keypoints, show_labels=True):
    """Draw keypoints, labels, and box edges on the frame."""
    pts = {}
    for kp in keypoints:
        pid = kp["point_id"]
        x, y = kp["x"], kp["y"]
        visible = kp.get("visible", 0)
        if visible > 0 and x >= 0 and y >= 0:
            pts[pid] = (int(x), int(y))

    # Draw box edges
    for edges, color in [(BOX1_EDGES, (0, 200, 0)), (BOX2_EDGES, (200, 0, 200))]:
        for (i, j) in edges:
            if i in pts and j in pts:
                cv2.line(frame, pts[i], pts[j], color, LINE_THICKNESS)

    # Draw keypoint circles and labels
    for kp in keypoints:
        pid = kp["point_id"]
        if pid not in pts:
            continue
        x, y = pts[pid]
        color = KP_COLORS[pid % len(KP_COLORS)]

        cv2.circle(frame, (x, y), RADIUS + 1, (255, 255, 255), -1)
        cv2.circle(frame, (x, y), RADIUS, color, -1)

        if show_labels:
            label = SHORT_LABELS[pid] if pid < len(SHORT_LABELS) else str(pid)
            label_pos = (x + RADIUS + 3, y - RADIUS)
            (tw, th), _ = cv2.getTextSize(label, FONT, FONT_SCALE, FONT_THICKNESS)
            cv2.rectangle(frame,
                          (label_pos[0] - 1, label_pos[1] - th - 2),
                          (label_pos[0] + tw + 1, label_pos[1] + 2),
                          (0, 0, 0), -1)
            cv2.putText(frame, label, label_pos, FONT, FONT_SCALE, color, FONT_THICKNESS)

    return frame


def draw_info_bar(frame, frame_idx, total_frames, image_id, mask_on):
    """Draw info bar at the top of the frame."""
    h, w = frame.shape[:2]
    bar_h = 30
    cv2.rectangle(frame, (0, 0), (w, bar_h), (30, 30, 30), -1)

    info_text = f"Frame {frame_idx + 1}/{total_frames}  |  {image_id}  |  Mask: {'ON' if mask_on else 'OFF'}  |  [A/D] Navigate  [M] Toggle Mask  [Q] Quit"
    cv2.putText(frame, info_text, (10, 20), FONT, 0.45, (200, 200, 200), 1)
    return frame


def blend_mask(frame, mask_path, alpha=0.4):
    """Blend the mask on top of the frame."""
    mask = cv2.imread(mask_path)
    if mask is None:
        return frame

    if mask.shape[:2] != frame.shape[:2]:
        mask = cv2.resize(mask, (frame.shape[1], frame.shape[0]))

    non_zero = np.any(mask > 10, axis=-1)
    blended = cv2.addWeighted(frame, 1 - alpha, mask, alpha, 0)
    result = frame.copy()
    result[non_zero] = blended[non_zero]

    gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    contours, _ = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(result, contours, -1, (0, 255, 0), 1)

    return result


def find_mask_file(mask_dir, image_id):
    """Find the matching mask file for an image (may differ in extension)."""
    base = os.path.splitext(image_id)[0]
    for ext in ['.png', '.jpg', '.jpeg']:
        p = os.path.join(mask_dir, base + ext)
        if os.path.exists(p):
            return p
    return None


def main():
    parser = argparse.ArgumentParser(description="Pose Verification Visualizer")
    # Default to working_dir in the root (2 levels up from utils/Tools)
    default_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "working_dir")
    parser.add_argument("--dir", default=default_dir,
                        help="Base directory (default: working_dir)")
    args = parser.parse_args()

    paths = find_data_paths(args.dir)
    if paths is None:
        print(f"ERROR: Could not find pose_labels.json and images in: {args.dir}")
        print("Make sure the pipeline has completed and pose data was exported.")
        sys.exit(1)

    print(f"Pose JSON : {paths['pose']}")
    print(f"Images dir: {paths['images']}")
    print(f"Masks dir : {paths['masks']}")

    pose_data = load_data(paths["pose"])
    total_frames = len(pose_data)
    idx = 0
    show_mask = True

    window_name = "Pose Verification"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    while True:
        entry = pose_data[idx]
        image_id = entry["image_id"]
        keypoints = entry["keypoints"]

        # Load image
        img_path = os.path.join(paths["images"], image_id)
        # Fallback: try without original extension
        if not os.path.exists(img_path):
            base = os.path.splitext(image_id)[0]
            for ext in ['.jpeg', '.jpg', '.png']:
                alt = os.path.join(paths["images"], base + ext)
                if os.path.exists(alt):
                    img_path = alt
                    break

        frame = cv2.imread(img_path)
        if frame is None:
            print(f"WARNING: Could not read image: {img_path}")
            idx = (idx + 1) % total_frames
            continue

        # Blend mask if enabled
        if show_mask and paths["masks"] and os.path.isdir(paths["masks"]):
            mask_path = find_mask_file(paths["masks"], image_id)
            if mask_path:
                frame = blend_mask(frame, mask_path)

        # Draw keypoints
        frame = draw_keypoints(frame, keypoints)

        # Draw info bar
        frame = draw_info_bar(frame, idx, total_frames, image_id, show_mask)

        cv2.imshow(window_name, frame)
        key = cv2.waitKey(0)

        if key == ord('q') or key == 27:
            break
        elif key == ord('d') or key == 83: # 'd' or Right Arrow
            idx = (idx + 1) % total_frames
        elif key == ord('a') or key == 81: # 'a' or Left Arrow
            idx = (idx - 1) % total_frames
        elif key == ord('m'):
            show_mask = not show_mask

    cv2.destroyAllWindows()
    print("Verification complete.")


if __name__ == "__main__":
    main()
