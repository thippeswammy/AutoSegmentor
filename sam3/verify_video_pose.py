"""
Video Pose Verification Visualizer
==================================
Loads a pose JSON (e.g., pose_label_video8.json) and a video file 
(e.g., OverlappedVideo8.mp4) to visualize keypoints frame-by-frame.

Usage:
    python verify_video_pose.py --video outputs/OverlappedVideo8.mp4 --json outputs/pose_label_video8.json

Controls:
    Right Arrow / 'd' : Next frame
    Left Arrow  / 'a' : Previous frame
    'q'               : Quit
"""

import argparse
import json
import os
import sys
import cv2
import numpy as np

# Adaptive colors (will wrap if more than 8)
KP_COLORS = [
    (0, 0, 255),      # Red
    (0, 128, 255),    # Orange
    (0, 255, 255),    # Yellow
    (0, 255, 0),      # Green
    (255, 0, 0),      # Blue
    (255, 0, 128),    # Purple
    (255, 128, 0),    # Teal
    (255, 255, 0),    # Cyan
    (128, 0, 255),    # Violet
    (0, 255, 128),    # Light Green
    (128, 255, 0),    # Lime
    (255, 128, 128),  # Pink
]

RADIUS = 5
LINE_THICKNESS = 2
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.4
FONT_THICKNESS = 1

def load_data(json_path):
    """Load pose JSON."""
    with open(json_path, 'r') as f:
        data = json.load(f)
    # Map index to entry
    return {entry.get("frame_index", i): entry for i, entry in enumerate(data)}

def draw_keypoints(frame, keypoints, show_labels=True):
    """Draw keypoints, labels, and connecting lines on the frame."""
    pts = {}
    valid_pids = []
    
    # Collect valid points
    for kp in keypoints:
        pid = kp["point_id"]
        x, y = int(kp["x"]), int(kp["y"])
        visible = kp.get("visible", 0)
        
        if visible > 0:
            pts[pid] = (x, y)
            valid_pids.append(pid)

    valid_pids.sort()

    # Draw connecting lines (1->2->...->n->1)
    if len(valid_pids) > 1:
        color_line = (0, 255, 0) # Green for sequence
        for i in range(len(valid_pids)):
            curr_pid = valid_pids[i]
            next_pid = valid_pids[(i + 1) % len(valid_pids)]
            
            # Draw line if both points exist
            if curr_pid in pts and next_pid in pts:
                cv2.line(frame, pts[curr_pid], pts[next_pid], color_line, LINE_THICKNESS)

    # Draw keypoint circles
    for pid in valid_pids:
        x, y = pts[pid]
        color = (0, 0, 255) # Fixed Red
        
        cv2.circle(frame, (x, y), RADIUS + 1, (255, 255, 255), -1)
        cv2.circle(frame, (x, y), RADIUS, color, -1)

    return frame

def draw_info_bar(frame, frame_idx, total_frames, image_id):
    """Draw info bar at the top of the frame."""
    h, w = frame.shape[:2]
    bar_h = 30
    cv2.rectangle(frame, (0, 0), (w, bar_h), (30, 30, 30), -1)

    info_text = f"Frame {frame_idx + 1}/{total_frames}  |  {image_id}  |  [A/D] Navigate  [Q] Quit"
    cv2.putText(frame, info_text, (10, 20), FONT, 0.45, (200, 200, 200), 1)
    return frame

import time
from collections import deque

class FrameCache:
    def __init__(self, cap, max_size=100):
        self.cap = cap
        self.max_size = max_size
        self.cache = {}  # index -> frame
        self.order = deque()

    def get_frame(self, idx):
        if idx in self.cache:
            return self.cache[idx]
        
        # Seek and read
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = self.cap.read()
        if not ret:
            return None
        
        # Add to cache
        self.cache[idx] = frame
        self.order.append(idx)
        if len(self.order) > self.max_size:
            oldest = self.order.popleft()
            del self.cache[oldest]
        
        return frame

def main():
    parser = argparse.ArgumentParser(description="Video Pose Visualizer")
    parser.add_argument("--video", required=True, help="Path to video file")
    parser.add_argument("--json", required=True, help="Path to pose JSON file")
    args = parser.parse_args()

    if not os.path.exists(args.video):
        print(f"ERROR: Video not found: {args.video}")
        sys.exit(1)
    if not os.path.exists(args.json):
        print(f"ERROR: JSON not found: {args.json}")
        sys.exit(1)

    pose_map = load_data(args.json)
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"ERROR: Could not open video: {args.video}")
        sys.exit(1)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Video frames: {total_frames}")
    print(f"JSON entries: {len(pose_map)}")

    frame_cache = FrameCache(cap, max_size=200) # Increased buffer size
    idx = 0
    window_name = "Video Pose Verification"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    last_idx = -1
    last_update_time = 0
    step_delay = 0.05 # 200ms repeat delay

    while True:
        current_time = time.time()
        
        if idx != last_idx:
            current_frame = frame_cache.get_frame(idx)
            if current_frame is None:
                print(f"WARNING: Could not read frame {idx}")
                idx = (idx + 1) % total_frames
                continue
            last_idx = idx

            display_frame = current_frame.copy()
            if idx in pose_map:
                entry = pose_map[idx]
                image_id = entry.get("image_id", f"frame_{idx}")
                instances = entry.get("instances", [])
                if instances:
                    for inst in instances:
                        keypoints = inst.get("keypoints", [])
                        display_frame = draw_keypoints(display_frame, keypoints)
                else:
                    keypoints = entry.get("keypoints", [])
                    display_frame = draw_keypoints(display_frame, keypoints)
                display_frame = draw_info_bar(display_frame, idx, total_frames, image_id)
            else:
                display_frame = draw_info_bar(display_frame, idx, total_frames, "NO DATA")
            
            cv2.imshow(window_name, display_frame)

        # Non-blocking wait to allow for smoother re-triggering
        key = cv2.waitKey(10) & 0xFF

        if key == ord('q') or key == 27:
            break
        elif key == ord('d') or key == 83: # 'd' or Right Arrow (sometimes 83)
            # Only step if enough time has passed (0.2s)
            if current_time - last_update_time > step_delay:
                idx = (idx + 1) % total_frames
                last_update_time = current_time
        elif key == ord('a') or key == 81: # 'a' or Left Arrow (sometimes 81)
            if current_time - last_update_time > step_delay:
                idx = (idx - 1) % total_frames
                last_update_time = current_time
        elif key == 255: # No key pressed
            pass
        else:
            # For any other single tap, reset timer so it responds immediately
            last_update_time = 0

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
