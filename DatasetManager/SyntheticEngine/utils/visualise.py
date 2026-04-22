"""
SyntheticEngine/utils/visualise.py
====================================
Debug utility to visualise generated YOLO Pose samples.
Overlays keypoints and bounding boxes on top of images.
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path

import cv2
import numpy as np

def draw_yolo_pose(image, label_path, class_names=None):
    """Parse YOLO Pose line and draw on image."""
    h, w = image.shape[:2]
    if not label_path.exists():
        return image
        
    lines = label_path.read_text().strip().split('\n')
    for line in lines:
        if not line: continue
        parts = line.split()
        cls_id = int(parts[0])
        cx, cy, bw, bh = map(float, parts[1:5])
        
        # Draw BBox
        x1 = int((cx - bw/2) * w)
        y1 = int((cy - bh/2) * h)
        x2 = int((cx + bw/2) * w)
        y2 = int((cy + bh/2) * h)
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Draw Keypoints
        kps = parts[5:]
        for i in range(0, len(kps), 3):
            kx, ky, kv = float(kps[i]), float(kps[i+1]), int(kps[i+2])
            if kv > 0:
                px = int(kx * w)
                py = int(ky * h)
                color = (0, 0, 255) if kv == 2 else (255, 255, 0) # Red for visible, Cyan for occluded
                cv2.circle(image, (px, py), 4, color, -1)
                cv2.putText(image, f"p{i//3 + 1}", (px+5, py+5), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    return image

def main():
    parser = argparse.ArgumentParser(description="Visualise generated YOLO Pose samples")
    parser.add_argument("--dataset", type=str, required=True, help="Path to generated dataset folder (e.g. outputs/pallet_synthetic_v1)")
    parser.add_argument("--split", type=str, default="train", help="Split to visualise (train/valid/test)")
    parser.add_argument("--count", type=int, default=10, help="Number of random samples to show")
    
    args = parser.parse_args()
    
    base_path = Path(args.dataset) / args.split
    img_dir = base_path / "images"
    lbl_dir = base_path / "labels_pose"
    
    if not img_dir.exists():
        print(f"Error: {img_dir} does not exist.")
        return

    images = list(img_dir.glob("*.jpg"))
    if not images:
        print("No images found.")
        return

    random.shuffle(images)
    samples = images[:args.count]

    for img_path in samples:
        image = cv2.imread(str(img_path))
        lbl_path = lbl_dir / (img_path.stem + ".txt")
        
        vis = draw_yolo_pose(image, lbl_path)
        
        cv2.imshow("Synthetic Sample (Press any key for next, ESC to quit)", vis)
        key = cv2.waitKey(0)
        if key == 27: # ESC
            break
            
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
