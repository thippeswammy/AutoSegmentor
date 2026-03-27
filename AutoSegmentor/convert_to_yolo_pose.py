"""
convert_to_yolo_pose.py
=======================
Converts pose labels (JSON) to YOLOv8-Pose format (.txt).
Optionally extracts original frames from video to create a full dataset.

Final Structure:
    <out_dir>/
        ├── images/   (original jpeg frames)
        └── labels/   (yolo .txt files)
"""

import json
import os
import sys
import argparse
from pathlib import Path

# ── Try to import an image/video library ──────────────────────────────────────
try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False
    try:
        from PIL import Image as PILImage
        HAS_PIL = True
    except ImportError:
        HAS_PIL = False

def get_image_size(path: Path):
    if HAS_PIL:
        try:
            with PILImage.open(path) as im:
                return im.size  # (width, height)
        except:
            pass
    if HAS_CV2:
        img = cv2.imread(str(path))
        if img is not None:
            h, w = img.shape[:2]
            return w, h
    return None

def get_video_size(path: Path):
    if not HAS_CV2:
        print("ERROR: OpenCV is required for video dimensions. Run: pip install opencv-python")
        return None
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return None
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return w, h

def extract_frames(video_path: Path, output_images_dir: Path):
    if not HAS_CV2:
        print("ERROR: OpenCV is required for frame extraction.")
        return False
    
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"ERROR: Could not open video {video_path}")
        return False
    
    output_images_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    print(f"Extracting frames to {output_images_dir}...")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # Match naming convention: Img<N>_00000.jpeg
        stem = video_path.stem.replace("Org", "") # e.g. OrgVideo6 -> Video6
        # If it matches OrgVideoN, use ImgN
        if "Video" in stem:
            n_str = stem.replace("Video", "")
            img_name = f"Img{n_str}_{count:05d}.jpeg"
        else:
            img_name = f"{stem}_{count:05d}.jpeg"
            
        out_path = output_images_dir / img_name
        cv2.imwrite(str(out_path), frame)
        
        if count % 100 == 0:
            print(f"  Extracted {count} frames...")
        count += 1
        
    cap.release()
    print(f"Successfully extracted {count} frames.")
    return True

# ── Config ────────────────────────────────────────────────────────────────────
CLASS_ID       = 0        # YOLO class index for 'pallet'
NUM_KEYPOINTS  = 12
BBOX_MARGIN    = 0.05     # 5 % expansion on each side

def f6(v: float) -> str:
    """Format a float to 6 decimal places."""
    return f"{v:.6f}"

def convert():
    parser = argparse.ArgumentParser(description="Convert pose JSON to YOLO pose txt.")
    parser.add_argument("--json", type=str, default=r"f:\RunningProjects\AutoSegmentor\sam3\outputs\pose_label_video6.json", help="Path to labels JSON")
    parser.add_argument("--video", type=str, default=r"f:\RunningProjects\AutoSegmentor\sam3\outputs\OrgVideo6.mp4", help="Path to video (for dimensions/extraction)")
    parser.add_argument("--images_in", type=str, default=None, help="Input images dir (alternative to --video)")
    parser.add_argument("--out_dir", type=str, default=r"f:\RunningProjects\AutoSegmentor\sam3\outputs\pose_dataset_video6", help="Output dataset directory")
    parser.add_argument("--extract", action="store_true", help="Extract original frames from video?")
    
    args = parser.parse_args()

    json_path = Path(args.json)
    base_out = Path(args.out_dir)
    images_out = base_out / "images"
    labels_out = base_out / "labels"
    video_path = Path(args.video) if args.video else None
    images_in = Path(args.images_in) if args.images_in else None

    if not json_path.exists():
        print(f"ERROR: JSON file not found: {json_path}")
        return

    images_out.mkdir(parents=True, exist_ok=True)
    labels_out.mkdir(parents=True, exist_ok=True)

    # ── Determine Dimensions ──────────────────────────────────────────────────
    video_w, video_h = None, None
    if video_path and video_path.exists():
        print(f"Detecting dimensions from video: {video_path}")
        video_w, video_h = get_video_size(video_path)
        if video_w:
            print(f"  Width: {video_w}, Height: {video_h}")
        
        if args.extract:
            extract_frames(video_path, images_out)

    with open(json_path, "r") as fh:
        records = json.load(fh)

    total   = len(records)
    written = 0
    skipped = 0

    print(f"Processing {total} label records...")

    for rec in records:
        image_id = rec["image_id"]
        instances = rec.get("instances", [])
        
        if not instances and "keypoints" in rec:
            instances = [{"keypoints": rec["keypoints"]}]

        if not instances:
            label_path = labels_out / (Path(image_id).stem + ".txt")
            label_path.write_text("")
            written += 1
            continue

        # ── Get Dimensions ───────────────────────────────────────────────────
        img_w, img_h = video_w, video_h
        if not img_w:
            possible_imgs = [images_in, images_out]
            for d in possible_imgs:
                if d:
                    path = d / image_id
                    if path.exists():
                        size = get_image_size(path)
                        if size:
                            img_w, img_h = size; break
            
        if not img_w:
            if "OrgVideo6" in str(json_path): img_w, img_h = 1920, 1080
            elif "OrgVideo7" in str(json_path): img_w, img_h = 2336, 1080
            else:
                print(f"  [WARN] Size unknown for {image_id}, skipping.")
                skipped += 1; continue

        # ── Process Instances ───────────────────────────────────────────────
        lines = []
        for inst in instances:
            keypoints = inst["keypoints"]
            kps_sorted = sorted(keypoints, key=lambda k: k.get("point_id", 0))

            kp_data, bbox_xs, bbox_ys = [], [], []
            for kp in kps_sorted:
                vis, x_raw, y_raw = int(kp.get("visible", 2)), float(kp["x"]), float(kp["y"])
                if vis == 0: kp_data.append((0.0, 0.0, 0)); continue

                x_norm, y_norm = max(0.0, min(1.0, x_raw/img_w)), max(0.0, min(1.0, y_raw/img_h))
                kp_data.append((x_norm, y_norm, vis))
                bbox_xs.append(x_norm); bbox_ys.append(y_norm)

            if not bbox_xs: continue

            min_x, max_x = max(0.0, min(bbox_xs)-BBOX_MARGIN), min(1.0, max(bbox_xs)+BBOX_MARGIN)
            min_y, max_y = max(0.0, min(bbox_ys)-BBOX_MARGIN), min(1.0, max(bbox_ys)+BBOX_MARGIN)

            x_c, y_c, bw, bh = (min_x+max_x)*0.5, (min_y+max_y)*0.5, max_x-min_x, max_y-min_y
            parts = [str(CLASS_ID), f6(x_c), f6(y_c), f6(bw), f6(bh)]
            for (kx, ky, kv) in kp_data: parts += [f6(kx), f6(ky), str(kv)]
            lines.append(" ".join(parts))

        label_path = labels_out / (Path(image_id).stem + ".txt")
        label_path.write_text("\n".join(lines) + ("\n" if lines else ""))
        written += 1

    print(f"\n[convert_to_yolo_pose] Done.")
    print(f"  Total records : {total}")
    print(f"  Labels written: {written}")
    print(f"  Output Base   : {base_out}")

if __name__ == "__main__":
    convert()
