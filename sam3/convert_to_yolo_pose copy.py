"""
convert_to_yolo_pose.py
=======================
Converts  sam3/working_dir/verified/pose_labels.json
       →  sam3/working_dir/verified/labels/<image_stem>.txt

Each output .txt line follows the YOLOv8-Pose format:
    <class_id> <x_c> <y_c> <w> <h>  <kp1_x> <kp1_y> <kp1_v>  ...  <kp12_x> <kp12_y> <kp12_v>

Where:
  • class_id  = 0  (pallet)
  • x_c, y_c  = normalised bounding-box centre  [0, 1]
  • w,   h    = normalised bounding-box width / height  [0, 1]
  • kpN_x/y   = normalised keypoint coords  [0, 1]
  • kpN_v     = visibility as stored in JSON (0 / 1 / 2)

Bounding box is derived from the convex extent of all *visible* (vis > 0)
keypoints, expanded by a 5 % margin, exactly as PalletDataGenerator.cs does.

Image dimensions are read from the actual JPEG files so no hard-coding is
needed.  Pillow (pip install Pillow) or OpenCV must be available.
"""

import json
import os
import sys
from pathlib import Path

# ── Try to import an image-reading library ────────────────────────────────────
try:
    from PIL import Image as PILImage
    def get_image_size(path: Path):
        with PILImage.open(path) as im:
            return im.size          # (width, height)
except ImportError:
    try:
        import cv2
        def get_image_size(path: Path):
            img = cv2.imread(str(path))
            if img is None:
                raise RuntimeError(f"cv2 could not open {path}")
            h, w = img.shape[:2]
            return w, h
    except ImportError:
        print("ERROR: Neither Pillow nor OpenCV is installed.  "
              "Run:  pip install Pillow", file=sys.stderr)
        sys.exit(1)


# ── Paths ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR   = Path(__file__).resolve().parent
VERIFIED_DIR = SCRIPT_DIR / "working_dir" / "verified"
IMAGES_DIR   = VERIFIED_DIR / "images"
LABELS_DIR   = VERIFIED_DIR / "labels"
JSON_PATH    = VERIFIED_DIR / "pose_labels.json"

# ── Config ────────────────────────────────────────────────────────────────────
CLASS_ID       = 0        # YOLO class index for 'pallet'
NUM_KEYPOINTS  = 12
BBOX_MARGIN    = 0.05     # 5 % expansion on each side (matches PalletDataGenerator.cs)


def f6(v: float) -> str:
    """Format a float to 6 decimal places (invariant locale)."""
    return f"{v:.6f}"


def convert():
    LABELS_DIR.mkdir(parents=True, exist_ok=True)

    with open(JSON_PATH, "r") as fh:
        records = json.load(fh)

    total   = len(records)
    written = 0
    skipped = 0

    for rec in records:
        image_id   = rec["image_id"]           # e.g. "Img6_00000.jpeg"
        keypoints  = rec["keypoints"]          # list of 12 dicts

        # ── Locate the image ──────────────────────────────────────────────────
        img_path = IMAGES_DIR / image_id
        if not img_path.exists():
            print(f"  [WARN] Image not found, skipping: {img_path}")
            skipped += 1
            continue

        img_w, img_h = get_image_size(img_path)

        # ── Build normalised keypoint array ───────────────────────────────────
        # Sort by point_id so order is always p1…p12 regardless of JSON order.
        kps_sorted = sorted(keypoints, key=lambda k: k["point_id"])

        kp_data   = []    # (x_norm, y_norm, vis)  × 12
        bbox_xs   = []
        bbox_ys   = []

        for kp in kps_sorted:
            vis   = int(kp["visible"])      # 0 / 1 / 2
            x_raw = float(kp["x"])
            y_raw = float(kp["y"])

            if vis == 0:
                kp_data.append((0.0, 0.0, 0))
                continue

            x_norm = x_raw / img_w
            y_norm = y_raw / img_h          # JSON y is already top-left origin

            # Clamp to [0, 1] for safety (keypoints very slightly outside frame).
            x_norm = max(0.0, min(1.0, x_norm))
            y_norm = max(0.0, min(1.0, y_norm))

            kp_data.append((x_norm, y_norm, vis))

            # Accumulate bbox extents from visible (even if occluded) points.
            bbox_xs.append(x_norm)
            bbox_ys.append(y_norm)

        # ── Skip frame if no visible keypoints ────────────────────────────────
        if not bbox_xs:
            label_path = LABELS_DIR / (Path(image_id).stem + ".txt")
            label_path.write_text("")
            written += 1
            continue

        # ── Bounding box with margin ──────────────────────────────────────────
        min_x = max(0.0, min(bbox_xs) - BBOX_MARGIN)
        max_x = min(1.0, max(bbox_xs) + BBOX_MARGIN)
        min_y = max(0.0, min(bbox_ys) - BBOX_MARGIN)
        max_y = min(1.0, max(bbox_ys) + BBOX_MARGIN)

        x_c  = (min_x + max_x) * 0.5
        y_c  = (min_y + max_y) * 0.5
        box_w = max_x - min_x
        box_h = max_y - min_y

        # ── Assemble YOLO line ────────────────────────────────────────────────
        parts = [
            str(CLASS_ID),
            f6(x_c), f6(y_c), f6(box_w), f6(box_h),
        ]
        for (kx, ky, kv) in kp_data:
            parts += [f6(kx), f6(ky), str(kv)]

        line = " ".join(parts)

        # ── Write label file ──────────────────────────────────────────────────
        label_path = LABELS_DIR / (Path(image_id).stem + ".txt")
        label_path.write_text(line + "\n")
        written += 1

    print(f"\n[convert_to_yolo_pose] Done.")
    print(f"  Total records : {total}")
    print(f"  Labels written: {written}")
    print(f"  Skipped       : {skipped}")
    print(f"  Output dir    : {LABELS_DIR}")


if __name__ == "__main__":
    convert()
