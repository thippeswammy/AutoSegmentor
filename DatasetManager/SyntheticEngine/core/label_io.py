"""
SyntheticEngine/core/label_io.py
==================================
Two responsibilities:
  1. LOAD  — Read pose_labels.json + matching images + color masks
             → List[SampleRecord]
  2. WRITE — Write YOLO pose / seg / box .txt files from a SampleRecord

Color mask convention (BGR, same as DatasetCreator.py's color_to_label):
  color_to_label = { (B, G, R): class_id }
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from core.sample_record import Keypoint, SampleRecord
from utils.mask_utils import (
    extract_binary_mask,
    get_object_bbox,
    mask_to_polygon,
)

log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _f6(v: float) -> str:
    return f"{v:.6f}"


def _find_mask_path(masks_dir: Path, stem: str) -> Optional[Path]:
    """Try common extensions; return first match or None."""
    for ext in (".png", ".jpg", ".jpeg"):
        p = masks_dir / (stem + ext)
        if p.exists():
            return p
    return None


# ─────────────────────────────────────────────────────────────────────────────
# LOAD
# ─────────────────────────────────────────────────────────────────────────────

def load_dataset(
    images_dir: Path,
    masks_dir: Path,
    pose_json: Path,
    color_to_label: Dict[Tuple[int, int, int], int],
    class_id: int = 0,
    num_keypoints: int = 12,
) -> List[SampleRecord]:
    """
    Build a SampleRecord for every annotated entry in pose_labels.json.

    Parameters
    ----------
    images_dir     : folder containing raw .jpeg / .jpg images
    masks_dir      : folder containing RGB color-mapped mask .png files
    pose_json      : path to pose_labels.json
    color_to_label : {(B,G,R): class_id} — BGR tuples as read by cv2.imread
    class_id       : YOLO class index to use for all records
    num_keypoints  : expected number of keypoints per object (default 12)
    """
    records: List[SampleRecord] = []

    with open(pose_json, "r") as fh:
        entries = json.load(fh)

    # Which BGR color maps to our class_id?
    target_color: Optional[Tuple[int, int, int]] = next(
        (c for c, cid in color_to_label.items() if cid == class_id), None
    )

    for entry in entries:
        image_id: str = entry["image_id"]          # e.g. "Img6_00000.jpeg"
        keypoints_raw: list = entry["keypoints"]   # list of dicts

        stem = Path(image_id).stem
        img_path = images_dir / image_id
        if not img_path.exists():
            # Try alternate extensions
            for ext in (".jpeg", ".jpg", ".png"):
                alt = images_dir / (stem + ext)
                if alt.exists():
                    img_path = alt
                    break
            else:
                log.warning("Image not found, skipping: %s", img_path)
                continue

        image = cv2.imread(str(img_path))
        if image is None:
            log.warning("Could not read image: %s", img_path)
            continue

        img_h, img_w = image.shape[:2]

        # ── Binary mask ──────────────────────────────────────────────────────
        binary_mask = np.zeros((img_h, img_w), dtype=np.uint8)
        mask_path = _find_mask_path(masks_dir, stem)
        if mask_path and target_color is not None:
            color_mask = cv2.imread(str(mask_path))
            if color_mask is not None:
                # Resize mask to match image if needed
                if color_mask.shape[:2] != (img_h, img_w):
                    color_mask = cv2.resize(
                        color_mask, (img_w, img_h), interpolation=cv2.INTER_NEAREST
                    )
                binary_mask = extract_binary_mask(color_mask, target_color)
        else:
            log.warning("Mask not found for: %s", image_id)

        # ── Keypoints ────────────────────────────────────────────────────────
        kps_sorted = sorted(keypoints_raw, key=lambda k: k["point_id"])
        keypoints: List[Keypoint] = []

        for kp in kps_sorted[:num_keypoints]:
            vis = int(kp.get("visible", 0))
            if vis == 0:
                keypoints.append(Keypoint(0.0, 0.0, 0, kp["point_id"]))
            else:
                x = float(kp["x"])
                y = float(kp["y"])
                x = max(0.0, min(float(img_w - 1), x))
                y = max(0.0, min(float(img_h - 1), y))
                keypoints.append(Keypoint(x, y, vis, kp["point_id"]))

        # Pad to num_keypoints if JSON has fewer entries
        while len(keypoints) < num_keypoints:
            pid = len(keypoints) + 1
            keypoints.append(Keypoint(0.0, 0.0, 0, pid))

        records.append(
            SampleRecord(
                image=image,
                mask=binary_mask,
                keypoints=keypoints,
                img_path=img_path,
                img_w=img_w,
                img_h=img_h,
                class_id=class_id,
                source_id=stem,
            )
        )

    log.info("Loaded %d SampleRecords from %s", len(records), pose_json)
    return records


# ─────────────────────────────────────────────────────────────────────────────
# WRITE
# ─────────────────────────────────────────────────────────────────────────────

def write_yolo_pose_label(
    record: SampleRecord,
    out_path: Path,
    class_id: Optional[int] = None,
    bbox_margin: float = 0.05,
) -> None:
    """Write one YOLO-Pose .txt line: class cx cy w h kp1_x kp1_y v1 …"""
    cid = class_id if class_id is not None else record.class_id
    w, h = record.img_w, record.img_h

    bbox_xs, bbox_ys, kp_parts = [], [], []

    for kp in record.keypoints:
        if kp.vis == 0:
            kp_parts.extend(["0.000000", "0.000000", "0"])
        else:
            nx = max(0.0, min(1.0, kp.x / w))
            ny = max(0.0, min(1.0, kp.y / h))
            kp_parts.extend([_f6(nx), _f6(ny), str(kp.vis)])
            bbox_xs.append(nx)
            bbox_ys.append(ny)

    if not bbox_xs:
        out_path.write_text("")
        return

    min_x = max(0.0, min(bbox_xs) - bbox_margin)
    max_x = min(1.0, max(bbox_xs) + bbox_margin)
    min_y = max(0.0, min(bbox_ys) - bbox_margin)
    max_y = min(1.0, max(bbox_ys) + bbox_margin)
    cx = (min_x + max_x) * 0.5
    cy = (min_y + max_y) * 0.5
    bw = max_x - min_x
    bh = max_y - min_y

    line = f"{cid} {_f6(cx)} {_f6(cy)} {_f6(bw)} {_f6(bh)} " + " ".join(kp_parts)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(line + "\n", encoding="utf-8")


def write_yolo_seg_label(
    record: SampleRecord,
    out_path: Path,
    class_id: Optional[int] = None,
) -> None:
    """Write one YOLO instance-seg .txt line from the binary mask contour."""
    cid = class_id if class_id is not None else record.class_id
    polygon = mask_to_polygon(record.mask)
    if polygon is None:
        out_path.write_text("")
        return
    coords_str = " ".join(f"{_f6(x)} {_f6(y)}" for x, y in polygon)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(f"{cid} {coords_str}\n", encoding="utf-8")


def write_yolo_box_label(
    record: SampleRecord,
    out_path: Path,
    class_id: Optional[int] = None,
    bbox_margin: float = 0.05,
) -> None:
    """Write one YOLO detection .txt line (bbox derived from keypoint extents)."""
    cid = class_id if class_id is not None else record.class_id
    w, h = record.img_w, record.img_h
    vis_kps = record.visible_keypoints()

    if not vis_kps:
        out_path.write_text("")
        return

    xs = [kp.x / w for kp in vis_kps]
    ys = [kp.y / h for kp in vis_kps]
    min_x = max(0.0, min(xs) - bbox_margin)
    max_x = min(1.0, max(xs) + bbox_margin)
    min_y = max(0.0, min(ys) - bbox_margin)
    max_y = min(1.0, max(ys) + bbox_margin)
    cx = (min_x + max_x) * 0.5
    cy = (min_y + max_y) * 0.5
    bw = max_x - min_x
    bh = max_y - min_y

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(f"{cid} {_f6(cx)} {_f6(cy)} {_f6(bw)} {_f6(bh)}\n", encoding="utf-8")
