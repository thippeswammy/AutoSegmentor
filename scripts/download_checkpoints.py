"""
download_checkpoints.py
=======================
One-click downloader for the SAM2 and CoTracker3 model checkpoints required by
AutoSegmentor.

Downloads into the project's expected checkpoint locations:

    external/segment_anything_2/checkpoints/sam2_hiera_large.pt
    external/co-tracker/checkpoints/scaled_offline.pth

Usage:
    python scripts/download_checkpoints.py            # download missing files
    python scripts/download_checkpoints.py --force    # re-download everything
    python scripts/download_checkpoints.py --check    # only verify presence
"""

import argparse
import os
import sys
import hashlib
import urllib.request

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from autosegmentor.models.model_info import (
    SAM2_CHECKPOINT, SAM2_URL, SAM2_REPO, COTRACKER_CHECKPOINT, COTRACKER_URL, COTRACKER_REPO,
)

SAM2_SHA256 = "7442e4e9b732a508f80e141e7c2913437a3610ee0c77381a66658c3a445df87b"
COTRACKER_SHA256 = None  # upstream does not publish a stable hash

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

TARGETS = [
    {
        "name": "SAM2 (sam2_hiera_large.pt)",
        "filename": SAM2_CHECKPOINT,
        "url": SAM2_URL,
        "repo": SAM2_REPO,
        "sha256": SAM2_SHA256,
        "dest": os.path.join(ROOT, "external", "segment_anything_2", "checkpoints", SAM2_CHECKPOINT),
    },
    {
        "name": "CoTracker3 (scaled_offline.pth)",
        "filename": COTRACKER_CHECKPOINT,
        "url": COTRACKER_URL,
        "repo": COTRACKER_REPO,
        "sha256": COTRACKER_SHA256,
        "dest": os.path.join(ROOT, "external", "co-tracker", "checkpoints", COTRACKER_CHECKPOINT),
    },
]


def sha256_of(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def already_valid(target):
    if not os.path.exists(target["dest"]):
        return False
    if os.path.getsize(target["dest"]) == 0:
        return False
    if target["sha256"] and sha256_of(target["dest"]) != target["sha256"]:
        return False
    return True


def download(target):
    os.makedirs(os.path.dirname(target["dest"]), exist_ok=True)
    print(f"Downloading {target['name']} ...")
    print(f"  from {target['url']}  (upstream: {target['repo']})")
    tmp = target["dest"] + ".part"
    try:
        urllib.request.urlretrieve(target["url"], tmp)
        os.replace(tmp, target["dest"])
    except Exception as e:
        if os.path.exists(tmp):
            os.remove(tmp)
        print(f"  FAILED: {e}", file=sys.stderr)
        return False
    size_mb = os.path.getsize(target["dest"]) / (1024 * 1024)
    print(f"  done ({size_mb:.1f} MB)")
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--force", action="store_true", help="Re-download even if present")
    parser.add_argument("--check", action="store_true", help="Only verify presence/size")
    args = parser.parse_args()

    all_ok = True
    for target in TARGETS:
        present = already_valid(target)
        if present and not args.force:
            size_mb = os.path.getsize(target["dest"]) / (1024 * 1024)
            print(f"[OK]   {target['name']} present ({size_mb:.1f} MB)")
            continue
        if args.check:
            print(f"[MISS] {target['name']} not present")
            all_ok = False
            continue
        if present and args.force:
            os.remove(target["dest"])
        if not download(target):
            all_ok = False

    if args.check:
        print()
        print("All checkpoints present." if all_ok else "Some checkpoints are missing.")
    print()
    print("Place checkpoints at:")
    for target in TARGETS:
        print(f"  {target['dest']}")
    print()
    print("Original sources (upstream repos):")
    for target in TARGETS:
        print(f"  {target['name']}: {target['repo']}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())