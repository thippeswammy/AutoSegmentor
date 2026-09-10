"""
install.py
===========
One-shot environment setup for AutoSegmentor. Works the same way on Windows
and Ubuntu/Linux since it's plain Python — no separate .bat/.sh scripts needed.

Assumes you already created and activated a virtual environment (see the
"Setup & Installation" section of README.md). This script then:

    1. Checks the Python version.
    2. Initializes git submodules (external/co-tracker, external/segment_anything_2).
    3. Installs Python dependencies from requirements-core.txt.
    4. Downloads the SAM2 + CoTracker3 checkpoints (skips files already present).
    5. Runs a GPU/CUDA diagnostic.

Usage:
    python install.py                      # full setup
    python install.py --cuda               # also install the `cuda` extra (flash-attn)
    python install.py --skip-submodules
    python install.py --skip-deps
    python install.py --skip-checkpoints
    python install.py --skip-gpu-check

    # Standalone equivalents of the old scripts/download_checkpoints.py
    # and scripts/gpu_diagnostic.py (now merged into this file):
    python install.py --checkpoints-only           # download missing checkpoints
    python install.py --checkpoints-only --check    # only verify presence
    python install.py --checkpoints-only --force    # re-download everything
    python install.py --gpu-check-only              # just run the GPU/system diagnostic
"""

import argparse
import hashlib
import os
import platform
import subprocess
import sys
import urllib.request

ROOT = os.path.dirname(os.path.abspath(__file__))

MIN_PYTHON = (3, 10)


def _fail(message):
    print(f"ERROR: {message}", file=sys.stderr)
    sys.exit(1)


def check_python_version():
    if sys.version_info < MIN_PYTHON:
        _fail(
            f"Python {MIN_PYTHON[0]}.{MIN_PYTHON[1]}+ is required, "
            f"found {platform.python_version()}."
        )
    print(f"[OK] Python {platform.python_version()}")


def init_submodules():
    print("\n== Git submodules (external/co-tracker, external/segment_anything_2) ==")
    try:
        subprocess.run(
            ["git", "submodule", "update", "--init", "--recursive"],
            cwd=ROOT, check=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        print(f"WARNING: could not run 'git submodule update': {e}", file=sys.stderr)
        print("If this checkout isn't a git clone, make sure external/co-tracker "
              "and external/segment_anything_2 are populated some other way.")


def install_dependencies(cuda_extra):
    print("\n== Python dependencies (requirements-core.txt) ==")
    req_file = os.path.join(ROOT, "requirements-core.txt")
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-r", req_file], check=True,
    )
    if cuda_extra:
        print("\n== CUDA extra (flash-attn) ==")
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-e", f"{ROOT}[cuda]"], check=True,
        )


# --- Checkpoint download (merged from the former scripts/download_checkpoints.py) ------

from autosegmentor.models.model_info import (
    SAM2_CHECKPOINT, SAM2_URL, SAM2_REPO, COTRACKER_CHECKPOINT, COTRACKER_URL, COTRACKER_REPO,
)

SAM2_SHA256 = "7442e4e9b732a508f80e141e7c2913437a3610ee0c77381a66658c3a445df87b"
COTRACKER_SHA256 = None  # upstream does not publish a stable hash

CHECKPOINT_TARGETS = [
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


def _sha256_of(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _checkpoint_valid(target):
    if not os.path.exists(target["dest"]):
        return False
    if os.path.getsize(target["dest"]) == 0:
        return False
    if target["sha256"] and _sha256_of(target["dest"]) != target["sha256"]:
        return False
    return True


def _download_checkpoint(target):
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


def handle_checkpoints(check_only=False, force=False):
    print("\n== Model checkpoints (SAM2 + CoTracker3) ==")
    all_ok = True
    for target in CHECKPOINT_TARGETS:
        present = _checkpoint_valid(target)
        if present and not force:
            size_mb = os.path.getsize(target["dest"]) / (1024 * 1024)
            print(f"[OK]   {target['name']} present ({size_mb:.1f} MB)")
            continue
        if check_only:
            print(f"[MISS] {target['name']} not present")
            all_ok = False
            continue
        if present and force:
            os.remove(target["dest"])
        if not _download_checkpoint(target):
            all_ok = False

    print()
    print("Checkpoint locations:")
    for target in CHECKPOINT_TARGETS:
        print(f"  {target['dest']}")
    return all_ok


# --- GPU/system diagnostic (merged from the former scripts/gpu_diagnostic.py) -----------

def run_gpu_diagnostic():
    print("\n== GPU / system diagnostic ==")

    print(f"OS: {platform.system()} {platform.release()} ({platform.version()})")
    print(f"Processor: {platform.processor()}")
    try:
        import psutil
        vm = psutil.virtual_memory()
        print(f"RAM: {vm.total / (1024**3):.2f} GB total, {vm.available / (1024**3):.2f} GB available")
    except ImportError:
        print("(psutil not installed, skipping RAM details)")

    try:
        import torch
        cuda_available = torch.cuda.is_available()
        print(f"PyTorch {torch.__version__} | CUDA available: {cuda_available}")
        if cuda_available:
            print(f"  GPU: {torch.cuda.get_device_name(0)} (device count: {torch.cuda.device_count()})")
    except ImportError:
        print("PyTorch not installed.")

    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,memory.used,utilization.gpu",
             "--format=csv,noheader,nounits"],
            stdout=subprocess.PIPE, text=True, check=True,
        )
        for i, line in enumerate(result.stdout.strip().splitlines()):
            name, total, used, util = line.split(", ")
            print(f"nvidia-smi GPU {i}: {name} — {used}/{total} MiB, {util}% load")
    except Exception:
        print("nvidia-smi not found or failed to execute (no NVIDIA GPU, or driver not installed).")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--cuda", action="store_true", help="Also install the `cuda` extra (flash-attn)")
    parser.add_argument("--skip-submodules", action="store_true")
    parser.add_argument("--skip-deps", action="store_true")
    parser.add_argument("--skip-checkpoints", action="store_true")
    parser.add_argument("--skip-gpu-check", action="store_true")
    parser.add_argument("--checkpoints-only", action="store_true",
                         help="Only run the checkpoint step, then exit")
    parser.add_argument("--gpu-check-only", action="store_true",
                         help="Only run the GPU/system diagnostic, then exit")
    parser.add_argument("--check", action="store_true", help="With --checkpoints-only: verify presence only")
    parser.add_argument("--force", action="store_true", help="With --checkpoints-only: re-download everything")
    args = parser.parse_args()

    if args.checkpoints_only:
        ok = handle_checkpoints(check_only=args.check, force=args.force)
        sys.exit(0 if ok else 1)

    if args.gpu_check_only:
        run_gpu_diagnostic()
        return

    check_python_version()
    if not args.skip_submodules:
        init_submodules()
    if not args.skip_deps:
        install_dependencies(cuda_extra=args.cuda)
    if not args.skip_checkpoints:
        handle_checkpoints()
    if not args.skip_gpu_check:
        run_gpu_diagnostic()

    print("\n" + "=" * 60)
    print("Setup complete. Run:  python run_main.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
