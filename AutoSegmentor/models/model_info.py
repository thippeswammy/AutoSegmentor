"""Shared metadata for the model checkpoints AutoSegmentor depends on.

Central place for the download URLs, original upstream repo links, and the
expected on-disk locations. Also builds the human-readable "model missing"
message used by the loaders, so a user always sees how to install/download the
weights no matter which loader first hits the problem.
"""

import os

try:
    from platformdirs import user_cache_dir
except ImportError:  # pragma: no cover - platformdirs is a core dependency
    user_cache_dir = None

# --- SAM2 ------------------------------------------------------------------
SAM2_CHECKPOINT = "sam2_hiera_large.pt"
SAM2_URL = "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2_hiera_large.pt"
SAM2_REPO = "https://github.com/facebookresearch/segment-anything-2"
SAM2_REL_DIR = os.path.join("external", "segment_anything_2", "checkpoints")

# --- CoTracker3 ------------------------------------------------------------
COTRACKER_CHECKPOINT = "scaled_offline.pth"
COTRACKER_URL = "https://huggingface.co/facebook/cotracker3/resolve/main/scaled_offline.pth"
COTRACKER_REPO = "https://github.com/facebookresearch/co-tracker"
COTRACKER_REL_DIR = os.path.join("external", "co-tracker", "checkpoints")

# --- Helpers ---------------------------------------------------------------
DOWNLOAD_SCRIPT = "python install.py --checkpoints-only"


def checkpoints_dir():
    """Directory where downloaded checkpoints should live when there's no
    repo checkout to place them next to (e.g. a plain `pip install`).

    Callers that care about a git-checkout layout should keep checking their
    existing SAM2_REL_DIR/COTRACKER_REL_DIR candidates first and only fall
    back to this location.
    """
    if user_cache_dir is None:
        return os.path.join(os.path.expanduser("~"), ".cache", "autosegmentor", "checkpoints")
    return os.path.join(user_cache_dir("autosegmentor"), "checkpoints")


def missing_model_message(name, tried_paths):
    """Build a log/console message telling the user a model is missing and how
    to install it (download link, upstream repo, and expected location)."""
    name_l = name.lower()
    if "sam" in name_l:
        url, repo, rel_dir, ckpt = SAM2_URL, SAM2_REPO, SAM2_REL_DIR, SAM2_CHECKPOINT
    elif "cotracker" in name_l:
        url, repo, rel_dir, ckpt = COTRACKER_URL, COTRACKER_REPO, COTRACKER_REL_DIR, COTRACKER_CHECKPOINT
    else:
        url = repo = rel_dir = ckpt = "?"

    expected = os.path.join(rel_dir, ckpt)
    tried = "\n    ".join(str(p) for p in tried_paths) if tried_paths else "n/a"

    return (
        f"{name} model checkpoint NOT FOUND.\n"
        f"  Tried paths:\n    {tried}\n"
        f"  Expected location: {expected}\n"
        f"  How to install / download:\n"
        f"    1. Run the helper script:  {DOWNLOAD_SCRIPT}\n"
        f"       (creates the folders and downloads both checkpoints automatically)\n"
        f"    2. Manual download from the original source:  {url}\n"
        f"    3. Upstream repo:  {repo}\n"
        f"  Place the file at: {expected}"
    )