# Installation

Tested on **Windows 11** and **Ubuntu 22.04 / 24.04**.

## Prerequisites

| | Requirement |
| :--- | :--- |
| Python | 3.10+ |
| GPU | NVIDIA GPU with CUDA 12.x (SAM2/CoTracker are far too slow on CPU alone) |
| RAM | 16GB+ recommended |
| Disk | ~20GB (model checkpoints + working directories) |
| Git | [Git LFS](https://git-lfs.com/) — only needed if you plan to contribute new demo footage |

## Quick install from PyPI

For the full annotation engine — SAM2 mask generation **and** CoTracker3 pose tracking —
without cloning the repo:

```bash
pip install autosegmentor
```

Both SAM2 and CoTracker3 are bundled as real packages (no `external/` checkout needed).
Model checkpoints are never part of the package regardless of install method; download them
with `python install.py --checkpoints-only` (covered further down), which now defaults to a
user cache directory rather than a path inside a checkout.

The rest of this page — the full git clone workflow — is what you want if you plan to run
the demos, use the Dataset Manager, the Synthetic Engine, or contribute changes. Bundling
those into the PyPI package too is planned for a future release.

## 1. Clone the repository

Cloning with `--recursive` fetches the vendored CoTracker3 submodule. (SAM2 is vendored
directly in the repo, not as a submodule.)

```bash
git clone --recursive https://github.com/thippeswammy/AutoSegmentor.git
cd AutoSegmentor
```

Forgot `--recursive`? Fetch the submodule after the fact:

```bash
git submodule update --init --recursive
```

## 2. Create a virtual environment

=== "Windows (PowerShell)"

    ```powershell
    python -m venv .venv
    .\.venv\Scripts\Activate.ps1
    ```

=== "Ubuntu / Linux"

    ```bash
    python3 -m venv .venv
    source .venv/bin/activate
    ```

## 3. Run the one-shot setup

```bash
python install.py
```

This single, cross-platform script — no separate `.bat`/`.sh` files to keep in sync —
takes care of everything else:

1. Checks your Python version.
2. Initializes git submodules.
3. Installs dependencies from `requirements-core.txt`.
4. Downloads the SAM2 and CoTracker3 model checkpoints.
5. Runs a GPU/CUDA diagnostic so you know before you launch whether inference will be fast
   or painfully slow.

Add `--cuda` to also install the optional `flash-attn` speedup — it needs a matching CUDA
build toolchain, which is more commonly already set up on Linux than on Windows, so treat it
as optional either way.

Only need part of it? Every step also runs standalone:

```bash
python install.py --checkpoints-only            # download missing checkpoints
python install.py --checkpoints-only --check    # just verify they're present
python install.py --gpu-check-only              # just the GPU/system diagnostic
```

## 4. Verify

```bash
python run_main.py --version     # AutoSegmentor 3.0.1
python run_main.py --demo list   # cat, road
```

If that prints a version and two demo names, you're ready — head to [Demos](demos.md).

## Running on your own video (not a demo)

The bundled demos (`--demo cat` / `--demo road`) are a fixed showcase. For your own
footage, it's just as simple:

1. Drop your video(s) into `workspace/VideoInputs/`.
2. Run:

   ```bash
   python run_main.py
   ```

That's it. With no `--demo` flag, this opens the **Setup Dialog** — a full configuration
window where you pick which video(s) to process, toggle SAM2 mask generation and CoTracker3
pose tracking on/off, set the run mode (full pipeline / mask-only / pose-only), batch size,
and pose keypoint classes. Confirm your settings there and the annotation window opens next
— click points, `Enter` to propagate, `Ctrl+S` to save, `Ctrl+E` to export, same as the
demos.

(The Setup Dialog reads/writes `workspace/inputs/config/default_config.yaml` — editing that
file directly works too, if you'd rather script it than click through the dialog each time.)

## Doing it by hand

Everything `install.py` automates, spelled out, in case you want full control:

```bash
python -m pip install --upgrade pip
pip install -r requirements-core.txt
```

SAM2 and CoTracker3 aren't listed in `requirements-core.txt` as separate dependencies —
both live under `external/` in the checkout. If you just run `python run_main.py` without
also installing the `autosegmentor` package itself, `run_main.py` adds them to `sys.path`
at launch and you only need the folders present (the clone step above handles that). If you
do run `pip install -e .` (see below), they're picked up as real installed packages
instead — the same `sam2`/`cotracker` packages that ship on PyPI, just sourced from your
checkout.

Download the two checkpoints manually if you'd rather not run the script:

| Model | File | Source |
| :--- | :--- | :--- |
| SAM2 | `sam2_hiera_large.pt` | [facebookresearch/segment-anything-2](https://github.com/facebookresearch/segment-anything-2) |
| CoTracker3 | `scaled_offline.pth` | [facebookresearch/co-tracker](https://github.com/facebookresearch/co-tracker) |

Place them at:

```text
external/segment_anything_2/checkpoints/sam2_hiera_large.pt
external/co-tracker/checkpoints/scaled_offline.pth
```

Optionally install AutoSegmentor as a package for the `autosegmentor` / `autosegmentor-demo`
console commands:

```bash
pip install -e .
```

## Troubleshooting

**Linux: `could not load the Qt platform plugin "xcb"`** — a minimal Ubuntu install is
missing PyQt5's system dependencies:

```bash
sudo apt-get install libxcb-xinerama0 libxcb-cursor0 libxkbcommon-x11-0
```

**VRAM out of memory** — lower `batch_size` in your config (try 8 or 16).

**Everything runs, but painfully slowly** — check `python install.py --gpu-check-only`
reports `CUDA available: True`. CPU-only inference works but is not a realistic way to run
this tool on real footage.
