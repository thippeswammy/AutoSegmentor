# AutoSegmentor Packaging (v3.0.0)

This directory contains everything needed to build a distributable
AutoSegmentor binary with PyInstaller for **Windows** and **Linux**.

## What is included in the build

- The full `autosegmentor` Python package and `run_main.py` launcher
- Vendored SAM2 (`sam2`) and CoTracker3 (`cotracker`) library code
- Bundled demo configs and demo videos (`demo/`)
- Runtime workspace configs (`workspace/inputs/config/`)

## What is NOT bundled

- **Model checkpoints** (`sam2_hiera_large.pt`, `scaled_offline.pth`) are too
  large to ship inside the binary. They are resolved at runtime from the
  project paths described in the README (place them in
  `external/segment_anything_2/checkpoints/` and `external/co-tracker/checkpoints/`).
- A GPU + CUDA runtime is required for performance.

## Building

### Windows
```bat
packaging\build_windows.bat
```

### Linux
```bash
chmod +x packaging/build_linux.sh
./packaging/build_linux.sh
```

Build output is written to `build/AutoSegmentor/`.

## Running the built binary

```
AutoSegmentor.exe                # Standard GUI
AutoSegmentor.exe --demo         # Default (cat) demo
AutoSegmentor.exe --demo industrial-warehouse
AutoSegmentor.exe --demo list
AutoSegmentor.exe --version
```
