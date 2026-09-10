# Changelog

All notable changes to **AutoSegmentor** are documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/).

---

## [3.0.1] - 2026-09-10

### Added

- **PyPI release**: `pip install autosegmentor` now works standalone, with no
  git checkout required — first public release to PyPI.
  ([autosegmentor on PyPI](https://pypi.org/project/autosegmentor/))
- SAM2 and CoTracker3 are bundled as real installable top-level packages
  (`sam2`, `sam2_configs`, `cotracker`) sourced from their existing
  `external/segment_anything_2/` and `external/co-tracker/` locations via
  `setup.py` `package_dir` mapping — no files were moved, so the git-checkout
  workflow is unaffected.
- `autosegmentor.models.model_info.checkpoints_dir()`: a platform-appropriate
  user cache directory (via `platformdirs`) used as the checkpoint download
  location when there's no `external/` checkout to place them next to (e.g. a
  plain `pip install`). Checkpoints already at the old repo-relative path are
  still recognized as-is.

### Fixed

- `requirements-core.txt` no longer lists `flash-attn` unconditionally — it
  requires `torch` to already be importable during its own isolated build, so
  a plain `pip install -r requirements-core.txt` (or `python install.py`)
  failed outright on a clean environment. It's installed separately via
  `python install.py --cuda` or the `cuda` extra, after torch is present.
- `torch`/`torchvision` pinned to a specific CUDA build (`+cu118`) instead of
  an unbounded `>=` constraint, which could let pip's resolver silently drift
  to an incompatible non-CUDA build.
- `install.py`'s docs/messages no longer incorrectly call SAM2 a git
  submodule (only `co-tracker` is one; SAM2 is bundled directly).
- Consolidated four separate copies of CoTracker checkpoint path-guessing
  logic (in `AutoSegmentorEngine.py`, `UserInteraction.py`,
  `PoseExporter.py`) into `CoTrackerPredictor.resolve_cotracker_checkpoint()`.

## [3.0.0] - 2026-09-08

### Added

- **Versioning metadata**
  - `autosegmentor/_version.py` as the single source of truth.
  - `autosegmentor.__version__` exposed via `autosegmentor/__init__.py`.
  - `python run_main.py --version` reports the installed version.
- **Demo**
  - `--demo cat`: classic segmentation/pose demo on the bundled cat clip.
  - `--demo road`: segmentation/pose demo on the bundled dashcam road clip.
  - Multi-demo registry (`autosegmentor/tools/demo_registry.py`) and
    `python run_main.py --demo list` to enumerate demos.
- **Packaging / distribution**
  - New `pyproject.toml`, `setup.py`, and `MANIFEST` configuration.
  - Clean `requirements-core.txt`.
  - `packaging/auto-segmentor.spec` + `build_windows.bat` / `build_linux.sh`
    for PyInstaller Windows and Linux builds.
  - `autosegmentor` and `autosegmentor-demo` console entry points.
- **Documentation**
  - `CHANGELOG.md`.
  - `README.md` "Automated Demo" and "Material Handling & Industrial
    Automation" sections.
  - `demo/README.md` and `demo/videos/README.md` with footage attribution.
  - `assets/AutoSegmenterCat.mp4` / `AutoSegmenterRoad.mp4`: full-quality
    showcase recordings (via Git LFS) embedded directly in the README and
    docs site, with poster thumbnails — full README walkthrough of the `cat`
    demo: UI annotation, SAM2/CoTracker auto-tracking, correction, and
    export to YOLO (detection + segmentation + pose). See `assets/README.md`.

### Changed

- `run_main.py` refactored into a callable `main()`.
- `--demo` now accepts an optional demo name for selecting scenarios.
- Demo session-state configs carry a `demo` metadata block (name, description, license).
- README clarifies that YOLO export covers detection (bbox), instance
  segmentation, and pose simultaneously.
- `pyproject.toml` and `requirements-core.txt` dependency lists reconciled so
  `pip install .` and `pip install -r requirements-core.txt` produce the same
  environment.

### Removed

- `requriments_i_used.txt`, a personal `pip freeze` dump of the author's local
  environment — not release material.
- The editable `cotracker` pip install from `requirements-core.txt`. CoTracker3
  is only ever loaded from the vendored `external/co-tracker` submodule (added
  to `sys.path` at launch), so the pip install was unused dead weight.

### Fixed

- `demo/road_demo_session_state.json` was accidentally deleted in a later
  refactor commit even though `--demo road` remained documented everywhere;
  restored so `python run_main.py --demo road` works again.

### Notes

- Large ML checkpoints are intentionally **not** bundled in PyInstaller builds;
  they are downloaded separately (see README).

---

## [2.0.0] - Prior releases (v0.1 → v2)

See git history and tags for earlier changes:
SAM2 + CoTracker3 integration, keypoint tracking with Lucas-Kanade fallback,
Scalable Model Routing, automated demo pipeline, YOLO dataset export, synthetic
data engine, and the modular `autosegmentor` package architecture.
