# Changelog

All notable changes to **AutoSegmentor** are documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/).

---

## [3.0.0] - 2026-09-08

### Added

- **Versioning metadata**
  - `autosegmentor/_version.py` as the single source of truth.
  - `autosegmentor.__version__` exposed via `autosegmentor/__init__.py`.
  - `python run_main.py --version` reports the installed version.
- **Industrial / material-handling demos** (flagship release focus)
  - `--demo industrial-warehouse`: moving forklift + pallet footage processed
    with SAM2 + CoTracker3 (pallet masks, 6-point pallet pose, 3-point forklift
    pose, YOLO pose export).
  - `--demo pallet-closeup`: a static, clearly visible wooden pallet stack
    ideal for **creating your own pallet labels** and training a pallet pose
    model.
  - Multi-demo registry (`autosegmentor/tools/demo_registry.py`) and
    `python run_main.py --demo list` to enumerate demos.
- **Packaging / distribution**
  - New `pyproject.toml`, `setup.py`, and `MANIFEST` configuration.
  - Clean `requirements-core.txt` (in addition to the frozen
    `requriments_i_used.txt`).
  - `packaging/auto-segmentor.spec` + `build_windows.bat` / `build_linux.sh`
    for PyInstaller Windows and Linux builds.
  - `autosegmentor` and `autosegmentor-demo` console entry points.
- **Documentation**
  - `CHANGELOG.md`.
  - `README.md` "Material Handling & Industrial Automation" section.
  - `demo/README.md` and `demo/videos/README.md` with footage attribution.

### Changed

- `run_main.py` refactored into a callable `main()`.
- `--demo` now accepts an optional demo name for selecting scenarios.
- Demo session-state configs carry a `demo` metadata block (name, description, license).

### Notes

- Large ML checkpoints are intentionally **not** bundled in PyInstaller builds;
  they are downloaded separately (see README).

---

## [2.0.0] - Prior releases (v0.1 → v2)

See git history and tags for earlier changes:
SAM2 + CoTracker3 integration, keypoint tracking with Lucas-Kanade fallback,
Scalable Model Routing, automated demo pipeline, YOLO dataset export, synthetic
data engine, and the modular `autosegmentor` package architecture.
