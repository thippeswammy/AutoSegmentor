# -*- mode: python ; coding: utf-8 -*-
#
# PyInstaller spec for AutoSegmentor 3.0.0.
#
# NOTE: Large ML model checkpoints are intentionally NOT bundled to keep the
# binary a reasonable size. They are resolved at runtime from
# `external/segment_anything_2/checkpoints/` and
# `external/co-tracker/checkpoints/` (see README download instructions).
#
# Build (Windows):
#     pyinstaller --clean --noconfirm packaging/auto-segmentor.spec
#
# Build (Linux):
#     pyinstaller --clean --noconfirm packaging/auto-segmentor.spec

import os
from PyInstaller.utils.hooks import collect_submodules, collect_data_files

root = os.path.abspath(os.path.join(SPECPATH, ".."))

# Collect all vendored third-party packages as hidden imports/data.
hidden_imports = []
datas = []

for pkg in ["sam2", "cotracker"]:
    hidden_imports += collect_submodules(pkg)
    datas += collect_data_files(pkg)

# Bundle the autosegmentor package itself.
for sub in [
    "autosegmentor.core",
    "autosegmentor.file_management",
    "autosegmentor.models.SAM",
    "autosegmentor.models.Tracking",
    "autosegmentor.ui",
    "autosegmentor.tools",
]:
    hidden_imports += collect_submodules(sub)
    datas += collect_data_files(sub)

# Bundle demo session configs and demo videos.
for entry in os.listdir(os.path.join(root, "demo")):
    p = os.path.join(root, "demo", entry)
    if os.path.isfile(p) and entry.endswith(("_session_state.json", ".md")):
        datas.append((p, "demo"))
videos_dir = os.path.join(root, "demo", "videos")
if os.path.isdir(videos_dir):
    for v in os.listdir(videos_dir):
        if v.endswith(".mp4"):
            datas.append((os.path.join(videos_dir, v), "demo/videos"))

# Bundle workspace runtime config (default_config.yaml, session_state.json).
cfg_dir = os.path.join(root, "workspace", "inputs", "config")
if os.path.isdir(cfg_dir):
    for cfg_file in os.listdir(cfg_dir):
        if cfg_file.endswith((".yaml", ".yml", ".json")):
            datas.append((os.path.join(cfg_dir, cfg_file), "workspace/inputs/config"))

block_cipher = None

a = Analysis(
    [os.path.join(root, "run_main.py")],
    pathex=[root],
    binaries=[],
    datas=datas,
    hiddenimports=hidden_imports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=["matplotlib.tests", "numpy.tests", "pytest", "tkinter"],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="AutoSegmentor",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name="AutoSegmentor",
)
