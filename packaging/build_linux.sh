#!/usr/bin/env bash
# ============================================================
#  AutoSegmentor 3.0.0 - PyInstaller build (Linux)
#  Produces build/AutoSegmentor/AutoSegmentor (one-folder)
# ============================================================
set -euo pipefail
cd "$(dirname "$0")/.."

echo "[1/3] Creating virtual environment (if missing)..."
if [ ! -x ".venv/bin/python" ]; then
    python3 -m venv .venv
fi

echo "[2/3] Installing dependencies..."
source .venv/bin/activate
python -m pip install --upgrade pip
pip install "pyinstaller>=6.0"
pip install -r requirements-core.txt

echo "[3/3] Building executable..."
pyinstaller --clean --noconfirm packaging/auto-segmentor.spec

echo
echo "Build complete. Output: build/AutoSegmentor/AutoSegmentor"
echo "NOTE: Model checkpoints are NOT bundled. Download them per README and"
echo "      place under external/.../checkpoints/."
