@echo off
REM ============================================================
REM  AutoSegmentor 3.0.0 - PyInstaller build (Windows)
REM  Produces build\AutoSegmentor\AutoSegmentor.exe (one-folder)
REM ============================================================
setlocal

cd /d "%~dp0\.."

echo [1/3] Creating virtual environment (if missing)...
if not exist ".venv\Scripts\python.exe" (
    python -m venv .venv
)

echo [2/3] Installing dependencies...
call ".venv\Scripts\activate.bat"
python -m pip install --upgrade pip
pip install "pyinstaller>=6.0"
pip install -r requirements-core.txt

echo [3/3] Building executable...
pyinstaller --clean --noconfirm packaging\auto-segmentor.spec

echo.
echo Build complete. Output: build\AutoSegmentor\AutoSegmentor.exe
echo NOTE: Model checkpoints are NOT bundled. Download them per README and
echo       place under external\...\checkpoints\.
endlocal
