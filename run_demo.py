"""
run_demo.py
===========
Main entry point for the AutoSegmentor application.
"""

import sys
import os

# Ensure the project root and sam2 source are in sys.path
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
# Adding the sam2 source directory directly to sys.path
SAM2_SRC = os.path.join(ROOT_DIR, "segment_anything_2")
SAM2_MODEL_SRC = os.path.join(SAM2_SRC, "sam2")

for d in [ROOT_DIR, SAM2_SRC, SAM2_MODEL_SRC]:
    if d not in sys.path:
        sys.path.insert(0, d)

from AutoSegmentor.utils.Tools.main_app import start_application

if __name__ == "__main__":
    start_application()
