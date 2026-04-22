"""
run_demo.py
===========
Root-level entry point for the AutoSegmentor application.
"""

import sys
import os

# Add the current directory to sys.path to allow absolute imports
sys.path.append(os.path.dirname(__file__))

from AutoSegmentor.utils.Tools.main_app import start_application

if __name__ == "__main__":
    start_application()
