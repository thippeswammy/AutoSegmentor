"""
run_demo.py
===========
Main entry point for the AutoSegmentor application.
"""

import sys
import os
import yaml
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("Bootstrap")

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(ROOT_DIR, "workspace", "inputs", "config", "default_config.yaml")

external_libs = []
try:
    with open(CONFIG_PATH, "r") as f:
        cfg = yaml.safe_load(f)
        if cfg and "external_libs" in cfg:
            external_libs = cfg["external_libs"]
except Exception as e:
    logger.error(f"Failed to load config from {CONFIG_PATH}: {e}")

# Resolve paths to absolute and add to sys.path
for lib in external_libs:
    lib_path = os.path.abspath(os.path.join(ROOT_DIR, lib))
    logger.info(f"Adding external library to sys.path: {lib_path}")
    if lib_path not in sys.path:
        sys.path.insert(0, lib_path)

if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from autosegmentor.tools.main_app import start_application

if __name__ == "__main__":
    start_application()
