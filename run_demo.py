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
import argparse
import json

def load_demo_config():
    demo_cfg_path = os.path.join(ROOT_DIR, "demo", "demo_session_state.json")
    try:
        with open(demo_cfg_path, "r") as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Failed to load demo config from {demo_cfg_path}: {e}")
        return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AutoSegmentor Launcher")
    parser.add_argument("--demo", action="store_true", help="Run in automated demo mode")
    args = parser.parse_args()

    if args.demo:
        logger.info("🚀 Launching Automated Demo Pipeline...")
        demo_config = load_demo_config()
        if demo_config:
            start_application(config_override=demo_config)
        else:
            logger.error("Could not start demo: Config missing or invalid.")
    else:
        logger.info("🚀 Launching Standard AutoSegmentor Application...")
        start_application()
