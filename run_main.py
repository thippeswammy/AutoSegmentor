"""
run_main.py
===========
Main entry point for the AutoSegmentor application.

Usage:
    python run_main.py                     # Standard GUI
    python run_main.py --demo              # Default (cat) demo
    python run_main.py --demo cat          # Cat demo
    python run_main.py --demo road         # Road/dashcam demo
    python run_main.py --demo list         # List available demos
    python run_main.py --version
"""

import sys
import os
import yaml
import logging
import argparse

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("Bootstrap")

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(ROOT_DIR, "workspace", "inputs", "config", "default_config.yaml")


def _setup_sys_path():
    """Register external libraries and project root on sys.path."""
    external_libs = []
    try:
        with open(CONFIG_PATH, "r") as f:
            cfg = yaml.safe_load(f)
            if cfg and "external_libs" in cfg:
                external_libs = cfg["external_libs"]
    except Exception as e:
        logger.error(f"Failed to load config from {CONFIG_PATH}: {e}")

    for lib in external_libs:
        lib_path = os.path.abspath(os.path.join(ROOT_DIR, lib))
        if not os.path.isdir(lib_path):
            # Expected for a plain `pip install` with no external/ checkout —
            # sam2 packages properly and doesn't need this, and co-tracker is
            # an optional plugin that may simply not be present.
            logger.info(f"Skipping missing external library path: {lib_path}")
            continue
        logger.info(f"Adding external library to sys.path: {lib_path}")
        if lib_path not in sys.path:
            sys.path.insert(0, lib_path)

    if ROOT_DIR not in sys.path:
        sys.path.insert(0, ROOT_DIR)


def _print_demos():
    from autosegmentor.tools.demo_registry import available_demos
    demos = available_demos()
    if not demos:
        logger.info("No bundled demos found.")
        return
    logger.info("Available demos:")
    for name, meta in demos.items():
        desc = meta.get("description") or ""
        license_note = meta.get("license") or ""
        logger.info(f"  - {name}: {desc} [{license_note}]")


def parse_args(argv=None):
    from autosegmentor import __version__

    parser = argparse.ArgumentParser(
        description="AutoSegmentor Launcher",
        epilog="Examples:\n"
               "  python run_main.py                     # Standard GUI\n"
               "  python run_main.py --demo              # Default (cat) demo\n"
               "  python run_main.py --demo cat          # Cat demo\n"
               "  python run_main.py --demo road         # Road/dashcam demo\n"
               "  python run_main.py --demo list         # List available demos",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--version", action="version", version=f"AutoSegmentor {__version__}"
    )
    parser.add_argument(
        "--demo", nargs="?", const="__default__",
        help="Run in automated demo mode. Optionally specify a demo name "
             "(e.g. cat, road, list).",
    )
    return parser.parse_args(argv)


def main(argv=None):
    from autosegmentor.tools.main_app import start_application
    from autosegmentor.tools.demo_registry import resolve_demo, default_demo_name
    from autosegmentor import __version__

    _setup_sys_path()
    args = parse_args(argv)

    if args.demo:
        demo_name = args.demo if args.demo != "__default__" else default_demo_name()
        if demo_name == "list":
            _print_demos()
            sys.exit(0)
        logger.info(f"🚀 Launching Automated Demo Pipeline (version {__version__})...")
        demo_config = resolve_demo(demo_name)
        if demo_config:
            start_application(config_override=demo_config)
        else:
            logger.error("Could not start demo: Config missing or invalid.")
            _print_demos()
            sys.exit(1)
    else:
        logger.info(f"🚀 Launching Standard AutoSegmentor Application (version {__version__})...")
        start_application()


if __name__ == "__main__":
    main()
