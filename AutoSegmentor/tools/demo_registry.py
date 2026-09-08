"""
demo_registry.py
================
Central registry and loader for the named AutoSegmentor demos.

Each demo is backed by a session-state JSON under the top-level ``demo/``
directory which the ``--demo <name>`` CLI flag can launch. This module resolves
a friendly demo name to its configuration file and exposes available demos.

NOTE: This module lives inside ``autosegmentor`` rather than the top-level
``demo/`` package on purpose. The ``external/co-tracker`` library ships a
``demo.py`` module that would shadow a top-level ``demo`` package on
``sys.path``, so we avoid importing a package by that name entirely.
"""

import os
import json
import logging

logger = logging.getLogger("DemoRegistry")

# Location of the session-state JSON files for each demo.
# __file__ = autosegmentor/tools/demo_registry.py -> ../.. = project root
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DEMO_DIR = os.path.join(_ROOT, "demo")


def available_demos():
    """Return a dict of demo-name -> session-state path for all bundled demos."""
    demos = {}
    if not os.path.isdir(DEMO_DIR):
        return demos
    for entry in sorted(os.listdir(DEMO_DIR)):
        if not entry.endswith("_session_state.json"):
            continue
        path = os.path.join(DEMO_DIR, entry)
        try:
            with open(path, "r", encoding="utf-8") as f:
                cfg = json.load(f)
        except Exception as e:
            logger.warning(f"Could not parse demo config {path}: {e}")
            continue
        demo_meta = cfg.get("demo", {})
        name = demo_meta.get("name") or os.path.splitext(entry)[0]
        demos[name] = {
            "path": path,
            "description": demo_meta.get("description", ""),
            "license": demo_meta.get("license", ""),
        }
    return demos


def resolve_demo(name):
    """Resolve a friendly demo name to its session-state config dict.

    Returns the loaded config dict, or None if the named demo does not exist.
    """
    demos = available_demos()
    if not name or name not in demos:
        logger.error(
            f"Unknown demo '{name}'. Available demos: {', '.join(demos) or 'none'}"
        )
        return None
    cfg_path = demos[name]["path"]
    try:
        with open(cfg_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Failed to load demo config from {cfg_path}: {e}")
        return None


def default_demo_name():
    """Return the name of the default demo (used when --demo is passed with no value)."""
    return "cat"
