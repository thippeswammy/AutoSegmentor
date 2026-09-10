"""
setup.py
========
Almost all packaging metadata lives declaratively in pyproject.toml
([project] / [tool.setuptools]). This file exists only to supply the one
thing pyproject.toml's declarative packages.find can't express: sam2,
sam2_configs, and cotracker are top-level Python packages (sam2/sam2_configs
are kept top-level because SAM2's own Hydra configs resolve them by that
name — see docs/architecture.md; cotracker is kept top-level to match its
own upstream import structure), but their source lives under
external/segment_anything_2/ and external/co-tracker/, not the repo root.
`package_dir` maps each independently, everything else stays declarative.
"""

from setuptools import find_packages, setup

SAM2_ROOT = "external/segment_anything_2"
COTRACKER_ROOT = "external/co-tracker"

packages = (
    find_packages(include=["autosegmentor", "autosegmentor.*"])
    + find_packages(where=SAM2_ROOT, include=["sam2", "sam2.*", "sam2_configs", "sam2_configs.*"])
    + find_packages(where=COTRACKER_ROOT, include=["cotracker", "cotracker.*"])
)

package_dir = {
    "sam2": f"{SAM2_ROOT}/sam2",
    "sam2_configs": f"{SAM2_ROOT}/sam2_configs",
    "cotracker": f"{COTRACKER_ROOT}/cotracker",
}

setup(
    packages=packages,
    package_dir=package_dir,
)
