"""
setup.py
========
Almost all packaging metadata lives declaratively in pyproject.toml
([project] / [tool.setuptools]). This file exists only to supply the one
thing pyproject.toml's declarative packages.find can't express: sam2 and
sam2_configs are top-level Python packages (kept top-level because SAM2's own
Hydra configs resolve them by that name — see docs/architecture.md), but
their source lives under external/segment_anything_2/, not the repo root.
`package_dir` maps the two independently, everything else stays declarative.
"""

from setuptools import find_packages, setup

SAM2_ROOT = "external/segment_anything_2"

packages = (
    find_packages(include=["autosegmentor", "autosegmentor.*"])
    + find_packages(where=SAM2_ROOT, include=["sam2", "sam2.*", "sam2_configs", "sam2_configs.*"])
)

package_dir = {
    "sam2": f"{SAM2_ROOT}/sam2",
    "sam2_configs": f"{SAM2_ROOT}/sam2_configs",
}

setup(
    packages=packages,
    package_dir=package_dir,
)
