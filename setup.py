"""Empty shim so legacy pip workflows (``pip install -e .``, setup.py-based tools) work.

The authoritative build configuration lives in ``pyproject.toml``.
"""

from setuptools import setup

if __name__ == "__main__":
    setup()
