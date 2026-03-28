import logging
import os

logger = logging.getLogger("SAM2")

# Avoid adding duplicate handlers when module is re-imported
if not logger.handlers:
    _fmt = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    # Console handler — INFO and above visible in terminal
    _ch = logging.StreamHandler()
    _ch.setLevel(logging.INFO)
    _ch.setFormatter(_fmt)
    logger.addHandler(_ch)

    # File handler — DEBUG and above written to autosegmentor.log
    # This captures all traceback errors even when the UI freezes
    _log_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'outputs', 'logs')
    os.makedirs(_log_dir, exist_ok=True)
    _fh = logging.FileHandler(os.path.join(_log_dir, 'autosegmentor.log'), encoding='utf-8')
    _fh.setLevel(logging.DEBUG)
    _fh.setFormatter(_fmt)
    logger.addHandler(_fh)

    logger.setLevel(logging.DEBUG)
    logger.propagate = False  # Prevent double-printing via root logger
