"""
Structured logging for benchmark runs.
Provides dual-handler logging: colored console output + persistent file logs.
"""

import logging
import os
from datetime import datetime


def setup_logger(name: str = "bench", log_dir: str = "logs",
                 verbose: bool = False) -> logging.Logger:
    """
    Configure and return a logger with console + file handlers.

    Args:
        name: Logger name
        log_dir: Directory for log files
        verbose: If True, set console to DEBUG level
    """
    logger = logging.getLogger(name)

    # Avoid duplicate handlers on repeated calls
    if logger.handlers:
        return logger

    logger.setLevel(logging.DEBUG)

    # ─── Console Handler ──────────────────────────────────
    console = logging.StreamHandler()
    console.setLevel(logging.DEBUG if verbose else logging.INFO)
    console_fmt = logging.Formatter(
        "%(asctime)s │ %(levelname)-7s │ %(message)s",
        datefmt="%H:%M:%S"
    )
    console.setFormatter(console_fmt)
    logger.addHandler(console)

    # ─── File Handler ─────────────────────────────────────
    os.makedirs(log_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath = os.path.join(log_dir, f"bench_{ts}.log")

    file_handler = logging.FileHandler(filepath, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_fmt = logging.Formatter(
        "%(asctime)s │ %(levelname)-7s │ %(name)s │ %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    file_handler.setFormatter(file_fmt)
    logger.addHandler(file_handler)

    logger.info(f"Log file: {filepath}")
    return logger
