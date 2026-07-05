"""
Centralized Logging Utility Module.

This module provides a unified logging interface for the entire application.
It configures a Python standard library logger with two handlers:
    1. A StreamHandler for console output (stdout).
    2. A RotatingFileHandler for persistent file-based logging.

The logger automatically respects the global configuration defined in
`settings.py`, adjusting log levels and output directories dynamically.
By centralizing the logger configuration here, we ensure consistent
formatting and behavior across all architectural layers.

Usage:
    from src.utils.logger import get_logger

    logger = get_logger(__name__)
    logger.info("Application started successfully.")
    logger.error("Failed to connect to the database.", exc_info=True)

Dependencies:
    - logging, logging.handlers (Python stdlib)
    - sys (for stdout stream handling)
    - src.config.settings (for application configuration)
"""

from __future__ import annotations

import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Optional

from src.config.settings import settings


# ===========================================================================
# Log Formatting Constants
# ===========================================================================
# Define a standard format for all log messages to ensure consistency
# across both console and file outputs.
# Format: [Timestamp] [LoggerName] [LogLevel] Message
# ===========================================================================

_LOG_FORMAT: str = "[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s"
_DATE_FORMAT: str = "%Y-%m-%d %H:%M:%S"

# RotatingFileHandler settings
_MAX_BYTES: int = 5 * 1024 * 1024  # 5 MB
_BACKUP_COUNT: int = 5             # Keep 5 backup files


def get_logger(name: str, log_file_name: str = "app.log") -> logging.Logger:
    """Retrieve a configured logger instance for the given module name.

    If the logger is already configured (has handlers), it is returned as-is
    to prevent duplicate log entries. Otherwise, it is initialized with a
    console handler and a rotating file handler based on application settings.

    Args:
        name: The name of the logger, typically __name__ of the calling module.
        log_file_name: The name of the log file to write to. Defaults to "app.log".

    Returns:
        A fully configured logging.Logger instance.
    """
    logger = logging.getLogger(name)

    # Prevent duplicate handlers if get_logger is called multiple times
    # for the same module name.
    if logger.hasHandlers():
        return logger

    # Map the string log level from settings to the logging module integer constant
    level_name = settings.app.log_level.upper()
    log_level = getattr(logging, level_name, logging.INFO)
    logger.setLevel(log_level)

    # Do not propagate to the root logger to avoid duplicate output
    # if the root logger is configured elsewhere.
    logger.propagate = False

    formatter = logging.Formatter(fmt=_LOG_FORMAT, datefmt=_DATE_FORMAT)

    # -----------------------------------------------------------------------
    # Console Handler (StreamHandler)
    # -----------------------------------------------------------------------
    try:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(log_level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    except Exception as e:
        # Fallback error handling if stdout stream attachment fails
        print(f"Failed to initialize console logger: {e}", file=sys.stderr)

    # -----------------------------------------------------------------------
    # File Handler (RotatingFileHandler)
    # -----------------------------------------------------------------------
    try:
        # Ensure the log directory exists before attempting to write
        log_dir: Path = settings.app.log_dir
        if not log_dir.exists():
            log_dir.mkdir(parents=True, exist_ok=True)

        log_file_path: Path = log_dir / log_file_name

        file_handler = RotatingFileHandler(
            filename=str(log_file_path),
            mode="a",
            maxBytes=_MAX_BYTES,
            backupCount=_BACKUP_COUNT,
            encoding="utf-8",
        )
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    except OSError as e:
        # If file permissions or path resolution fails, log to console
        logger.warning(
            f"Failed to initialize file logger at {log_dir}. "
            f"Logging will continue to console only. Error: {e}"
        )
    except Exception as e:
        logger.error(f"Unexpected error initializing file logger: {e}")

    return logger
