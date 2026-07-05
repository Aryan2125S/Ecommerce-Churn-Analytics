"""
Application Settings and Runtime Initialization Module.

This module serves as the application bootstrap layer that sits above
config.py and constants.py. While config.py loads environment variables
into typed dataclasses and constants.py defines immutable domain values,
this module is responsible for:

    1. Validating that loaded configuration values are within acceptable
       ranges and constraints before the application proceeds.
    2. Initializing the runtime environment: creating required directories,
       setting global random seeds, configuring library defaults (matplotlib
       backend, pandas display options, warning filters).
    3. Providing a single ``initialize()`` entry point that downstream
       modules (main.py, dashboard app.py) call once at startup to ensure
       the application environment is fully prepared.
    4. Re-exporting the ``settings`` singleton and key constant classes
       for convenient single-import access.

Usage:
    from src.config.settings import initialize, settings, ColumnNames

    # Call once at application startup
    initialize()

    # Then use settings and constants as needed
    raw_path = settings.paths.raw_data_dir
    target = ColumnNames.CHURN

Dependencies:
    - src.config.config (Settings singleton)
    - src.config.constants (Domain constants)
    - pathlib, os, warnings (stdlib)
"""

from __future__ import annotations

import os
import warnings
from pathlib import Path
from typing import Optional

from src.config.config import Settings, _load_settings, settings
from src.config.constants import (
    BusinessRules,
    ColorPalette,
    ColumnNames,
    DashboardPages,
    FeatureNames,
    FileNames,
    Geography,
    KPILabels,
    MetricNames,
    ModelRegistry,
    UITokens,
)


# ===========================================================================
# Re-exports for Convenience
# ===========================================================================
# Downstream modules can import everything they need from this single module:
#   from src.config.settings import settings, ColumnNames, ColorPalette
# ===========================================================================

__all__ = [
    "settings",
    "initialize",
    "validate_settings",
    "ensure_directories",
    "configure_runtime",
    "ColumnNames",
    "FeatureNames",
    "ColorPalette",
    "UITokens",
    "ModelRegistry",
    "MetricNames",
    "FileNames",
    "DashboardPages",
    "KPILabels",
    "BusinessRules",
    "Geography",
]


# ===========================================================================
# Configuration Validation
# ===========================================================================


class SettingsValidationError(Exception):
    """Raised when a configuration value fails validation constraints."""


def validate_settings(config: Optional[Settings] = None) -> list[str]:
    """Validate all configuration values against business constraints.

    Checks that numeric settings fall within acceptable ranges, required
    string values are non-empty, and path configurations are syntactically
    valid. This function does NOT check whether directories or files
    actually exist on disk — that responsibility belongs to
    ``ensure_directories()``.

    Args:
        config: The Settings instance to validate. Defaults to the
                module-level singleton if not provided.

    Returns:
        A list of human-readable warning messages for non-critical issues.
        Critical failures raise SettingsValidationError immediately.

    Raises:
        SettingsValidationError: If any configuration value violates a
            hard constraint that would cause application failure.
    """
    cfg = config or settings
    validation_warnings: list[str] = []

    # --- Model Configuration Validation ---
    if not (0.0 < cfg.model.test_size < 1.0):
        raise SettingsValidationError(
            f"TEST_SIZE must be between 0.0 and 1.0 exclusive, "
            f"got {cfg.model.test_size}"
        )

    if cfg.model.cv_folds < 2:
        raise SettingsValidationError(
            f"CV_FOLDS must be at least 2 for cross-validation, "
            f"got {cfg.model.cv_folds}"
        )

    if cfg.model.cv_folds > 20:
        validation_warnings.append(
            f"CV_FOLDS={cfg.model.cv_folds} is unusually high. "
            f"Values above 10 significantly increase training time."
        )

    if cfg.model.random_state < 0:
        raise SettingsValidationError(
            f"RANDOM_STATE must be a non-negative integer, "
            f"got {cfg.model.random_state}"
        )

    # --- Data Generation Validation ---
    if cfg.data_generation.dataset_size < 1000:
        raise SettingsValidationError(
            f"DATASET_SIZE must be at least 1000 for meaningful "
            f"model training, got {cfg.data_generation.dataset_size}"
        )

    if cfg.data_generation.dataset_size > 500000:
        validation_warnings.append(
            f"DATASET_SIZE={cfg.data_generation.dataset_size} is very large. "
            f"Generation and training may take considerable time."
        )

    # --- Database Configuration Validation ---
    valid_engines = ("sqlite", "mysql")
    if cfg.database.engine.lower() not in valid_engines:
        raise SettingsValidationError(
            f"DB_ENGINE must be one of {valid_engines}, "
            f"got '{cfg.database.engine}'"
        )

    if cfg.database.is_mysql and not cfg.database.password:
        validation_warnings.append(
            "DB_PASSWORD is empty for MySQL engine. "
            "Connection may fail without valid credentials."
        )

    if cfg.database.is_mysql and cfg.database.port < 1:
        raise SettingsValidationError(
            f"DB_PORT must be a positive integer for MySQL, "
            f"got {cfg.database.port}"
        )

    # --- Application Configuration Validation ---
    valid_log_levels = ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL")
    if cfg.app.log_level not in valid_log_levels:
        raise SettingsValidationError(
            f"APP_LOG_LEVEL must be one of {valid_log_levels}, "
            f"got '{cfg.app.log_level}'"
        )

    valid_environments = ("development", "staging", "production")
    if cfg.app.env.lower() not in valid_environments:
        validation_warnings.append(
            f"APP_ENV='{cfg.app.env}' is not a recognized environment. "
            f"Expected one of {valid_environments}."
        )

    # --- Streamlit Validation ---
    if not (1024 <= cfg.streamlit.port <= 65535):
        validation_warnings.append(
            f"STREAMLIT_PORT={cfg.streamlit.port} is outside the "
            f"recommended range [1024, 65535]."
        )

    # --- Best Model Name Validation ---
    if not cfg.model.best_model_name.endswith(".joblib"):
        validation_warnings.append(
            f"BEST_MODEL_NAME='{cfg.model.best_model_name}' does not end "
            f"with '.joblib'. Serialization may use an unexpected format."
        )

    return validation_warnings


# ===========================================================================
# Directory Initialization
# ===========================================================================


def ensure_directories(config: Optional[Settings] = None) -> list[Path]:
    """Create all required application directories if they do not exist.

    This function is idempotent — calling it multiple times is safe.
    It creates data, model, log, report, and image directories to ensure
    the application can write outputs without encountering missing
    directory errors at runtime.

    Args:
        config: The Settings instance to read paths from. Defaults to
                the module-level singleton if not provided.

    Returns:
        A list of Path objects for directories that were newly created.
        Existing directories are silently skipped.
    """
    cfg = config or settings

    required_dirs: list[Path] = [
        cfg.paths.raw_data_dir,
        cfg.paths.processed_data_dir,
        cfg.paths.external_data_dir,
        cfg.paths.database_dir,
        cfg.paths.reports_dir,
        cfg.paths.reports_assets_dir,
        cfg.paths.images_dir,
        cfg.app.log_dir,
        cfg.model.model_dir,
    ]

    created_dirs: list[Path] = []

    for directory in required_dirs:
        if not directory.exists():
            directory.mkdir(parents=True, exist_ok=True)
            created_dirs.append(directory)

    return created_dirs


# ===========================================================================
# Runtime Environment Configuration
# ===========================================================================


def configure_runtime(config: Optional[Settings] = None) -> None:
    """Configure global runtime defaults for third-party libraries.

    Sets matplotlib to a non-interactive backend (Agg) for headless
    chart generation, adjusts pandas display options for readable
    DataFrame output in logs, seeds numpy's global random state for
    reproducibility, and suppresses noisy library warnings that clutter
    production logs.

    Args:
        config: The Settings instance to read parameters from. Defaults
                to the module-level singleton if not provided.
    """
    cfg = config or settings

    # --- Matplotlib: Use non-interactive backend for server-side rendering ---
    try:
        import matplotlib
        matplotlib.use("Agg")
    except ImportError:
        pass  # matplotlib is optional at import time

    # --- Pandas: Readable display defaults for logging and debugging ---
    try:
        import pandas as pd
        pd.set_option("display.max_columns", 50)
        pd.set_option("display.max_rows", 100)
        pd.set_option("display.width", 200)
        pd.set_option("display.float_format", lambda x: f"{x:.4f}")
        pd.set_option("display.max_colwidth", 50)
    except ImportError:
        pass  # pandas is optional at import time

    # --- NumPy: Global random seed for reproducibility ---
    try:
        import numpy as np
        np.random.seed(cfg.model.random_state)
    except ImportError:
        pass  # numpy is optional at import time

    # --- Warnings: Suppress noisy convergence and deprecation warnings ---
    if not cfg.app.debug:
        warnings.filterwarnings("ignore", category=FutureWarning)
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        warnings.filterwarnings(
            "ignore",
            message=".*convergence.*",
            category=UserWarning,
        )

    # --- Environment Variables for Library Configuration ---
    # Suppress TensorFlow/XGBoost verbose logging in production
    if cfg.app.is_production:
        os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
        os.environ.setdefault("PYTHONHASHSEED", str(cfg.model.random_state))


# ===========================================================================
# Unified Initialization Entry Point
# ===========================================================================


_initialized: bool = False


def initialize(config: Optional[Settings] = None) -> None:
    """Initialize the complete application environment.

    This function must be called exactly once at application startup,
    before any data processing, model training, or dashboard rendering
    begins. It performs the following steps in order:

        1. Validate all configuration values.
        2. Create required output directories.
        3. Configure runtime library defaults.

    Subsequent calls are silently ignored to prevent duplicate
    initialization in environments where multiple modules import
    this function.

    Args:
        config: Optional Settings instance override. Defaults to the
                module-level singleton.

    Raises:
        SettingsValidationError: If any critical configuration constraint
            is violated.
    """
    global _initialized

    if _initialized:
        return

    cfg = config or settings

    # Step 1: Validate configuration
    validation_warnings = validate_settings(cfg)

    # Step 2: Ensure directories exist
    ensure_directories(cfg)

    # Step 3: Configure runtime environment
    configure_runtime(cfg)

    # Step 4: Print validation warnings to stderr (logger not yet available)
    for warning_msg in validation_warnings:
        warnings.warn(warning_msg, UserWarning, stacklevel=2)

    _initialized = True
