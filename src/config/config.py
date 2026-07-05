"""
Centralized Configuration Management Module.

This module provides a single source of truth for all application settings.
It loads environment variables from a .env file, applies type-safe defaults,
resolves filesystem paths relative to the project root, and exposes structured
configuration groups via immutable dataclasses.

Usage:
    from src.config.config import settings

    db_url = settings.database.connection_url
    seed = settings.model.random_state
    raw_path = settings.paths.raw_data_dir

Design Decisions:
    - Singleton pattern ensures configuration is loaded exactly once and
      shared across all modules without redundant disk I/O.
    - Frozen dataclasses enforce immutability after initialization,
      preventing accidental mutation of settings at runtime.
    - All paths are resolved to absolute pathlib.Path objects at load time
      so downstream consumers never handle relative path ambiguity.
    - Every setting has a sensible default so the application can run
      in a zero-configuration local environment without a .env file.

Dependencies:
    - python-dotenv (loads .env files into os.environ)
    - pathlib (stdlib, cross-platform path resolution)
    - dataclasses (stdlib, structured immutable containers)
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv


# ---------------------------------------------------------------------------
# Project Root Resolution
# ---------------------------------------------------------------------------
# Resolve the project root directory by walking up from this file's location.
# This file lives at: <project_root>/src/config/config.py
# Therefore, project root is three levels up from this file.
# ---------------------------------------------------------------------------
_PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent.parent


def _resolve_path(relative_path: str) -> Path:
    """Resolve a relative path string against the project root directory.

    Args:
        relative_path: A path string relative to the project root.

    Returns:
        An absolute Path object anchored to the project root.
    """
    return (_PROJECT_ROOT / relative_path).resolve()


def _get_env_str(key: str, default: str = "") -> str:
    """Retrieve an environment variable as a string.

    Args:
        key: The environment variable name.
        default: Fallback value if the variable is not set.

    Returns:
        The environment variable value or the default.
    """
    return os.getenv(key, default)


def _get_env_int(key: str, default: int = 0) -> int:
    """Retrieve an environment variable as an integer.

    Args:
        key: The environment variable name.
        default: Fallback value if the variable is not set or not numeric.

    Returns:
        The parsed integer value or the default.
    """
    raw = os.getenv(key)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _get_env_float(key: str, default: float = 0.0) -> float:
    """Retrieve an environment variable as a float.

    Args:
        key: The environment variable name.
        default: Fallback value if the variable is not set or not numeric.

    Returns:
        The parsed float value or the default.
    """
    raw = os.getenv(key)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _get_env_bool(key: str, default: bool = False) -> bool:
    """Retrieve an environment variable as a boolean.

    Truthy values: "true", "1", "yes" (case-insensitive).
    All other non-empty values are treated as False.

    Args:
        key: The environment variable name.
        default: Fallback value if the variable is not set.

    Returns:
        The parsed boolean value or the default.
    """
    raw = os.getenv(key)
    if raw is None:
        return default
    return raw.strip().lower() in ("true", "1", "yes")


# ===========================================================================
# Configuration Dataclasses
# ===========================================================================
# Each group is a frozen dataclass to enforce immutability after creation.
# Fields are populated from environment variables with safe defaults.
# ===========================================================================


@dataclass(frozen=True)
class DatabaseConfig:
    """Database connection and engine configuration.

    Attributes:
        engine: Database backend identifier ('sqlite' or 'mysql').
        host: Database server hostname.
        port: Database server port number.
        name: Database / schema name.
        user: Database authentication username.
        password: Database authentication password.
    """

    engine: str
    host: str
    port: int
    name: str
    user: str
    password: str

    @property
    def is_sqlite(self) -> bool:
        """Check if the configured engine is SQLite."""
        return self.engine.lower() == "sqlite"

    @property
    def is_mysql(self) -> bool:
        """Check if the configured engine is MySQL."""
        return self.engine.lower() == "mysql"

    @property
    def connection_url(self) -> str:
        """Build a SQLAlchemy-compatible connection URL.

        Returns:
            A fully qualified database connection string.
            For SQLite, returns a file-based URL pointing to the database
            directory inside the project root.
            For MySQL, returns a PyMySQL-based connection URL.
        """
        if self.is_sqlite:
            db_path = _resolve_path("database") / f"{self.name}.db"
            return f"sqlite:///{db_path}"
        return (
            f"mysql+pymysql://{self.user}:{self.password}"
            f"@{self.host}:{self.port}/{self.name}"
        )


@dataclass(frozen=True)
class AppConfig:
    """Application-level settings controlling runtime behavior.

    Attributes:
        env: Deployment environment name (development, staging, production).
        debug: Flag to enable verbose debug-level output.
        log_level: Minimum severity for log emission (DEBUG, INFO, WARNING, ERROR).
        log_dir: Absolute path to the rotating log file directory.
    """

    env: str
    debug: bool
    log_level: str
    log_dir: Path

    @property
    def is_production(self) -> bool:
        """Check if the application is running in production mode."""
        return self.env.lower() == "production"

    @property
    def is_development(self) -> bool:
        """Check if the application is running in development mode."""
        return self.env.lower() == "development"


@dataclass(frozen=True)
class ModelConfig:
    """Machine learning pipeline configuration.

    Attributes:
        model_dir: Absolute path to the serialized model export directory.
        best_model_name: Filename for the best trained model artifact.
        random_state: Global random seed for reproducibility across all
                      stochastic operations (data splitting, model training,
                      SMOTE sampling).
        test_size: Proportion of the dataset reserved for the holdout test set.
        cv_folds: Number of stratified cross-validation folds.
    """

    model_dir: Path
    best_model_name: str
    random_state: int
    test_size: float
    cv_folds: int

    @property
    def best_model_path(self) -> Path:
        """Full absolute path to the best model artifact file."""
        return self.model_dir / self.best_model_name

    @property
    def feature_names_path(self) -> Path:
        """Full absolute path to the serialized feature names artifact."""
        return self.model_dir / "feature_names.joblib"

    @property
    def label_encoders_path(self) -> Path:
        """Full absolute path to the serialized label encoders artifact."""
        return self.model_dir / "label_encoders.joblib"

    @property
    def scaler_path(self) -> Path:
        """Full absolute path to the serialized scaler artifact."""
        return self.model_dir / "scaler.joblib"

    @property
    def model_config_path(self) -> Path:
        """Full absolute path to the serialized model configuration snapshot."""
        return self.model_dir / "model_config.joblib"


@dataclass(frozen=True)
class PathsConfig:
    """Filesystem path configuration for data directories.

    Attributes:
        project_root: Absolute path to the project root directory.
        raw_data_dir: Absolute path to the raw data input directory.
        processed_data_dir: Absolute path to the processed data output directory.
        external_data_dir: Absolute path to external/supplementary data directory.
        database_dir: Absolute path to the SQL scripts directory.
        reports_dir: Absolute path to the reports output directory.
        reports_assets_dir: Absolute path to the report image assets directory.
        images_dir: Absolute path to the project images directory.
    """

    project_root: Path
    raw_data_dir: Path
    processed_data_dir: Path
    external_data_dir: Path
    database_dir: Path
    reports_dir: Path
    reports_assets_dir: Path
    images_dir: Path

    @property
    def raw_dataset_path(self) -> Path:
        """Full path to the primary raw dataset CSV file."""
        return self.raw_data_dir / "ecommerce_customers.csv"

    @property
    def cleaned_dataset_path(self) -> Path:
        """Full path to the cleaned dataset CSV file."""
        return self.processed_data_dir / "cleaned_customers.csv"

    @property
    def engineered_dataset_path(self) -> Path:
        """Full path to the feature-engineered dataset CSV file."""
        return self.processed_data_dir / "engineered_customers.csv"


@dataclass(frozen=True)
class DataGenerationConfig:
    """Synthetic data generator configuration.

    Attributes:
        dataset_size: Number of customer records to generate.
    """

    dataset_size: int


@dataclass(frozen=True)
class StreamlitConfig:
    """Streamlit dashboard application configuration.

    Attributes:
        theme: Dashboard color theme identifier ('light' or 'dark').
        port: Local development server port number.
    """

    theme: str
    port: int


# ===========================================================================
# Root Settings Container
# ===========================================================================


@dataclass(frozen=True)
class Settings:
    """Root configuration container aggregating all setting groups.

    This is the primary interface for accessing application configuration.
    All downstream modules should import and reference the module-level
    ``settings`` singleton rather than constructing Settings directly.

    Attributes:
        database: Database connection parameters.
        app: Application runtime flags and paths.
        model: Machine learning pipeline parameters.
        paths: Filesystem directory paths.
        data_generation: Data generator parameters.
        streamlit: Dashboard application parameters.
    """

    database: DatabaseConfig
    app: AppConfig
    model: ModelConfig
    paths: PathsConfig
    data_generation: DataGenerationConfig
    streamlit: StreamlitConfig


def _load_settings() -> Settings:
    """Load all configuration from environment variables and .env file.

    This function is called exactly once at module import time. It reads
    the .env file (if present) from the project root, parses all environment
    variables into their respective typed dataclasses, and returns a fully
    initialized, immutable Settings instance.

    Returns:
        A frozen Settings dataclass containing all application configuration.
    """
    # Load .env file from project root (silently skip if not present)
    dotenv_path = _PROJECT_ROOT / ".env"
    load_dotenv(dotenv_path=dotenv_path, override=False)

    # --- Database ---
    database = DatabaseConfig(
        engine=_get_env_str("DB_ENGINE", "sqlite"),
        host=_get_env_str("DB_HOST", "localhost"),
        port=_get_env_int("DB_PORT", 3306),
        name=_get_env_str("DB_NAME", "ecommerce_churn"),
        user=_get_env_str("DB_USER", "root"),
        password=_get_env_str("DB_PASSWORD", ""),
    )

    # --- Application ---
    app = AppConfig(
        env=_get_env_str("APP_ENV", "development"),
        debug=_get_env_bool("APP_DEBUG", True),
        log_level=_get_env_str("APP_LOG_LEVEL", "INFO").upper(),
        log_dir=_resolve_path(_get_env_str("LOG_DIR", "logs")),
    )

    # --- Model ---
    model = ModelConfig(
        model_dir=_resolve_path(_get_env_str("MODEL_DIR", "models/export")),
        best_model_name=_get_env_str(
            "BEST_MODEL_NAME", "best_churn_model.joblib"
        ),
        random_state=_get_env_int("RANDOM_STATE", 42),
        test_size=_get_env_float("TEST_SIZE", 0.2),
        cv_folds=_get_env_int("CV_FOLDS", 5),
    )

    # --- Paths ---
    paths = PathsConfig(
        project_root=_PROJECT_ROOT,
        raw_data_dir=_resolve_path(
            _get_env_str("RAW_DATA_DIR", "data/raw")
        ),
        processed_data_dir=_resolve_path(
            _get_env_str("PROCESSED_DATA_DIR", "data/processed")
        ),
        external_data_dir=_resolve_path(
            _get_env_str("EXTERNAL_DATA_DIR", "data/external")
        ),
        database_dir=_resolve_path("database"),
        reports_dir=_resolve_path("reports"),
        reports_assets_dir=_resolve_path("reports/assets"),
        images_dir=_resolve_path("images"),
    )

    # --- Data Generation ---
    data_generation = DataGenerationConfig(
        dataset_size=_get_env_int("DATASET_SIZE", 50000),
    )

    # --- Streamlit ---
    streamlit = StreamlitConfig(
        theme=_get_env_str("STREAMLIT_THEME", "light"),
        port=_get_env_int("STREAMLIT_PORT", 8501),
    )

    return Settings(
        database=database,
        app=app,
        model=model,
        paths=paths,
        data_generation=data_generation,
        streamlit=streamlit,
    )


# ===========================================================================
# Module-Level Singleton
# ===========================================================================
# Instantiated once on first import. All modules should use this instance.
# ===========================================================================

settings: Settings = _load_settings()
