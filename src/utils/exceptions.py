"""
Custom Exception Hierarchy Module.

This module defines a domain-specific exception hierarchy for the Enterprise
E-Commerce Customer Churn Analytics Platform. By inheriting from a single
base class, we ensure that all custom exceptions can be caught globally
at the application boundaries (e.g., top-level try/except in main scripts
or Streamlit global error handlers).

The hierarchy is organized by architectural layers:
    - Base Error
    - Configuration & Environment
    - Data Infrastructure (Database, Generators)
    - Data Processing (Cleaning, Feature Engineering)
    - Machine Learning (Training, Evaluation, Prediction, Artifacts)
    - Reporting & Visualization

Usage:
    from src.utils.exceptions import DatabaseConnectionError

    try:
        engine.connect()
    except Exception as e:
        raise DatabaseConnectionError(f"Failed to reach DB: {e}") from e

Dependencies:
    - None (Standard library only)
"""

from __future__ import annotations


# ===========================================================================
# Base Exception
# ===========================================================================


class ChurnAnalyticsBaseError(Exception):
    """Base class for all custom exceptions in the Churn Analytics Platform.

    All custom exceptions should inherit from this class. This allows
    global error handlers to easily distinguish between domain-specific
    application errors and standard Python runtime exceptions.
    """


# ===========================================================================
# Configuration & Environment Errors
# ===========================================================================


class ConfigurationError(ChurnAnalyticsBaseError):
    """Raised when there is an invalid application configuration.

    This includes missing environment variables, conflicting settings,
    or out-of-bounds configuration values.
    """


# ===========================================================================
# Data Infrastructure Errors
# ===========================================================================


class DataGenerationError(ChurnAnalyticsBaseError):
    """Raised when the synthetic data generator fails.

    This could be due to invalid statistical parameters, memory allocation
    failures during large dataset generation, or I/O errors during saving.
    """


class DatabaseError(ChurnAnalyticsBaseError):
    """Base class for all database-related operations."""


class DatabaseConnectionError(DatabaseError):
    """Raised when the application fails to connect to the SQL database.

    Examples: Incorrect credentials, unreachable host, or missing SQLite file.
    """


class DatabaseQueryError(DatabaseError):
    """Raised when an SQL query or DataFrame insertion fails.

    Examples: Schema mismatch, duplicate primary keys, or invalid SQL syntax.
    """


# ===========================================================================
# Data Processing Errors
# ===========================================================================


class PreprocessingError(ChurnAnalyticsBaseError):
    """Base class for data cleaning and transformation errors."""


class DataCleaningError(PreprocessingError):
    """Raised during the data cleaning pipeline.

    Examples: Missing mandatory columns, invalid data types, or failure
    during outlier winsorization.
    """


class FeatureEngineeringError(PreprocessingError):
    """Raised during the feature engineering calculations.

    Examples: Division by zero when calculating ratios, missing demographic
    data required for RFM scoring, or invalid aggregation logic.
    """


# ===========================================================================
# Machine Learning Pipeline Errors
# ===========================================================================


class ModelPipelineError(ChurnAnalyticsBaseError):
    """Base class for machine learning pipeline errors."""


class ModelTrainingError(ModelPipelineError):
    """Raised when an ML algorithm fails during the training phase.

    Examples: Convergence failures, memory exhaustion during cross-validation,
    or invalid hyperparameter combinations.
    """


class ModelEvaluationError(ModelPipelineError):
    """Raised when metric calculation or SHAP explainability fails.

    Examples: Mismatched shape between predictions and true labels, or
    TreeExplainer failures on unsupported model architectures.
    """


class PredictionError(ModelPipelineError):
    """Raised during real-time inference on new customer data.

    Examples: Missing features in the prediction payload, unrecognized
    categorical levels, or scaling transformation failures.
    """


class ModelArtifactError(ModelPipelineError):
    """Raised when saving or loading serialized model artifacts fails.

    Examples: Corrupted .joblib files, missing artifact directories, or
    version mismatches between scikit-learn training and inference envs.
    """


# ===========================================================================
# Reporting & Visualization Errors
# ===========================================================================


class VisualizationError(ChurnAnalyticsBaseError):
    """Raised when chart generation fails.

    Examples: Matplotlib backend errors, invalid plot dimensions, or
    missing data series required for plotting.
    """


class ReportGenerationError(ChurnAnalyticsBaseError):
    """Raised when exporting PDF, PPTX, or CSV reports fails.

    Examples: Missing image assets required for the slide deck, PDF layout
    overflow, or file permission denied during export.
    """
