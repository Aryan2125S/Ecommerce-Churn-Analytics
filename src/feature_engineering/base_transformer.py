"""
Base Feature Transformer Module.

Provides an abstract base class for all Phase 5 Feature Engineering modules.
By inheriting from scikit-learn's BaseEstimator and TransformerMixin, this
base class ensures that all custom feature calculators (RFM, Engagement, Risk)
are fully compatible with scikit-learn Pipelines.

It enforces the architectural rule of separating `.fit()` from `.transform()`
to prevent data leakage, and centralizes validation and state tracking.

Usage:
    class MyCustomTransformer(BaseFeatureTransformer):
        def __init__(self, required_columns):
            super().__init__(required_columns)
            
        def fit(self, X, y=None):
            self._validate_input(X)
            # Learn boundaries...
            self.is_fitted = True
            return self
            
        def transform(self, X):
            self._check_is_fitted()
            self._validate_input(X)
            # Apply transformations...
            return X_transformed
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from src.utils.exceptions import PreprocessingError
from src.utils.logger import get_logger

logger = get_logger(__name__)


class BaseFeatureTransformer(ABC, BaseEstimator, TransformerMixin):
    """Abstract base class for all custom feature engineering transformers."""

    def __init__(self, required_columns: list[str] | None = None) -> None:
        """Initialize the transformer with required columns.

        Args:
            required_columns: List of columns that must exist in the DataFrame.
        """
        self.required_columns = required_columns or []
        self.is_fitted = False

    @abstractmethod
    def fit(self, X: pd.DataFrame, y: Any = None) -> BaseFeatureTransformer:
        """Learn parameters from the training data.

        Args:
            X: The training pandas DataFrame.
            y: Ignored. Exists for scikit-learn pipeline compatibility.

        Returns:
            The fitted transformer instance.
        """
        pass

    @abstractmethod
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply learned parameters to transform the data.

        Args:
            X: The data to transform.

        Returns:
            The transformed pandas DataFrame.
        """
        pass

    def _check_is_fitted(self) -> None:
        """Verify the transformer has been fitted before calling transform.
        
        Raises:
            PreprocessingError: If transform is called before fit.
        """
        if not getattr(self, "is_fitted", False):
            logger.error(f"Attempted to transform using unfitted {self.__class__.__name__}.")
            raise PreprocessingError(
                f"This {self.__class__.__name__} instance is not fitted yet. "
                "Call 'fit' with appropriate arguments before using this estimator."
            )

    def _validate_input(self, X: pd.DataFrame) -> None:
        """Ensure the input is a valid DataFrame and contains required columns.
        
        Raises:
            PreprocessingError: If input is invalid or missing columns.
        """
        if X is None or X.empty:
            raise PreprocessingError(f"{self.__class__.__name__} received an empty DataFrame.")
            
        if self.required_columns:
            missing = set(self.required_columns) - set(X.columns)
            if missing:
                raise PreprocessingError(
                    f"{self.__class__.__name__} is missing required columns: {missing}"
                )
