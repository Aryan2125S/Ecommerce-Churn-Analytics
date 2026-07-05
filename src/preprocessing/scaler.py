"""
Continuous Feature Scaling Module.

This module provides a reusable, configurable scaling class for normalizing
continuous numerical features. It supports multiple scaling strategies such as
Standardization, Min-Max normalization, and Robust scaling.

CRITICAL ARCHITECTURAL RULE (DATA LEAKAGE PREVENTION):
The `fit()` operation must NEVER be executed on the entire dataset prior to 
the train/test split. This class is designed to be instantiated in the 
preprocessing pipeline but explicitly fitted ONLY on `X_train` during Phase 7.
The test set (`X_test`) must solely use `.transform()`.

Usage:
    from src.preprocessing.scaler import FeatureScaler
    
    scaler = FeatureScaler(numerical_columns=["age", "total_spend"], method="robust")
    
    # IN PHASE 7 ONLY:
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Persist the fitted scaler for production inference
    scaler.save("models/export/scaler.joblib")

Dependencies:
    - pandas
    - scikit-learn (StandardScaler, MinMaxScaler, RobustScaler)
    - joblib (for artifact persistence)
    - src.utils.exceptions (PreprocessingError)
    - src.utils.logger
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import joblib
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler

from src.utils.exceptions import PreprocessingError
from src.utils.logger import get_logger

logger = get_logger(__name__)


class FeatureScaler:
    """Configurable scaler for normalizing continuous numerical features."""

    def __init__(
        self,
        numerical_columns: list[str],
        method: Literal["standard", "minmax", "robust"] = "standard"
    ) -> None:
        """Initialize the feature scaler.

        Args:
            numerical_columns: A list of continuous numerical columns to scale.
            method: The scaling strategy to apply ("standard", "minmax", or "robust").
            
        Raises:
            PreprocessingError: If the numerical_columns list is empty.
        """
        if not numerical_columns:
            raise PreprocessingError("numerical_columns list cannot be empty.")
            
        self.numerical_columns = numerical_columns
        self.method = method
        self.is_fitted = False
        
        # Instantiate the appropriate scikit-learn scaler
        if self.method == "standard":
            self.scaler = StandardScaler()
        elif self.method == "minmax":
            self.scaler = MinMaxScaler()
        elif self.method == "robust":
            # RobustScaler handles remaining outliers elegantly using IQR
            self.scaler = RobustScaler()
        else:
            raise PreprocessingError(f"Unsupported scaling method: {self.method}")
            
        logger.debug(f"Initialized FeatureScaler using method: '{self.method}'.")

    def fit(self, df: pd.DataFrame) -> FeatureScaler:
        """Fit the scaler strictly on the training data.

        Args:
            df: The training pandas DataFrame.

        Returns:
            The fitted FeatureScaler instance.

        Raises:
            PreprocessingError: If any target columns are missing or non-numerical.
        """
        self._validate_columns(df)
        
        logger.info(
            f"Fitting {self.scaler.__class__.__name__} on "
            f"{len(self.numerical_columns)} columns."
        )
        self.scaler.fit(df[self.numerical_columns])
        self.is_fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform data using the fitted scaler.

        Args:
            df: The DataFrame to transform (can be train, test, or inference set).

        Returns:
            A new pandas DataFrame with the specified numerical columns scaled.

        Raises:
            PreprocessingError: If transform is called before fit.
        """
        if not self.is_fitted:
            raise PreprocessingError(
                "Scaler must be fitted on training data before calling transform()."
            )
            
        self._validate_columns(df)
        
        logger.debug(f"Transforming dataset of shape {df.shape} via {self.method} scaler.")
        
        # Operate on a copy to avoid SettingWithCopyWarnings
        df_out = df.copy()
        scaled_array = self.scaler.transform(df[self.numerical_columns])
        
        # Replace original columns with scaled values
        df_out[self.numerical_columns] = scaled_array
        return df_out

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fit the scaler and immediately transform the training data.

        Args:
            df: The training pandas DataFrame.

        Returns:
            The scaled training DataFrame.
        """
        return self.fit(df).transform(df)

    def save(self, filepath: str | Path) -> None:
        """Persist the fitted scaler to disk using Joblib.

        Args:
            filepath: The destination file path (.joblib).

        Raises:
            PreprocessingError: If the scaler has not been fitted or save fails.
        """
        if not self.is_fitted:
            raise PreprocessingError("Cannot save an unfitted scaler.")
            
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            joblib.dump(self, path)
            logger.info(f"Fitted FeatureScaler successfully saved to {path}.")
        except Exception as e:
            logger.error(f"Failed to persist scaler: {e}")
            raise PreprocessingError(f"Joblib save failed: {e}") from e

    @classmethod
    def load(cls, filepath: str | Path) -> FeatureScaler:
        """Load a fitted scaler from disk.

        Args:
            filepath: The file path of the saved .joblib scaler.

        Returns:
            The loaded FeatureScaler instance.

        Raises:
            PreprocessingError: If the file does not exist or load fails.
        """
        path = Path(filepath)
        if not path.exists():
            raise PreprocessingError(f"Scaler file not found at {path}.")
            
        try:
            scaler = joblib.load(path)
            logger.info(f"FeatureScaler successfully loaded from {path}.")
            return scaler
        except Exception as e:
            logger.error(f"Failed to load scaler: {e}")
            raise PreprocessingError(f"Joblib load failed: {e}") from e

    def _validate_columns(self, df: pd.DataFrame) -> None:
        """Ensure all required columns exist and are numerical."""
        missing = set(self.numerical_columns) - set(df.columns)
        if missing:
            raise PreprocessingError(f"Cannot scale missing columns: {missing}")
            
        for col in self.numerical_columns:
            if not pd.api.types.is_numeric_dtype(df[col]):
                raise PreprocessingError(
                    f"Column '{col}' must be numerical to apply scaling. "
                    f"Found dtype: {df[col].dtype}"
                )
