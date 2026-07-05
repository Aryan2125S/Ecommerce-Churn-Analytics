"""
Outlier Handling Module.

This module is responsible for detecting and treating statistical outliers
in continuous numerical variables prior to model training. Left untreated,
extreme outliers can severely distort gradient-based models (like Logistic
Regression or Neural Networks) and skew scaling algorithms.

This class supports multiple detection methods (IQR, Z-Score) and treatment
strategies (Keep, Remove, Cap/Winsorize), allowing configuration per feature
to preserve business realism (e.g., highly loyal customers naturally have
extreme total spend, which should be capped rather than removed).

Usage:
    from src.preprocessing.outlier_handler import OutlierHandler
    
    handler = OutlierHandler(df)
    df_treated = handler.treat_feature("total_spend", method="iqr", strategy="cap")

Dependencies:
    - pandas (for vector manipulation)
    - numpy (for statistical calculations)
    - src.utils.exceptions (for PreprocessingError)
    - src.utils.logger (for execution logging)
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd

from src.utils.exceptions import PreprocessingError
from src.utils.logger import get_logger

logger = get_logger(__name__)


class OutlierHandler:
    """Detects and handles outliers in numerical pandas DataFrames."""

    def __init__(self, df: pd.DataFrame) -> None:
        """Initialize the OutlierHandler.

        Args:
            df: The pandas DataFrame to process.
        
        Raises:
            PreprocessingError: If the DataFrame is empty.
        """
        if df is None or df.empty:
            logger.error("OutlierHandler initialized with an empty DataFrame.")
            raise PreprocessingError("Input DataFrame cannot be empty.")
        
        self.df = df.copy()

    def get_dataframe(self) -> pd.DataFrame:
        """Retrieve the modified DataFrame.

        Returns:
            The pandas DataFrame containing the treated features.
        """
        return self.df

    def treat_feature(
        self,
        column_name: str,
        method: Literal["iqr", "zscore"] = "iqr",
        strategy: Literal["keep", "remove", "cap"] = "cap",
        threshold: float | None = None
    ) -> pd.DataFrame:
        """Detect and treat outliers for a specific numerical column.

        Args:
            column_name: The name of the column to treat.
            method: The statistical detection method ("iqr" or "zscore").
            strategy: The treatment strategy ("keep", "remove", "cap").
                      "cap" applies Winsorization at the boundary limits.
            threshold: The multiplier for the method. Defaults to 1.5 for IQR
                       and 3.0 for Z-Score if not explicitly provided.

        Returns:
            The updated DataFrame.

        Raises:
            PreprocessingError: If the column is missing or non-numerical.
        """
        if column_name not in self.df.columns:
            raise PreprocessingError(f"Column '{column_name}' not found in DataFrame.")

        if not pd.api.types.is_numeric_dtype(self.df[column_name]):
            raise PreprocessingError(
                f"Column '{column_name}' must be numeric to perform outlier handling. "
                f"Found dtype: {self.df[column_name].dtype}"
            )

        logger.info(
            f"Treating outliers for '{column_name}' using method='{method}' "
            f"and strategy='{strategy}'."
        )

        series = self.df[column_name]
        
        # 1. Calculate boundaries based on the selected method
        if method == "iqr":
            lower_bound, upper_bound = self._calculate_iqr_bounds(series, threshold or 1.5)
        elif method == "zscore":
            lower_bound, upper_bound = self._calculate_zscore_bounds(series, threshold or 3.0)
        else:
            raise PreprocessingError(f"Unsupported detection method: '{method}'")

        # 2. Identify outliers
        outlier_mask = (series < lower_bound) | (series > upper_bound)
        outlier_count = outlier_mask.sum()
        
        if outlier_count == 0:
            logger.info(f"No outliers detected in '{column_name}'.")
            return self.df

        # 3. Apply the treatment strategy
        if strategy == "keep":
            logger.info(
                f"Strategy is 'keep'. Identified {outlier_count} outliers in "
                f"'{column_name}' but applied no transformations."
            )
        elif strategy == "remove":
            self.df = self.df[~outlier_mask].copy()
            logger.warning(f"Removed {outlier_count} rows containing outliers in '{column_name}'.")
        elif strategy == "cap":
            # Winsorization: clip values to the calculated bounds
            self.df[column_name] = np.clip(series, lower_bound, upper_bound)
            logger.info(
                f"Capped (Winsorized) {outlier_count} outliers in '{column_name}' "
                f"to bounds [{lower_bound:.2f}, {upper_bound:.2f}]."
            )
        else:
            raise PreprocessingError(f"Unsupported treatment strategy: '{strategy}'")

        return self.df

    def _calculate_iqr_bounds(self, series: pd.Series, multiplier: float) -> tuple[float, float]:
        """Calculate the lower and upper bounds using the Interquartile Range."""
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        
        lower_bound = q1 - (multiplier * iqr)
        upper_bound = q3 + (multiplier * iqr)
        
        return lower_bound, upper_bound

    def _calculate_zscore_bounds(self, series: pd.Series, z_threshold: float) -> tuple[float, float]:
        """Calculate the lower and upper bounds using Z-Score."""
        mean = series.mean()
        std = series.std()
        
        # Guard against zero standard deviation
        if std == 0:
            return mean, mean
            
        lower_bound = mean - (z_threshold * std)
        upper_bound = mean + (z_threshold * std)
        
        return lower_bound, upper_bound
