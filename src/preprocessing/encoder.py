"""
Categorical Encoding Module.

This module provides reusable encoding classes for transforming categorical 
variables into numerical representations suitable for machine learning models.

CRITICAL ARCHITECTURAL RULE:
To prevent data leakage, encoders defined in this module must NEVER be fitted
on the entire dataset prior to a train/test split. They are designed to be 
instantiated here, but fitted exclusively on the training fold during Phase 7 
(Model Training). The test fold must only call `.transform()`.

Usage:
    from src.preprocessing.encoder import CategoricalEncoder
    
    encoder = CategoricalEncoder(categorical_columns=["city", "gender"])
    # IN PHASE 7 ONLY:
    X_train_encoded = encoder.fit_transform(X_train)
    X_test_encoded = encoder.transform(X_test)

Dependencies:
    - pandas
    - scikit-learn (OneHotEncoder)
    - src.utils.exceptions
    - src.utils.logger
"""

from __future__ import annotations

import pandas as pd
from sklearn.preprocessing import OneHotEncoder

from src.utils.exceptions import PreprocessingError
from src.utils.logger import get_logger

logger = get_logger(__name__)


class CategoricalEncoder:
    """Wrapper for scikit-learn encoders to process pandas DataFrames safely."""

    def __init__(self, categorical_columns: list[str]) -> None:
        """Initialize the encoder.

        Args:
            categorical_columns: List of column names to encode.
        """
        if not categorical_columns:
            raise PreprocessingError("categorical_columns list cannot be empty.")
            
        self.categorical_columns = categorical_columns
        
        # We use sparse_output=False (formerly sparse=False) to yield standard pandas DataFrames
        # handle_unknown='ignore' prevents breaking on unseen test categories
        self.encoder = OneHotEncoder(
            sparse_output=False,
            handle_unknown="ignore",
            drop="first"  # Drop first to avoid dummy variable trap (multicollinearity)
        )
        self.is_fitted = False
        self.feature_names_out_: list[str] = []

    def fit(self, df: pd.DataFrame) -> CategoricalEncoder:
        """Fit the encoder strictly on training data.

        Args:
            df: The training pandas DataFrame.

        Returns:
            The fitted CategoricalEncoder instance.
            
        Raises:
            PreprocessingError: If requested columns are missing.
        """
        self._validate_columns(df)
        
        logger.info(f"Fitting OneHotEncoder on {len(self.categorical_columns)} columns.")
        self.encoder.fit(df[self.categorical_columns])
        
        self.is_fitted = True
        self.feature_names_out_ = list(
            self.encoder.get_feature_names_out(self.categorical_columns)
        )
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform data using the previously fitted encoder.

        Args:
            df: The DataFrame to transform (can be train or test set).

        Returns:
            A new DataFrame with original categorical columns dropped 
            and one-hot encoded columns appended.
            
        Raises:
            PreprocessingError: If transform is called before fit.
        """
        if not self.is_fitted:
            raise PreprocessingError(
                "Encoder must be fitted on training data before calling transform()."
            )
            
        self._validate_columns(df)
        
        logger.debug(f"Transforming DataFrame of shape {df.shape} via OneHotEncoder.")
        
        # Transform returns a numpy array since sparse_output=False
        encoded_array = self.encoder.transform(df[self.categorical_columns])
        
        # Convert back to DataFrame retaining original index
        encoded_df = pd.DataFrame(
            encoded_array,
            columns=self.feature_names_out_,
            index=df.index
        )
        
        # Drop original categories and concatenate new features
        df_out = df.drop(columns=self.categorical_columns)
        df_out = pd.concat([df_out, encoded_df], axis=1)
        
        return df_out

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fit the encoder and immediately transform the training data.

        Args:
            df: The training pandas DataFrame.

        Returns:
            The transformed training DataFrame.
        """
        return self.fit(df).transform(df)

    def _validate_columns(self, df: pd.DataFrame) -> None:
        """Ensure all requested categorical columns exist in the DataFrame."""
        missing = set(self.categorical_columns) - set(df.columns)
        if missing:
            raise PreprocessingError(f"Cannot encode missing columns: {missing}")
