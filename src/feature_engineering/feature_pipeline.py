"""
Feature Engineering Pipeline Module.

Orchestrates the sequential application of all Phase 5 feature engineering 
transformers (RFM and Engagement). Provides a unified `fit_transform` and 
`transform` interface to strictly prevent ML data leakage while preparing
the final dataframe for model training.
"""

from typing import Any

import pandas as pd

from src.feature_engineering.engagement_scorer import EngagementScorer
from src.feature_engineering.rfm_calculator import RFMCalculator
from src.utils.logger import get_logger
from src.utils.exceptions import PreprocessingError

logger = get_logger(__name__)


class FeatureEngineeringPipeline:
    """Orchestrates all feature engineering transformations."""

    def __init__(self) -> None:
        """Initialize the feature engineering modules."""
        self.rfm_calculator = RFMCalculator()
        self.engagement_scorer = EngagementScorer()
        self.is_fitted = False

    def fit_transform(self, X: pd.DataFrame, y: Any = None) -> pd.DataFrame:
        """Fit all feature transformers and apply them to the training set.

        Args:
            X: Cleaned training DataFrame.
            y: Ignored.

        Returns:
            DataFrame augmented with engineered features.
        """
        logger.info("Starting feature engineering fit_transform...")
        
        # Sequentially fit_transform each module
        X_engineered = self.rfm_calculator.fit_transform(X)
        X_engineered = self.engagement_scorer.fit_transform(X_engineered)
        
        self.is_fitted = True
        logger.info("Feature engineering complete.")
        return X_engineered

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply the fitted feature transformers to test data.

        Args:
            X: Cleaned test DataFrame.

        Returns:
            DataFrame augmented with engineered features using frozen boundaries.
        """
        if not self.is_fitted:
            raise PreprocessingError("FeatureEngineeringPipeline must be fitted before transform.")
            
        logger.info("Applying fitted feature engineering transformations...")
        
        # Sequentially transform using frozen states
        X_engineered = self.rfm_calculator.transform(X)
        X_engineered = self.engagement_scorer.transform(X_engineered)
        
        return X_engineered
