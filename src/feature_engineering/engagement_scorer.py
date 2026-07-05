"""
Engagement Scorer Module.

Calculates a composite 'Engagement Score' based on user interaction metrics.
Inherits from `BaseFeatureTransformer` to guarantee zero data leakage by 
learning normalization bounds purely on the training dataset.

Business Justification for Formula:
    Engagement is a multi-dimensional behavioral trait. A highly engaged user 
    spends more time on the platform, views more pages, and interacts with 
    platform features like wishlists and discounts. 
    
    Because these raw metrics reside on completely different mathematical scales 
    (e.g., `session_duration` might be in hundreds, while `wishlist_items` is < 20), 
    we first normalize each feature to a 0-1 scale. We then apply a weighted sum 
    prioritizing active exploration (`pages_viewed`) and time investment. 
    The final score is scaled to a configurable range (default 0-100).
"""

from typing import Any

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from src.config.constants import ColumnNames
from src.feature_engineering.base_transformer import BaseFeatureTransformer
from src.utils.logger import get_logger

logger = get_logger(__name__)


class EngagementScorer(BaseFeatureTransformer):
    """Calculates a normalized Customer Engagement Score without data leakage."""

    def __init__(self, target_range: tuple[float, float] = (0.0, 100.0)) -> None:
        """Initialize the Engagement Scorer.

        Args:
            target_range: The min and max bounds for the final score. Default is 0 to 100.
        """
        super().__init__(required_columns=[
            ColumnNames.SESSION_DURATION,
            ColumnNames.PAGES_VIEWED,
            ColumnNames.WISHLIST_ITEMS,
            ColumnNames.DISCOUNT_USAGE
        ])
        
        self.target_range = target_range
        
        # Scikit-learn scalers to maintain state strictly on the train set
        self.feature_scaler = MinMaxScaler(feature_range=(0, 1))
        self.final_score_scaler = MinMaxScaler(feature_range=target_range)
        
        # Business weights for the composite score
        self.weights = np.array([
            0.3,  # session_duration (Time investment)
            0.4,  # pages_viewed (Active exploration)
            0.2,  # wishlist_items (Intent to purchase)
            0.1   # discount_usage (Interaction with promotions)
        ])

    def fit(self, X: pd.DataFrame, y: Any = None) -> 'EngagementScorer':
        """Learn the min/max bounds for the engagement metrics from training data.

        Args:
            X: The training pandas DataFrame.
            y: Ignored.

        Returns:
            The fitted EngagementScorer instance.
        """
        self._validate_input(X)
        logger.info("Fitting Engagement bounds on training dataset...")
        
        # Extract the relevant columns for calculation
        features = X[[
            ColumnNames.SESSION_DURATION,
            ColumnNames.PAGES_VIEWED,
            ColumnNames.WISHLIST_ITEMS,
            ColumnNames.DISCOUNT_USAGE
        ]].astype(float)
        
        # Fit the feature scaler to learn 0-1 boundaries for the raw inputs
        self.feature_scaler.fit(features)
        
        # To fit the final scaler, we calculate the raw weighted scores on the train set
        normalized_features = self.feature_scaler.transform(features)
        raw_composite_scores = np.dot(normalized_features, self.weights).reshape(-1, 1)
        
        # Fit the final score scaler to learn the bounds of the raw composite score
        self.final_score_scaler.fit(raw_composite_scores)
        
        self.is_fitted = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply the fitted bounds to calculate the engagement score.

        Args:
            X: The train or test DataFrame to transform.

        Returns:
            A new DataFrame containing the `engagement_score`.
        """
        self._check_is_fitted()
        self._validate_input(X)
        logger.debug(f"Applying Engagement Scoring to {len(X)} records.")
        
        df_out = X.copy()
        
        features = df_out[[
            ColumnNames.SESSION_DURATION,
            ColumnNames.PAGES_VIEWED,
            ColumnNames.WISHLIST_ITEMS,
            ColumnNames.DISCOUNT_USAGE
        ]].astype(float)
        
        # 1. Normalize the raw features using the bounds learned during fit()
        normalized_features = self.feature_scaler.transform(features)
        
        # 2. Calculate the raw composite score using business weights
        raw_composite_scores = np.dot(normalized_features, self.weights).reshape(-1, 1)
        
        # 3. Scale the final score to the target range (e.g., 0-100)
        final_scores = self.final_score_scaler.transform(raw_composite_scores)
        
        # Ensure scores don't exceed target range if test set had crazy outliers
        min_val, max_val = self.target_range
        final_scores = np.clip(final_scores, min_val, max_val)
        
        df_out['engagement_score'] = np.round(final_scores, 2)
        
        return df_out
