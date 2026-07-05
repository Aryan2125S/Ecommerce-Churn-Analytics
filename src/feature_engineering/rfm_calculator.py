"""
RFM Calculator Module.

Calculates Recency, Frequency, and Monetary (RFM) metrics for E-commerce 
customers. To strictly prevent Data Leakage, this module follows the same 
architecture as ML transformers: it calculates the reference date and 
quintile bin edges purely on the training dataset (`fit()`), and applies 
those static boundaries to the test dataset (`transform()`).

Calculations:
    - Recency: Days between `last_purchase_date` and the dataset's max date.
    - Frequency: Directly utilizes `total_orders`.
    - Monetary: Directly utilizes `total_spend`.
    - RFM Score: A concatenated string of quintiles (e.g., "555" for best).
    - RFM Segment: A categorical classification (e.g., "Champions", "At Risk").

Usage:
    from src.feature_engineering.rfm_calculator import RFMCalculator
    
    rfm = RFMCalculator()
    df_train = rfm.fit_transform(df_train)
    df_test = rfm.transform(df_test)
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.config.constants import ColumnNames
from src.feature_engineering.base_transformer import BaseFeatureTransformer
from src.utils.logger import get_logger

logger = get_logger(__name__)


class RFMCalculator(BaseFeatureTransformer):
    """Calculates robust RFM metrics without introducing ML Data Leakage."""

    def __init__(self) -> None:
        """Initialize the RFM Calculator with required columns."""
        super().__init__(required_columns=[
            ColumnNames.LAST_PURCHASE_DATE,
            ColumnNames.TOTAL_ORDERS,
            ColumnNames.TOTAL_SPEND
        ])
        self.reference_date: pd.Timestamp | None = None
        self.r_bins: np.ndarray | None = None
        self.f_bins: np.ndarray | None = None
        self.m_bins: np.ndarray | None = None

    def fit(self, X: pd.DataFrame, y: Any = None) -> RFMCalculator:
        """Learn the reference date and quintile edges from the training data.

        Args:
            X: The training pandas DataFrame.
            y: Ignored.

        Returns:
            The fitted RFMCalculator instance.
        """
        self._validate_input(X)
        
        logger.info("Fitting RFM boundaries on training dataset...")
        
        # Ensure datetime format for calculation
        dates = pd.to_datetime(X[ColumnNames.LAST_PURCHASE_DATE])
        
        # 1. Reference Date (Maximum purchase date in the train set)
        self.reference_date = dates.max()
        logger.debug(f"RFM Reference Date fixed at: {self.reference_date}")
        
        # Calculate raw Recency for bin calculation
        recency_days = (self.reference_date - dates).dt.days
        
        # 2. Calculate Quintile Bins using qcut
        _, self.r_bins = pd.qcut(recency_days, q=5, retbins=True, duplicates='drop')
        _, self.f_bins = pd.qcut(X[ColumnNames.TOTAL_ORDERS], q=5, retbins=True, duplicates='drop')
        _, self.m_bins = pd.qcut(X[ColumnNames.TOTAL_SPEND], q=5, retbins=True, duplicates='drop')
        
        # Extend absolute outer edges to handle test set values that exceed train bounds
        self.r_bins[0], self.r_bins[-1] = -np.inf, np.inf
        self.f_bins[0], self.f_bins[-1] = -np.inf, np.inf
        self.m_bins[0], self.m_bins[-1] = -np.inf, np.inf
        
        self.is_fitted = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply RFM calculations and assign segments using fitted boundaries.

        Args:
            X: The data to transform.

        Returns:
            A new DataFrame containing the new RFM columns.
        """
        self._check_is_fitted()
        self._validate_input(X)
        
        logger.debug(f"Applying RFM transformations to {len(X)} records.")
        
        df_out = X.copy()
        dates = pd.to_datetime(df_out[ColumnNames.LAST_PURCHASE_DATE])
        
        # 1. Raw Value Extraction
        df_out['recency_days'] = (self.reference_date - dates).dt.days
        df_out['frequency_raw'] = df_out[ColumnNames.TOTAL_ORDERS]
        df_out['monetary_raw'] = df_out[ColumnNames.TOTAL_SPEND]
        
        # 2. Scoring (1 to 5) using pd.cut with frozen bins
        r_labels = list(range(len(self.r_bins) - 1, 0, -1))
        f_labels = list(range(1, len(self.f_bins)))
        m_labels = list(range(1, len(self.m_bins)))
        
        df_out['R_Score'] = pd.cut(df_out['recency_days'], bins=self.r_bins, labels=r_labels, include_lowest=True).astype(str)
        df_out['F_Score'] = pd.cut(df_out['frequency_raw'], bins=self.f_bins, labels=f_labels, include_lowest=True).astype(str)
        df_out['M_Score'] = pd.cut(df_out['monetary_raw'], bins=self.m_bins, labels=m_labels, include_lowest=True).astype(str)
        
        # 3. Concatenate RFM string
        df_out['rfm_score_concat'] = df_out['R_Score'] + df_out['F_Score'] + df_out['M_Score']
        
        # 4. Map Segmentation
        df_out['rfm_segment'] = df_out['rfm_score_concat'].apply(self._map_segment)
        
        df_out.drop(columns=['frequency_raw', 'monetary_raw', 'R_Score', 'F_Score', 'M_Score', 'rfm_score_concat'], inplace=True)
        return df_out

    def _map_segment(self, score_str: str) -> str:
        """Map the concatenated string to an E-commerce business segment."""
        try:
            r = int(score_str[0])
            f = int(score_str[1])
            m = int(score_str[2])
            
            if r >= 4 and f >= 4 and m >= 4:
                return "Champions"
            elif r >= 3 and f >= 3:
                return "Loyal Customers"
            elif r >= 4 and f <= 2:
                return "Recent Users"
            elif r <= 2 and f >= 4:
                return "At Risk"
            elif r <= 2 and f <= 2:
                return "Lost"
            else:
                return "Average"
        except (ValueError, IndexError):
            return "Unknown"
