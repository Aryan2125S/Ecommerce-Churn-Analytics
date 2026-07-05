"""
Data Cleaning Module.

This module acts as the first line of defense in the Data Preprocessing
pipeline (Phase 4). It is responsible for taking raw data ingested from the
database (or CSV) and standardizing it. This includes structural type casting,
handling missing values (imputation), removing hard duplicates, and enforcing
schema constraints before the data proceeds to advanced Feature Engineering.

Usage:
    from src.preprocessing.data_cleaner import DataCleaner
    
    cleaner = DataCleaner(df_raw)
    df_clean = cleaner.execute_cleaning_pipeline()

Dependencies:
    - pandas (for DataFrame manipulation)
    - src.config.constants (for schema definitions)
    - src.utils.exceptions (for DataCleaningError)
    - src.utils.logger (for execution logging)
"""

from __future__ import annotations

import pandas as pd

from src.config.constants import ColumnNames, Geography, PaymentMethod
from src.utils.exceptions import DataCleaningError
from src.utils.logger import get_logger

logger = get_logger(__name__)


class DataCleaner:
    """Encapsulates all data cleaning logic for the Customer Churn dataset."""

    def __init__(self, raw_data: pd.DataFrame) -> None:
        """Initialize the cleaner with raw data.

        Args:
            raw_data: The raw pandas DataFrame ingested from the database.
        
        Raises:
            DataCleaningError: If the DataFrame is empty.
        """
        if raw_data is None or raw_data.empty:
            logger.error("Attempted to initialize DataCleaner with an empty DataFrame.")
            raise DataCleaningError("Input DataFrame cannot be empty.")
            
        # Operate on a copy to prevent mutating the original reference
        self.df = raw_data.copy()
        
    def execute_cleaning_pipeline(self) -> pd.DataFrame:
        """Run the full cleaning pipeline in the correct logical sequence.

        Returns:
            A thoroughly cleaned pandas DataFrame.
            
        Raises:
            DataCleaningError: If any pipeline step fails.
        """
        logger.info(f"Starting cleaning pipeline for {len(self.df)} records.")
        try:
            self._validate_schema()
            self._drop_duplicates()
            self._handle_missing_values()
            self._standardize_data_types()
            self._clean_categorical_strings()
            
            logger.info("Data cleaning pipeline completed successfully.")
            return self.df
            
        except Exception as e:
            logger.error(f"Cleaning pipeline failed: {e}", exc_info=True)
            raise DataCleaningError(f"Pipeline execution failed: {e}") from e

    def _validate_schema(self) -> None:
        """Ensure all required columns are present."""
        missing_cols = set(ColumnNames.ALL_COLUMNS) - set(self.df.columns)
        if missing_cols:
            raise DataCleaningError(f"Missing mandatory columns: {missing_cols}")
            
    def _drop_duplicates(self) -> None:
        """Remove identical row-level duplicates, keeping the first occurrence."""
        initial_count = len(self.df)
        # We explicitly drop duplicates based on customer_id if it exists
        if ColumnNames.CUSTOMER_ID in self.df.columns:
            self.df.drop_duplicates(subset=[ColumnNames.CUSTOMER_ID], keep="first", inplace=True)
        else:
            self.df.drop_duplicates(inplace=True)
            
        dropped = initial_count - len(self.df)
        if dropped > 0:
            logger.warning(f"Dropped {dropped} duplicate records.")

    def _handle_missing_values(self) -> None:
        """Impute or drop missing values using domain-specific logic."""
        # Numeric structural features: fill with 0 (e.g., no complaints means 0)
        zero_fill_cols = [
            ColumnNames.COMPLAINTS,
            ColumnNames.SUPPORT_CALLS,
            ColumnNames.RETURNED_ORDERS,
            ColumnNames.REFUND_AMOUNT,
            ColumnNames.COUPON_USAGE,
            ColumnNames.DISCOUNT_USAGE,
            ColumnNames.CART_ABANDONMENT,
        ]
        for col in zero_fill_cols:
            self.df[col] = self.df[col].fillna(0)

        # Behavioral medians: fill continuous numeric with median to avoid skew
        median_fill_cols = [
            ColumnNames.AGE,
            ColumnNames.CUSTOMER_SATISFACTION,
            ColumnNames.CUSTOMER_RATING,
            ColumnNames.DELIVERY_RATING,
            ColumnNames.BROWSING_TIME,
            ColumnNames.SESSION_DURATION,
        ]
        for col in median_fill_cols:
            median_val = self.df[col].median()
            self.df[col] = self.df[col].fillna(median_val)
            
        # Categorical modes: fill with most frequent value
        mode_fill_cols = [
            ColumnNames.GENDER,
            ColumnNames.PAYMENT_METHOD,
            ColumnNames.CITY,
            ColumnNames.STATE,
        ]
        for col in mode_fill_cols:
            mode_val = self.df[col].mode()[0]
            self.df[col] = self.df[col].fillna(mode_val)

        # Drop any remaining rows that are fundamentally corrupted
        # (e.g., missing critical IDs or dates that cannot be imputed)
        critical_cols = [
            ColumnNames.CUSTOMER_ID,
            ColumnNames.REGISTRATION_DATE,
            ColumnNames.CHURN
        ]
        self.df.dropna(subset=critical_cols, inplace=True)

    def _standardize_data_types(self) -> None:
        """Enforce strict dtype casting per the TDD schema."""
        # Integers
        int_cols = [
            ColumnNames.AGE,
            ColumnNames.TOTAL_ORDERS,
            ColumnNames.BROWSING_TIME,
            ColumnNames.PAGES_VIEWED,
            ColumnNames.WISHLIST_ITEMS,
            ColumnNames.CART_ABANDONMENT,
            ColumnNames.RETURNED_ORDERS,
            ColumnNames.SUPPORT_CALLS,
            ColumnNames.COMPLAINTS,
            ColumnNames.DELIVERY_RATING,
            ColumnNames.CUSTOMER_SATISFACTION,
            ColumnNames.LOYALTY_SCORE,
            ColumnNames.CHURN,
            ColumnNames.COUPON_USAGE,
            ColumnNames.DISCOUNT_USAGE,
        ]
        for col in int_cols:
            self.df[col] = pd.to_numeric(self.df[col], errors='coerce').fillna(0).astype(int)

        # Floats
        float_cols = [
            ColumnNames.PURCHASE_FREQUENCY,
            ColumnNames.AVERAGE_ORDER_VALUE,
            ColumnNames.TOTAL_SPEND,
            ColumnNames.SESSION_DURATION,
            ColumnNames.REFUND_AMOUNT,
            ColumnNames.CUSTOMER_RATING,
            ColumnNames.CUSTOMER_LIFETIME_VALUE,
        ]
        for col in float_cols:
            self.df[col] = pd.to_numeric(self.df[col], errors='coerce').fillna(0.0).astype(float)

        # Datetimes
        date_cols = [ColumnNames.REGISTRATION_DATE, ColumnNames.LAST_PURCHASE_DATE]
        for col in date_cols:
            self.df[col] = pd.to_datetime(self.df[col], errors='coerce')

        # Categoricals (Optimized strings)
        cat_cols = [
            ColumnNames.GENDER,
            ColumnNames.CITY,
            ColumnNames.STATE,
            ColumnNames.COUNTRY,
            ColumnNames.MEMBERSHIP_TYPE,
            ColumnNames.PAYMENT_METHOD,
            ColumnNames.CUSTOMER_SEGMENT,
        ]
        for col in cat_cols:
            self.df[col] = self.df[col].astype('category')

    def _clean_categorical_strings(self) -> None:
        """Normalize categorical text cases (strip whitespace, title case)."""
        # We only clean string/object/category columns
        for col in self.df.select_dtypes(include=['category', 'object']).columns:
            if col != ColumnNames.CUSTOMER_ID:
                # Convert to string, strip whitespace, title-case, then back to category
                self.df[col] = self.df[col].astype(str).str.strip().str.title().astype('category')
                
        # Fix known acronyms that Title() ruins (e.g. "Upi", "Cod")
        if ColumnNames.PAYMENT_METHOD in self.df.columns:
            pay_fixes = {
                "Upi": PaymentMethod.UPI.value,
                "Cod": PaymentMethod.COD.value,
            }
            # Replace explicitly if it exists
            if self.df[ColumnNames.PAYMENT_METHOD].dtype.name == 'category':
                # Add new categories before replacing to avoid ValueError
                for correct_val in pay_fixes.values():
                    if correct_val not in self.df[ColumnNames.PAYMENT_METHOD].cat.categories:
                        self.df[ColumnNames.PAYMENT_METHOD] = self.df[ColumnNames.PAYMENT_METHOD].cat.add_categories([correct_val])
                        
            self.df[ColumnNames.PAYMENT_METHOD] = self.df[ColumnNames.PAYMENT_METHOD].replace(pay_fixes)
