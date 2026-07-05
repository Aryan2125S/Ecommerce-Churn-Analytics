"""
Unit Tests for Phase 4 Preprocessing Pipeline.

This test suite verifies the structural integrity of the Data Cleaning,
Outlier Handling, and Orchestration modules. It utilizes pytest fixtures
to mock database extractions and simulate corrupted raw data.

Coverage:
    - Missing value imputation
    - Duplicate record removal
    - Outlier Winsorization
    - Strict data type enforcement
    - End-to-end pipeline execution
    - Exception handling for invalid inputs
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from src.config.constants import ColumnNames, PaymentMethod
from src.preprocessing.data_cleaner import DataCleaner
from src.preprocessing.outlier_handler import OutlierHandler
from src.preprocessing.pipeline import PreprocessingPipeline
from src.utils.exceptions import DataCleaningError, PreprocessingError


@pytest.fixture
def sample_raw_data() -> pd.DataFrame:
    """Fixture providing a mock raw dataset with known corruptions."""
    # Create a base schema that matches the TDD 32 columns
    data = {col: [None] * 6 for col in ColumnNames.ALL_COLUMNS}
    df = pd.DataFrame(data)
    
    # Populate essential valid data for 6 rows
    df[ColumnNames.CUSTOMER_ID] = ["C001", "C002", "C003", "C001", "C004", "C005"] # Notice C001 is duplicated
    df[ColumnNames.AGE] = [25, 45, np.nan, 25, 30, 40] # 1 missing age
    df[ColumnNames.GENDER] = ["Male", "Female", "Female", "Male", "Male", "Female"]
    df[ColumnNames.CITY] = ["New York", "Los Angeles", None, "New York", "Chicago", "Miami"] # 1 missing city
    df[ColumnNames.STATE] = ["NY", "CA", "TX", "NY", "IL", "FL"]
    df[ColumnNames.COUNTRY] = ["USA", "USA", "USA", "USA", "USA", "USA"]
    df[ColumnNames.MEMBERSHIP_TYPE] = ["Premium", "Basic", "Premium", "Premium", "Basic", "Premium"]
    df[ColumnNames.REGISTRATION_DATE] = ["2023-01-01", "2023-05-15", "2023-11-20", "2023-01-01", "2023-02-10", "2023-08-05"]
    df[ColumnNames.LAST_PURCHASE_DATE] = ["2023-12-01", "2023-12-15", "2023-12-20", "2023-12-01", "2023-11-01", "2023-12-05"]
    df[ColumnNames.PURCHASE_FREQUENCY] = [1.5, 2.0, 3.5, 1.5, 1.0, 4.0]
    df[ColumnNames.TOTAL_ORDERS] = [10, 5, 20, 10, 2, 30]
    df[ColumnNames.AVERAGE_ORDER_VALUE] = [50.0, 20.0, 100.0, 50.0, 75.0, 60.0]
    df[ColumnNames.TOTAL_SPEND] = [500.0, 100.0, 999999.0, 500.0, 150.0, 1800.0] # Massive outlier for C003
    df[ColumnNames.DISCOUNT_USAGE] = [2, 0, 5, 2, 0, 10]
    df[ColumnNames.COUPON_USAGE] = [1, 0, 2, 1, 0, 5]
    df[ColumnNames.SESSION_DURATION] = [300.5, 120.0, 600.0, 300.5, 150.0, 450.0]
    df[ColumnNames.BROWSING_TIME] = [15, 5, 45, 15, 10, 30]
    df[ColumnNames.PAGES_VIEWED] = [10, 3, 25, 10, 5, 20]
    df[ColumnNames.WISHLIST_ITEMS] = [5, 1, 15, 5, 2, 10]
    df[ColumnNames.CART_ABANDONMENT] = [2, 0, 5, 2, 1, 3]
    df[ColumnNames.PAYMENT_METHOD] = [PaymentMethod.CREDIT_CARD.value, PaymentMethod.UPI.value, " upi ", PaymentMethod.CREDIT_CARD.value, PaymentMethod.CREDIT_CARD.value, PaymentMethod.CREDIT_CARD.value]
    df[ColumnNames.RETURNED_ORDERS] = [0, 0, 1, 0, 0, 2]
    df[ColumnNames.REFUND_AMOUNT] = [0.0, 0.0, 50.0, 0.0, 0.0, 100.0]
    df[ColumnNames.CUSTOMER_RATING] = [4.5, 3.0, 5.0, 4.5, 4.0, 4.8]
    df[ColumnNames.DELIVERY_RATING] = [5, 3, 4, 5, 4, 5]
    df[ColumnNames.SUPPORT_CALLS] = [0, np.nan, 2, 0, 1, 0] # Missing support calls
    df[ColumnNames.COMPLAINTS] = [0, 1, 0, 0, 0, 0]
    df[ColumnNames.CUSTOMER_SATISFACTION] = [4, 3, 5, 4, 4, 5]
    df[ColumnNames.CUSTOMER_LIFETIME_VALUE] = [1000.0, 300.0, 2500.0, 1000.0, 500.0, 4000.0]
    df[ColumnNames.CUSTOMER_SEGMENT] = ["High", "Low", "VIP", "High", "Mid", "VIP"]
    df[ColumnNames.LOYALTY_SCORE] = [80, 40, 95, 80, 50, 90]
    df[ColumnNames.CHURN] = [0, 1, 0, 0, 1, 0]
    
    return df


def test_invalid_input_handling() -> None:
    """Test that modules raise appropriate domain exceptions on empty data."""
    empty_df = pd.DataFrame()
    
    with pytest.raises(DataCleaningError, match="cannot be empty"):
        DataCleaner(empty_df)
        
    with pytest.raises(PreprocessingError, match="cannot be empty"):
        OutlierHandler(empty_df)


def test_duplicate_removal(sample_raw_data: pd.DataFrame) -> None:
    """Test that identical customer records are dropped."""
    cleaner = DataCleaner(sample_raw_data)
    assert len(cleaner.df) == 6
    
    cleaner._drop_duplicates()
    
    assert len(cleaner.df) == 5
    assert list(cleaner.df[ColumnNames.CUSTOMER_ID]) == ["C001", "C002", "C003", "C004", "C005"]


def test_missing_value_handling(sample_raw_data: pd.DataFrame) -> None:
    """Test domain-aware imputation logic."""
    cleaner = DataCleaner(sample_raw_data)
    cleaner._drop_duplicates()
    cleaner._handle_missing_values()
    
    # Support calls (count) should be filled with 0
    assert cleaner.df.loc[1, ColumnNames.SUPPORT_CALLS] == 0
    
    # Age (continuous) should be filled with median (25 and 45 -> median 35.0)
    assert cleaner.df.loc[2, ColumnNames.AGE] == 35.0
    
    # City (categorical) should be filled with mode
    assert pd.notna(cleaner.df.loc[2, ColumnNames.CITY])


def test_data_type_validation_and_cleaning(sample_raw_data: pd.DataFrame) -> None:
    """Test string normalization and strict Pandas type casting."""
    cleaner = DataCleaner(sample_raw_data)
    cleaner._drop_duplicates()
    cleaner._handle_missing_values()
    cleaner._standardize_data_types()
    cleaner._clean_categorical_strings()
    
    df = cleaner.df
    
    # Check types
    assert pd.api.types.is_integer_dtype(df[ColumnNames.AGE])
    assert pd.api.types.is_float_dtype(df[ColumnNames.TOTAL_SPEND])
    assert pd.api.types.is_datetime64_any_dtype(df[ColumnNames.REGISTRATION_DATE])
    assert isinstance(df[ColumnNames.CITY].dtype, pd.CategoricalDtype)
    
    # Check string cleaning (Messy " upi " should become Title case or properly matched Enum)
    assert df.loc[2, ColumnNames.PAYMENT_METHOD] == PaymentMethod.UPI.value


def test_outlier_handling(sample_raw_data: pd.DataFrame) -> None:
    """Test that IQR Winsorization caps extreme values without deleting rows."""
    cleaner = DataCleaner(sample_raw_data)
    df_clean = cleaner.execute_cleaning_pipeline()
    
    handler = OutlierHandler(df_clean)
    
    # The total_spend for C003 is 999999.0 which is a massive outlier
    # The normal values are 500.0 and 100.0
    df_treated = handler.treat_feature(ColumnNames.TOTAL_SPEND, method="iqr", strategy="cap")
    
    # The row should still exist (no removal)
    assert len(df_treated) == 5
    
    # The massive outlier should be capped significantly below 999999.0
    capped_val = df_treated.loc[df_treated[ColumnNames.CUSTOMER_ID] == "C003", ColumnNames.TOTAL_SPEND].values[0]
    assert capped_val < 999999.0
    
    # The non-outlier should remain untouched
    normal_val = df_treated.loc[df_treated[ColumnNames.CUSTOMER_ID] == "C002", ColumnNames.TOTAL_SPEND].values[0]
    assert normal_val == 100.0


@patch("src.preprocessing.pipeline.DatabaseManager")
@patch("src.preprocessing.pipeline.Path")
def test_pipeline_execution(MockPath: MagicMock, MockDBManager: MagicMock, sample_raw_data: pd.DataFrame) -> None:
    """Test the end-to-end PreprocessingPipeline orchestrator."""
    # Mock the DB returning our sample data
    mock_db_instance = MockDBManager.return_value
    mock_db_instance.fetch_all_as_dataframe.return_value = sample_raw_data
    
    pipeline = PreprocessingPipeline(table_name="test_customers")
    df_final = pipeline.run(save_to_disk=True)
    
    # Assert DB was called
    mock_db_instance.fetch_all_as_dataframe.assert_called_once_with(table_name="test_customers")
    
    # Assert structural pipeline was fully applied (dupe dropped, 5 rows remain)
    assert len(df_final) == 5
    
    # Assert missing values handled
    assert not df_final.isnull().values.any()
    
    # Assert outlier capping was applied (C003 total spend is no longer 999999.0)
    capped_val = df_final.loc[df_final[ColumnNames.CUSTOMER_ID] == "C003", ColumnNames.TOTAL_SPEND].values[0]
    assert capped_val < 999999.0
