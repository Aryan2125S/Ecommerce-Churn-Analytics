"""
Preprocessing Orchestration Pipeline.

This module orchestrates Phase 4 of the E-commerce Customer Churn project.
It is strictly limited to structural preprocessing: data loading, schema
validation, missing value imputation, type casting, and outlier capping.

CRITICAL DESIGN CONSTRAINT:
This pipeline does NOT perform train/test splits, model-specific scaling,
categorical encoding, or SMOTE augmentation. Those steps strictly belong 
in the ML pipeline (Phase 7) to prevent data leakage.

Usage:
    from src.preprocessing.pipeline import PreprocessingPipeline
    
    pipeline = PreprocessingPipeline()
    df_clean = pipeline.run(save_to_disk=True)

Dependencies:
    - pandas
    - src.database.db_manager (DatabaseManager)
    - src.preprocessing.data_cleaner (DataCleaner)
    - src.preprocessing.outlier_handler (OutlierHandler)
    - src.config.settings (for save paths)
    - src.config.constants (ColumnNames)
    - src.utils.exceptions
    - src.utils.logger
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.config.constants import ColumnNames
from src.config.settings import settings
from src.database.db_manager import DatabaseManager
from src.preprocessing.data_cleaner import DataCleaner
from src.preprocessing.outlier_handler import OutlierHandler
from src.utils.exceptions import PreprocessingError
from src.utils.logger import get_logger

logger = get_logger(__name__)


class PreprocessingPipeline:
    """Orchestrator for structural data cleaning and outlier handling."""

    def __init__(self, table_name: str = "customers") -> None:
        """Initialize the Preprocessing Pipeline.

        Args:
            table_name: The database table to pull raw data from.
        """
        self.table_name = table_name
        self.db_manager = DatabaseManager()

    def run(self, save_to_disk: bool = True) -> pd.DataFrame:
        """Execute the structural preprocessing pipeline sequence.

        Steps:
            1. Extract raw data from database.
            2. Clean data (imputation, deduplication, typing).
            3. Handle outliers (Winsorization on continuous features).
            4. Optionally persist the clean dataset.

        Args:
            save_to_disk: If True, saves the processed DataFrame using paths
                          configured in project settings.

        Returns:
            The fully cleaned and standardized pandas DataFrame.

        Raises:
            PreprocessingError: If any pipeline stage fails.
        """
        logger.info("Starting Phase 4 Preprocessing Pipeline...")

        try:
            # 1. Data Loading
            logger.info(f"Extracting raw dataset from table: {self.table_name}")
            df_raw = self.db_manager.fetch_all_as_dataframe(table_name=self.table_name)
            
            # 2. Data Cleaning
            logger.info("Executing DataCleaner module...")
            cleaner = DataCleaner(df_raw)
            df_cleaned = cleaner.execute_cleaning_pipeline()
            
            # 3. Outlier Handling
            logger.info("Executing OutlierHandler module...")
            handler = OutlierHandler(df_cleaned)
            df_treated = self._apply_outlier_treatment(handler)
            
            # 4. Optional Persistence
            if save_to_disk:
                self._persist_dataset(df_treated)
                
            logger.info("Phase 4 Preprocessing Pipeline completed successfully.")
            return df_treated

        except Exception as e:
            logger.error(f"Preprocessing Pipeline failed: {e}", exc_info=True)
            raise PreprocessingError(f"Pipeline execution aborted: {e}") from e

    def _apply_outlier_treatment(self, handler: OutlierHandler) -> pd.DataFrame:
        """Apply business-aware outlier handling to continuous features.
        
        We cap (Winsorize) extreme financial and behavioral metrics using IQR.
        We do NOT treat categorical integers or bounded ratings.
        """
        # Define strictly continuous features prone to extreme right-tail skew
        continuous_features = [
            ColumnNames.TOTAL_SPEND,
            ColumnNames.AVERAGE_ORDER_VALUE,
            ColumnNames.PURCHASE_FREQUENCY,
            ColumnNames.SESSION_DURATION,
            ColumnNames.BROWSING_TIME,
            ColumnNames.REFUND_AMOUNT,
            ColumnNames.CUSTOMER_LIFETIME_VALUE,
        ]
        
        current_df = handler.get_dataframe()
        
        for feature in continuous_features:
            if feature in current_df.columns:
                handler.treat_feature(
                    column_name=feature,
                    method="iqr",
                    strategy="cap",
                    threshold=1.5
                )
                
        return handler.get_dataframe()

    def _persist_dataset(self, df: pd.DataFrame) -> None:
        """Save the processed dataset utilizing project settings configuration."""
        # settings.paths.processed_data_dir points to `data/processed`
        output_dir = Path(settings.paths.processed_data_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        output_path = output_dir / "ecommerce_processed.csv"
        
        logger.info(f"Persisting cleaned dataset to {output_path}...")
        df.to_csv(output_path, index=False)
        logger.info(f"Successfully saved {len(df)} records to disk.")
