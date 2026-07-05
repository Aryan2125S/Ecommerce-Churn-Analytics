"""
Machine Learning Training Pipeline.

This script strictly aligns with the official internship project requirements.
It orchestrates the end-to-end data flow: loading, cleaning, splitting, feature 
engineering, encoding, scaling, and training a predictive churn model.

Key Mathematical Constraints (Leakage Prevention):
    - Structural cleaning (imputation, dropping dupes) occurs BEFORE the split.
    - Feature Engineering (RFM, Engagement) occurs AFTER the split (fitted on train).
    - Encoding and Scaling occur AFTER the split (fitted on train).
"""

import os
import joblib
import pandas as pd
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

from src.config.constants import ColumnNames
from src.config.settings import settings
from src.database.db_manager import DatabaseManager
from src.preprocessing.pipeline import PreprocessingPipeline
from src.preprocessing.encoder import CategoricalEncoder
from src.preprocessing.scaler import FeatureScaler
from src.feature_engineering.feature_pipeline import FeatureEngineeringPipeline
from src.utils.logger import get_logger

logger = get_logger(__name__)


def run_training_pipeline():
    """Execute the end-to-end Machine Learning pipeline."""
    logger.info("Initializing ML Training Pipeline...")

    # ---------------------------------------------------------
    # 1. Data Ingestion & Structural Cleaning (Pre-Split)
    # ---------------------------------------------------------
    logger.info("Step 1: Ingesting and structurally cleaning raw data...")
    # Initialize DB (assumes data_generator was previously run)
    db = DatabaseManager()
    
    # Run the cleaning pipeline (Missing Values, Outliers)
    # We load directly via the PreprocessingPipeline which pulls from DB internally
    preprocessor = PreprocessingPipeline(table_name="customers")
    df_clean = preprocessor.run(save_to_disk=False)
    
    # ---------------------------------------------------------
    # 2. Train / Test Split
    # ---------------------------------------------------------
    logger.info("Step 2: Performing Train/Test Split...")
    # Separate Features and Target
    X = df_clean.drop(columns=[ColumnNames.CHURN, ColumnNames.CUSTOMER_ID])
    y = df_clean[ColumnNames.CHURN]
    
    # We must split NOW before applying distribution-dependent transformers
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # We inject customer_id back into X to allow RFM logic to map to user, but wait, RFM relies on dates and orders.
    # Actually, RFM calculator expects dates, orders, and spend which are in X. customer_id is not strictly required.
    
    # ---------------------------------------------------------
    # 3. Feature Engineering (Post-Split, Fitted on Train)
    # ---------------------------------------------------------
    logger.info("Step 3: Executing Feature Engineering...")
    fe_pipeline = FeatureEngineeringPipeline()
    X_train_fe = fe_pipeline.fit_transform(X_train)
    X_test_fe = fe_pipeline.transform(X_test)
    
    # We must drop DateTime and String identifier columns that algorithms can't process natively
    cols_to_drop = [
        ColumnNames.REGISTRATION_DATE, 
        ColumnNames.LAST_PURCHASE_DATE,
    ]
    X_train_fe = X_train_fe.drop(columns=cols_to_drop, errors='ignore')
    X_test_fe = X_test_fe.drop(columns=cols_to_drop, errors='ignore')
    
    # ---------------------------------------------------------
    # 4. Encoding and Scaling
    # ---------------------------------------------------------
    logger.info("Step 4: Encoding and Scaling...")
    # Define categorical and numerical splits
    # Exclude segment variables created by FE
    cat_cols = X_train_fe.select_dtypes(include=['category', 'object']).columns.tolist()
    num_cols = X_train_fe.select_dtypes(include=['int64', 'float64']).columns.tolist()
    
    encoder = CategoricalEncoder(categorical_columns=cat_cols)
    X_train_enc = encoder.fit_transform(X_train_fe)
    X_test_enc = encoder.transform(X_test_fe)
    
    # Scaler requires lists of column names for the num_cols
    scaler = FeatureScaler(numerical_columns=num_cols, method="standard")
    
    # We need to explicitly scale only the numeric columns, but our FeatureScaler wrapper 
    # was built to take the whole df and scale everything numeric. Let's pass the encoded DF.
    X_train_final = scaler.fit_transform(X_train_enc)
    X_test_final = scaler.transform(X_test_enc)
    
    # ---------------------------------------------------------
    # 5. Model Training
    # ---------------------------------------------------------
    logger.info("Step 5: Training Random Forest Model...")
    # Random Forest is highly robust, handles non-linearities, and yields Feature Importances
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        class_weight="balanced", # Crucial for churn prediction imbalance
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train_final, y_train)
    
    # ---------------------------------------------------------
    # 6. Evaluation
    # ---------------------------------------------------------
    logger.info("Step 6: Evaluating Model...")
    y_pred = model.predict(X_test_final)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    
    logger.info(f"Model Accuracy: {accuracy:.4f}")
    logger.info(f"\nClassification Report:\n{report}")
    
    # ---------------------------------------------------------
    # 7. Exporting Artifacts
    # ---------------------------------------------------------
    logger.info("Step 7: Exporting Model and Pipelines...")
    os.makedirs(settings.model.model_dir, exist_ok=True)
    
    joblib.dump(model, settings.model.model_dir / 'rf_churn_model.pkl')
    joblib.dump(fe_pipeline, settings.model.model_dir / 'fe_pipeline.pkl')
    joblib.dump(encoder, settings.model.model_dir / 'encoder.pkl')
    joblib.dump(scaler, settings.model.model_dir / 'scaler.pkl')
    
    # Also save the complete dataset with predictions for PowerBI (Step 7 requirement)
    logger.info("Exporting final combined predictions for Power BI...")
    
    # For PowerBI, we want the ORIGINAL dataframe joined with predictions and FE features
    # Since X_test has the exact indices of df_clean, we can map predictions back.
    powerbi_df = df_clean.copy()
    
    # Apply FE to entire dataset strictly for the dashboard (not for training)
    # Using the fitted fe_pipeline to avoid leakage on the actual values
    powerbi_fe = fe_pipeline.transform(powerbi_df.drop(columns=[ColumnNames.CHURN, ColumnNames.CUSTOMER_ID], errors='ignore'))
    
    # Join engineered features back to PowerBI DF
    powerbi_df['rfm_segment'] = powerbi_fe.get('rfm_segment', 'Unknown')
    powerbi_df['engagement_score'] = powerbi_fe.get('engagement_score', 0.0)
    
    # Add predictions where available (test set), and train set predictions
    y_pred_all = model.predict(scaler.transform(encoder.transform(powerbi_fe.drop(columns=cols_to_drop, errors='ignore'))))
    powerbi_df['predicted_churn'] = y_pred_all
    
    powerbi_path = settings.paths.processed_data_dir / 'powerbi_export.csv'
    powerbi_df.to_csv(powerbi_path, index=False)
    logger.info(f"Power BI export saved to {powerbi_path}")
    
    logger.info("ML Training Pipeline Completed Successfully.")


if __name__ == "__main__":
    run_training_pipeline()
