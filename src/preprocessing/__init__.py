"""
Data Preprocessing Package.

Provides sequential data cleaning and transformation pipelines that prepare
raw customer data for downstream feature engineering and model training.

Modules:
    data_cleaner: Implements the DataCleaner class with methods for null
                  value imputation, duplicate removal, outlier winsorization,
                  date parsing, skewness correction, categorical encoding,
                  and feature scaling.

Dependencies:
    - pandas, numpy, scipy (for statistical transformations)
    - scikit-learn (for StandardScaler and encoding utilities)
    - src.config (for column definitions and paths)
    - src.utils (for logging and exception handling)
"""
