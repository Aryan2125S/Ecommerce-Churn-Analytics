"""
Utilities Package.

Provides shared infrastructure services used across all application layers
including centralized logging, custom exception hierarchy, and synthetic
data generation.

Modules:
    logger: Configures a centralized logging utility with RotatingFileHandler
            that writes timestamped log entries to both console and the
            logs/ directory. Exposes a get_logger(name) factory function.
    exceptions: Defines a domain-specific exception class hierarchy rooted
                in ChurnAnalyticsBaseError, with subclasses for data
                generation, database, cleaning, feature engineering,
                model training, evaluation, prediction, and export errors.
    data_generator: Implements the DataGenerator class that synthesizes
                    50,000+ realistic customer records with correlated
                    demographic, transactional, and behavioral features.

Dependencies:
    - Python standard library logging module
    - numpy (for statistical distributions in data generation)
    - pandas (for DataFrame construction)
    - src.config (for paths, random seed, and column definitions)
"""
