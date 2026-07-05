"""
Test Suite Package.

Contains unit and integration tests for all application modules using
the pytest framework. Tests are organized by application layer to mirror
the source code structure.

Test Modules:
    test_preprocessing: Validates DataCleaner operations including null
                        handling, outlier capping, encoding, and scaling.
    test_feature_engineering: Validates FeatureEngineer calculations
                              including RFM scores, risk scores, engagement
                              scores, ratios, and flag classifications.
    test_models: Validates PredictionService artifact loading, input
                 validation, and inference output format.

Dependencies:
    - pytest (test runner)
    - pytest-cov (coverage reporting)
    - pandas, numpy (for test data construction)
"""
