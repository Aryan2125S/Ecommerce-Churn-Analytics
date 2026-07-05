"""
Feature Engineering Package.

Computes business-derived analytical features from cleaned customer data.
All features are designed to provide predictive signals for churn modeling
and actionable insights for business intelligence.

Modules:
    features: Implements the FeatureEngineer class that calculates RFM
              metrics, engagement scores, customer risk scores, discount
              dependency indexes, complaint and return ratios, VIP/premium
              flags, activity level classifications, and other derived
              business features.

Dependencies:
    - pandas, numpy (for vectorized computations)
    - src.config (for feature definitions and column name constants)
    - src.utils (for logging and exception handling)
"""
