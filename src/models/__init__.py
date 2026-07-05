"""
Machine Learning Models Package.

Provides the complete ML training pipeline and real-time prediction service
for customer churn classification. Supports nine classification algorithms
with automated hyperparameter tuning, cross-validation, model comparison,
and artifact serialization.

Modules:
    train: Implements the ModelTrainer class orchestrating Pipeline-based
           training of Logistic Regression, Decision Tree, Random Forest,
           Extra Trees, AdaBoost, Gradient Boosting, XGBoost, LightGBM,
           and CatBoost with SMOTE imbalance handling and GridSearchCV /
           RandomizedSearchCV hyperparameter optimization.
    predict: Implements the PredictionService class that loads serialized
             model artifacts, validates input features, executes inference,
             and returns churn probability, risk level, and confidence score.

Dependencies:
    - scikit-learn, xgboost, lightgbm, catboost (model implementations)
    - imbalanced-learn (SMOTE for class imbalance)
    - joblib (model serialization)
    - src.config (for model paths, random state, CV folds)
    - src.utils (for logging and exception handling)
"""
