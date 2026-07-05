"""
Project-Wide Constants and Enumerations Module.

This module serves as the single authoritative registry for all immutable
values used across the application. By centralizing constants here, we
eliminate magic strings, prevent typos, and ensure consistent references
across the data pipeline, ML training, evaluation, and dashboard layers.

Organization:
    - Column Names: Raw dataset and engineered feature column identifiers.
    - Categorical Values: Allowed values for gender, membership, payment, etc.
    - Business Rules: RFM thresholds, age group bins, activity level labels.
    - Color Palette: SaaS-grade UI hex codes from the design specification.
    - UI Tokens: Typography, spacing, shadow, and transition constants.
    - Model Registry: Classifier names and tuning method assignments.
    - File Names: Standardized artifact and output filenames.
    - Dashboard Pages: Page identifiers and display labels.

Usage:
    from src.config.constants import ColumnNames, ColorPalette, ModelRegistry

    target_col = ColumnNames.CHURN
    primary_color = ColorPalette.PRIMARY_ACCENT
    models = ModelRegistry.ALL_MODELS

Dependencies:
    - enum (stdlib)
"""

from __future__ import annotations

from enum import Enum, unique
from typing import Final


# ===========================================================================
# Raw Dataset Column Names
# ===========================================================================
# Identifiers for every column in the generated 50,000+ row dataset.
# These must match the CSV headers produced by the DataGenerator.
# ===========================================================================


class ColumnNames:
    """Centralized registry of raw dataset column name strings.

    These constants are used by the data generator, preprocessor,
    feature engineer, database manager, and dashboard pages to ensure
    column references remain consistent across the entire codebase.
    """

    # --- Identifiers ---
    CUSTOMER_ID: Final[str] = "customer_id"

    # --- Demographics ---
    AGE: Final[str] = "age"
    GENDER: Final[str] = "gender"
    CITY: Final[str] = "city"
    STATE: Final[str] = "state"
    COUNTRY: Final[str] = "country"

    # --- Membership & Lifecycle ---
    MEMBERSHIP_TYPE: Final[str] = "membership_type"
    REGISTRATION_DATE: Final[str] = "registration_date"
    LAST_PURCHASE_DATE: Final[str] = "last_purchase_date"

    # --- Transaction Metrics ---
    PURCHASE_FREQUENCY: Final[str] = "purchase_frequency"
    TOTAL_ORDERS: Final[str] = "total_orders"
    AVERAGE_ORDER_VALUE: Final[str] = "average_order_value"
    TOTAL_SPEND: Final[str] = "total_spend"

    # --- Discounts & Coupons ---
    DISCOUNT_USAGE: Final[str] = "discount_usage"
    COUPON_USAGE: Final[str] = "coupon_usage"

    # --- Browsing Activity ---
    SESSION_DURATION: Final[str] = "session_duration"
    BROWSING_TIME: Final[str] = "browsing_time"
    PAGES_VIEWED: Final[str] = "pages_viewed"

    # --- Shopping Behavior ---
    WISHLIST_ITEMS: Final[str] = "wishlist_items"
    CART_ABANDONMENT: Final[str] = "cart_abandonment"

    # --- Payment ---
    PAYMENT_METHOD: Final[str] = "payment_method"

    # --- Returns & Refunds ---
    RETURNED_ORDERS: Final[str] = "returned_orders"
    REFUND_AMOUNT: Final[str] = "refund_amount"

    # --- Ratings & Satisfaction ---
    CUSTOMER_RATING: Final[str] = "customer_rating"
    DELIVERY_RATING: Final[str] = "delivery_rating"

    # --- Customer Support ---
    SUPPORT_CALLS: Final[str] = "support_calls"
    COMPLAINTS: Final[str] = "complaints"
    CUSTOMER_SATISFACTION: Final[str] = "customer_satisfaction"

    # --- Business Value ---
    CUSTOMER_LIFETIME_VALUE: Final[str] = "customer_lifetime_value"
    CUSTOMER_SEGMENT: Final[str] = "customer_segment"
    LOYALTY_SCORE: Final[str] = "loyalty_score"

    # --- Target Variable ---
    CHURN: Final[str] = "churn"

    # Ordered tuple of all raw dataset columns for schema validation
    ALL_COLUMNS: Final[tuple[str, ...]] = (
        CUSTOMER_ID, AGE, GENDER, CITY, STATE, COUNTRY,
        MEMBERSHIP_TYPE, REGISTRATION_DATE, LAST_PURCHASE_DATE,
        PURCHASE_FREQUENCY, TOTAL_ORDERS, AVERAGE_ORDER_VALUE,
        TOTAL_SPEND, DISCOUNT_USAGE, COUPON_USAGE,
        SESSION_DURATION, BROWSING_TIME, PAGES_VIEWED,
        WISHLIST_ITEMS, CART_ABANDONMENT, PAYMENT_METHOD,
        RETURNED_ORDERS, REFUND_AMOUNT,
        CUSTOMER_RATING, DELIVERY_RATING,
        SUPPORT_CALLS, COMPLAINTS, CUSTOMER_SATISFACTION,
        CUSTOMER_LIFETIME_VALUE, CUSTOMER_SEGMENT, LOYALTY_SCORE,
        CHURN,
    )

    # Date columns requiring datetime parsing
    DATE_COLUMNS: Final[tuple[str, ...]] = (
        REGISTRATION_DATE,
        LAST_PURCHASE_DATE,
    )

    # Categorical columns requiring encoding
    CATEGORICAL_COLUMNS: Final[tuple[str, ...]] = (
        GENDER,
        CITY,
        STATE,
        COUNTRY,
        MEMBERSHIP_TYPE,
        PAYMENT_METHOD,
        CUSTOMER_SEGMENT,
    )

    # Numeric columns subject to outlier detection and scaling
    NUMERIC_COLUMNS: Final[tuple[str, ...]] = (
        AGE, PURCHASE_FREQUENCY, TOTAL_ORDERS, AVERAGE_ORDER_VALUE,
        TOTAL_SPEND, DISCOUNT_USAGE, COUPON_USAGE,
        SESSION_DURATION, BROWSING_TIME, PAGES_VIEWED,
        WISHLIST_ITEMS, CART_ABANDONMENT,
        RETURNED_ORDERS, REFUND_AMOUNT,
        CUSTOMER_RATING, DELIVERY_RATING,
        SUPPORT_CALLS, COMPLAINTS, CUSTOMER_SATISFACTION,
        CUSTOMER_LIFETIME_VALUE, LOYALTY_SCORE,
    )

    # Columns with high right-skew requiring log/sqrt transformation
    SKEWED_COLUMNS: Final[tuple[str, ...]] = (
        TOTAL_SPEND,
        REFUND_AMOUNT,
        CUSTOMER_LIFETIME_VALUE,
        SESSION_DURATION,
        BROWSING_TIME,
    )


# ===========================================================================
# Engineered Feature Names
# ===========================================================================
# Identifiers for all business-derived features appended by the
# FeatureEngineer class during Phase 6.
# ===========================================================================


class FeatureNames:
    """Registry of all engineered feature column names.

    Each feature is documented with its business purpose. These constants
    are referenced by the feature engineering module, the ML training
    pipeline, the prediction service, and the dashboard analytics pages.
    """

    # Demographic segmentation bin label
    CUSTOMER_AGE_GROUP: Final[str] = "customer_age_group"

    # Tenure: days between registration and last purchase
    CUSTOMER_LIFETIME_DAYS: Final[str] = "customer_lifetime_days"

    # Spending velocity: total_spend / lifetime_months
    AVG_MONTHLY_SPEND: Final[str] = "avg_monthly_spend"

    # RFM components
    RECENCY: Final[str] = "recency"
    FREQUENCY: Final[str] = "frequency"
    MONETARY: Final[str] = "monetary"
    RECENCY_SCORE: Final[str] = "recency_score"
    FREQUENCY_SCORE: Final[str] = "frequency_score"
    MONETARY_SCORE: Final[str] = "monetary_score"

    # Composite RFM ranking: (R*100) + (F*10) + M
    RFM_SCORE: Final[str] = "rfm_score"

    # Average gap between consecutive purchases
    PURCHASE_INTERVAL: Final[str] = "purchase_interval"

    # Weighted composite of browsing, pages, frequency
    ENGAGEMENT_SCORE: Final[str] = "engagement_score"

    # Weighted churn hazard indicator
    CUSTOMER_RISK_SCORE: Final[str] = "customer_risk_score"

    # Promotion sensitivity: discount_usage / total_orders
    DISCOUNT_DEPENDENCY: Final[str] = "discount_dependency"

    # Service friction: complaints / support_calls
    COMPLAINT_RATIO: Final[str] = "complaint_ratio"

    # Product satisfaction proxy: returned_orders / total_orders
    RETURN_RATIO: Final[str] = "return_ratio"

    # Normalized digital engagement composite
    BROWSING_SCORE: Final[str] = "browsing_score"

    # Top 10% customers by CLV
    VIP_FLAG: Final[str] = "vip_flag"

    # membership_type == 'Premium'
    PREMIUM_FLAG: Final[str] = "premium_flag"

    # Quintile-based behavioral segmentation label
    ACTIVITY_LEVEL: Final[str] = "activity_level"

    # Ordered tuple of all engineered features
    ALL_FEATURES: Final[tuple[str, ...]] = (
        CUSTOMER_AGE_GROUP, CUSTOMER_LIFETIME_DAYS, AVG_MONTHLY_SPEND,
        RECENCY, FREQUENCY, MONETARY,
        RECENCY_SCORE, FREQUENCY_SCORE, MONETARY_SCORE, RFM_SCORE,
        PURCHASE_INTERVAL, ENGAGEMENT_SCORE, CUSTOMER_RISK_SCORE,
        DISCOUNT_DEPENDENCY, COMPLAINT_RATIO, RETURN_RATIO,
        BROWSING_SCORE, VIP_FLAG, PREMIUM_FLAG, ACTIVITY_LEVEL,
    )


# ===========================================================================
# Categorical Domain Values
# ===========================================================================
# Allowed values for categorical columns. Used by the data generator
# to produce valid data and by the preprocessor for encoding validation.
# ===========================================================================


@unique
class Gender(Enum):
    """Allowed gender classifications."""
    MALE = "Male"
    FEMALE = "Female"
    OTHER = "Other"


@unique
class MembershipType(Enum):
    """Customer membership tier levels."""
    BASIC = "Basic"
    SILVER = "Silver"
    GOLD = "Gold"
    PREMIUM = "Premium"


@unique
class PaymentMethod(Enum):
    """Supported payment instrument types."""
    CREDIT_CARD = "Credit Card"
    DEBIT_CARD = "Debit Card"
    UPI = "UPI"
    NET_BANKING = "Net Banking"
    WALLET = "Wallet"
    COD = "Cash on Delivery"


@unique
class CustomerSegment(Enum):
    """Business-defined customer value segments."""
    NEW = "New"
    REGULAR = "Regular"
    LOYAL = "Loyal"
    VIP = "VIP"
    AT_RISK = "At Risk"
    DORMANT = "Dormant"


@unique
class ActivityLevel(Enum):
    """Engagement-based activity classification labels.

    Derived from engagement score quintile binning.
    """
    VERY_LOW = "Very Low"
    LOW = "Low"
    MEDIUM = "Medium"
    HIGH = "High"
    VERY_HIGH = "Very High"


# ===========================================================================
# Geographic Domains
# ===========================================================================
# Representative Indian metro cities and states for the e-commerce context.
# ===========================================================================


class Geography:
    """Geographic domain values for customer demographics."""

    COUNTRY: Final[str] = "India"

    CITIES: Final[tuple[str, ...]] = (
        "Mumbai", "Delhi", "Bangalore", "Hyderabad", "Chennai",
        "Kolkata", "Pune", "Ahmedabad", "Jaipur", "Lucknow",
        "Chandigarh", "Indore", "Bhopal", "Surat", "Nagpur",
        "Patna", "Kochi", "Coimbatore", "Visakhapatnam", "Vadodara",
    )

    # Mapping of city to state for referential integrity
    CITY_STATE_MAP: Final[dict[str, str]] = {
        "Mumbai": "Maharashtra",
        "Delhi": "Delhi",
        "Bangalore": "Karnataka",
        "Hyderabad": "Telangana",
        "Chennai": "Tamil Nadu",
        "Kolkata": "West Bengal",
        "Pune": "Maharashtra",
        "Ahmedabad": "Gujarat",
        "Jaipur": "Rajasthan",
        "Lucknow": "Uttar Pradesh",
        "Chandigarh": "Punjab",
        "Indore": "Madhya Pradesh",
        "Bhopal": "Madhya Pradesh",
        "Surat": "Gujarat",
        "Nagpur": "Maharashtra",
        "Patna": "Bihar",
        "Kochi": "Kerala",
        "Coimbatore": "Tamil Nadu",
        "Visakhapatnam": "Andhra Pradesh",
        "Vadodara": "Gujarat",
    }

    STATES: Final[tuple[str, ...]] = tuple(
        sorted(set(CITY_STATE_MAP.values()))
    )


# ===========================================================================
# Business Rule Constants
# ===========================================================================
# Thresholds, weights, and bins used by feature engineering formulas.
# All values match the Technical Design Document specifications.
# ===========================================================================


class BusinessRules:
    """Domain-specific thresholds and formula weights.

    These constants encode the exact formulas specified in the TDD
    for feature engineering calculations.
    """

    # --- RFM Scoring ---
    RFM_QUANTILE_COUNT: Final[int] = 5
    RFM_SCORE_LABELS: Final[tuple[int, ...]] = (1, 2, 3, 4, 5)

    # --- Age Group Bins ---
    # Boundaries: 18-24, 25-34, 35-44, 45-54, 55-64, 65+
    AGE_BIN_EDGES: Final[tuple[int, ...]] = (0, 24, 34, 44, 54, 64, 100)
    AGE_BIN_LABELS: Final[tuple[str, ...]] = (
        "18-24", "25-34", "35-44", "45-54", "55-64", "65+",
    )

    # --- Customer Risk Score Weights (TDD §4.2) ---
    # Risk = 0.4*complaints + 0.3*cart_abandon + 0.2*support + 0.1*(5-satisfaction)
    RISK_WEIGHT_COMPLAINTS: Final[float] = 0.4
    RISK_WEIGHT_CART_ABANDONMENT: Final[float] = 0.3
    RISK_WEIGHT_SUPPORT_CALLS: Final[float] = 0.2
    RISK_WEIGHT_DISSATISFACTION: Final[float] = 0.1
    MAX_SATISFACTION_SCORE: Final[int] = 5

    # --- Engagement Score Weights (TDD §4.3) ---
    # Engagement = 0.5*(browsing/max) + 0.3*(pages/max) + 0.2*frequency
    ENGAGEMENT_WEIGHT_BROWSING: Final[float] = 0.5
    ENGAGEMENT_WEIGHT_PAGES: Final[float] = 0.3
    ENGAGEMENT_WEIGHT_FREQUENCY: Final[float] = 0.2

    # --- VIP Classification ---
    VIP_CLV_PERCENTILE: Final[float] = 0.90

    # --- Outlier Winsorization ---
    WINSORIZE_LOWER_PERCENTILE: Final[float] = 0.01
    WINSORIZE_UPPER_PERCENTILE: Final[float] = 0.99

    # --- Churn Distribution Target ---
    CHURN_POSITIVE_RATE_MIN: Final[float] = 0.15
    CHURN_POSITIVE_RATE_MAX: Final[float] = 0.25

    # --- Data Generation ---
    AGE_MEAN: Final[int] = 35
    AGE_STD: Final[int] = 12
    AGE_MIN: Final[int] = 18
    AGE_MAX: Final[int] = 70


# ===========================================================================
# Color Palette
# ===========================================================================
# SaaS-grade premium pastel color system from the design specification.
# Inspired by Stripe, Linear, and Notion design language.
# ===========================================================================


class ColorPalette:
    """UI color hex codes for the Streamlit dashboard and visualizations.

    All values match the design specification exactly. Charts, cards,
    metric indicators, and status badges reference these constants
    to maintain visual consistency across the application.
    """

    # --- Backgrounds ---
    PRIMARY_BG: Final[str] = "#F8FAFC"
    SECONDARY_BG: Final[str] = "#EEF2FF"
    CARD_BG: Final[str] = "#FFFFFF"

    # --- Accents ---
    PRIMARY_ACCENT: Final[str] = "#7C83FD"
    SECONDARY_ACCENT: Final[str] = "#A5B4FC"

    # --- Status Colors ---
    SUCCESS: Final[str] = "#86EFAC"
    WARNING: Final[str] = "#FDE68A"
    ERROR: Final[str] = "#FCA5A5"

    # --- Typography ---
    TEXT_PRIMARY: Final[str] = "#1F2937"
    TEXT_SECONDARY: Final[str] = "#6B7280"
    TEXT_MUTED: Final[str] = "#9CA3AF"

    # --- Chart Color Sequence ---
    # A harmonious palette for Plotly multi-series visualizations
    CHART_COLORS: Final[tuple[str, ...]] = (
        "#7C83FD",  # Soft Indigo
        "#A5B4FC",  # Light Pastel Blue
        "#86EFAC",  # Mint Green
        "#FDE68A",  # Warm Amber
        "#FCA5A5",  # Soft Coral
        "#C4B5FD",  # Lavender
        "#67E8F9",  # Cyan
        "#FDBA74",  # Peach
        "#F9A8D4",  # Pink
        "#6EE7B7",  # Emerald
    )

    # --- Gradient Pairs ---
    GRADIENT_PRIMARY: Final[tuple[str, str]] = ("#7C83FD", "#A5B4FC")
    GRADIENT_SUCCESS: Final[tuple[str, str]] = ("#86EFAC", "#6EE7B7")
    GRADIENT_WARNING: Final[tuple[str, str]] = ("#FDE68A", "#FDBA74")
    GRADIENT_ERROR: Final[tuple[str, str]] = ("#FCA5A5", "#F87171")


# ===========================================================================
# UI Design Tokens
# ===========================================================================
# CSS values for Streamlit custom styling and component rendering.
# ===========================================================================


class UITokens:
    """CSS design tokens for the Streamlit dashboard.

    These values are injected into Streamlit via st.markdown() to override
    default styling and achieve the premium SaaS look specified in the TDD.
    """

    # --- Typography ---
    FONT_FAMILY: Final[str] = (
        "'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', "
        "Roboto, 'Helvetica Neue', Arial, sans-serif"
    )
    FONT_SIZE_SM: Final[str] = "0.875rem"
    FONT_SIZE_BASE: Final[str] = "1rem"
    FONT_SIZE_LG: Final[str] = "1.125rem"
    FONT_SIZE_XL: Final[str] = "1.5rem"
    FONT_SIZE_2XL: Final[str] = "2rem"
    FONT_SIZE_3XL: Final[str] = "2.5rem"

    # --- Spacing ---
    SPACING_XS: Final[str] = "0.25rem"
    SPACING_SM: Final[str] = "0.5rem"
    SPACING_MD: Final[str] = "1rem"
    SPACING_LG: Final[str] = "1.5rem"
    SPACING_XL: Final[str] = "2rem"
    SPACING_2XL: Final[str] = "3rem"

    # --- Border Radius ---
    RADIUS_SM: Final[str] = "6px"
    RADIUS_MD: Final[str] = "8px"
    RADIUS_LG: Final[str] = "12px"
    RADIUS_XL: Final[str] = "16px"
    RADIUS_FULL: Final[str] = "9999px"

    # --- Shadows ---
    SHADOW_SM: Final[str] = "0 1px 2px 0 rgba(0, 0, 0, 0.05)"
    SHADOW_MD: Final[str] = "0 4px 6px -1px rgba(0, 0, 0, 0.05)"
    SHADOW_LG: Final[str] = (
        "0 10px 15px -3px rgba(0, 0, 0, 0.08), "
        "0 4px 6px -4px rgba(0, 0, 0, 0.05)"
    )
    SHADOW_GLASS: Final[str] = (
        "0 8px 32px 0 rgba(124, 131, 253, 0.10)"
    )

    # --- Glassmorphism ---
    GLASS_BACKDROP: Final[str] = "blur(12px)"
    GLASS_BG: Final[str] = "rgba(255, 255, 255, 0.75)"
    GLASS_BORDER: Final[str] = "1px solid rgba(255, 255, 255, 0.18)"

    # --- Transitions ---
    TRANSITION_FAST: Final[str] = "all 0.15s ease"
    TRANSITION_BASE: Final[str] = "all 0.2s ease"
    TRANSITION_SLOW: Final[str] = "all 0.3s ease"

    # --- Hover Effects ---
    HOVER_TRANSLATE: Final[str] = "translateY(-2px)"
    HOVER_SCALE: Final[str] = "scale(1.02)"


# ===========================================================================
# Model Registry
# ===========================================================================
# Classifier identifiers and tuning method assignments from the TDD.
# ===========================================================================


@unique
class TuningMethod(Enum):
    """Hyperparameter search strategy identifiers."""
    GRID_SEARCH = "GridSearchCV"
    RANDOM_SEARCH = "RandomizedSearchCV"


class ModelRegistry:
    """Machine learning model name constants and tuning assignments.

    The model names defined here are used as dictionary keys in the
    ModelTrainer comparison tables, evaluation reports, and dashboard
    model selection dropdowns.
    """

    LOGISTIC_REGRESSION: Final[str] = "Logistic Regression"
    DECISION_TREE: Final[str] = "Decision Tree"
    RANDOM_FOREST: Final[str] = "Random Forest"
    EXTRA_TREES: Final[str] = "Extra Trees"
    ADABOOST: Final[str] = "AdaBoost"
    GRADIENT_BOOSTING: Final[str] = "Gradient Boosting"
    XGBOOST: Final[str] = "XGBoost"
    LIGHTGBM: Final[str] = "LightGBM"
    CATBOOST: Final[str] = "CatBoost"

    # Ordered tuple of all model names
    ALL_MODELS: Final[tuple[str, ...]] = (
        LOGISTIC_REGRESSION,
        DECISION_TREE,
        RANDOM_FOREST,
        EXTRA_TREES,
        ADABOOST,
        GRADIENT_BOOSTING,
        XGBOOST,
        LIGHTGBM,
        CATBOOST,
    )

    # Model-to-tuning-method mapping per TDD specification
    TUNING_METHODS: Final[dict[str, TuningMethod]] = {
        LOGISTIC_REGRESSION: TuningMethod.GRID_SEARCH,
        DECISION_TREE: TuningMethod.GRID_SEARCH,
        RANDOM_FOREST: TuningMethod.RANDOM_SEARCH,
        EXTRA_TREES: TuningMethod.RANDOM_SEARCH,
        ADABOOST: TuningMethod.GRID_SEARCH,
        GRADIENT_BOOSTING: TuningMethod.RANDOM_SEARCH,
        XGBOOST: TuningMethod.RANDOM_SEARCH,
        LIGHTGBM: TuningMethod.RANDOM_SEARCH,
        CATBOOST: TuningMethod.RANDOM_SEARCH,
    }


# ===========================================================================
# Evaluation Metric Names
# ===========================================================================
# Standardized metric label strings used in comparison tables and reports.
# ===========================================================================


class MetricNames:
    """Evaluation metric label constants.

    Used as column headers in model comparison DataFrames, chart axis
    labels, and classification report sections.
    """

    ACCURACY: Final[str] = "Accuracy"
    PRECISION: Final[str] = "Precision"
    RECALL: Final[str] = "Recall"
    F1_SCORE: Final[str] = "F1-Score"
    ROC_AUC: Final[str] = "ROC AUC"

    ALL_METRICS: Final[tuple[str, ...]] = (
        ACCURACY, PRECISION, RECALL, F1_SCORE, ROC_AUC,
    )

    # The primary metric used for automatic best model selection
    PRIMARY_SELECTION_METRIC: Final[str] = F1_SCORE


# ===========================================================================
# File Name Constants
# ===========================================================================
# Standardized filenames for generated datasets, model artifacts,
# and report outputs.
# ===========================================================================


class FileNames:
    """Standardized output file name constants.

    Centralizes all generated file names so that producers and consumers
    always reference the same identifiers without path duplication.
    """

    # --- Data Files ---
    RAW_DATASET: Final[str] = "ecommerce_customers.csv"
    CLEANED_DATASET: Final[str] = "cleaned_customers.csv"
    ENGINEERED_DATASET: Final[str] = "engineered_customers.csv"

    # --- Model Artifacts ---
    BEST_MODEL: Final[str] = "best_churn_model.joblib"
    FEATURE_NAMES: Final[str] = "feature_names.joblib"
    LABEL_ENCODERS: Final[str] = "label_encoders.joblib"
    SCALER: Final[str] = "scaler.joblib"
    MODEL_CONFIG: Final[str] = "model_config.joblib"

    # --- Evaluation Assets ---
    CONFUSION_MATRIX: Final[str] = "confusion_matrix.png"
    ROC_CURVE: Final[str] = "roc_curve.png"
    PR_CURVE: Final[str] = "pr_curve.png"
    LEARNING_CURVE: Final[str] = "learning_curve.png"
    VALIDATION_CURVE: Final[str] = "validation_curve.png"
    FEATURE_IMPORTANCE: Final[str] = "feature_importance.png"
    SHAP_SUMMARY: Final[str] = "shap_summary.png"
    MODEL_COMPARISON: Final[str] = "model_comparison.png"
    MODEL_COMPARISON_CSV: Final[str] = "model_comparison.csv"

    # --- Reports ---
    TECHNICAL_REPORT: Final[str] = "technical_report.md"
    PRESENTATION: Final[str] = "churn_analytics_presentation.pptx"


# ===========================================================================
# Dashboard Page Registry
# ===========================================================================
# Page identifiers and human-readable labels for Streamlit navigation.
# ===========================================================================


class DashboardPages:
    """Streamlit multi-page navigation identifiers.

    Each constant is a tuple of (page_key, display_label, icon_emoji)
    used by the sidebar navigation builder in app.py.
    """

    HOME: Final[tuple[str, str, str]] = (
        "home", "Home", "🏠"
    )
    EXECUTIVE_DASHBOARD: Final[tuple[str, str, str]] = (
        "dashboard", "Executive Dashboard", "📊"
    )
    CUSTOMER_ANALYTICS: Final[tuple[str, str, str]] = (
        "customer_analytics", "Customer Analytics", "👥"
    )
    SALES_ANALYTICS: Final[tuple[str, str, str]] = (
        "sales_analytics", "Sales Analytics", "💰"
    )
    PREDICTION: Final[tuple[str, str, str]] = (
        "prediction", "Prediction", "🔮"
    )
    BUSINESS_INSIGHTS: Final[tuple[str, str, str]] = (
        "business_insights", "Business Insights", "💡"
    )
    REPORTS: Final[tuple[str, str, str]] = (
        "reports", "Reports", "📄"
    )
    ABOUT: Final[tuple[str, str, str]] = (
        "about", "About", "ℹ️"
    )

    # Ordered list for sidebar rendering
    ALL_PAGES: Final[tuple[tuple[str, str, str], ...]] = (
        HOME, EXECUTIVE_DASHBOARD, CUSTOMER_ANALYTICS,
        SALES_ANALYTICS, PREDICTION, BUSINESS_INSIGHTS,
        REPORTS, ABOUT,
    )


# ===========================================================================
# KPI Label Constants
# ===========================================================================
# Display labels for business KPI cards on the executive dashboard.
# ===========================================================================


class KPILabels:
    """Business KPI display label constants.

    Used by the dashboard metric cards and report tables to ensure
    consistent KPI naming across the platform.
    """

    REVENUE: Final[str] = "Total Revenue"
    AVG_ORDER_VALUE: Final[str] = "Average Order Value"
    CUSTOMER_COUNT: Final[str] = "Total Customers"
    ACTIVE_CUSTOMERS: Final[str] = "Active Customers"
    INACTIVE_CUSTOMERS: Final[str] = "Inactive Customers"
    REPEAT_CUSTOMERS: Final[str] = "Repeat Customers"
    RETENTION_RATE: Final[str] = "Retention Rate"
    CHURN_RATE: Final[str] = "Churn Rate"
    GROWTH_RATE: Final[str] = "Growth Rate"
    REFUND_RATE: Final[str] = "Refund Rate"
    CONVERSION_RATE: Final[str] = "Conversion Rate"
    CLV: Final[str] = "Customer Lifetime Value"
    AVG_SESSION_DURATION: Final[str] = "Avg Session Duration"
    NET_REVENUE: Final[str] = "Net Revenue"
    MONTHLY_SALES: Final[str] = "Monthly Sales"
    QUARTERLY_SALES: Final[str] = "Quarterly Sales"
    YEARLY_SALES: Final[str] = "Yearly Sales"
