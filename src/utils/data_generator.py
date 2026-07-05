"""
Data Ingestion and Augmentation Utility.

Following the Realism-First Data Strategy, this module is responsible for
ingesting a real-world base dataset (Kaggle E-Commerce Churn Dataset) and
statistically augmenting it to fulfill the enterprise 32-column schema
required by the TDD. 

It does NOT synthetically scale the dataset size. Instead, it maps existing
real behavioral features to our schema and carefully derives missing
business attributes using conditional probability and deterministic logic
anchored to the real data (e.g., deriving `total_spend` from real
`CashbackAmount`, or establishing timestamps from real `Tenure`).

Usage:
    from src.utils.data_generator import DataAugmenter
    
    augmenter = DataAugmenter()
    # Generates the fully mapped and augmented baseline CSV
    augmenter.generate_baseline_dataset()

Dependencies:
    - pandas, numpy (for statistical distributions and mapping)
    - src.config.settings (for configuration and paths)
    - src.utils.logger (for execution logging)
    - src.utils.exceptions (for error handling)
"""

from __future__ import annotations

import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from src.config.constants import (
    ColumnNames,
    FileNames,
    Gender,
    Geography,
    MembershipType,
    PaymentMethod,
)
from src.config.settings import settings
from src.utils.exceptions import DataGenerationError
from src.utils.logger import get_logger

logger = get_logger(__name__)


class DataAugmenter:
    """Ingests and augments a Kaggle base dataset into the target schema."""

    def __init__(self, base_file_path: Optional[Path] = None):
        """Initialize the data augmenter.

        Args:
            base_file_path: Path to the raw Kaggle dataset. If None, it
                            expects 'E-Commerce_Dataset.csv' in data/external.
        """
        self.base_path = base_file_path or (
            settings.paths.external_data_dir / "E-Commerce_Dataset.csv"
        )
        self.output_path = settings.paths.raw_dataset_path

        # Set the global random seed for reproducible augmentation
        np.random.seed(settings.model.random_state)

    def generate_baseline_dataset(self) -> Path:
        """Execute the full ingestion and augmentation pipeline.

        Returns:
            The absolute path to the generated raw dataset.
            
        Raises:
            DataGenerationError: If the base dataset is missing or augmentation fails.
        """
        logger.info(f"Starting data augmentation using base: {self.base_path}")

        # 1. Validate Base File Presence
        if not self.base_path.exists():
            raise DataGenerationError(
                f"Base Kaggle dataset missing at {self.base_path}. "
                "Please download the 'E-Commerce Dataset' by Ankush Pandey "
                "and place it in the data/external/ directory."
            )

        try:
            # 2. Load Base Data
            df_base = pd.read_csv(self.base_path)
            logger.info(f"Loaded {len(df_base)} base records.")

            # 3. Initialize Target DataFrame
            df_target = pd.DataFrame()

            # 4. Map Exact/Direct Equivalents
            self._map_direct_features(df_base, df_target)

            # 5. Derive Time & Date Features
            self._derive_temporal_features(df_base, df_target)

            # 6. Derive Demographic & Geographic Features
            self._derive_demographics(df_base, df_target)

            # 7. Derive Transactional & Monetary Features
            self._derive_transactional_features(df_base, df_target)

            # 8. Derive Behavioral & Support Features
            self._derive_behavioral_features(df_base, df_target)

            # 9. Derive Complex Business Rules & KPIs
            self._derive_business_metrics(df_base, df_target)

            # 10. Reorder columns to match the strict TDD schema
            df_target = df_target[list(ColumnNames.ALL_COLUMNS)]

            # 11. Save to Disk
            df_target.to_csv(self.output_path, index=False)
            logger.info(
                f"Successfully augmented and saved {len(df_target)} records "
                f"with {len(df_target.columns)} features to {self.output_path}"
            )

            return self.output_path

        except Exception as e:
            logger.error(f"Augmentation pipeline failed: {e}", exc_info=True)
            raise DataGenerationError(f"Failed to augment dataset: {e}") from e

    def _map_direct_features(self, df_base: pd.DataFrame, df_target: pd.DataFrame) -> None:
        """Map Kaggle columns directly to TDD columns where possible."""
        df_target[ColumnNames.CUSTOMER_ID] = df_base["CustomerID"]
        df_target[ColumnNames.CHURN] = df_base["Churn"]
        
        # Gender mapping with normalization
        df_target[ColumnNames.GENDER] = df_base["Gender"].replace(
            {"Male": Gender.MALE.value, "Female": Gender.FEMALE.value}
        ).fillna(Gender.OTHER.value)
        
        # Satisfaction and Complaints
        df_target[ColumnNames.CUSTOMER_SATISFACTION] = df_base["SatisfactionScore"].fillna(3)
        df_target[ColumnNames.COMPLAINTS] = df_base["Complain"].fillna(0)
        df_target[ColumnNames.TOTAL_ORDERS] = df_base["OrderCount"].fillna(1)
        df_target[ColumnNames.COUPON_USAGE] = df_base["CouponUsed"].fillna(0)
        
        # Payment Method Mapping
        pay_map = {
            "Debit Card": PaymentMethod.DEBIT_CARD.value,
            "Credit Card": PaymentMethod.CREDIT_CARD.value,
            "E wallet": PaymentMethod.WALLET.value,
            "UPI": PaymentMethod.UPI.value,
            "COD": PaymentMethod.COD.value,
            "Cash on Delivery": PaymentMethod.COD.value,
        }
        df_target[ColumnNames.PAYMENT_METHOD] = df_base["PreferredPaymentMode"].map(pay_map).fillna(PaymentMethod.CREDIT_CARD.value)
        
        # Browsing time mapping (HourSpendOnApp -> minutes)
        df_target[ColumnNames.BROWSING_TIME] = (df_base["HourSpendOnApp"].fillna(2) * 60).astype(int)

    def _derive_temporal_features(self, df_base: pd.DataFrame, df_target: pd.DataFrame) -> None:
        """Derive registration and last purchase dates backwards from today."""
        # Anchor date: Current date for realistic generation
        today = datetime.datetime.now()
        
        # Kaggle Tenure is in months. Convert to days.
        tenure_days = df_base["Tenure"].fillna(0) * 30
        
        # Registration Date = today - tenure_days
        df_target[ColumnNames.REGISTRATION_DATE] = tenure_days.apply(
            lambda days: (today - datetime.timedelta(days=int(days))).strftime("%Y-%m-%d")
        )
        
        # Last Purchase Date = today - DaySinceLastOrder
        days_since = df_base["DaySinceLastOrder"].fillna(0)
        df_target[ColumnNames.LAST_PURCHASE_DATE] = days_since.apply(
            lambda days: (today - datetime.timedelta(days=int(days))).strftime("%Y-%m-%d")
        )

    def _derive_demographics(self, df_base: pd.DataFrame, df_target: pd.DataFrame) -> None:
        """Derive age, geography, and membership tier."""
        n = len(df_base)
        
        # Age: Normal distribution anchored at 35, std 12 (bounded 18-70)
        ages = np.random.normal(loc=35, scale=12, size=n)
        df_target[ColumnNames.AGE] = np.clip(ages, 18, 70).astype(int)
        
        # Geography: Use Kaggle CityTier to assign realistic Metro/Non-Metro Indian cities
        tier1_cities = ["Mumbai", "Delhi", "Bangalore", "Hyderabad", "Chennai"]
        tier2_cities = ["Pune", "Ahmedabad", "Jaipur", "Lucknow", "Chandigarh"]
        tier3_cities = ["Indore", "Bhopal", "Surat", "Nagpur", "Patna", "Kochi"]
        
        def assign_city(tier):
            if tier == 1:
                return np.random.choice(tier1_cities)
            elif tier == 2:
                return np.random.choice(tier2_cities)
            else:
                return np.random.choice(tier3_cities)
                
        cities = df_base["CityTier"].fillna(3).apply(assign_city)
        df_target[ColumnNames.CITY] = cities
        df_target[ColumnNames.STATE] = cities.map(Geography.CITY_STATE_MAP)
        df_target[ColumnNames.COUNTRY] = Geography.COUNTRY

    def _derive_transactional_features(self, df_base: pd.DataFrame, df_target: pd.DataFrame) -> None:
        """Derive spend, AOV, and discount usage using Kaggle CashbackAmount."""
        n = len(df_base)
        
        # Cashback is realistically 2-5% of total spend. Reverse engineer Total Spend.
        cashback = df_base["CashbackAmount"].fillna(100)
        cashback_rate = np.random.uniform(0.02, 0.05, size=n)
        total_spend = cashback / cashback_rate
        df_target[ColumnNames.TOTAL_SPEND] = total_spend.round(2)
        
        # Average Order Value = Total Spend / Order Count
        orders = df_target[ColumnNames.TOTAL_ORDERS]
        df_target[ColumnNames.AVERAGE_ORDER_VALUE] = (total_spend / np.maximum(orders, 1)).round(2)
        
        # Purchase Frequency (orders per year)
        tenure_years = np.maximum(df_base["Tenure"].fillna(1) / 12.0, 0.1)
        df_target[ColumnNames.PURCHASE_FREQUENCY] = (orders / tenure_years).round(2)
        
        # Discount Usage: Proportional to CouponUsed
        df_target[ColumnNames.DISCOUNT_USAGE] = (df_target[ColumnNames.COUPON_USAGE] * np.random.uniform(1.2, 2.0, size=n)).astype(int)

    def _derive_behavioral_features(self, df_base: pd.DataFrame, df_target: pd.DataFrame) -> None:
        """Derive digital interactions, support calls, and returns."""
        n = len(df_base)
        
        # Session Duration: Correlated with browsing time (minutes per session)
        browsing = df_target[ColumnNames.BROWSING_TIME]
        orders = df_target[ColumnNames.TOTAL_ORDERS]
        estimated_sessions = np.maximum(orders * np.random.uniform(2, 5, size=n), 1)
        df_target[ColumnNames.SESSION_DURATION] = (browsing / estimated_sessions).round(1)
        
        # Pages Viewed: Correlated with browsing time
        df_target[ColumnNames.PAGES_VIEWED] = (df_target[ColumnNames.SESSION_DURATION] * np.random.uniform(1.5, 3.0, size=n)).astype(int)
        
        # Wishlist & Cart Abandonment
        df_target[ColumnNames.WISHLIST_ITEMS] = np.random.poisson(lam=3, size=n)
        df_target[ColumnNames.CART_ABANDONMENT] = np.random.poisson(lam=2, size=n)
        
        # Returns & Refunds: Correlated with Complaints
        complaints = df_target[ColumnNames.COMPLAINTS]
        # Customers who complained have a higher chance of returns
        return_prob = np.where(complaints > 0, 0.4, 0.05)
        has_return = np.random.binomial(1, return_prob, size=n)
        
        # Max returned orders cannot exceed total orders
        returned = has_return * np.random.randint(1, np.maximum(orders + 1, 2))
        df_target[ColumnNames.RETURNED_ORDERS] = np.minimum(returned, orders)
        
        # Refund amount is a fraction of total spend if there are returns
        aov = df_target[ColumnNames.AVERAGE_ORDER_VALUE]
        df_target[ColumnNames.REFUND_AMOUNT] = (df_target[ColumnNames.RETURNED_ORDERS] * aov * np.random.uniform(0.8, 1.0, size=n)).round(2)

        # Support Calls: Highly correlated with Complaints
        df_target[ColumnNames.SUPPORT_CALLS] = (complaints * np.random.uniform(1.5, 3.0, size=n)).astype(int)
        
        # Customer & Delivery Ratings: Anchored to real SatisfactionScore
        sat_score = df_target[ColumnNames.CUSTOMER_SATISFACTION]
        # Delivery rating fluctuates slightly from overall satisfaction
        delivery = sat_score + np.random.normal(0, 0.5, size=n)
        df_target[ColumnNames.DELIVERY_RATING] = np.clip(delivery.round(), 1, 5).astype(int)
        
        # Granular Customer Rating (1-5 float) based on integer satisfaction
        cust_rating = sat_score + np.random.normal(0, 0.2, size=n)
        df_target[ColumnNames.CUSTOMER_RATING] = np.clip(cust_rating, 1.0, 5.0).round(1)

    def _derive_business_metrics(self, df_base: pd.DataFrame, df_target: pd.DataFrame) -> None:
        """Derive macro KPIs like CLV, segments, and membership tiers."""
        n = len(df_base)
        
        # Customer Lifetime Value = Total Spend + Projected Future Margin
        # We simplify for raw data: CLV is slightly higher than total spend for active users
        total_spend = df_target[ColumnNames.TOTAL_SPEND]
        churn = df_target[ColumnNames.CHURN]
        
        # Churned customers have static CLV, active customers have projected higher CLV
        clv_multiplier = np.where(churn == 1, 1.0, np.random.uniform(1.1, 1.5, size=n))
        df_target[ColumnNames.CUSTOMER_LIFETIME_VALUE] = (total_spend * clv_multiplier).round(2)
        
        # Loyalty Score: 1-100 based on orders, tenure, and spend
        orders = df_target[ColumnNames.TOTAL_ORDERS]
        tenure = df_base["Tenure"].fillna(0)
        loyalty_raw = (orders * 5) + (tenure * 2) + (total_spend / 1000)
        # Normalize to 1-100 scale
        loyalty_norm = (loyalty_raw - loyalty_raw.min()) / (loyalty_raw.max() - loyalty_raw.min() + 1e-5) * 99 + 1
        df_target[ColumnNames.LOYALTY_SCORE] = loyalty_norm.astype(int)
        
        # Membership Type & Customer Segment logic
        def assign_membership(loyalty):
            if loyalty > 85: return MembershipType.PREMIUM.value
            if loyalty > 60: return MembershipType.GOLD.value
            if loyalty > 30: return MembershipType.SILVER.value
            return MembershipType.BASIC.value
            
        df_target[ColumnNames.MEMBERSHIP_TYPE] = df_target[ColumnNames.LOYALTY_SCORE].apply(assign_membership)
        
        # Standardize baseline segments (Phase 6 FeatureEngineer adds advanced ones)
        df_target[ColumnNames.CUSTOMER_SEGMENT] = "Regular"
