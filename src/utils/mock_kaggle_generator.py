import pandas as pd
import numpy as np
import os
from src.config.settings import settings

def generate_mock_kaggle_data():
    os.makedirs(settings.paths.external_data_dir, exist_ok=True)
    n = 1000
    np.random.seed(42)
    
    df = pd.DataFrame({
        "CustomerID": [f"C{str(i).zfill(5)}" for i in range(n)],
        "Churn": np.random.choice([0, 1], size=n, p=[0.8, 0.2]),
        "Gender": np.random.choice(["Male", "Female"], size=n),
        "SatisfactionScore": np.random.randint(1, 6, size=n),
        "Complain": np.random.choice([0, 1], size=n, p=[0.8, 0.2]),
        "OrderCount": np.random.randint(1, 16, size=n),
        "CouponUsed": np.random.randint(0, 5, size=n),
        "PreferredPaymentMode": np.random.choice(["Debit Card", "Credit Card", "E wallet", "UPI", "COD"], size=n),
        "HourSpendOnApp": np.random.uniform(0.5, 4.0, size=n).round(1),
        "Tenure": np.random.randint(0, 30, size=n),
        "DaySinceLastOrder": np.random.randint(1, 30, size=n),
        "CityTier": np.random.choice([1, 2, 3], size=n),
        "CashbackAmount": np.random.uniform(50, 300, size=n).round(2),
    })
    
    out_path = settings.paths.external_data_dir / "E-Commerce_Dataset.csv"
    df.to_csv(out_path, index=False)
    print(f"Mock Kaggle data saved to {out_path}")

if __name__ == "__main__":
    generate_mock_kaggle_data()
