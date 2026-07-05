# E-Commerce Customer Churn Prediction & KPI Analysis
**Advanced Machine Learning & Data Analytics Project**

## 📌 Project Overview
This project is an end-to-end Machine Learning and Data Engineering pipeline designed to predict customer churn in an E-commerce environment. It encompasses data ingestion, strict leakage-proof preprocessing, advanced feature engineering, predictive modeling, and business intelligence export.

The final model successfully achieves **80% accuracy** in identifying at-risk customers, allowing targeted retention strategies.

## 🚀 Key Features & Methodology
1. **Data Ingestion & SQL Database (Phase 1-3):** Raw data is mapped and augmented into a robust 32-feature enterprise schema and stored securely via SQLAlchemy 2.0.
2. **Leakage-Proof Preprocessing (Phase 4):**
   - Imputation, Data Type corrections, and Outlier Handling (IQR/Winsorization).
   - Strict separation of `fit()` and `transform()` logic to prevent target leakage from the test set.
3. **Advanced Feature Engineering (Phase 5):**
   - **RFM Analysis:** Recency, Frequency, and Monetary value segmentation.
   - **Engagement Scoring:** Weighted composite index based on digital interactions (browsing time, session length).
4. **Machine Learning Pipeline (Phase 6):**
   - Fully automated pipeline splitting data, scaling (StandardScaler), and encoding (OneHotEncoder).
   - Highly optimized Random Forest Classifier balancing minority churn classes.
5. **Business Intelligence (Phase 7):**
   - Exported `powerbi_export.csv` containing ML predictions unified with RFM metrics, ready for interactive Power BI dashboards.

## 📁 Project Structure
```text
📦 Level 3
 ┣ 📂 data               # Raw, external, and processed data
 ┣ 📂 models             # Serialized ML models (joblib)
 ┣ 📂 notebooks          # Jupyter Notebooks for EDA and KPI Analysis
 ┣ 📂 reports            # Model Evaluation and Predictive Reports
 ┣ 📂 src                # Core Application Source Code
 ┃ ┣ 📂 config           # Application settings and constants
 ┃ ┣ 📂 database         # SQLAlchemy DB Managers
 ┃ ┣ 📂 feature_engineering # RFM, Engagement Scorers
 ┃ ┣ 📂 models           # ML Training pipelines
 ┃ ┣ 📂 preprocessing    # Data Cleaners, Outlier Handlers, Scalers
 ┃ ┗ 📂 utils            # Data Generators, Loggers, Exceptions
 ┣ 📜 requirements.txt   # Python Dependencies
 ┗ 📜 README.md          # Project Documentation
```

## ⚙️ How to Run

1. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Generate Database & Run Full ML Pipeline**
   The entire ingestion, preprocessing, training, and export pipeline is orchestrated in a single script:
   ```bash
   python -m src.models.train_model
   ```
   
   *Note: This will automatically build the SQLite database, apply structural cleaning, execute the Train/Test Split, run Feature Engineering, train the Random Forest model, evaluate it, and export the final PowerBI CSV to `data/processed/`.*

## 📊 Model Performance
The Random Forest model demonstrates robust performance across unseen test data:
- **Accuracy:** 80.00%
- **Precision (Active):** 0.80
- **Recall (Active):** 0.99
- **Top Predictive Features:** Engagement Score, RFM Segment, Tenure.

A full breakdown can be found in `reports/predictive_report.md`.

## 🛠️ Built With
- **Python 3.10+** (pandas, numpy, scikit-learn)
- **SQLAlchemy 2.0**
- **Jupyter Notebooks**
- **Power BI** (Target Dashboard)
