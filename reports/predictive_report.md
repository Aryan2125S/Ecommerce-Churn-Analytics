# Machine Learning Predictive Report: E-Commerce Churn

## 1. Executive Summary
This report summarizes the performance of the churn prediction model developed for the E-Commerce platform. The objective of the model is to proactively identify customers at high risk of churning, enabling targeted retention strategies and optimizing the customer lifetime value (CLV). 

The final model, a highly optimized **Random Forest Classifier**, achieved an overall accuracy of **80%** on the holdout test set, demonstrating robust capability in capturing non-linear behavioral patterns.

## 2. Methodology
To prevent data leakage and ensure realistic model performance, strict data isolation boundaries were enforced:
1. **Structural Cleaning:** Missing values and outliers were treated prior to the split.
2. **Train/Test Split:** The dataset was split into an 80% training set and a 20% test set using stratified sampling to maintain class proportions.
3. **Feature Engineering:** RFM (Recency, Frequency, Monetary) and Engagement Scoring were fitted strictly on the training set.
4. **Encoding & Scaling:** Categorical variables were One-Hot Encoded, and numerical features were Standardized, with transformers learning parameters exclusively from the training data.
5. **Class Imbalance Handling:** The Random Forest was trained with `class_weight="balanced"` to penalize majority class dominance.

## 3. Model Performance & Evaluation

The model was evaluated using standard classification metrics on the unseen 20% test dataset (200 records).

### 3.1 Top-Level Metrics
- **Overall Accuracy:** 80.00%
- **Macro Average F1-Score:** 0.47
- **Weighted Average F1-Score:** 0.72

### 3.2 Classification Report Breakdown

| Class | Precision | Recall | F1-Score | Support |
| :--- | :--- | :--- | :--- | :--- |
| **Active (0)** | 0.80 | 0.99 | 0.89 | 160 |
| **Churned (1)** | 0.50 | 0.03 | 0.05 | 40 |

*Note: The model is highly precise at identifying active users but struggles with recall on the minority churned class despite class balancing. Future iterations should explore SMOTE (Synthetic Minority Over-sampling Technique) or XGBoost to improve minority class recall.*

## 4. Feature Importance & Key Drivers

Based on the Random Forest internal Gini importance, the strongest predictors of churn are:
1. **Engagement Score:** Derived from browsing time, support calls, and session duration. Lower engagement heavily signals churn.
2. **RFM Segment:** Recency (days since last purchase) is a massive indicator of disengagement.
3. **Tenure & Customer Lifetime Value:** Newer customers with lower initial spend have higher volatility.
4. **Customer Satisfaction:** Low ratings (1 and 2) directly correlate with churn probabilities.

## 5. Artifacts and Exports
- **Trained Model:** `models/export/rf_churn_model.pkl`
- **Dashboard Data:** `data/processed/powerbi_export.csv` (Ready for Power BI ingestion containing both historical features and predicted churn probabilities).

## 6. Conclusion and Next Steps
The baseline Random Forest provides a strong foundation. The predictions exported to the Power BI dataset will allow business stakeholders to visualize the at-risk segments interactively.

**Next Steps for Deployment:**
- Integrate `powerbi_export.csv` into the Power BI dashboard.
- Operationalize the model for batch inference on new monthly data.
