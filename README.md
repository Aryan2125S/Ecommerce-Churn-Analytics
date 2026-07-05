# E-Commerce Customer Churn Analytics

This project is an end-to-end Machine Learning and Data Engineering solution designed to predict customer churn in an e-commerce ecosystem. By processing raw customer data, engineering complex behavior features (like RFM segments and Engagement Scores), and orchestrating a robust machine learning pipeline, this project accurately identifies at-risk customers. The results are visualized in a premium, interactive Streamlit dashboard, providing actionable business intelligence to retention and marketing teams.

## Features
- **Customer Churn Prediction**: High-accuracy Random Forest classifier to identify at-risk users.
- **Data Preprocessing**: Strict, leakage-proof data cleaning, imputation, and outlier handling.
- **Feature Engineering**: Dynamic generation of RFM (Recency, Frequency, Monetary) metrics and custom Engagement Scores.
- **SQL Database Integration**: Scalable SQLite database managed entirely through SQLAlchemy ORM.
- **Machine Learning Pipeline**: Automated model training, scaling, encoding, and evaluation workflows.
- **Interactive Streamlit Dashboard**: A professional, dynamic UI to explore customer analytics and high-risk user profiles.
- **Power BI Compatible Export**: Generates clean, flat CSVs (`powerbi_export.csv`) for immediate BI tool ingestion.
- **Customer Behavior Analysis**: In-depth exploratory data analysis inside Jupyter notebooks.
- **Predictive Reporting**: Automated markdown report generation containing model metrics and feature importance.

## Tech Stack
- **Python**: Core programming language.
- **Pandas / NumPy**: Data manipulation and numerical operations.
- **Scikit-learn**: Machine learning model development and pipeline orchestration.
- **SQLAlchemy / SQLite**: Relational database mapping and local storage.
- **Streamlit**: Web application framework for the interactive dashboard.
- **Plotly**: Highly interactive, theme-adaptive charts and graphs.
- **Matplotlib / Seaborn**: Static data visualization used in exploratory notebooks.

## Project Structure

```text
📦 Ecommerce-Churn-Analytics
 ┣ 📂 data               # Contains raw, external, and processed data (including powerbi_export.csv).
 ┣ 📂 models             # Serialized Machine Learning models (e.g., random_forest.joblib).
 ┣ 📂 notebooks          # Jupyter Notebooks utilized for initial EDA and KPI analysis.
 ┣ 📂 reports            # Output directory for automated model evaluation and predictive reports.
 ┣ 📂 src                # Core application source code.
 ┃ ┣ 📂 config           # Application settings, constants, and database configurations.
 ┃ ┣ 📂 database         # SQLAlchemy database models, session managers, and initializers.
 ┃ ┣ 📂 dashboard        # Streamlit interactive dashboard UI (app.py).
 ┃ ┣ 📂 feature_engineering # Modules to compute RFM and Engagement metrics.
 ┃ ┣ 📂 models           # ML training pipelines and prediction scripts.
 ┃ ┣ 📂 preprocessing    # Data cleaners, outlier handlers, and feature scalers.
 ┃ ┗ 📂 utils            # Reusable utility functions, logging, and custom exceptions.
 ┣ 📜 requirements.txt   # Complete list of Python dependencies.
 ┗ 📜 README.md          # Project documentation (this file).
```

## Installation & Usage

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd Ecommerce-Churn-Analytics
   ```

2. **Install the required dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Generate Database & Train the Model**
   Run the orchestration script to clean data, engineer features, train the model, and export the processed datasets:
   ```bash
   python -m src.models.train_model
   ```

4. **Launch the Interactive Dashboard**
   Once the pipeline finishes and `powerbi_export.csv` is generated, start the Streamlit application:
   ```bash
   python -m streamlit run src/dashboard/app.py
   ```
