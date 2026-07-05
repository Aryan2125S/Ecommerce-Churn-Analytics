"""
Dashboard Pages Sub-package.

Contains individual page modules for the Streamlit multi-page application.
Each module exports a single render function that constructs the full page
layout including headers, metric cards, charts, forms, and action elements.

Pages:
    home: Hero banner, project overview, tech stack grid, animated KPI cards.
    dashboard: Executive summary with revenue, churn, retention, and CLV KPIs.
    customer_analytics: RFM analysis, cohort retention, demographic segments.
    sales_analytics: Revenue trends, payment methods, regional performance.
    prediction: Churn prediction form, probability gauge, risk level display.
    business_insights: Structured recommendation cards with priority levels.
    reports: Data export controls for CSV, PDF, and evaluation assets.
    about: Project credits, architecture overview, methodology, and contact.

Dependencies:
    - streamlit (for page rendering)
    - plotly (for interactive visualizations)
    - src.dashboard.components (for reusable UI elements)
    - src.config.constants (for color palette and column references)
"""
