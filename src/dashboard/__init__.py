"""
Streamlit Dashboard Package.

Implements the premium SaaS-grade web application using Streamlit with
custom CSS overrides that replicate Stripe/Linear/Notion design language.
Contains reusable UI components, page routing, session state management,
and all eight application pages.

Modules:
    app: Main entry point managing sidebar navigation, global CSS injection,
         page routing, session state initialization, and application lifecycle.
    components: Reusable UI component library including glassmorphism metric
                cards, styled containers, progress indicators, toast alerts,
                recommendation cards, and custom button classes.

Sub-packages:
    pages: Individual page modules for each dashboard view (Home, Executive
           Dashboard, Customer Analytics, Sales Analytics, Prediction,
           Business Insights, Reports, About).

Dependencies:
    - streamlit (for web application framework)
    - plotly (for interactive chart rendering)
    - src.config (for color palette, paths, and UI tokens)
    - src.models.predict (for real-time inference on Prediction page)
    - src.utils (for logging and exception handling)
"""
