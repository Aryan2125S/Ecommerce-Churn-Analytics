"""
Model Evaluation Package.

Generates comprehensive evaluation metrics, comparison tables, and
publication-quality visualizations for trained classification models.
Includes SHAP-based model explainability analysis.

Modules:
    evaluate: Implements the ModelEvaluator class that produces classification
              reports, confusion matrix heatmaps, ROC curves, Precision-Recall
              curves, learning curves, validation curves, feature importance
              bar charts, SHAP summary/force/waterfall plots, and multi-model
              comparison tables.

Dependencies:
    - scikit-learn (for metric calculations)
    - matplotlib, seaborn (for static visualizations)
    - plotly (for interactive charts)
    - shap (for model explainability)
    - src.config (for color palette and output paths)
    - src.utils (for logging and exception handling)
"""
