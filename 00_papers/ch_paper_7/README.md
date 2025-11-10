Trustworthy ROC/PR Curves Project
=================================

This folder was auto-generated to rebuild the ROC and PR curves with a realistic, reproducible pipeline.
Key choices:

- Data: Synthetic but statistically plausible 5-class dataset with class imbalance (weights=[0.35, 0.25, 0.18, 0.13, 0.09]).
- Model: One-vs-Rest calibrated Logistic Regression (sigmoid, 5-fold CV).
- Splitting: Stratified 70/30 train/test.
- Scaling: Standardization on features.
- Curves: Macro-averaged ROC and PR computed by interpolating per-class curves on a common grid and averaging.
- Baselines: ROC chance diagonal; PR baseline uses macro prevalence (mean positive rate across classes).
- Reproducibility: RANDOM_STATE = 42.

Quick Metrics (from results/metrics.json):
- Macro ROC AUC: 0.951
- Macro PR AUC (trapz): 1.000
- Macro AP (sklearn): 0.837
- Micro AP: 0.887

Artifacts:
- img/roc_curve_macro.png (macro ROC)
- img/pr_curve_macro.png (macro PR)
- img/roc_curve_per_class.png
- img/pr_curves_per_class.png
- img/confusion_matrix.png
- img/reliability_diagram.png
- results/roc_macro_curve.csv (FPR-TPR grid)
- results/pr_macro_curve.csv (Recall-Precision grid)
- results/metrics.json (summary numbers)
