# Predictive Modeling of Online Purchase Behavior
---
## Project Overview

This project builds an end-to-end **conversion prediction pipeline** on anonymized e-commerce session data. The goal is to predict whether a given browsing session will result in a purchase, enabling targeted marketing and personalization strategies.

### Key Results
- Improved ROC-AUC from **0.62 → 0.74** via hyperparameter tuning and feature engineering
- Engineered behavioral features: purchase recency, cart frequency, session duration, and more
- Deployed interactive Tableau dashboards with drill-down funnel views
- Final model: **Random Forest** with calibrated probability outputs
---
## Project Structure
```
online_purchase_prediction/
│
├── data/
│   ├── raw/                    # Raw session data (gitignored)
│   ├── processed/              # Cleaned, feature-engineered datasets
│   └── generate_data.py        # Synthetic data generator (mirrors real schema)
│
├── notebooks/
│   ├── 01_EDA.ipynb            # Exploratory Data Analysis
│   ├── 02_feature_engineering.ipynb
│   ├── 03_modeling.ipynb       # Baseline → tuned models
│   └── 04_evaluation.ipynb     # Final evaluation, ROC curves, SHAP
│
├── src/
│   ├── __init__.py
│   ├── data_loader.py          # SQL queries + data loading utilities
│   ├── feature_engineering.py  # Feature construction pipeline
│   ├── model.py                # Model training, tuning, serialization
│   ├── evaluate.py             # Metrics, calibration, plotting
│   └── predict.py              # Inference on new sessions
│
├── models/
│   ├── baseline_dt.pkl         # Baseline decision tree
│   ├── random_forest_tuned.pkl # Final tuned model
│   └── model_metadata.json     # Training params + performance log
│
├── sql/
│   ├── create_tables.sql       # Schema definition
│   ├── session_features.sql    # Feature extraction query
│   └── funnel_analysis.sql     # Conversion funnel metrics
│
├── reports/
│   ├── model_report.md         # Full model write-up
│   └── figures/                # Saved plots (ROC, confusion matrix, SHAP)
│
├── tests/
│   ├── test_features.py
│   └── test_model.py
│
├── requirements.txt
├── config.py
└── README.md
```
---
## Quickstart

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Generate synthetic data
python data/generate_data.py

# 3. Run the full pipeline
python src/model.py

# 4. Run tests
pytest tests/
```
---
## Feature Engineering Summary

| Feature | Description |
|---|---|
| `session_duration_min` | Total time spent in session (minutes) |
| `pages_viewed` | Number of product pages visited |
| `cart_add_count` | Items added to cart during session |
| `cart_abandon_rate` | Historical cart-to-purchase ratio |
| `purchase_recency_days` | Days since last purchase |
| `visit_frequency_30d` | Sessions in past 30 days |
| `avg_order_value` | Average historical spend |
| `device_type` | Mobile / Desktop / Tablet (encoded) |
| `traffic_source` | Organic / Paid / Email / Direct (encoded) |
| `time_of_day` | Hour bucket (Morning/Afternoon/Evening/Night) |
| `is_returning_user` | Binary: has prior purchase history |
| `category_affinity` | Most visited product category |
---
## Model Performance

| Model | ROC-AUC | Precision | Recall | F1 |
|---|---|---|---|---|
| Baseline Decision Tree | 0.62 | 0.58 | 0.54 | 0.56 |
| Random Forest (default) | 0.69 | 0.64 | 0.61 | 0.62 |
| Random Forest (tuned) | **0.74** | **0.70** | **0.66** | **0.68** |
| Gradient Boosting | 0.73 | 0.69 | 0.64 | 0.66 |
