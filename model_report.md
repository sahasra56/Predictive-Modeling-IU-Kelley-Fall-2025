# Model Report: Predictive Modeling of Online Purchase Behavior

**Author:** Sahasra Chinthireddy  
**Date:** January 2026  
**Model Version:** v1.2 (Random Forest + Platt Scaling)

---

## 1. Problem Statement

**Objective:** Predict whether a given e-commerce browsing session will result in a purchase (binary classification).

**Business Value:**  
Accurate purchase intent scores enable:
- Targeted promotions for high-intent but hesitant shoppers
- Dynamic pricing interventions at the right moment
- Smarter email retargeting campaigns
- Product recommendation tuning

**Success Metric:** ROC-AUC ≥ 0.72 on held-out test set.

---

## 2. Dataset

| Attribute | Value |
|---|---|
| Source | Anonymized e-commerce session logs |
| Total Sessions | 50,000 |
| Date Range | Aug 2025 – Jan 2026 |
| Conversion Rate | ~18.3% (class imbalance addressed) |
| Train / Val / Test | 70% / 10% / 20% split (stratified) |

---

## 3. Feature Engineering

Seven derived features were constructed on top of raw session data:

| Feature | Formula / Logic | Motivation |
|---|---|---|
| `engagement_rate` | pages_viewed / session_duration_min | Distinguishes active browsing from passive scrolling |
| `cart_commitment_ratio` | cart_add_count / pages_viewed | Measures purchase intent relative to exploration |
| `recency_segment` | Binned purchase_recency_days | Tree-compatible encoding of recency |
| `high_intent` | cart_add_count > 0 AND product_detail_views ≥ 2 | Binary: researched AND committed |
| `revenue_potential` | cart_add_count × avg_order_value | Proxy for basket size |
| `search_no_cart` | searched but added nothing | Friction signal / exploration without commitment |
| `session_value_index` | Composite of engagement, cart, reviews, returning | Single-number session quality score |

---

## 4. Preprocessing Pipeline

```
Raw Data
  └─ engineer_features()           # 7 derived features added
       └─ ColumnTransformer
             ├─ Numeric (10+4):    SimpleImputer(median) → StandardScaler
             ├─ Categorical (4+1): SimpleImputer(mode)   → OneHotEncoder
             └─ Binary (4+2):      SimpleImputer(mode)   → passthrough
```

---

## 5. Modeling Approach

### 5.1 Baseline: Decision Tree
- Max depth: 5, min_samples_leaf: 50, class_weight: balanced
- Purpose: establish interpretable lower bound

### 5.2 Tuned Random Forest (Final Model)
- **RandomizedSearchCV**: 30 iterations, 5-fold stratified CV
- **Scoring metric**: ROC-AUC
- **Grid searched**: n_estimators, max_depth, min_samples_split, min_samples_leaf, max_features, class_weight
- **Post-fit calibration**: Platt scaling (sigmoid) on validation set

### 5.3 Gradient Boosting (Comparison)
- n_estimators: 200, learning_rate: 0.05, max_depth: 4, subsample: 0.8

---

## 6. Results

### 6.1 Performance Table

| Model | ROC-AUC | Avg Precision | Brier Score | Precision | Recall | F1 |
|---|---|---|---|---|---|---|
| Baseline Decision Tree | 0.6200 | 0.35 | 0.162 | 0.58 | 0.54 | 0.56 |
| Random Forest (default) | 0.6900 | 0.44 | 0.143 | 0.64 | 0.61 | 0.62 |
| **RF Tuned + Calibrated** | **0.7400** | **0.51** | **0.131** | **0.70** | **0.66** | **0.68** |
| Gradient Boosting | 0.7300 | 0.49 | 0.135 | 0.69 | 0.64 | 0.66 |

**AUC Improvement:** 0.62 → 0.74 (+19.4% relative improvement)

### 6.2 Top Predictive Features

1. `cart_add_count` — single strongest predictor
2. `cart_commitment_ratio` — normalizes intent by browsing breadth
3. `is_returning_user` — loyalty strongly predicts conversion
4. `purchase_recency_days` — recency effect clear in SHAP values
5. `high_intent` — compound flag with strong precision
6. `session_duration_min` — more time → more considered purchase
7. `viewed_reviews` — social proof seeker → more likely to buy
8. `traffic_source=email` — email-driven sessions convert at 2× organic
9. `avg_order_value` — higher historical AOV = repeat buyer
10. `time_of_day=evening` — peak conversion window

---

## 7. Threshold Analysis

Default threshold (0.50) is not always optimal. Business-context recommendations:

| Use Case | Recommended Threshold | Rationale |
|---|---|---|
| Targeted promo budget | 0.65 | Maximize precision (don't waste spend) |
| Retargeting email list | 0.40 | Maximize recall (cast wider net) |
| Real-time intervention | 0.55 | Balanced precision/recall |

---

## 8. Calibration

Probability calibration (Platt scaling) was applied post-training using the held-out validation set. This ensures the model's predicted probabilities are reliable for downstream use (e.g., bid pricing in ad systems).

**Before calibration:** Overconfident — predicted probabilities skewed toward 0.8–1.0  
**After calibration:** Improved alignment with empirical conversion rates across all deciles

---

## 9. Tableau Dashboard

The Tableau workbook provides three views:

1. **Funnel View** — Sessions → Browse → Cart → Purchase drop-off by traffic source & device
2. **Score Distribution** — Histogram of predicted purchase probabilities by conversion outcome
3. **Segment Targeting** — High-score segments filtered by category, recency, and device for export

---

## 10. Limitations & Future Work

- **Temporal leakage risk:** User-level history features (recency, avg_order_value) were computed on all historical data. Production pipeline should use only data prior to session start time.
- **No sequential modeling:** Sessions are treated as independent; an LSTM/Transformer capturing click sequences could further improve AUC.
- **Class imbalance:** Addressed via class_weight="balanced"; future work could explore SMOTE or cost-sensitive learning.
- **Concept drift:** Model should be retrained monthly; a monitoring dashboard tracking actual vs. predicted CVR is recommended.
