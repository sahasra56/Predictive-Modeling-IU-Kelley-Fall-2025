"""
notebooks/01_EDA.py
-------------------
Exploratory Data Analysis of the e-commerce session dataset.
Mirrors the logic of 01_EDA.ipynb.

Sections
--------
1. Data Overview
2. Target Distribution
3. Conversion by Category
4. Correlation Analysis
5. Feature Distributions by Conversion
6. Missing Values
"""

import sys
import warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, "..")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sys.path.insert(0, "../src")
from data_loader import load_raw_sessions
from feature_engineering import engineer_features

plt.style.use("seaborn-v0_8-whitegrid")
FIGURES = "../reports/figures/"

# ── 1. Load Data ───────────────────────────────────────────────────────────
df = engineer_features(load_raw_sessions())

print("=" * 60)
print(f"Dataset: {df.shape[0]:,} sessions, {df.shape[1]} columns")
print(f"Conversion rate: {df['converted'].mean():.2%}")
print(f"Missing values: {df.isnull().sum().sum()}")
print(df.dtypes.value_counts())

# ── 2. Target Distribution ─────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(11, 4))

df["converted"].value_counts().plot.bar(
    ax=axes[0], color=["#E84855", "#2E86AB"], edgecolor="white"
)
axes[0].set_xticklabels(["No Purchase", "Purchase"], rotation=0)
axes[0].set_title("Conversion Count")
axes[0].set_ylabel("Sessions")

df.groupby("converted")["session_duration_min"].plot.kde(ax=axes[1])
axes[1].set_title("Session Duration by Conversion")
axes[1].set_xlabel("Minutes")
axes[1].legend(["No Purchase", "Purchase"])

fig.tight_layout()
fig.savefig(f"{FIGURES}eda_target.png", dpi=150)
print("Saved: eda_target.png")

# ── 3. Conversion by Category ──────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

for ax, col in zip(axes, ["traffic_source", "device_type", "time_of_day"]):
    cvr = df.groupby(col)["converted"].mean().sort_values(ascending=False)
    cvr.plot.bar(ax=ax, color="#2E86AB", edgecolor="white")
    ax.set_title(f"CVR by {col.replace('_', ' ').title()}")
    ax.set_ylabel("Conversion Rate")
    ax.tick_params(axis="x", rotation=30)
    for bar in ax.patches:
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.002,
            f"{bar.get_height():.1%}",
            ha="center", va="bottom", fontsize=8,
        )

fig.tight_layout()
fig.savefig(f"{FIGURES}eda_cvr_by_category.png", dpi=150)
print("Saved: eda_cvr_by_category.png")

# ── 4. Numeric Feature Distributions ──────────────────────────────────────
numeric_cols = [
    "session_duration_min", "pages_viewed", "cart_add_count",
    "cart_abandon_rate", "visit_frequency_30d", "engagement_rate",
    "cart_commitment_ratio", "session_value_index",
]

fig, axes = plt.subplots(2, 4, figsize=(16, 8))
for ax, col in zip(axes.flat, numeric_cols):
    for val, color, label in [(0, "#9b9b9b", "No Purchase"), (1, "#2E86AB", "Purchase")]:
        df[df["converted"] == val][col].clip(upper=df[col].quantile(0.99)).plot.kde(
            ax=ax, color=color, label=label, alpha=0.7
        )
    ax.set_title(col.replace("_", " ").title())
    ax.set_ylabel("")
    ax.legend(fontsize=8)

fig.suptitle("Feature Distributions by Conversion Outcome", y=1.01, fontsize=13)
fig.tight_layout()
fig.savefig(f"{FIGURES}eda_feature_distributions.png", dpi=150)
print("Saved: eda_feature_distributions.png")

# ── 5. Correlation Heatmap ─────────────────────────────────────────────────
corr_cols = numeric_cols + ["is_returning_user", "used_promo_code",
                             "viewed_reviews", "added_to_wishlist", "converted"]
corr = df[corr_cols].corr()

fig, ax = plt.subplots(figsize=(12, 10))
mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
sns.heatmap(
    corr, mask=mask, annot=True, fmt=".2f",
    cmap="RdBu_r", center=0, vmin=-1, vmax=1,
    linewidths=0.5, ax=ax,
)
ax.set_title("Feature Correlation Matrix")
fig.tight_layout()
fig.savefig(f"{FIGURES}eda_correlation.png", dpi=150)
print("Saved: eda_correlation.png")

# ── 6. Summary Stats ───────────────────────────────────────────────────────
print("\n── Converted vs Not — Key Stats ──")
print(df.groupby("converted")[numeric_cols[:6]].mean().round(2).T.to_string())
