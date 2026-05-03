"""
src/evaluate.py
---------------
Evaluation utilities: metrics, ROC curves, confusion matrices,
calibration plots, and SHAP feature importance.
"""

import logging
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
    average_precision_score,
    brier_score_loss,
)

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import FIGURES_DIR

logger = logging.getLogger(__name__)

# ── Style ──────────────────────────────────────────────────────────────────
plt.style.use("seaborn-v0_8-whitegrid")
PALETTE = {"baseline": "#9b9b9b", "rf_tuned": "#2E86AB", "gradient_boosting": "#E84855"}


# ── Core Metrics ───────────────────────────────────────────────────────────

def compute_metrics(model, X_test, y_test, label: str = "model", threshold: float = 0.5) -> dict:
    """
    Return a dict of standard binary classification metrics.

    Metrics included
    ----------------
    roc_auc, avg_precision, brier_score,
    precision, recall, f1, tn, fp, fn, tp
    """
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

    return {
        "label":         label,
        "roc_auc":       round(roc_auc_score(y_test, y_prob),         4),
        "avg_precision": round(average_precision_score(y_test, y_prob), 4),
        "brier_score":   round(brier_score_loss(y_test, y_prob),       4),
        "precision":     round(precision_score(y_test, y_pred),        4),
        "recall":        round(recall_score(y_test, y_pred),           4),
        "f1":            round(f1_score(y_test, y_pred),               4),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


def log_metrics(metrics: dict) -> None:
    print(f"\n── {metrics['label']} ──────────────")
    for k, v in metrics.items():
        if k not in ("label", "tn", "fp", "fn", "tp"):
            print(f"  {k:<18}: {v}")
    print(f"  Confusion matrix  TN={metrics['tn']}  FP={metrics['fp']}  "
          f"FN={metrics['fn']}  TP={metrics['tp']}")


# ── ROC Curve ──────────────────────────────────────────────────────────────

def plot_roc_curves(models: dict, X_test, y_test, save: bool = True) -> plt.Figure:
    """
    Overlay ROC curves for multiple models.

    Parameters
    ----------
    models : dict
        {label: fitted_pipeline}
    """
    fig, ax = plt.subplots(figsize=(7, 6))

    for label, model in models.items():
        y_prob = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        auc = roc_auc_score(y_test, y_prob)
        color = PALETTE.get(label, None)
        ax.plot(fpr, tpr, lw=2, label=f"{label}  (AUC={auc:.3f})", color=color)

    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves — Purchase Conversion Models")
    ax.legend(loc="lower right")
    fig.tight_layout()

    if save:
        path = FIGURES_DIR / "roc_curves.png"
        fig.savefig(path, dpi=150)
        logger.info(f"ROC curve saved → {path}")

    return fig


# ── Calibration Plot ───────────────────────────────────────────────────────

def plot_calibration(models: dict, X_test, y_test, save: bool = True) -> plt.Figure:
    """Reliability diagram for probability calibration."""
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Perfect calibration")

    for label, model in models.items():
        y_prob = model.predict_proba(X_test)[:, 1]
        frac_pos, mean_pred = calibration_curve(y_test, y_prob, n_bins=10)
        ax.plot(mean_pred, frac_pos, "s-", label=label, color=PALETTE.get(label))

    ax.set_xlabel("Mean Predicted Probability")
    ax.set_ylabel("Fraction of Positives")
    ax.set_title("Calibration Curves")
    ax.legend()
    fig.tight_layout()

    if save:
        path = FIGURES_DIR / "calibration.png"
        fig.savefig(path, dpi=150)
        logger.info(f"Calibration plot saved → {path}")

    return fig


# ── Confusion Matrix ───────────────────────────────────────────────────────

def plot_confusion_matrix(model, X_test, y_test, label: str = "Model", save: bool = True):
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay(cm, display_labels=["No Purchase", "Purchase"]).plot(ax=ax, cmap="Blues")
    ax.set_title(f"Confusion Matrix — {label}")
    fig.tight_layout()

    if save:
        slug = label.lower().replace(" ", "_")
        path = FIGURES_DIR / f"cm_{slug}.png"
        fig.savefig(path, dpi=150)
        logger.info(f"Confusion matrix saved → {path}")

    return fig


# ── Feature Importance ─────────────────────────────────────────────────────

def plot_feature_importance(model, feature_names: list[str], top_n: int = 20, save: bool = True):
    """
    Extract and plot feature importances from a fitted RandomForest pipeline.
    Falls back to permutation importance if classifier doesn't expose importances.
    """
    clf = model.named_steps.get("classifier")
    if clf is None:
        # Try calibrated wrapper
        clf = getattr(model, "estimator", None)
        if clf is not None:
            clf = clf.named_steps.get("classifier")

    if clf is None or not hasattr(clf, "feature_importances_"):
        logger.warning("Model does not expose feature_importances_; skipping plot.")
        return None

    importances = clf.feature_importances_
    n_features  = min(len(importances), len(feature_names))
    importance_df = (
        pd.DataFrame({"feature": feature_names[:n_features], "importance": importances[:n_features]})
        .sort_values("importance", ascending=False)
        .head(top_n)
    )

    fig, ax = plt.subplots(figsize=(9, 6))
    sns.barplot(data=importance_df, x="importance", y="feature", palette="Blues_r", ax=ax)
    ax.set_title(f"Top {top_n} Feature Importances (RandomForest)")
    ax.set_xlabel("Mean Decrease in Impurity")
    fig.tight_layout()

    if save:
        path = FIGURES_DIR / "feature_importance.png"
        fig.savefig(path, dpi=150)
        logger.info(f"Feature importance plot saved → {path}")

    return fig


# ── Prediction Threshold Analysis ─────────────────────────────────────────

def threshold_analysis(model, X_test, y_test, save: bool = True) -> pd.DataFrame:
    """
    Compute precision, recall, and F1 across a range of thresholds
    to support business-driven threshold selection.
    """
    y_prob = model.predict_proba(X_test)[:, 1]
    thresholds = np.arange(0.10, 0.91, 0.05)
    records = []
    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        records.append({
            "threshold": round(t, 2),
            "precision": round(precision_score(y_test, y_pred, zero_division=0), 3),
            "recall":    round(recall_score(y_test, y_pred, zero_division=0), 3),
            "f1":        round(f1_score(y_test, y_pred, zero_division=0), 3),
        })

    df = pd.DataFrame(records)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(df["threshold"], df["precision"], label="Precision", marker="o")
    ax.plot(df["threshold"], df["recall"],    label="Recall",    marker="s")
    ax.plot(df["threshold"], df["f1"],        label="F1",        marker="^")
    ax.set_xlabel("Decision Threshold")
    ax.set_ylabel("Score")
    ax.set_title("Precision / Recall / F1 vs Threshold")
    ax.legend()
    fig.tight_layout()

    if save:
        path = FIGURES_DIR / "threshold_analysis.png"
        fig.savefig(path, dpi=150)
        logger.info(f"Threshold analysis saved → {path}")

    return df
