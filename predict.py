"""
src/predict.py
--------------
Inference module: load the trained model and score new sessions.
Can be used as a library or run directly from the command line.

Usage
-----
    python src/predict.py --input data/raw/new_sessions.csv --output predictions.csv
"""

import argparse
import logging
from pathlib import Path

import joblib
import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import FINAL_MODEL_PATH
from feature_engineering import engineer_features

logger = logging.getLogger(__name__)


def load_model(path: Path = FINAL_MODEL_PATH):
    """Load the serialized pipeline from disk."""
    logger.info(f"Loading model from {path}")
    return joblib.load(path)


def predict(df_raw: pd.DataFrame, model=None, threshold: float = 0.5) -> pd.DataFrame:
    """
    Score a batch of raw session records.

    Parameters
    ----------
    df_raw : pd.DataFrame
        Raw sessions (same schema as training data).
    model  : fitted pipeline (loaded if None).
    threshold : float
        Decision threshold for binary label.

    Returns
    -------
    pd.DataFrame
        Original data + `prob_purchase` and `predicted_converted` columns.
    """
    if model is None:
        model = load_model()

    df_eng = engineer_features(df_raw)

    # Identify feature columns (drop metadata)
    drop_cols = {"session_id", "user_id", "converted"}
    X = df_eng.drop(columns=[c for c in drop_cols if c in df_eng.columns])

    probs = model.predict_proba(X)[:, 1]
    preds = (probs >= threshold).astype(int)

    df_out = df_raw.copy()
    df_out["prob_purchase"]      = probs.round(4)
    df_out["predicted_converted"] = preds

    logger.info(
        f"Scored {len(df_out):,} sessions  |  "
        f"Predicted conversion rate: {preds.mean():.2%}"
    )
    return df_out


def score_single_session(session: dict, model=None, threshold: float = 0.5) -> dict:
    """
    Score a single session dict. Useful for API/real-time inference.

    Parameters
    ----------
    session : dict  — raw session feature values.

    Returns
    -------
    dict with keys: prob_purchase, predicted_converted
    """
    df = pd.DataFrame([session])
    result = predict(df, model=model, threshold=threshold)
    return {
        "prob_purchase":      float(result["prob_purchase"].iloc[0]),
        "predicted_converted": int(result["predicted_converted"].iloc[0]),
    }


# ── CLI ───────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="Score new sessions with the purchase prediction model.")
    parser.add_argument("--input",     required=True,  help="Path to CSV of new sessions.")
    parser.add_argument("--output",    required=True,  help="Path to write scored CSV.")
    parser.add_argument("--threshold", default=0.5, type=float, help="Decision threshold.")
    parser.add_argument("--model",     default=str(FINAL_MODEL_PATH), help="Path to model .pkl file.")
    return parser.parse_args()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
    args = parse_args()

    model   = load_model(Path(args.model))
    df_raw  = pd.read_csv(args.input)
    df_out  = predict(df_raw, model=model, threshold=args.threshold)

    df_out.to_csv(args.output, index=False)
    print(f"\nScored output saved → {args.output}")
    print(df_out[["session_id", "prob_purchase", "predicted_converted"]].head(10).to_string(index=False))
