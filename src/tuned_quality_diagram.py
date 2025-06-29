#!/usr/bin/env python
"""
tuned_lightgbm_quality_diagram.py

Generates a 2×2 panel figure of:
  1. Actual vs Predicted quartile matrix
  2. Residuals vs Fitted
  3. Q–Q plot of residuals
  4. MAPE by predicted-value quintile

Supports LightGBM text models and Python-pickled (.pkl) Booster objects.
Automatically falls back among common model filenames if the one given is missing.
"""

import argparse
import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
import lightgbm as lgb


def resolve_model_path(path):
    # If it exists literally, use it
    if os.path.exists(path):
        return path

    # Otherwise, try common alternatives
    base_dir = os.path.dirname(path) or "artefacts"
    for fname in ("lightgbm_tuned_weighted.pkl",
                  "lightgbm_tuned.pkl",
                  "lightgbm.pkl"):
        alt = os.path.join(base_dir, fname)
        if os.path.exists(alt):
            print(f"> Using fallback model file: {alt}")
            return alt

    raise FileNotFoundError(f"Model file not found: {path} or any fallback in {base_dir}")


def load_data(feat_path):
    df = pd.read_parquet(feat_path)
    y = df["target_next7"].values
    X = df.drop(columns=["target_next7", "itemid", "date"], errors="ignore")
    return X, y


def load_model(model_path):
    model_path = resolve_model_path(model_path)
    if model_path.endswith(".pkl"):
        with open(model_path, "rb") as f:
            obj = pickle.load(f)
        if isinstance(obj, dict) and "booster" in obj:
            return obj["booster"]
        if isinstance(obj, lgb.Booster):
            return obj
        raise ValueError(f"Pickle did not contain a Booster: {model_path}")
    return lgb.Booster(model_file=model_path)


def predict(model, X):
    return model.predict(X)


def plot_diagnostics(y_true, y_pred, out_path):
    resid = y_true - y_pred

    # 1. Quartile confusion matrix (bins from true-values)
    bins = np.quantile(y_true, [0, 0.25, 0.5, 0.75, 1.0])
    true_q = np.digitize(y_true, bins) - 1
    pred_q = np.digitize(y_pred, bins) - 1
    matrix = pd.crosstab(true_q, pred_q, rownames=["Actual"], colnames=["Predicted"])

    # 4. MAPE by predicted quintile
    quint_edges = np.quantile(y_pred, np.linspace(0, 1, 6))
    quint = np.digitize(y_pred, quint_edges) - 1
    mape_by_q = {}
    for q in range(5):
        mask = (quint == q)
        # skip bins with <2 samples
        if mask.sum() < 2:
            continue
        y_t = y_true[mask]
        y_p = y_pred[mask]
        nonzero = y_t != 0
        if nonzero.sum() < 1:
            continue
        mape_by_q[q] = np.mean(np.abs(y_t[nonzero] - y_p[nonzero]) / y_t[nonzero]) * 100

    # Plot
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    ax1, ax2, ax3, ax4 = axes.flatten()

    # Panel 1: Quartile matrix
    im = ax1.imshow(matrix, cmap="Blues", origin="lower")
    ax1.set_xticks(range(4)); ax1.set_yticks(range(4))
    ax1.set_xlabel("Predicted Quartile"); ax1.set_ylabel("Actual Quartile")
    ax1.set_title("Actual vs Predicted Quartile")
    for (i, j), val in np.ndenumerate(matrix.values):
        ax1.text(j, i, int(val),
                 ha="center", va="center",
                 color="white" if val > matrix.values.max()/2 else "black")
    fig.colorbar(im, ax=ax1, fraction=0.046, pad=0.04)

    # Panel 2: Residuals vs. Fitted
    ax2.scatter(y_pred, resid, alpha=0.3, s=5)
    ax2.axhline(0, color="red", linewidth=1)
    ax2.set_xlabel("Predicted Value"); ax2.set_ylabel("Residual (True–Pred)")
    ax2.set_title("Residuals vs Fitted")

    # Panel 3: Q–Q Plot of Residuals
    stats.probplot(resid, dist="norm", plot=ax3)
    ax3.set_title("Q–Q Plot of Residuals")

    # Panel 4: MAPE by Predicted Quintile
    qs, vals = zip(*sorted(mape_by_q.items()))
    ax4.bar([str(q) for q in qs], vals, color="skyblue")
    ax4.set_xlabel("Predicted Quintile (0=lowest)"); ax4.set_ylabel("MAPE (%)")
    ax4.set_title("MAPE by Predicted-Value Quintile")

    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"Saved tuned diagnostics to {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Plot tuned-lightgbm quality diagnostics"
    )
    parser.add_argument(
        "--model", required=True,
        help="Path to LightGBM model file (text) or pickled Booster (.pkl)"
    )
    parser.add_argument(
        "--features", required=True,
        help="Parquet file containing features and 'target_next7' column"
    )
    parser.add_argument(
        "--output", default="reports/tuned_prediction_quality.png",
        help="Output path for the diagnostics PNG"
    )
    args = parser.parse_args()

    model = load_model(args.model)
    X, y = load_data(args.features)
    y_pred = predict(model, X)
    plot_diagnostics(y, y_pred, args.output)


if __name__ == "__main__":
    main()
