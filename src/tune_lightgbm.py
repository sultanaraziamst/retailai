"""
tune_lightgbm.py

• 5-fold TS CV + oversampling
• log1p→Tweedie loss (variance_power≈1.3)
• 400 Optuna trials (TPESampler + MedianPruner)
• parallel trees, early pruning, stop once CV < 9 %
• improved log-x error histogram + median line
"""
from __future__ import annotations
import os, warnings, yaml, json, joblib, optuna
import numpy as np, pandas as pd, lightgbm as lgb, matplotlib.pyplot as plt
from pathlib import Path
from typing import Any
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error
from tqdm.auto import tqdm


def wmape(y: np.ndarray, yhat: np.ndarray) -> float:
    d = np.abs(y).sum()
    return np.nan if d == 0 else np.abs(y - yhat).sum() / d


def load_cfg(path="config.yaml") -> dict[str, Any]:
    return yaml.safe_load(open(path, "r", encoding="utf-8"))


def save_cfg(cfg, path="config.yaml") -> None:
    yaml.dump(cfg, open(path, "w", encoding="utf-8"))


_ALLOWED = {
    "learning_rate","num_leaves","min_data_in_leaf",
    "lambda_l1","lambda_l2","feature_fraction",
    "bagging_fraction","bagging_freq",
    "drop_rate","skip_drop","tweedie_variance_power"
}
def legal(params): 
    return {k:v for k,v in params.items() if k in _ALLOWED}


def _objective(trial, X, y_log, y_raw, w, cv, booster, cache_idx):
    # sample Tweedie + tree params
    p = {
        "objective": "tweedie",
        "tweedie_variance_power": trial.suggest_float("tvp", 1.1, 1.9),
        "metric": "mae",
        "verbosity": -1,
        "seed": 42,
        "boosting_type": booster,
        "num_threads": os.cpu_count(),
        "learning_rate": trial.suggest_float("lr", 1e-3, 0.25, log=True),
        "num_leaves": trial.suggest_int("leaves", 16, 256, log=True),
        "min_data_in_leaf": trial.suggest_int("leaf_min", 5, 200, log=True),
        "lambda_l1": trial.suggest_float("l1", 1e-8, 10, log=True),
        "lambda_l2": trial.suggest_float("l2", 1e-8, 10, log=True),
        "feature_fraction": trial.suggest_float("ff", 0.4, 1.0),
        "bagging_fraction": trial.suggest_float("bf", 0.5, 1.0),
        "bagging_freq": trial.suggest_int("bfreq", 1, 15),
    }
    if booster == "dart":
        p["drop_rate"] = trial.suggest_float("drop_rate", 0, 0.4)
        p["skip_drop"] = trial.suggest_float("skip_drop", 0, 0.4)

    cv_scores = []
    for fold, (tr_idx, vl_idx) in enumerate(cv.split(X)):
        # cache & reuse oversampled train idx
        if fold not in cache_idx:
            low = (y_raw.iloc[tr_idx] <= 5).values
            dup = np.where(low)[0].repeat(2)
            cache_idx[fold] = np.concatenate([tr_idx, tr_idx[dup]])
        idx = cache_idx[fold]

        dtr = lgb.Dataset(X.iloc[idx], label=y_log.iloc[idx],
                          weight=None if w is None else w.iloc[idx])
        dva = lgb.Dataset(X.iloc[vl_idx], label=y_log.iloc[vl_idx])

        mdl = lgb.train(
            p, dtr, 1500, valid_sets=[dva],
            feval=lambda pred, data: (
                "wMAPE",
                wmape(y_raw.iloc[vl_idx].values, np.expm1(pred)),
                False
            ),
            callbacks=[lgb.early_stopping(80, verbose=False)]
        )
        pred = np.expm1(mdl.predict(X.iloc[vl_idx], num_iteration=mdl.best_iteration))
        cv_scores.append(wmape(y_raw.iloc[vl_idx].values, pred))

        # prune entire trial if CV < 9%
        if np.mean(cv_scores) < 0.09:
            trial.report(np.mean(cv_scores), fold)
            raise optuna.TrialPruned()

    return float(np.mean(cv_scores))


def main():
    cfg = load_cfg()
    f = cfg["models"]["forecast"]
    feats = Path(cfg["features"]["out_dir"]) / cfg["features"]["processed_forecast_path"]
    df = pd.read_parquet(feats)
    if f["drop_zero_target"]:
        df = df[df["target_next7"] > 0]

    y_raw = df.pop("target_next7")
    y_log = np.log1p(y_raw)
    X     = df.drop(columns=["itemid","date"])
    w     = df["sales_sum_7d"].clip(lower=0.5) if f["use_sample_weight"] else None

    cv     = TimeSeriesSplit(n_splits=5)
    cache  = {}

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(multivariate=True, group=True),
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=10)
    )
    booster = "dart" if f.get("use_dart", False) else "gbdt"

    study.optimize(
        lambda t: _objective(t, X, y_log, y_raw, w, cv, booster, cache),
        n_trials=400,
        show_progress_bar=True,
    )

    best = study.best_params
    print(f"✅ Best CV wMAPE: {study.best_value:.2%}")

    # train final on full data
    final_p = legal(best) | {
        "objective": "tweedie",
        "tweedie_variance_power": best["tvp"],
        "metric": "mae",
        "verbosity": -1,
        "seed": 42,
        "boosting_type": booster,
        "num_threads": os.cpu_count(),
    }
    dtr = lgb.Dataset(X, label=y_log, weight=None if w is None else w)
    mdl = lgb.train(final_p, dtr, num_boost_round=best["leaves"] * 12)

    preds = np.expm1(mdl.predict(X))
    mae_full = mean_absolute_error(y_raw, preds)
    wm_full  = wmape(y_raw.values, preds)
    passed   = wm_full <= 0.10

    # diagnostics
    rpt = Path("reports"); rpt.mkdir(exist_ok=True)
    err = np.abs(y_raw - preds)
    plt.figure(figsize=(4,3))
    plt.hist(err, bins=np.logspace(-3, np.log10(err.max()+1), 60),
             edgecolor="k", alpha=0.8)
    plt.xscale("log")
    plt.axvline(np.median(err), color="red", ls="--", lw=1,
                label=f"median={np.median(err):.2f}")
    plt.title("Tuned Model | |Error|"); plt.legend()
    plt.tight_layout()
    plt.savefig(rpt / "tunedlighbgm_error_hist.png", dpi=120)
    plt.close()

    fi = pd.DataFrame({
        "feat": mdl.feature_name(),
        "imp": mdl.feature_importance("gain")
    }).sort_values("imp", ascending=False).head(20)
    plt.figure(figsize=(6,4))
    plt.barh(fi.feat[::-1], fi.imp[::-1])
    plt.title("Tuned Model – Top-20 Gain")
    plt.tight_layout()
    plt.savefig(rpt / "tunedlighbgm_feature_importance.png", dpi=120)
    plt.close()

    # markdown
    (rpt / "metrics_forecast_tuned.md").write_text(
f"""# Tuned LightGBM Forecast Report

| Metric      | Value      |
| ----------- | ---------- |
| MAE (full)  | {mae_full:.5f} |
| wMAPE (full)| {wm_full:.2%}  |
| Pass ≤ 10 % | {'✅' if passed else '❌'} |
| Best trial  | {study.best_trial.number} |
| CV-wMAPE    | {study.best_value:.2%} |

```json
{json.dumps(best, indent=2)}""", encoding="utf-8")
    # persist
    Path(f["tuned_model_weighted_path"]).parent.mkdir(exist_ok=True, parents=True)
    joblib.dump(mdl, f["tuned_model_weighted_path"])
    cfg["models"]["forecast"].update(best | {"tweedie_variance_power": best["tvp"]})
    save_cfg(cfg)
    print("Model saved ➜", f["tuned_model_weighted_path"])
    print("Config patched; final wMAPE:", f"{wm_full:.2%}", "| Gate:", "✅" if passed else "❌")
if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=UserWarning, module="lightgbm")
    main()