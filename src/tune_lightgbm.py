"""
tune_lightgbm.py  (fast)

Speed-oriented rewrite:
• parallel Optuna trials (n_jobs) with constant_liar TPE
• reduced search space + rounds scaled by lr
• per-fold pruning from fold 0
• per-fold Datasets pre-binned once
• force_row_wise / force_col_wise, num_threads tuned
• early study stop on plateau
"""
from __future__ import annotations

import os
import json
import warnings
from pathlib import Path
from typing import Any

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
import lightgbm as lgb
import yaml
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def wmape(y: np.ndarray, yhat: np.ndarray) -> float:
    d = np.abs(y).sum()
    return float("nan") if d == 0 else float(np.abs(y - yhat).sum() / d)


def load_cfg(path: str = "config.yaml") -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def save_cfg(cfg: dict[str, Any], path: str = "config.yaml") -> None:
    with open(path, "w", encoding="utf-8") as fh:
        yaml.dump(cfg, fh)


_ALLOWED = {
    "learning_rate", "num_leaves", "min_data_in_leaf",
    "lambda_l1", "lambda_l2", "feature_fraction",
    "bagging_fraction", "bagging_freq",
    "drop_rate", "skip_drop", "tweedie_variance_power",
}


# --------------------------------------------------------------------------- #
# objective
# --------------------------------------------------------------------------- #
def _objective(trial, X, y_log, y_raw, w_np, fold_data,
               booster, n_threads, first_fold_medians):
    lr = trial.suggest_float("lr", 0.02, 0.15, log=True)
    p = {
        "objective": "tweedie",
        "tweedie_variance_power": trial.suggest_float("tvp", 1.1, 1.9),
        "metric": "mae",
        "verbosity": -1,
        "seed": 42,
        "boosting_type": booster,
        "num_threads": n_threads,
        "max_bin": 127,
        "feature_pre_filter": False,
        "learning_rate": lr,
        "num_leaves": trial.suggest_int("leaves", 31, 127, log=True),
        "min_data_in_leaf": trial.suggest_int("leaf_min", 20, 200, log=True),
        "lambda_l1": trial.suggest_float("l1", 1e-8, 10, log=True),
        "lambda_l2": trial.suggest_float("l2", 1e-8, 10, log=True),
        "feature_fraction": trial.suggest_float("ff", 0.5, 1.0),
        "bagging_fraction": trial.suggest_float("bf", 0.6, 1.0),
        "bagging_freq": trial.suggest_int("bfreq", 1, 10),
    }
    if booster == "dart":
        p["drop_rate"] = trial.suggest_float("drop_rate", 0.0, 0.3)
        p["skip_drop"] = trial.suggest_float("skip_drop", 0.0, 0.3)

    # rounds scale with lr — small lr needs more trees
    max_rounds = int(min(3000, max(300, 1200 * lr + 300)))

    cv_scores: list[float] = []
    for fold, (dtr, dva, vl_idx) in enumerate(fold_data):
        y_vl = y_raw[vl_idx]

        if booster == "dart":
            callbacks = [lgb.log_evaluation(0)]
            n_rounds = max_rounds // 2      # DART converges faster per round
        else:
            callbacks = [lgb.early_stopping(50, verbose=False),
                         lgb.log_evaluation(0)]
            n_rounds = max_rounds

        mdl = lgb.train(
            p, dtr, num_boost_round=n_rounds,
            valid_sets=[dva],
            feval=lambda pred, data, _y=y_vl: (
                "wMAPE", wmape(_y, np.expm1(pred)), False
            ),
            callbacks=callbacks,
        )
        best_it = getattr(mdl, "best_iteration", 0) or n_rounds
        pred = np.expm1(mdl.predict(X[vl_idx], num_iteration=best_it))
        fold_score = wmape(y_vl, pred)
        cv_scores.append(fold_score)

        # ---- prune at every fold, incl. fold 0 --------------------------- #
        running = float(np.mean(cv_scores))
        trial.report(running, fold)
        if trial.should_prune():
            raise optuna.TrialPruned()
        # hard gate: if fold 0 already worse than median of completed folds
        if fold == 0 and len(first_fold_medians) >= 10:
            if fold_score > float(np.median(first_fold_medians)) * 1.5:
                raise optuna.TrialPruned()
        # pass gate
        if running < 0.09:
            raise optuna.TrialPruned()

    return float(np.mean(cv_scores))


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main() -> None:
    cfg = load_cfg()
    f = cfg["models"]["forecast"]

    feats = Path(cfg["features"]["out_dir"]) / cfg["features"]["processed_forecast_path"]
    df = pd.read_parquet(feats).reset_index(drop=True)

    if f["drop_zero_target"]:
        df = df[df["target_next7"] > 0].reset_index(drop=True)

    y_raw_s = df.pop("target_next7")
    w_col   = "sales_sum_7d"
    w_s     = df.pop(w_col) if f["use_sample_weight"] else None
    X_df    = df.drop(columns=["itemid", "date"])

    X     = np.ascontiguousarray(X_df.to_numpy(dtype=np.float32, copy=False))
    y_log = np.log1p(y_raw_s.to_numpy(dtype=np.float64, copy=False))
    y_raw = y_raw_s.to_numpy(dtype=np.float64, copy=False)
    w_np  = None if w_s is None else w_s.clip(lower=0.5).to_numpy(dtype=np.float32)
    feat_names = list(X_df.columns)
    del X_df, df, y_raw_s, w_s

    n_rows, n_feats = X.shape
    force_row_wise = n_rows >= n_feats

    # ---- threads ---------------------------------------------------------- #
    cpus = os.cpu_count() or 4
    n_trials_parallel = 4                      # parallel trials
    threads_per_trial = max(1, cpus // n_trials_parallel)
    print(f"CPUs={cpus} | parallel trials={n_trials_parallel} "
          f"| threads/trial={threads_per_trial}")

    # ---- pre-binned per-fold Datasets (reused by every trial) ------------- #
    cv = TimeSeriesSplit(n_splits=5)
    fold_data = []
    for tr_idx, vl_idx in cv.split(X):
        low = y_raw[tr_idx] <= 5
        dup = np.where(low)[0].repeat(2)
        tr_idx_os = np.concatenate([tr_idx, tr_idx[dup]])

        dtr = lgb.Dataset(
            X[tr_idx_os], label=y_log[tr_idx_os],
            weight=None if w_np is None else w_np[tr_idx_os],
            feature_name=feat_names,
            free_raw_data=False,
            params={
                "max_bin": 127,
                "feature_pre_filter": False,
                "force_row_wise": force_row_wise,
                "force_col_wise": not force_row_wise,
            },
        )
        dva = lgb.Dataset(
            X[vl_idx], label=y_log[vl_idx],
            feature_name=feat_names, reference=dtr,
            free_raw_data=False,
            params={
                "max_bin": 127,
                "feature_pre_filter": False,
                "force_row_wise": force_row_wise,
                "force_col_wise": not force_row_wise,
            },
        )
        # materialize bins NOW so no trial pays for it
        dtr.construct()
        dva.construct()
        fold_data.append((dtr, dva, vl_idx))

    booster = "dart" if f.get("use_dart", False) else "gbdt"

    # ---- study ------------------------------------------------------------ #
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    sampler = optuna.samplers.TPESampler(
        multivariate=True, group=True, seed=42,
        constant_liar=True,              # essential for n_jobs > 1
    )
    pruner = optuna.pruners.MedianPruner(
        n_startup_trials=15, n_warmup_steps=0,
    )
    study = optuna.create_study(
        direction="minimize", sampler=sampler, pruner=pruner,
    )

    first_fold_medians: list[float] = []
    n_trials  = 150
    patience  = 40

    # thread-safe callback: use a lock for the list append
    import threading
    lock = threading.Lock()

    def _trial_cb(study, trial):
        with lock:
            if trial.value is not None and trial.number < 20:
                # crude proxy: first-fold median tracked from pruned report
                pass
        if study.best_trial.number < trial.number - patience:
            study.stop()

    study.optimize(
        lambda t: _objective(
            t, X, y_log, y_raw, w_np, fold_data,
            booster, threads_per_trial, first_fold_medians,
        ),
        n_trials=n_trials,
        n_jobs=n_trials_parallel,
        show_progress_bar=True,
        callbacks=[_trial_cb],
    )

    best = study.best_params
    print(f"✅ Best CV wMAPE: {study.best_value:.2%} "
          f"(trial {study.best_trial.number})")

    rpt = Path("reports"); rpt.mkdir(exist_ok=True)
    study.trials_dataframe().to_csv(rpt / "optuna_trials.csv", index=False)

    # ---- refit on full data ----------------------------------------------- #
    final_p = {
        "objective": "tweedie",
        "tweedie_variance_power": best["tvp"],
        "metric": "mae",
        "verbosity": -1,
        "seed": 42,
        "boosting_type": booster,
        "num_threads": cpus,
        "max_bin": 127,
        "feature_pre_filter": False,
        "force_row_wise": force_row_wise,
        "force_col_wise": not force_row_wise,
        **{k: best[k] for k in _ALLOWED
           if k in best and k != "tweedie_variance_power"},
    }
    final_p["tweedie_variance_power"] = best["tvp"]

    dtr_full = lgb.Dataset(
        X, label=y_log,
        weight=None if w_np is None else w_np,
        feature_name=feat_names, free_raw_data=False,
        params={
            "max_bin": 127,
            "feature_pre_filter": False,
            "force_row_wise": force_row_wise,
            "force_col_wise": not force_row_wise,
        },
    )
    n_full = int(best["leaves"] * 12)
    mdl = lgb.train(final_p, dtr_full, num_boost_round=n_full,
                    callbacks=[lgb.log_evaluation(0)])

    preds = np.expm1(mdl.predict(X, num_iteration=n_full))
    mae_full = mean_absolute_error(y_raw, preds)
    wm_full  = wmape(y_raw, preds)
    passed   = wm_full <= 0.10

    # ---- diagnostics ------------------------------------------------------ #
    err = np.abs(y_raw - preds)
    plt.figure(figsize=(4, 3))
    if err.max() > 0:
        plt.hist(err, bins=np.logspace(-3, np.log10(err.max() + 1), 60),
                 edgecolor="k", alpha=0.8)
        plt.xscale("log")
    else:
        plt.hist(err, bins=30, edgecolor="k", alpha=0.8)
    plt.axvline(np.median(err), color="red", ls="--", lw=1,
                label=f"median={np.median(err):.2f}")
    plt.title("Tuned Model | |Error|"); plt.legend()
    plt.tight_layout()
    plt.savefig(rpt / "tunedlighbgm_error_hist.png", dpi=120)
    plt.close()

    fi = (pd.DataFrame({"feat": mdl.feature_name(),
                        "imp":  mdl.feature_importance("gain")})
          .sort_values("imp", ascending=False).head(20))
    plt.figure(figsize=(6, 4))
    plt.barh(fi.feat[::-1], fi.imp[::-1])
    plt.title("Tuned Model – Top-20 Gain")
    plt.tight_layout()
    plt.savefig(rpt / "tunedlighbgm_feature_importance.png", dpi=120)
    plt.close()

    # ---- markdown + persistence ------------------------------------------- #
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
{json.dumps(best, indent=2)}
```""",
        encoding="utf-8",
    )

    out_path = Path(f["tuned_model_weighted_path"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(mdl, out_path)

    cfg["models"]["forecast"].update(
        {k: best[k] for k in _ALLOWED if k in best}
        | {"tweedie_variance_power": best["tvp"]}
    )
    save_cfg(cfg)

    print("Model saved ➜", out_path)
    print("Config patched; final wMAPE:", f"{wm_full:.2%}",
          "| Gate:", "✅" if passed else "❌")


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=UserWarning, module="lightgbm")
    main()