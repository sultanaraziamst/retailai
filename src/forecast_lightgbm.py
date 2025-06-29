"""
Baseline LightGBM forecaster
• Poisson objective + wMAPE early‑stop
• Comprehensive diagnostics (3 PNGs, prefix: lightgbm_)
"""
from __future__ import annotations
import warnings, yaml, joblib, lightgbm as lgb, matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, Any

import numpy as np, pandas as pd
from sklearn.metrics import mean_absolute_error
from tqdm import tqdm


# ─── helpers ──────────────────────────────────────────────────────────────────
def wmape(y, yhat): d=np.abs(y).sum(); return np.abs(y-yhat).sum()/d if d else np.nan
def _cfg(p="config.yaml"): return yaml.safe_load(open(p,"r",encoding="utf-8"))
class _Bar:               # tqdm callback
    def __init__(self,t): self.t=tqdm(total=t,desc="Train",unit="iter",leave=False)
    def __call__(self,env): self.t.update(1); self.t.close() if env.iteration+1==self.t.total else None


# ─── visual block ────────────────────────────────────────────────────────────
def _diag_plots(val: pd.DataFrame, pred: np.ndarray, mdl: lgb.Booster, out: Path, prefix: str):
    y = val["target_next7"].values
    # 1️⃣ feature importance
    gain = mdl.feature_importance("gain"); names = mdl.feature_name()
    top = (pd.DataFrame({"f":names,"g":gain})
             .sort_values("g",ascending=False).head(20))
    plt.figure(figsize=(6,4)); plt.barh(top["f"][::-1], top["g"][::-1])
    plt.title("Top‑20 Feature Importance"); plt.tight_layout()
    plt.savefig(out/f"{prefix}feature_importance.png", dpi=120); plt.close()

    # 2️⃣ error histogram
    plt.figure(figsize=(4,3)); plt.hist(np.abs(y-pred),bins=40,edgecolor="k")
    plt.title("Absolute‑Error"); plt.tight_layout()
    plt.savefig(out/f"{prefix}error_hist.png", dpi=120); plt.close()

    # 3️⃣ 4‑panel quality figure
    fig = plt.figure(figsize=(9,7))
    # a) prediction‑quality matrix (quartile bins)
    ax1 = plt.subplot2grid((2,2),(0,0))
    q_actual = pd.qcut(y, 3, labels=False, duplicates="drop")
    q_pred   = pd.qcut(pred,3, labels=False, duplicates="drop")
    m = pd.crosstab(q_actual, q_pred)
    ax1.imshow(m, cmap="Blues")
    for (i,j),v in np.ndenumerate(m.values):
        ax1.text(j, i, str(v), ha="center", va="center", color="k")
    ax1.set_title("Prediction Quality Matrix\n(Quartile Bins)")
    ax1.set_xlabel("Predicted Quartile"); ax1.set_ylabel("Actual Quartile")

    # b) residuals vs fitted
    ax2 = plt.subplot2grid((2,2),(0,1))
    res = y - pred
    ax2.scatter(pred, res, s=8, alpha=.4)
    ax2.axhline(0, ls="--", c="r", lw=1)
    ax2.set_title("Residuals vs Fitted")
    ax2.set_xlabel("Predicted Values"); ax2.set_ylabel("Residuals")

    # c) Q‑Q plot of residuals
    ax3 = plt.subplot2grid((2,2),(1,0))
    from scipy import stats
    stats.probplot(res, dist="norm", plot=ax3)
    ax3.set_title("Q‑Q Plot of Residuals")

    # d) MAPE by value‐range (percentile bins)
    ax4 = plt.subplot2grid((2,2),(1,1))
    bins = pd.qcut(pred, 5, labels=False, duplicates="drop")
    # Convert to pandas Series for groupby
    res_series = pd.Series(np.abs(res)/np.maximum(y,1e-9))
    bins_series = pd.Series(bins)
    mape = res_series.groupby(bins_series).mean()*100
    mape.plot.bar(ax=ax4)
    ax4.set_title("Prediction Accuracy by Value Range")
    ax4.set_xlabel("Value Range (Percentile Bins)")
    ax4.set_ylabel("MAPE (%)")

    plt.tight_layout()
    fig.savefig(out/f"{prefix}prediction_quality.png", dpi=120)
    plt.close(fig)


# ─── training ────────────────────────────────────────────────────────────────
def train(df: pd.DataFrame, cfg: Dict[str,Any]) -> bool:
    f,cfgm = cfg["models"]["forecast"], cfg["models"]
    if f["drop_zero_target"]: df=df[df["target_next7"]>0]
    df = df.copy()  # Avoid SettingWithCopyWarning
    df["date"]=pd.to_datetime(df["date"])
    split=df["date"].max()-pd.Timedelta(days=cfgm["val_split_days"])
    tr,va=df[df["date"]<split], df[df["date"]>=split]
    Xtr=tr.drop(columns=["itemid","date","target_next7"]); ytr=tr["target_next7"]
    Xva=va.drop(columns=["itemid","date","target_next7"]); yva=va["target_next7"]
    w=tr["sales_sum_7d"].clip(lower=.1) if f["use_sample_weight"] else None
    dtr=lgb.Dataset(Xtr,label=ytr,weight=w); dva=lgb.Dataset(Xva,label=yva)

    legal={"learning_rate","num_leaves","min_data_in_leaf","lambda_l1","lambda_l2",
           "feature_fraction","bagging_fraction","bagging_freq"}
    params={k:f[k] for k in legal if k in f}|{"objective":"poisson","metric":"mae",
            "verbosity":-1,"seed":42}

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        mdl=lgb.train(params,dtr,4000,[dva],
                      feval=lambda p,d:("wMAPE",wmape(d.get_label(),p),False),
                      callbacks=[lgb.early_stopping(150,verbose=False),_Bar(4000)])

    pred=mdl.predict(Xva, num_iteration=mdl.best_iteration)
    mae=mean_absolute_error(yva,pred); w_err=wmape(yva.values,pred); passed=w_err<=0.10

    rpt=Path("reports"); rpt.mkdir(exist_ok=True)
    _diag_plots(va,pred,mdl,rpt,"lightgbm_")
    (rpt/"metrics_forecast_final.md").write_text(
f"""# Baseline Forecast Report

| Metric | Value |
| ------ | ----- |
| MAE | {mae:.5f} |
| wMAPE | {w_err:.2%} |
| Best iteration | {mdl.best_iteration} |
| Pass ≤ 10 %? | {'✅' if passed else '❌'} |
""",encoding="utf-8")

    Path(f["weighted_model_path"]).parent.mkdir(parents=True,exist_ok=True)
    joblib.dump(mdl,f["weighted_model_path"])
    return passed


if __name__=="__main__":
    cfg=_cfg()
    feats=Path(cfg["features"]["out_dir"])/cfg["features"]["processed_forecast_path"]
    ok=train(pd.read_parquet(feats),cfg)
    print("wMAPE gate:", "PASSED 🎉" if ok else "NOT met 😔")
