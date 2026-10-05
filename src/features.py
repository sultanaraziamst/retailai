"""
Enhanced feature-engineering for the RetailRocket dataset
(FORECAST + RECOMMENDATION) with visible TQDM progress bars.

CHANGES vs. original (leakage fixes):
  1. _smooth() is REMOVED from this file. Category-level smoothed ratios
     (ctr_sm_7d, buyrate_sm_7d, sales_sm_7d) used to be computed here over
     the ENTIRE date range (train + validation together) before any
     train/val split existed. That let validation rows' features be built
     from category means that included validation-period sales -> leakage.
     build_forecast_features() now emits the RAW ingredients only:
       ctr_7d, buyrate_7d, sales_sum_7d, cat_sales_7d, categoryid
     Smoothing is done later, in train.py, AFTER the chronological
     train/val split, fit on training rows only, then applied to
     validation/test rows. See train_lightgbm_fixed.py.
  2. _rolling_sum() now applies .shift(1) after the rolling sum, so the
     window at day t covers days [t-window, t-1] and EXCLUDES day t's own
     views/adds/sales. Previously the window included day t itself, which
     is a same-day proxy for future sales given daily sales autocorrelation
     -- this was the larger of the two leaks (bigger driver of the 0.94%
     vs 15.24% wMAPE gap than the smoothing leak). ctr_7d, buyrate_7d, and
     cat_sales_7d are all derived from the (now shifted) rolling sums, so
     they inherit the fix automatically -- no separate change needed there,
     only confirmation that they run after the rolling-window block (they
     do, and still do after this edit).
"""
from __future__ import annotations
import argparse, json, logging
from pathlib import Path
from typing import Dict, Any

import numpy as np
import pandas as pd
import yaml
from tqdm import tqdm


# ──────────────────────────────────────────────────────────────────────────────
def _load_cfg(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def _rolling_sum(df: pd.DataFrame, window: int, col: str) -> pd.Series:
    """Trailing rolling sum over [t-window, t-1] per item.

    Shift the raw column by one day *per item* first, then roll. Shifting
    after the groupby-rolling instead of before would shift across item
    boundaries in the flattened MultiIndex, leaking one item's history
    into the next item's first row.
    """
    d = df.sort_values(["itemid", "date"]).copy()
    d["_lag"] = d.groupby("itemid")[col].shift(1)
    return (
        d.groupby("itemid")["_lag"]
         .rolling(window, min_periods=1)
         .sum()
         .reset_index(level=0, drop=True)
    )


# def _rolling_sum(df, window, col):
#     d = df.sort_values(["itemid", "date"]).copy()
#     d["_lag"] = d.groupby("itemid")[col].shift(1)
#     return (
#         d.groupby("itemid")["_lag"]
#          .rolling(window, min_periods=1)
#          .sum()
#          .reset_index(level=0, drop=True)
#     )


# def _rolling_sum(df: pd.DataFrame, window: int, col: str) -> pd.Series:
#     """Trailing rolling sum EXCLUDING the current day.

#     The window at day t covers [t-window, t-1]. Without the .shift(1),
#     day t's own views/adds/sales would leak into a feature used to predict
#     sales at day t+7 -- daily sales are autocorrelated, so an unshifted
#     sum is effectively a proxy for the target. .shift(1) must be applied
#     per item, i.e. after .sum() but before .reset_index (the groupby index
#     is still active at that point, so the shift respects item boundaries).
#     """
#     return (
#         df.sort_values(["itemid", "date"])
#           .groupby("itemid")[col]
#           .rolling(window, min_periods=1)
#           .sum()
#           .shift(1)
#           .reset_index(level=0, drop=True)
#     )


# ──────────────────────────────────────────────────────────────────────────────
def build_forecast_features(events: pd.DataFrame, props: pd.DataFrame,
                             cfg: Dict[str, Any]) -> pd.DataFrame:
    

    df = events.copy()
    df["date"] = df["timestamp"].dt.normalize()

    # Daily counts
    daily = (
        df.pivot_table(index=["itemid", "date"],
                        columns="event",
                        values="visitorid",
                        aggfunc="count",
                        fill_value=0)
          .rename(columns={"view": "views",
                            "addtocart": "adds",
                            "transaction": "sales"})
          .reset_index()
    )
    for c in ["views", "adds", "sales"]:
        daily[c] = daily.get(c, 0)

    # Price
    price_df = props.loc[props["property"] == "price", ["itemid", "value"]]\
                    .rename(columns={"value": "price"})
    price_df["price"] = pd.to_numeric(price_df["price"], errors="coerce")
    daily = daily.merge(price_df.drop_duplicates("itemid"), on="itemid", how="left")
    daily["price"] = daily["price"].ffill()

    # Category
    cat_df = props.loc[props["property"] == "categoryid", ["itemid", "value"]]\
                  .rename(columns={"value": "categoryid"})\
                  .drop_duplicates("itemid")
    daily = daily.merge(cat_df, on="itemid", how="left")

    # Rolling windows (trailing, excludes current day -- see _rolling_sum)
    for w in tqdm(cfg["features"].get("rolling_windows", [3, 7, 14, 30, 60]),
                  desc="Rolling windows", unit="window"):
        for c in ["views", "adds", "sales"]:
            # .shift(1) leaves the first row per item as NaN (true cold-start:
            # no prior history exists yet) -- fill with 0 rather than leaving
            # NaN, since 0 is the correct "no observed history" value here.
            daily[f"{c}_sum_{w}d"] = _rolling_sum(daily, w, c).fillna(0)
    _first = daily.sort_values(["itemid", "date"]).groupby("itemid").head(1)
    assert (_first["sales_sum_7d"] == 0).all(), \
        "Leak: first-day sales_sum_7d must be zero for every item"

    # Lags
    daily = daily.sort_values(["itemid", "date"])
    for lag in cfg["features"].get("lag_days", [1, 7, 14]):
        daily[f"sales_lag_{lag}d"] = daily.groupby("itemid")["sales"].shift(lag).fillna(0)
    daily["sales_lag_dow"] = (
        daily.groupby("itemid")["sales"].shift(7)
             .where(daily["date"].dt.dayofweek ==
                    (daily["date"] - pd.Timedelta(days=7)).dt.dayofweek, 0)
             .fillna(0)
    )

    # Raw ratios -- NOT smoothed here anymore (see module docstring)
    daily["ctr_7d"] = daily["adds_sum_7d"] / daily["views_sum_7d"].replace(0, np.nan)
    daily["ctr_7d"] = daily["ctr_7d"].fillna(0)
    daily["buyrate_7d"] = daily["sales_sum_7d"] / daily["views_sum_7d"].replace(0, np.nan)
    daily["buyrate_7d"] = daily["buyrate_7d"].fillna(0)

    # Category-level raw sales aggregate (still fine to keep raw; smoothing
    # of this happens post-split too, since it's derived straight from sales)
    cat_sales = (
        daily.groupby(["categoryid", "date"])["sales_sum_7d"]
             .sum()
             .rename("cat_sales_7d")
             .reset_index()
    )
    daily = daily.merge(cat_sales, on=["categoryid", "date"], how="left")

    # Price change
    daily["price_lag_7d"] = daily.groupby("itemid")["price"].shift(7)
    daily["price_pct_chg_7d"] = (daily["price"] - daily["price_lag_7d"]) \
                                 / daily["price_lag_7d"].replace(0, np.nan)

    # Calendar
    daily["dow"] = daily["date"].dt.dayofweek
    daily["is_weekend"] = daily["dow"].isin([5, 6]).astype(int)
    daily["weekofyear"] = daily["date"].dt.isocalendar().week.astype(int)
    daily["month"] = daily["date"].dt.month
    daily["day_of_year"] = daily["date"].dt.day_of_year
    daily = pd.concat([daily,
                        pd.get_dummies(daily["dow"], prefix="dow", dtype="int8")],
                       axis=1)

    # Target
    tgt = daily[["itemid", "date", "sales"]].copy()
    tgt["date"] = tgt["date"] - pd.Timedelta(days=7)
    tgt = tgt.rename(columns={"sales": "target_next7"})
    daily = daily.merge(tgt, on=["itemid", "date"], how="left")
    daily = daily.dropna(subset=["target_next7"]).reset_index(drop=True)

    # tgt = daily[["itemid", "date", "sales"]].copy()
    # tgt["date"] = tgt["date"] - pd.Timedelta(days=7)
    # tgt = tgt.rename(columns={"sales": "target_next7"})
    # daily = daily.merge(tgt, on=["itemid", "date"], how="left") \
                #  .fillna({"target_next7": 0})

    return daily.drop(columns=["views", "adds", "sales", "price_lag_7d", "dow"])


# ──────────────────────────────────────────────────────────────────────────────
def build_reco_sequences(events: pd.DataFrame, props: pd.DataFrame,
                          cats: pd.DataFrame, cfg: Dict[str, Any]):

    df = events.sort_values(["visitorid", "timestamp"]).copy()
    item2idx = {int(i): idx + 1 for idx, i in enumerate(df["itemid"].unique())}
    df["item_idx"] = df["itemid"].map(item2idx)

    cat_map = props.loc[props["property"] == "categoryid", ["itemid", "value"]]\
                    .rename(columns={"value": "categoryid"})
    cat2idx = {int(c): ix + 1 for ix, c in enumerate(cats["categoryid"].unique())}
    df = df.merge(cat_map, on="itemid", how="left")
    df["cat_idx"] = df["categoryid"].map(cat2idx).fillna(0).astype(int)

    rows = []
    for visitor, g in tqdm(df.groupby("visitorid"),
                            desc="Reco Sequences", unit="visitor"):
        g = g.tail(cfg["features"]["max_seq_length"])
        rows.append({
            "visitorid": visitor,
            "item_seq": list(g["item_idx"]),
            "cat_seq": list(g["cat_idx"]),
            "time_seq": list((g["timestamp"] - g["timestamp"].iloc[0])
                              .dt.total_seconds() // 60),
            "session_end": g["timestamp"].max()
        })

    return pd.DataFrame(rows), item2idx


# ──────────────────────────────────────────────────────────────────────────────
def main() -> None:
    p = argparse.ArgumentParser(description="Feature engineering (leakage-fixed)")
    p.add_argument("--cfg", "--config", default="config.yaml")
    p.add_argument("--in_dir", default=None)
    p.add_argument("--out_dir", default=None)
    p.add_argument("--art_dir", default=None)
    a = p.parse_args()

    cfg = _load_cfg(Path(a.cfg))
    in_dir = Path(a.in_dir or cfg["features"]["in_dir"])
    out_dir = Path(a.out_dir or cfg["features"]["out_dir"])
    art_dir = Path(a.art_dir or cfg["features"].get("artefacts_dir", "artefacts"))
    out_dir.mkdir(parents=True, exist_ok=True)
    art_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(level=logging.INFO,
                         format="%(asctime)s %(levelname)s: %(message)s")
    logging.info("Loading data...")
    events = pd.read_parquet(in_dir / "events_clean.parquet")
    props = pd.read_parquet(in_dir / "item_properties.parquet")
    cats = pd.read_parquet(in_dir / "category_tree.parquet")

    logging.info("Building forecast features (raw ratios, no cross-split smoothing)...")
    ff = build_forecast_features(events, props, cfg)
    ff.to_parquet(out_dir / "forecast_features.parquet", index=False)

    logging.info("Building recommendation sequences...")
    reco_df, item2idx = build_reco_sequences(events, props, cats, cfg)
    reco_df.to_parquet(out_dir / "reco_sequences.parquet", index=False)
    with open(art_dir / "item2idx.json", "w", encoding="utf-8") as f:
        json.dump(item2idx, f)

    logging.info("Feature engineering done (leakage-fixed).")


if __name__ == "__main__":
    main()