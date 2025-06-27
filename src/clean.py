"""
src/clean.py
~~~~~~~~~~~~
Cleaning pipeline for the Retailrocket dataset.

Reads raw CSVs from data/raw/:
  • events.csv
  • item_properties_part1.csv
  • item_properties_part2.csv
  • category_tree.csv

Performs dataset-specific cleaning on the original column names:
  • Parse `timestamp` (ms) → UTC datetime
  • Whitelist `event` values
  • Extract numeric from `value`
  • Drop rows with missing critical fields
  • Deduplicate
  • Cast IDs to integers (where applicable)

Writes cleaned Parquets to data/interim/:
  • events_clean.parquet
  • item_properties.parquet
  • category_tree.parquet

And Markdown summaries to reports/:
  • clean_events_summary.md
  • clean_item_properties_summary.md
  • clean_category_summary.md
"""

from __future__ import annotations
import argparse
import logging
import re
import textwrap
from pathlib import Path
from typing import Any, Dict

import pandas as pd
import yaml

def _load_cfg(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="python -m src.clean",
        description="Clean Retailrocket CSVs without renaming"
    )
    p.add_argument("--cfg",    type=Path, default=Path("config.yaml"))
    p.add_argument("--raw_dir", type=Path, default=None)
    p.add_argument("--out_dir", type=Path, default=None)
    return p.parse_args()

def _write_md(stats: Dict[str, Any], title: str, path: Path) -> None:
    header = f"# {title}\n\n| Metric | Value |\n| ------ | ----- |\n"
    lines = [f"| {k.replace('_',' ').capitalize()} | {v} |" for k, v in stats.items()]
    path.write_text(header + "\n".join(lines) + "\n", encoding="utf-8")


#Event.csv

def clean_events(df: pd.DataFrame) -> tuple[pd.DataFrame, Dict[str, Any]]:
    stats: Dict[str, Any] = {"rows_before": len(df)}

    # parse timestamp (ms)
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    stats["rows_dropped_bad_ts"] = int(df["timestamp"].isna().sum())
    df = df.dropna(subset=["timestamp"])

    # whitelist event types
    allowed = {"view", "addtocart", "removefromcart", "transaction"}
    before = len(df)
    df = df[df["event"].isin(allowed)]
    stats["rows_dropped_bad_event"] = before - len(df)

    # drop missing visitorid or itemid
    before = len(df)
    df = df.dropna(subset=["visitorid"])
    stats["rows_dropped_missing_visitor"] = before - len(df)

    # dedupe
    stats["duplicates_dropped"] = int(df.duplicated().sum())
    df = df.drop_duplicates()

    # finalize
    df = df.sort_values("timestamp")
    stats.update({
        "rows_after":      len(df),
        "unique_visitors": int(df["visitorid"].nunique()),
        "unique_items":    int(df["itemid"].nunique()),
        "date_min":        df["timestamp"].min().date().isoformat(),
        "date_max":        df["timestamp"].max().date().isoformat(),
    })
    return df, stats

#clean the properties

# ───────────────────────────────────────────────────────────────────────────

_NUM_RE = re.compile(r"-?\d+\.?\d*")

def clean_properties(df: pd.DataFrame) -> tuple[pd.DataFrame, Dict[str, Any]]:
    stats: Dict[str, Any] = {"rows_before": len(df)}

    # parse timestamp
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    stats["rows_dropped_bad_ts"] = int(df["timestamp"].isna().sum())
    df = df.dropna(subset=["timestamp"])

    # extract first numeric token from 'value'
    def _first_num(x: Any) -> float | None:
        s = str(x)
        nums = _NUM_RE.findall(s)
        return float(nums[0]) if nums else None

    df["value"] = df["value"].apply(_first_num)
    stats["rows_dropped_bad_value"] = int(df["value"].isna().sum())
    df = df.dropna(subset=["value"])

    # drop missing itemid or property
    before = len(df)
    df = df.dropna(subset=["itemid", "property"])
    stats["rows_dropped_missing_ids"] = before - len(df)

    # dedupe on itemid, property, timestamp
    before = len(df)
    df = df.drop_duplicates(subset=["itemid", "property", "timestamp"])
    stats["duplicates_dropped"] = before - len(df)

    stats.update({
        "rows_after":        len(df),
        "unique_items":      int(df["itemid"].nunique()),
        "unique_properties": int(df["property"].nunique()),
    })
    return df, stats



#Function to clean the csv catagory file

# ──────────────────────────────────────────────────────────────────────────────
def clean_category(df: pd.DataFrame) -> tuple[pd.DataFrame, Dict[str, Any]]:
    stats: Dict[str, Any] = {"rows_before": len(df)}

    # drop missing categoryid
    before = len(df)
    df = df.dropna(subset=["categoryid"])
    stats["rows_dropped_missing"] = before - len(df)

    # dedupe on categoryid
    before = len(df)
    df = df.drop_duplicates(subset=["categoryid"])
    stats["duplicates_dropped"] = before - len(df)

    # cast to Int64 and fill missing parentid with -1
    df["categoryid"] = pd.to_numeric(df["categoryid"], errors="coerce").astype("Int64")
    if "parentid" in df.columns:
        df["parentid"] = (
            pd.to_numeric(df["parentid"], errors="coerce")
              .fillna(-1)               # treat NaN as root category
              .astype("Int64")
        )

    stats.update({
        "rows_after":        len(df),
        "unique_categories": int(df["categoryid"].nunique()),
        "unique_parents":    int(df["parentid"].nunique()) if "parentid" in df.columns else 0,
    })
    return df, stats


#Calling the main function

def main() -> None:
    args = _parse_args()
    cfg  = _load_cfg(args.cfg)
    raw     = args.raw_dir or Path(cfg["clean"]["raw_dir"])
    interim = args.out_dir  or Path(cfg["clean"]["out_dir"])
    reports = Path("reports")
    interim.mkdir(parents=True, exist_ok=True)
    reports.mkdir(exist_ok=True)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

    # events
    logging.info("Cleaning events.csv …")
    df_ev = pd.read_csv(raw / "events.csv", low_memory=False)
    ev_clean, ev_stats = clean_events(df_ev)
    ev_clean.to_parquet(interim / "events_clean.parquet", index=False)
    _write_md(ev_stats, "Events Cleaning Summary", reports / "clean_events_summary.md")

    # item_properties
    logging.info("Cleaning item_properties …")
    parts = [raw/"item_properties_part1.csv", raw/"item_properties_part2.csv"]
    df_pr = pd.concat([pd.read_csv(p, low_memory=False) for p in parts], ignore_index=True)
    pr_clean, pr_stats = clean_properties(df_pr)
    pr_clean.to_parquet(interim / "item_properties.parquet", index=False)
    _write_md(pr_stats, "Item Properties Cleaning Summary", reports / "clean_item_properties_summary.md")

    # category_tree
    logging.info("Cleaning category_tree.csv …")
    df_ct = pd.read_csv(raw / "category_tree.csv", low_memory=False)
    ct_clean, ct_stats = clean_category(df_ct)
    ct_clean.to_parquet(interim / "category_tree.parquet", index=False)
    _write_md(ct_stats, "Category Tree Cleaning Summary", reports / "clean_category_summary.md")

    logging.info("All files cleaned.")
    
if __name__ == "__main__":
    main()
