"""
item_knn_baseline.py  –  item-kNN co-occurrence baseline
========================================================
Reproduces the exact same chronological split used by GRU4Rec_baseline.py
(sort by session_end, 80th-percentile cutoff, keep length >= 2 sequences),
then evaluates item-kNN under the identical leave-one-out, full-catalogue
ranking protocol (no negative sampling).

Outputs
-------
reports/metrics_reco_knn.md
reports/item_knn_top_examples.csv   (optional sanity check)
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import yaml
from scipy.sparse import csr_matrix, lil_matrix


# ────────────────────────── utils ──────────────────────────
def _load_cfg(p: Path = Path("config.yaml")) -> Dict:
    with open(p, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def _ndcg(rank: int, k: int = 20) -> float:
    return 1.0 / math.log2(rank + 1) if rank <= k else 0.0


# ────────────────────────── split (same as GRU4Rec) ────────────────────────
def chronological_split(df: pd.DataFrame, q: float = 0.8,
                        min_len: int = 2) -> Tuple[List[List[int]],
                                                   List[List[int]],
                                                   pd.Timestamp]:
    """Return (train_seqs, val_seqs, cutoff) identical to GRU4Rec_baseline.py."""
    df = df.sort_values("session_end").reset_index(drop=True)
    cutoff = df["session_end"].quantile(q)
    tr = df.loc[df["session_end"] <  cutoff, "item_seq"].tolist()
    va = df.loc[df["session_end"] >= cutoff, "item_seq"].tolist()
    # length filter for next-item prediction
    tr = [s for s in tr if len(s) >= min_len]
    va = [s for s in va if len(s) >= min_len]
    return tr, va, cutoff


# ────────────────────────── item-kNN model ──────────────────────────
class ItemKNN:
    """
    Symmetric co-occurrence counter over training sequences.
    For a query item i, scores are cooc[i, :]; top-k items are recommended.
    Falls back to global popularity when i has no co-occurrences.
    """
    def __init__(self, n_items: int):
        self.n_items = n_items
        self.cooc = lil_matrix((n_items, n_items), dtype=np.float32)
        self.pop_top: np.ndarray | None = None

    def fit(self, train_seqs: List[List[int]], pop_k: int = 20) -> "ItemKNN":
        # 1) co-occurrence, deduplicated within each session
        for seq in train_seqs:
            items = list(dict.fromkeys(seq))
            for i in items:
                row = self.cooc.rows[i]
                row_set = set(row)
                for j in items:
                    if i == j:
                        continue
                    self.cooc[i, j] += 1.0
        self.cooc = self.cooc.tocsr()

        # 2) popularity fallback
        freq = np.zeros(self.n_items, dtype=np.int64)
        for seq in train_seqs:
            for i in seq:
                freq[i] += 1
        self.pop_top = np.argsort(-freq)[:pop_k]
        return self

    def recommend(self, last_item: int, k: int = 20) -> np.ndarray:
        row = self.cooc.getrow(last_item).toarray().ravel()
        if row.sum() == 0:
            return self.pop_top.copy()
        order = np.argsort(-row)
        top = order[:k]
        if top.size < k:                       # pad with popularity
            pad = np.array([i for i in self.pop_top if i not in set(top.tolist())])
            top = np.concatenate([top, pad[: k - top.size]])
        return top


# ────────────────────────── evaluation ──────────────────────────
def evaluate(model: ItemKNN, val_seqs: List[List[int]],
             k: int = 20) -> Tuple[float, float]:
    hits, ndcg, n = 0, 0.0, 0
    for seq in val_seqs:
        last_item, target = seq[-2], seq[-1]   # leave-one-out
        recs = model.recommend(last_item, k)
        pos = np.where(recs == target)[0]
        if pos.size:
            rank = int(pos[0]) + 1
            hits += 1
            ndcg += _ndcg(rank, k)
        n += 1
    return hits / n, ndcg / n


def popularity_baseline(val_seqs: List[List[int]],
                        train_seqs: List[List[int]],
                        k: int = 20) -> Tuple[float, float]:
    freq = np.zeros(max(max(s) for s in train_seqs) + 1, dtype=np.int64)
    for seq in train_seqs:
        for i in seq:
            freq[i] += 1
    top = np.argsort(-freq)[:k]
    hits, ndcg, n = 0, 0.0, 0
    for seq in val_seqs:
        target = seq[-1]
        pos = np.where(top == target)[0]
        if pos.size:
            rank = int(pos[0]) + 1
            hits += 1
            ndcg += _ndcg(rank, k)
        n += 1
    return hits / n, ndcg / n


# ────────────────────────── main ──────────────────────────
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", default="config.yaml")
    ap.add_argument("--k",   type=int, default=20)
    a = ap.parse_args()

    cfg = _load_cfg(Path(a.cfg))
    seq_path = (Path(cfg["features"]["out_dir"])
                / cfg["features"].get("processed_reco_path",
                                      "reco_sequences.parquet"))
    df = pd.read_parquet(seq_path)

    tr_seqs, va_seqs, cutoff = chronological_split(df, q=0.8, min_len=2)
    print(f"Cutoff {cutoff} | train seqs {len(tr_seqs):,} | "
          f"val seqs {len(va_seqs):,}")

    n_items = int(max(max(s) for s in tr_seqs + va_seqs)) + 1

    # popularity
    hr_pop, ndcg_pop = popularity_baseline(va_seqs, tr_seqs, k=a.k)

    # item-kNN
    knn = ItemKNN(n_items).fit(tr_seqs, pop_k=a.k)
    hr_knn, ndcg_knn = evaluate(knn, va_seqs, k=a.k)

    print(f"Popularity : HR@{a.k}={hr_pop:.4f}  NDCG@{a.k}={ndcg_pop:.4f}")
    print(f"Item-kNN   : HR@{a.k}={hr_knn:.4f}  NDCG@{a.k}={ndcg_knn:.4f}")

    Path("reports").mkdir(exist_ok=True)
    Path("reports/metrics_reco_knn.md").write_text(
        f"""# Recommender Baselines Under Corrected Chronological Split

| Model                | Hit Rate@{a.k} (%) | NDCG@{a.k} (%) |
| -------------------- | ------------------ | -------------- |
| Popularity (Top-{a.k})   | {hr_pop*100:5.2f} | {ndcg_pop*100:5.2f} |
| Item-kNN (k={a.k}, co-occurrence) | {hr_knn*100:5.2f} | {ndcg_knn*100:5.2f} |

Split cutoff: {cutoff}
Validation sequences (length >= 2): {len(va_seqs):,}
Protocol: leave-one-out, full-catalogue ranking, no negative sampling.
""", encoding="utf-8")

    # optional: save top-20 neighbours of a few sample items
    sample_items = [s[-1] for s in va_seqs[:200]]
    rows = []
    for it in sample_items:
        rows.append({"item": it,
                     "top20": ",".join(map(str, knn.recommend(it, a.k).tolist()))})
    pd.DataFrame(rows).to_csv("reports/item_knn_top_examples.csv", index=False)


if __name__ == "__main__":
    main()