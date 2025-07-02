"""
GRU4Rec_baseline.py  –  GPU-aware baseline sequential recommender
=================================================================

• Uses CUDA automatically if available
• Weight-tying only when emb_dim == hidden_size
• Batch-level progress via tqdm
• Outputs:                                          (same as before)
    artefacts/gru4rec_baseline.pt
    reports/metrics_reco_baseline.md
    reports/gru4rec_training_curve.png
"""

from __future__ import annotations
import argparse, math, os, random, time, platform
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm


# ────────────────────────── utils ──────────────────────────
def _load_cfg(p: Path = Path("config.yaml")) -> Dict:
    return yaml.safe_load(open(p, "r", encoding="utf-8"))

def _set_seed(seed: int = 42) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)

def _ndcg(rank: int, k: int = 20) -> float:
    return 1 / math.log2(rank + 1) if rank <= k else 0.0


# ───────────────────────── dataset ─────────────────────────
class _SeqDataset(Dataset):
    def __init__(self, seqs: List[List[int]], max_len: int = 50):
        self.samples = [(s[:-1][-max_len:], s[-1]) for s in seqs if len(s) >= 2]
    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        h, t = self.samples[idx]
        return torch.tensor(h, dtype=torch.long), torch.tensor(t, dtype=torch.long)

def _collate(batch, pad: int = 0):
    hists, tgts = zip(*batch)
    max_len = max(map(len, hists))
    padded = torch.full((len(batch), max_len), pad, dtype=torch.long)
    for i, h in enumerate(hists):
        padded[i, -len(h):] = torch.as_tensor(h, dtype=torch.long)
    return padded, torch.stack(tgts)


# ────────────────────────── model ──────────────────────────
class GRU4Rec(nn.Module):
    def __init__(self, n_items: int, emb: int, hid: int,
                 layers: int, dropout: float):
        super().__init__()
        self.item_emb = nn.Embedding(n_items + 1, emb, padding_idx=0)
        self.gru = nn.GRU(emb, hid, num_layers=layers,
                          batch_first=True,
                          dropout=dropout if layers > 1 else 0.0)

        self.fc = nn.Linear(hid, n_items + 1, bias=False)
        if emb == hid:      # safe weight-tying
            self.fc.weight = self.item_emb.weight

    def forward(self, seq: torch.Tensor) -> torch.Tensor:
        out, _ = self.gru(self.item_emb(seq))
        return self.fc(out[:, -1, :])          # [B, n_items+1]


# ─────────────────────── train & eval ──────────────────────
def _train_epoch(model, loader, opt, device) -> float:
    ce = nn.CrossEntropyLoss(ignore_index=0)
    model.train(); tot, loss_sum = 0, 0.0
    for seq, tgt in tqdm(loader, desc="Train", leave=False, unit="batch"):
        seq, tgt = seq.to(device, non_blocking=True), tgt.to(device, non_blocking=True)
        opt.zero_grad()
        loss = ce(model(seq), tgt)
        loss.backward(); opt.step()
        bs = seq.size(0); tot += bs; loss_sum += loss.item() * bs
    return loss_sum / tot

@torch.no_grad()
def _evaluate(model, loader, device, k=20) -> Tuple[float, float]:
    model.eval(); hits = ndcgs = tot = 0
    for seq, tgt in loader:
        seq, tgt = seq.to(device, non_blocking=True), tgt.to(device, non_blocking=True)
        topk = torch.topk(model(seq), k, dim=1).indices
        for cand, t in zip(topk, tgt):
            tot += 1
            pos = (cand == t).nonzero(as_tuple=True)
            if pos[0].numel():
                rank = pos[0].item() + 1
                hits += 1; ndcgs += _ndcg(rank, k)
    return hits / tot, ndcgs / tot


# ────────────────────────── main ───────────────────────────
def main() -> None:
    arg = argparse.ArgumentParser()
    arg.add_argument("--cfg", default="config.yaml"); a = arg.parse_args()

    cfg = _load_cfg(Path(a.cfg))
    reco = cfg.setdefault("models", {}).setdefault("reco", {})
    _set_seed(reco.get("seed", 42))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tqdm.write(f"🚀 Using device: {device} "
               f"({torch.cuda.get_device_name() if device.type=='cuda' else platform.processor()})")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True  # speed boost for fixed-size batches

    # data paths
    seq_path = Path(cfg["features"]["out_dir"]) / cfg["features"].get(
        "processed_reco_path", "reco_sequences.parquet")
    df = pd.read_parquet(seq_path)
    seqs = df["item_seq"].tolist()
    n_items = int(max(max(s) for s in seqs))
    max_len = reco.get("max_seq_len", 50)

    # split
    rng = np.random.RandomState(42)
    mask = rng.rand(len(seqs)) < 0.8
    tr_ds = _SeqDataset([s for m, s in zip(mask, seqs) if m], max_len)
    va_ds = _SeqDataset([s for m, s in zip(mask, seqs) if not m], max_len)

    tr_loader = DataLoader(tr_ds, batch_size=reco.get("batch", 256),
                           shuffle=True, num_workers=os.cpu_count(),
                           pin_memory=(device.type == "cuda"), collate_fn=_collate)
    va_loader = DataLoader(va_ds, batch_size=512, shuffle=False,
                           num_workers=os.cpu_count(),
                           pin_memory=(device.type == "cuda"), collate_fn=_collate)

    # model
    emb = reco.get("embedding_dim", 64); hid = reco.get("hidden", 64)
    model = GRU4Rec(n_items, emb, hid, reco.get("layers", 1),
                    reco.get("dropout", 0.2)).to(device)
    opt = torch.optim.Adam(model.parameters(),
                           lr=reco.get("lr", 1e-3),
                           weight_decay=reco.get("wd", 1e-5))

    # training loop
    epochs = reco.get("epochs", 20)
    best_ndcg, best_state = 0.0, None
    hist_hr, hist_ndcg = [], []
    for ep in range(1, epochs + 1):
        t0 = time.time()
        loss = _train_epoch(model, tr_loader, opt, device)
        hr, ndcg = _evaluate(model, va_loader, device)
        hist_hr.append(hr); hist_ndcg.append(ndcg)
        if ndcg > best_ndcg: best_ndcg, best_state = ndcg, model.state_dict()
        tqdm.write(f"[{ep:02d}/{epochs}] loss={loss:.4f} "
                   f"HR@20={hr:.3f} NDCG@20={ndcg:.3f} "
                   f"(best={best_ndcg:.3f}) {time.time()-t0:5.1f}s")

    # save artefacts
    Path("artefacts").mkdir(exist_ok=True)
    model.load_state_dict(best_state)
    torch.save({"state_dict": model.state_dict(),
                "n_items": n_items,
                "cfg": cfg},
               "artefacts/gru4rec_baseline.pt")

    # PNG
    Path("reports").mkdir(exist_ok=True)
    plt.figure(figsize=(6,4))
    plt.plot(range(1, epochs+1), hist_hr,   label="HitRate@20")
    plt.plot(range(1, epochs+1), hist_ndcg, label="NDCG@20")
    plt.xlabel("Epoch"); plt.ylabel("Metric"); plt.title("GRU4Rec Training")
    plt.legend(); plt.tight_layout()
    plt.savefig("reports/gru4rec_training_curve.png", dpi=120); plt.close()

    hr, ndcg = _evaluate(model, va_loader, device)
    ok = ndcg >= 0.14 and hr >= 0.35
    Path("reports/metrics_reco_baseline.md").write_text(
f"""# Baseline GRU4Rec Report

| Metric        | Value |
|---------------|-------|
| HitRate@20    | {hr:.3f} |
| NDCG@20       | {ndcg:.3f} |
| Pass Gates?   | {'✅' if ok else '❌'} |
| Train samples | {len(tr_ds):,} |
| Val samples   | {len(va_ds):,} |
| Epochs        | {epochs} |
| Device        | {device} |
""", encoding="utf-8")

if __name__ == "__main__":
    main()
