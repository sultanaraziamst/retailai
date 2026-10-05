# """
# tune_GRU4Rec.py  –  GPU-aware Optuna sweep for GRU4Rec
# ======================================================

# CLI
# ---
# python tune_GRU4Rec.py --mode quick           #  ~30-40 min  GPU
# python tune_GRU4Rec.py --mode full  --trials 80   # ≤ 1 h  GPU

# Outputs
# -------
# artefacts/optuna_gru4rec.db        – Optuna study (SQLite + RDBStorage)
# artefacts/gru4rec_tuned.pt         – final model (best params, 20-epoch refit)
# reports/tunedgru4rec_study_curve.png
# reports/metrics_reco_tuned.md
# """
# from __future__ import annotations
# import argparse, json, os, platform
# from pathlib import Path
# from typing import Dict, List, Tuple

# import matplotlib.pyplot as plt
# import numpy as np
# import optuna
# import pandas as pd
# import torch
# import yaml
# from torch.utils.data import DataLoader
# from tqdm.auto import tqdm

# # ─── local helpers (from baseline file) ─────────────────────────────────────
# from .GRU4REC_baseline import (GRU4Rec, _SeqDataset, _collate,
#                                _evaluate, _load_cfg, _set_seed)

# # ════════════════════════════════════════════════════════════════════════════
# #                           DATA HELPERS
# # ════════════════════════════════════════════════════════════════════════════
# # def _load_seqs(cfg: Dict) -> Tuple[List[List[int]], int]:
# #     seq_file = (Path(cfg["features"]["out_dir"]) /
# #                 cfg["features"].get("processed_reco_path", "reco_sequences.parquet"))
# #     df = pd.read_parquet(seq_file)
# #     seqs = df["item_seq"].tolist()
# #     return seqs, int(max(max(s) for s in seqs))

# def _load_seqs(cfg: Dict):
#     seq_file = (Path(cfg["features"]["out_dir"]) /
#                 cfg["features"].get("processed_reco_path", "reco_sequences.parquet"))
#     df = pd.read_parquet(seq_file)
#     n_items = int(max(max(s) for s in df["item_seq"]))
#     return df, n_items


# def _split(df, q: float = 0.8):
#     """Chronological split by session_end timestamp."""
#     df = df.sort_values("session_end").reset_index(drop=True)
#     cutoff = df["session_end"].quantile(q)
#     tr = df.loc[df["session_end"] <  cutoff, "item_seq"].tolist()
#     va = df.loc[df["session_end"] >= cutoff, "item_seq"].tolist()
#     return tr, va

# # def _split(seqs, keep: float = .8, seed=42):
# #     rng = np.random.RandomState(seed)
# #     mask = rng.rand(len(seqs)) < keep
# #     return ([s for f, s in zip(mask, seqs) if f],
# #             [s for f, s in zip(mask, seqs) if not f])

# def _train_one_epoch(model, loader, opt, device, trial_num=None, epoch=None):
#     ce = torch.nn.CrossEntropyLoss(ignore_index=0)
#     model.train()
    
#     # Simplified progress bar - update less frequently for speed
#     show_progress = trial_num is None or trial_num == "Final"  # Only show detailed progress for final fit
#     if show_progress:
#         desc = f"Trial {trial_num}, Epoch {epoch}" if trial_num is not None and epoch is not None else "Training"
#         pbar = tqdm(loader, desc=desc, leave=False, unit="batch", mininterval=1.0)
    
#     total_loss = 0.0
#     for i, (seq, tgt) in enumerate(loader):
#         seq, tgt = seq.to(device, non_blocking=True), tgt.to(device, non_blocking=True)
#         opt.zero_grad(set_to_none=True)  # Faster than zero_grad()
#         loss = ce(model(seq), tgt)
#         loss.backward()
#         opt.step()
        
#         total_loss += loss.item()
#         # Update progress less frequently for speed
#         if show_progress and i % 50 == 0:
#             pbar.set_postfix({"loss": f"{loss.item():.4f}", "avg_loss": f"{total_loss/(i+1):.4f}"})
    
#     if show_progress:
#         pbar.close()
#     return total_loss / len(loader)

# # ════════════════════════════════════════════════════════════════════════════
# #                          OPTUNA OBJECTIVE
# # ════════════════════════════════════════════════════════════════════════════
# def _objective(trial: optuna.Trial, cfg: Dict,
#                seqs: List[List[int]], n_items: int,
#                device: torch.device, space: Dict) -> float:

#     # Reduced logging for speed - only log key info
#     if trial.number % 5 == 0:  # Log every 5th trial
#         tqdm.write(f"🏃 Trial {trial.number} starting...")
    
#     emb     = trial.suggest_categorical("emb",    space["emb"])
#     hid     = trial.suggest_categorical("hid",    space["hid"])
#     layers  = trial.suggest_int        ("layers", *space["layers"])
#     drop    = trial.suggest_float      ("drop",   *space["drop"])
#     lr      = trial.suggest_float      ("lr",     *space["lr"],  log=True)
#     wd      = trial.suggest_float      ("wd",     *space["wd"],  log=True)
#     batch   = trial.suggest_categorical("batch",  space["batch"])
#     epochs  = space["epochs"]

#     # tr_seqs, va_seqs = _split(seqs, keep=space["train_frac"])
#     tr_seqs, va_seqs = _split(df, q=space["train_frac"])
#     max_len = cfg["models"]["reco"].get("max_seq_len", 50)

#     # Optimized DataLoaders - smaller batch for validation, persistent workers
#     tr_loader = DataLoader(_SeqDataset(tr_seqs, max_len), batch_size=batch,
#                            shuffle=True, num_workers=1, persistent_workers=True,
#                            pin_memory=(device.type == "cuda"), collate_fn=_collate)
#     va_loader = DataLoader(_SeqDataset(va_seqs, max_len), batch_size=512,  # Smaller batch for faster eval
#                            shuffle=False, num_workers=1, persistent_workers=True,
#                            pin_memory=(device.type == "cuda"), collate_fn=_collate)

#     model = GRU4Rec(n_items, emb, hid, layers, drop).to(device)
#     # Use AdamW with compile for speed
#     opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    
#     # Compile model for faster execution (PyTorch 2.0+)
#     if hasattr(torch, 'compile'):
#         model = torch.compile(model, mode='reduce-overhead')

#     best = 0.0
#     # Simplified epoch tracking
#     for ep in range(1, epochs + 1):
#         avg_loss = _train_one_epoch(model, tr_loader, opt, device, trial.number, ep)
#         _, ndcg = _evaluate(model, va_loader, device)
        
#         trial.report(ndcg, ep)
#         if trial.should_prune():
#             if trial.number % 5 == 0:  # Only log pruning for every 5th trial
#                 tqdm.write(f"  ✂️  Trial {trial.number} pruned at epoch {ep}")
#             raise optuna.TrialPruned()
#         best = max(best, ndcg)
    
#     if trial.number % 10 == 0:  # Log completion for every 10th trial
#         tqdm.write(f"  ✅ Trial {trial.number} completed! NDCG: {best:.4f}")
#     return best

# # ════════════════════════════════════════════════════════════════════════════
# #                                    MAIN
# # ════════════════════════════════════════════════════════════════════════════
# def main():
#     pa = argparse.ArgumentParser()
#     pa.add_argument("--cfg", default="config.yaml")
#     pa.add_argument("--mode", choices=["quick", "full"], default="quick")
#     pa.add_argument("--trials", type=int, default=80,
#                     help="trial budget for full mode")
#     args = pa.parse_args()

#     cfg = _load_cfg(Path(args.cfg)); _set_seed(cfg.get("seed", 42))
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     tqdm.write(f"🚀  Device {device} "
#                f"({torch.cuda.get_device_name(0) if device.type=='cuda' else platform.processor()})")

#     tqdm.write("📂 Loading sequences...")
#     seqs, n_items = _load_seqs(cfg)
#     tqdm.write(f"📊 Loaded {len(seqs):,} sequences, {n_items:,} unique items")
    
#     Path("artefacts").mkdir(exist_ok=True)
#     storage = optuna.storages.RDBStorage(url="sqlite:///artefacts/optuna_gru4rec.db",
#                                          engine_kwargs={"connect_args": {"check_same_thread": False}})

#     # ── pruner setup ────────────────────────────────────────────────────────
#     pruner = optuna.pruners.SuccessiveHalvingPruner(
#         min_resource=1,  reduction_factor=3, min_early_stopping_rate=0)

#     study = optuna.create_study(study_name="gru4rec_tuning",
#                                 storage=storage,
#                                 load_if_exists=True,
#                                 direction="maximize",
#                                 pruner=pruner,
#                                 sampler=optuna.samplers.TPESampler(multivariate=True, group=True))

#     # ── search spaces ───────────────────────────────────────────────────────
#     # Optimized search spaces - fewer options for faster convergence
#     QUICK_PILOT = dict(emb=[64], hid=[64, 96], layers=(1, 1),
#                        drop=(0.0, 0.3), lr=(1e-4, 2e-3), wd=(1e-6, 1e-4),
#                        batch=[256], epochs=3, train_frac=0.5)  # Smaller batch, fewer epochs
#     QUICK_FOCUS = dict(emb=[64], hid=[96], layers=(1, 2),
#                        drop=(0.0, 0.4), lr=(5e-4, 3e-3), wd=(1e-6, 5e-5),
#                        batch=[384], epochs=6, train_frac=0.8)  # Reduced epochs
#     FULL_FAST   = dict(emb=[64], hid=[64, 96], layers=(1, 2),  # Fewer hidden sizes
#                        drop=(0.0, 0.5), lr=(1e-4, 3e-3), wd=(1e-6, 1e-4),
#                        batch=[384], epochs=4, train_frac=0.8)  # Smaller batch, fewer epochs

#     # ── optimisation runs ──────────────────────────────────────────────────
#     if args.mode == "quick":
#         tqdm.write("🔍 Running QUICK mode: pilot phase...")
#         study.optimize(lambda t: _objective(t, cfg, seqs, n_items, device, QUICK_PILOT),
#                        n_trials=15, timeout=600, show_progress_bar=True)  # Reduced trials/time
#         tqdm.write("🎯 Running QUICK mode: focused phase...")
#         study.optimize(lambda t: _objective(t, cfg, seqs, n_items, device, QUICK_FOCUS),
#                        n_trials=5, timeout=600, show_progress_bar=True)   # Reduced trials/time

#     else:  # full mode (fast variant)
#         tqdm.write(f"🚄 Running FULL mode: {args.trials} trials")
#         study.optimize(lambda t: _objective(t, cfg, seqs, n_items, device, FULL_FAST),
#                        n_trials=args.trials,
#                        timeout=3600,          # 1-hour ceiling
#                        show_progress_bar=True)

#     # ── plot study progress ────────────────────────────────────────────────
#     Path("reports").mkdir(exist_ok=True)
#     df = study.trials_dataframe(attrs=("number", "value"))
#     plt.figure(figsize=(6, 4))
#     plt.plot(df["number"], df["value"])
#     plt.xlabel("Trial"); plt.ylabel("Best NDCG@20 so far")
#     plt.title("Optuna GRU4Rec Progress"); plt.tight_layout()
#     plt.savefig("reports/tunedgru4rec_study_curve.png", dpi=120); plt.close()

#     # ── final 20-epoch refit with best params ──────────────────────────────
#     tqdm.write(f"🏆 Best trial: {study.best_trial.number}, NDCG: {study.best_value:.4f}")
#     tqdm.write("🔄 Starting final 15-epoch refit with best parameters...")  # Reduced epochs
    
#     best = study.best_params
#     max_len = cfg["models"]["reco"].get("max_seq_len", 50)
#     # Use smaller batch size for final fit to avoid memory issues
#     final_batch = min(best["batch"], 256)  
#     loader = DataLoader(_SeqDataset(seqs, max_len), batch_size=final_batch,
#                         shuffle=True, num_workers=0,
#                         pin_memory=(device.type == "cuda"), collate_fn=_collate)

#     model = GRU4Rec(n_items, best["emb"], best["hid"],
#                     best["layers"], best["drop"]).to(device)
#     opt = torch.optim.AdamW(model.parameters(), lr=best["lr"], weight_decay=best["wd"])
    
#     # Compile model for faster final training
#     if hasattr(torch, 'compile'):
#         model = torch.compile(model, mode='reduce-overhead')
    
#     for epoch in tqdm(range(15), desc="Final fit", unit="epoch"):  # Reduced from 20 to 15
#         _train_one_epoch(model, loader, opt, device, "Final", epoch+1)

#     torch.save({"state_dict": model.state_dict(),
#                 "n_items": n_items,
#                 "best_params": best,
#                 "cfg": cfg},
#                "artefacts/gru4rec_tuned.pt")

#     # ── evaluate full set ──────────────────────────────────────────────────
#     full_loader = DataLoader(_SeqDataset(seqs, max_len), batch_size=512,  # Smaller batch
#                              shuffle=False, num_workers=0,
#                              pin_memory=(device.type == "cuda"), collate_fn=_collate)
#     hr, ndcg = _evaluate(model, full_loader, device)
#     Path("reports/metrics_reco_tuned.md").write_text(
# f"""# Tuned GRU4Rec Report

# | Metric      | Value |
# |-------------|-------|
# | HitRate@20  | {hr:.3f} |
# | NDCG@20     | {ndcg:.3f} |
# | Trials      | {len(study.trials)} |
# | Best trial  | {study.best_trial.number} |
# | Device      | {device} |
# | Pass Gates? | {'✅' if ndcg>=0.14 and hr>=0.35 else '❌'} |

# ```json
# {json.dumps(best, indent=2)}
# ```""", encoding="utf-8")

#     print("Model saved ➜ artefacts/gru4rec_tuned.pt")

# if __name__ == "__main__":
#     main()


"""
tune_GRU4Rec.py  –  GPU-aware Optuna sweep for GRU4Rec
======================================================

CLI
---
python tune_GRU4Rec.py --mode quick           #  ~30-40 min  GPU
python tune_GRU4Rec.py --mode full  --trials 80   # ≤ 1 h  GPU

Outputs
-------
artefacts/optuna_gru4rec.db        – Optuna study (SQLite + RDBStorage)
artefacts/gru4rec_tuned.pt         – final model (best params, 15-epoch refit)
reports/tunedgru4rec_study_curve.png
reports/metrics_reco_tuned.md

Leakage audit (2026-09 revision)
--------------------------------
Two changes to eliminate evaluation leaks identified in the original script:

  1. Session split is now CHRONOLOGICAL by `session_end` timestamp, so a
     visitor's later sessions cannot appear in training while their earlier
     sessions appear in validation. `_split(df, q)` replaces the previous
     random mask.

  2. The final evaluation for `metrics_reco_tuned.md` now uses the HELD-OUT
     validation split (`va_seqs_final`), not the full sequence set. The
     original script evaluated the tuned model on `seqs`, which included the
     training sequences and inflated Hit Rate@20 through partial memorization.
"""
from __future__ import annotations
import argparse, json, os, platform
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

# ─── local helpers (from baseline file) ─────────────────────────────────────
from .GRU4REC_baseline import (GRU4Rec, _SeqDataset, _collate,
                               _evaluate, _load_cfg, _set_seed)


# ════════════════════════════════════════════════════════════════════════════
#                           DATA HELPERS
# ════════════════════════════════════════════════════════════════════════════
def _load_seqs(cfg: Dict):
    """Load reco sequences plus their session_end timestamps.

    Requires `session_end` to have been added to `reco_sequences.parquet` by
    src/features.py. If the column is missing, a clear error is raised.
    """
    seq_file = (Path(cfg["features"]["out_dir"]) /
                cfg["features"].get("processed_reco_path", "reco_sequences.parquet"))
    df = pd.read_parquet(seq_file)
    if "session_end" not in df.columns:
        raise KeyError(
            "reco_sequences.parquet has no 'session_end' column. "
            "Re-run `python3 -m src.features` after applying the "
            "build_reco_sequences patch that adds session_end."
        )
    n_items = int(max(max(s) for s in df["item_seq"]))
    return df, n_items


def _split(df: pd.DataFrame, q: float = 0.8):
    """Chronological split by session_end timestamp.

    Train = sessions ending strictly before the q-quantile of session_end.
    Val   = sessions ending at or after that cutoff.
    No session appears in both.
    """
    df = df.sort_values("session_end").reset_index(drop=True)
    cutoff = df["session_end"].quantile(q)
    tr = df.loc[df["session_end"] <  cutoff, "item_seq"].tolist()
    va = df.loc[df["session_end"] >= cutoff, "item_seq"].tolist()
    return tr, va


def _train_one_epoch(model, loader, opt, device, trial_num=None, epoch=None):
    ce = torch.nn.CrossEntropyLoss(ignore_index=0)
    model.train()

    show_progress = trial_num is None or trial_num == "Final"
    if show_progress:
        desc = (f"Trial {trial_num}, Epoch {epoch}"
                if trial_num is not None and epoch is not None
                else "Training")
        pbar = tqdm(loader, desc=desc, leave=False, unit="batch", mininterval=1.0)

    total_loss = 0.0
    for i, (seq, tgt) in enumerate(loader):
        seq, tgt = seq.to(device, non_blocking=True), tgt.to(device, non_blocking=True)
        opt.zero_grad(set_to_none=True)
        loss = ce(model(seq), tgt)
        loss.backward()
        opt.step()

        total_loss += loss.item()
        if show_progress and i % 50 == 0:
            pbar.set_postfix({"loss": f"{loss.item():.4f}",
                              "avg_loss": f"{total_loss/(i+1):.4f}"})

    if show_progress:
        pbar.close()
    return total_loss / len(loader)


# ════════════════════════════════════════════════════════════════════════════
#                          OPTUNA OBJECTIVE
# ════════════════════════════════════════════════════════════════════════════
def _objective(trial: optuna.Trial, cfg: Dict,
               df: pd.DataFrame, n_items: int,
               device: torch.device, space: Dict) -> float:

    if trial.number % 5 == 0:
        tqdm.write(f"🏃 Trial {trial.number} starting...")

    emb    = trial.suggest_categorical("emb",   space["emb"])
    hid    = trial.suggest_categorical("hid",   space["hid"])
    layers = trial.suggest_int        ("layers", *space["layers"])
    drop   = trial.suggest_float      ("drop",   *space["drop"])
    lr     = trial.suggest_float      ("lr",     *space["lr"],  log=True)
    wd     = trial.suggest_float      ("wd",     *space["wd"],  log=True)
    batch  = trial.suggest_categorical("batch",  space["batch"])
    epochs = space["epochs"]

    tr_seqs, va_seqs = _split(df, q=space["train_frac"])
    max_len = cfg["models"]["reco"].get("max_seq_len", 50)

    tr_loader = DataLoader(_SeqDataset(tr_seqs, max_len), batch_size=batch,
                           shuffle=True, num_workers=1, persistent_workers=True,
                           pin_memory=(device.type == "cuda"), collate_fn=_collate)
    va_loader = DataLoader(_SeqDataset(va_seqs, max_len), batch_size=512,
                           shuffle=False, num_workers=1, persistent_workers=True,
                           pin_memory=(device.type == "cuda"), collate_fn=_collate)

    model = GRU4Rec(n_items, emb, hid, layers, drop).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)

    if hasattr(torch, 'compile'):
        model = torch.compile(model, mode='reduce-overhead')

    best = 0.0
    for ep in range(1, epochs + 1):
        _train_one_epoch(model, tr_loader, opt, device, trial.number, ep)
        _, ndcg = _evaluate(model, va_loader, device)

        trial.report(ndcg, ep)
        if trial.should_prune():
            if trial.number % 5 == 0:
                tqdm.write(f"  ✂️  Trial {trial.number} pruned at epoch {ep}")
            raise optuna.TrialPruned()
        best = max(best, ndcg)

    if trial.number % 10 == 0:
        tqdm.write(f"  ✅ Trial {trial.number} completed! NDCG: {best:.4f}")
    return best


# ════════════════════════════════════════════════════════════════════════════
#                                    MAIN
# ════════════════════════════════════════════════════════════════════════════
def main():
    pa = argparse.ArgumentParser()
    pa.add_argument("--cfg", default="config.yaml")
    pa.add_argument("--mode", choices=["quick", "full"], default="quick")
    pa.add_argument("--trials", type=int, default=80,
                    help="trial budget for full mode")
    args = pa.parse_args()

    cfg = _load_cfg(Path(args.cfg)); _set_seed(cfg.get("seed", 42))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tqdm.write(f"🚀  Device {device} "
               f"({torch.cuda.get_device_name(0) if device.type=='cuda' else platform.processor()})")

    tqdm.write("📂 Loading sequences...")
    reco_df, n_items = _load_seqs(cfg)
    tqdm.write(f"📊 Loaded {len(reco_df):,} sessions, {n_items:,} unique items")

    Path("artefacts").mkdir(exist_ok=True)
    storage = optuna.storages.RDBStorage(
        url="sqlite:///artefacts/optuna_gru4rec.db",
        engine_kwargs={"connect_args": {"check_same_thread": False}})

    pruner = optuna.pruners.SuccessiveHalvingPruner(
        min_resource=1, reduction_factor=3, min_early_stopping_rate=0)

    study = optuna.create_study(study_name="gru4rec_tuning",
                                storage=storage,
                                load_if_exists=True,
                                direction="maximize",
                                pruner=pruner,
                                sampler=optuna.samplers.TPESampler(
                                    multivariate=True, group=True))

    QUICK_PILOT = dict(emb=[64], hid=[64, 96], layers=(1, 1),
                       drop=(0.0, 0.3), lr=(1e-4, 2e-3), wd=(1e-6, 1e-4),
                       batch=[256], epochs=3, train_frac=0.5)
    QUICK_FOCUS = dict(emb=[64], hid=[96], layers=(1, 2),
                       drop=(0.0, 0.4), lr=(5e-4, 3e-3), wd=(1e-6, 5e-5),
                       batch=[384], epochs=6, train_frac=0.8)
    FULL_FAST   = dict(emb=[64], hid=[64, 96], layers=(1, 2),
                       drop=(0.0, 0.5), lr=(1e-4, 3e-3), wd=(1e-6, 1e-4),
                       batch=[384], epochs=4, train_frac=0.8)

    if args.mode == "quick":
        tqdm.write("🔍 Running QUICK mode: pilot phase...")
        study.optimize(lambda t: _objective(t, cfg, reco_df, n_items,
                                            device, QUICK_PILOT),
                       n_trials=15, timeout=600, show_progress_bar=True)
        tqdm.write("🎯 Running QUICK mode: focused phase...")
        study.optimize(lambda t: _objective(t, cfg, reco_df, n_items,
                                            device, QUICK_FOCUS),
                       n_trials=5, timeout=600, show_progress_bar=True)
    else:
        tqdm.write(f"🚄 Running FULL mode: {args.trials} trials")
        study.optimize(lambda t: _objective(t, cfg, reco_df, n_items,
                                            device, FULL_FAST),
                       n_trials=args.trials,
                       timeout=3600,
                       show_progress_bar=True)

    # ── plot study progress ────────────────────────────────────────────────
    Path("reports").mkdir(exist_ok=True)
    trials_df = study.trials_dataframe(attrs=("number", "value"))
    plt.figure(figsize=(6, 4))
    plt.plot(trials_df["number"], trials_df["value"])
    plt.xlabel("Trial"); plt.ylabel("Best NDCG@20 so far")
    plt.title("Optuna GRU4Rec Progress"); plt.tight_layout()
    plt.savefig("reports/tunedgru4rec_study_curve.png", dpi=120); plt.close()

    tqdm.write(f"🏆 Best trial: {study.best_trial.number}, NDCG: {study.best_value:.4f}")
    tqdm.write("🔄 Starting final 15-epoch refit with best parameters...")

    best = study.best_params
    max_len = cfg["models"]["reco"].get("max_seq_len", 50)
    final_batch = min(best["batch"], 256)

    # Refit on the TRAINING split only (do not refit on the full set,
    # or the final held-out evaluation becomes meaningless).
    tr_seqs_final, va_seqs_final = _split(reco_df, q=0.8)
    loader = DataLoader(_SeqDataset(tr_seqs_final, max_len), batch_size=final_batch,
                        shuffle=True, num_workers=0,
                        pin_memory=(device.type == "cuda"), collate_fn=_collate)

    model = GRU4Rec(n_items, best["emb"], best["hid"],
                    best["layers"], best["drop"]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=best["lr"], weight_decay=best["wd"])

    if hasattr(torch, 'compile'):
        model = torch.compile(model, mode='reduce-overhead')

    for epoch in tqdm(range(15), desc="Final fit", unit="epoch"):
        _train_one_epoch(model, loader, opt, device, "Final", epoch + 1)

    torch.save({"state_dict": model.state_dict(),
                "n_items": n_items,
                "best_params": best,
                "cfg": cfg},
               "artefacts/gru4rec_tuned.pt")

    # ── evaluate on HELD-OUT validation split (leakage fix) ────────────────
    va_loader = DataLoader(_SeqDataset(va_seqs_final, max_len), batch_size=512,
                           shuffle=False, num_workers=0,
                           pin_memory=(device.type == "cuda"), collate_fn=_collate)
    hr, ndcg = _evaluate(model, va_loader, device)

    Path("reports/metrics_reco_tuned.md").write_text(
f"""# Tuned GRU4Rec Report (chronological split, held-out evaluation)

| Metric      | Value |
|-------------|-------|
| HitRate@20  | {hr:.3f} |
| NDCG@20     | {ndcg:.3f} |
| Trials      | {len(study.trials)} |
| Best trial  | {study.best_trial.number} |
| Device      | {device} |
| Train sessions | {len(tr_seqs_final):,} |
| Val sessions   | {len(va_seqs_final):,} |
| Pass Gates? | {'✅' if ndcg>=0.14 and hr>=0.35 else '❌'} |

```json
{json.dumps(best, indent=2)}
```""", encoding="utf-8")

    print(f"Tuned: HR@20={hr:.3f}  NDCG@20={ndcg:.3f}")
    print("Model saved ➜ artefacts/gru4rec_tuned.pt")


if __name__ == "__main__":
    main()