# Pipeline - Quick Command Reference

## Features

- **Real-time ETA Display**: See estimated time remaining for each stage and overall pipeline
- **Smart Progress Tracking**: Progress bar shows current stage and completion percentage
- **Adaptive Time Estimation**: Pipeline learns from actual execution times to improve estimates

## Essential Commands

### Pipeline Execution

```bash
# Run full pipeline (with ETA tracking)
python pipeline.py

# Check status
python pipeline.py --status

# Force re-run
python pipeline.py --force
```

### Individual Stage Commands

```bash
# Data cleaning (~30s)
python -m src.clean --cfg config.yaml

# Feature engineering (~45s)
python -m src.features --cfg config.yaml

# EDA (~1m)
python -m src.EDA

# Train LightGBM (~2m)
python -m src.forecast_lightgbm

# Train GRU4Rec (~3m)
python -m src.GRU4REC_baseline

# Tune LightGBM (~5m)
python -m src.tune_lightgbm

# Tune GRU4Rec (~10m)
python -m src.tune_GRU4REC

# Launch dashboard
streamlit run app.py
```

### Pipeline Modes

```bash
# Data processing only (~2m)
python pipeline.py --mode data

# Model training only (~5m)
python pipeline.py --mode models

# Model tuning only (~15m)
python pipeline.py --mode tune

# Evaluation mode (~22m)
python pipeline.py --mode evaluate
```

### Progress Tracking

The pipeline now shows real-time progress with ETA:

```
Stage 3/9: eda:  33%|████████████████████████████████████████████████████████████████                                                    | 3/9 [02:15<04:32] ETA: ✓ eda | ETA: 00:04:32
```

### File Existence Check Results

When you run the pipeline, it will show:

- `[SKIP]` - Files already exist, skipping stage
- `[EXISTS]` - Specific files that already exist
- `[SUCCESS]` - Stage completed successfully
- `[FAILED]` - Stage failed to execute
- `[ETA]` - Estimated time for current stage and remaining pipeline

Use `--force` to override existing files and re-run stages.

## 📁 Key Files & Their Purpose

**Pipeline Control:**

- `pipeline.py` - Main orchestrator (9 stages)
- `config.yaml` - Configuration settings

**Data Processing:**

- `src/clean.py` - Data cleaning → `data/interim/*.parquet`
- `src/features.py` - Feature engineering → `data/processed/*.parquet`
- `src/EDA.py` - Exploratory analysis → `reports/eda/`

**Model Training:**

- `src/forecast_lightgbm.py` - Baseline forecasting → `artefacts/lightgbm_weighted.pkl`
- `src/tune_lightgbm.py` - Tuned forecasting → `artefacts/lightgbm_tuned_weighted.pkl`
- `src/GRU4REC_baseline.py` - Baseline recommendations → `artefacts/gru4rec_baseline.pt`
- `src/tune_GRU4REC.py` - Tuned recommendations → `artefacts/gru4rec_tuned.pt`

**Results & Metrics:**

- `reports/metrics_forecast_*.md` - Forecasting performance metrics
- `reports/metrics_reco_*.md` - Recommendation performance metrics
- `reports/eda/` - Data analysis charts and summaries

**Dashboard:**

- `app.py` - Interactive Streamlit dashboard
