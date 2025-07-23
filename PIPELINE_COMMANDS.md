# SmartRocket Pipeline Commands Reference

This document lists all the commands executed by the pipeline.py script in each stage.

## Features

- **Smart File Existence Checking**: Automatically skips stages if outputs already exist
- **Real-time Progress Tracking**: Progress bar with estimated time remaining (ETA)
- **Individual Stage ETA**: Shows estimated time for each stage before execution
- **Adaptive Time Estimation**: Learns from actual execution times to improve future estimates
- **Flexible Execution**: Run full pipeline, specific stages, or skip certain stages

## Pipeline Usage Commands

### Basic Pipeline Commands

```bash
# Run complete pipeline with ETA tracking
python pipeline.py

# Run complete pipeline (explicit)
python pipeline.py --mode full

# Check pipeline status
python pipeline.py --status

# Run with verbose logging
python pipeline.py --verbose

# Force re-execution of all stages
python pipeline.py --force
```

### Progress Tracking Features

The pipeline now provides detailed progress information:

**Example Output:**

```
Stage 3/9: eda:  33%|████████████████████████████████████████████████████████████████                                                    | 3/9 [02:15<04:32] ETA: ✓ eda | ETA: 00:04:32
[3/9] ===== STAGE: EDA =====
[INFO] Exploratory Data Analysis
[ETA] Stage estimated time: 1m00s
[ETA] Pipeline remaining time: 00:04:32
```

**Stage Duration Estimates:**

| Stage          | Script                     | Input                                      | Output                                  | Metrics                             |
| -------------- | -------------------------- | ------------------------------------------ | --------------------------------------- | ----------------------------------- |
| clean          | `src/clean.py`             | `data/raw/*.csv`                           | `data/interim/*.parquet`                | `reports/clean_*_summary.md`        |
| features       | `src/features.py`          | `data/interim/*.parquet`                   | `data/processed/*.parquet`              | -                                   |
| eda            | `src/EDA.py`               | `data/interim/*.parquet`                   | `reports/eda/`                          | `reports/eda/eda_summary.md`        |
| train_lightgbm | `src/forecast_lightgbm.py` | `data/processed/forecast_features.parquet` | `artefacts/lightgbm_weighted.pkl`       | `reports/metrics_forecast_final.md` |
| train_gru4rec  | `src/GRU4REC_baseline.py`  | `data/processed/reco_sequences.parquet`    | `artefacts/gru4rec_baseline.pt`         | `reports/metrics_reco_baseline.md`  |
| tune_lightgbm  | `src/tune_lightgbm.py`     | `data/processed/forecast_features.parquet` | `artefacts/lightgbm_tuned_weighted.pkl` | `reports/metrics_forecast_tuned.md` |
| tune_gru4rec   | `src/tune_GRU4REC.py`      | `data/processed/reco_sequences.parquet`    | `artefacts/gru4rec_tuned.pt`            | `reports/metrics_reco_tuned.md`     |
| evaluate       | Built into pipeline        | `artefacts/*`                              | Console output                          | -                                   |
| dashboard      | `app.py` check             | All above files                            | Dashboard ready                         | -                                   |

### Mode-specific Commands

```bash
# Data processing only
python pipeline.py --mode data

# Model training only
python pipeline.py --mode models

# Model tuning only
python pipeline.py --mode tune

# Evaluation mode (all stages except dashboard)
python pipeline.py --mode evaluate
```

### Stage-specific Commands

```bash
# Run specific stages
python pipeline.py --stage clean
python pipeline.py --stage features
python pipeline.py --stage clean features eda

# Skip specific stages
python pipeline.py --skip-stages eda reports

# Custom configuration
python pipeline.py --config custom_config.yaml
```

## 📊 Performance Metrics (Actual Results)

**Forecasting Model Performance:**

- **Baseline** (see `reports/metrics_forecast_final.md`): 18.50% wMAPE
- **Tuned** (see `reports/metrics_forecast_tuned.md`): 0.94% wMAPE (95% improvement!)

**Recommendation Model Performance:**

- **Baseline** (see `reports/metrics_reco_baseline.md`): 39.2% Hit Rate@20
- **Tuned** (see `reports/metrics_reco_tuned.md`): 84.3% Hit Rate@20 (115% improvement!)

**Key Insight**: The tuned models achieve production-ready performance with significant improvements over baseline models.

## Internal Commands Executed by Pipeline

### Stage 1: Data Cleaning (`clean`)

```bash
# Command executed internally:
python -m src.clean --cfg config.yaml

# Optional parameters:
python -m src.clean --cfg config.yaml --raw_dir data/raw --out_dir data/interim
```

**Input Files:**

- `data/raw/events.csv`
- `data/raw/category_tree.csv`
- `data/raw/item_properties_part1.csv`
- `data/raw/item_properties_part2.csv`

**Output Files:**

- `data/interim/events_clean.parquet`
- `data/interim/item_properties.parquet`
- `data/interim/category_tree.parquet`

### Stage 2: Feature Engineering (`features`)

```bash
# Command executed internally:
python -m src.features --cfg config.yaml
```

**Input Files:**

- `data/interim/events_clean.parquet`
- `data/interim/item_properties.parquet`
- `data/interim/category_tree.parquet`

**Output Files:**

- `data/processed/forecast_features.parquet`
- `data/processed/reco_sequences.parquet`

### Stage 3: Exploratory Data Analysis (`eda`)

```bash
# Command executed internally:
python -m src.EDA
```

**Input Files:**

- `data/interim/events_clean.parquet`
- `data/interim/item_properties.parquet`
- `data/interim/category_tree.parquet`

**Output Files:**

- `reports/eda/eda_summary.md`
- `reports/eda/figures/daily_volume.png`
- `reports/eda/figures/hour_weekday_heatmap.png`
- `reports/eda/figures/inter_event_time_hist.png`
- `reports/eda/figures/session_length_hist.png`

### Stage 4: Train LightGBM Model (`train_lightgbm`)

```bash
# Command executed internally:
python -m src.forecast_lightgbm
```

**Input Files:**

- `data/processed/forecast_features.parquet`

**Output Files:**

- `artefacts/lightgbm_weighted.pkl`
- `reports/lightgbm_error_hist.png`
- `reports/lightgbm_feature_importance.png`
- `reports/lightgbm_prediction_quality.png`

### Stage 5: Train GRU4Rec Model (`train_gru4rec`)

```bash
# Command executed internally:
python -m src.GRU4REC_baseline
```

**Input Files:**

- `data/processed/reco_sequences.parquet`

**Output Files:**

- `artefacts/gru4rec_baseline.pt`
- `artefacts/item2idx.json`
- `reports/gru4rec_training_curve.png`
- `reports/metrics_reco_baseline.md`

### Stage 6: Tune LightGBM Model (`tune_lightgbm`)

```bash
# Command executed internally:
python -m src.tune_lightgbm
```

**Input Files:**

- `data/processed/forecast_features.parquet`

**Output Files:**

- `artefacts/lightgbm_tuned_weighted.pkl`
- `reports/metrics_forecast_tuned.md`
- `reports/tuned_prediction_quality.png`
- `reports/tunedlighbgm_error_hist.png`
- `reports/tunedlighbgm_feature_importance.png`

### Stage 7: Tune GRU4Rec Model (`tune_gru4rec`)

```bash
# Command executed internally:
python -m src.tune_GRU4REC
```

**Input Files:**

- `data/processed/reco_sequences.parquet`

**Output Files:**

- `artefacts/gru4rec_tuned.pt`
- `artefacts/optuna_gru4rec.db`
- `reports/metrics_reco_tuned.md`
- `reports/tunedgru4rec_study_curve.png`

### Stage 8: Model Evaluation (`evaluate`)

```bash
# This stage is handled internally by the pipeline
# It checks for the existence of all model files and validates them
```

**Input Files:**

- `artefacts/lightgbm_weighted.pkl`
- `artefacts/lightgbm_tuned_weighted.pkl`
- `artefacts/gru4rec_baseline.pt`
- `artefacts/gru4rec_tuned.pt`

**Output Files:**

- Validation logs in `pipeline.log`

### Stage 9: Dashboard Preparation (`dashboard`)

```bash
# This stage validates dashboard requirements
# No external commands executed, just file validation
```

**Required Files:**

- `data/processed/forecast_features.parquet`
- `data/processed/reco_sequences.parquet`
- `config.yaml`
- `app.py`

**To Launch Dashboard:**

```bash
streamlit run app.py
```

## Manual Commands for Individual Modules

### Run Individual Source Modules

```bash
# Clean data manually
python -m src.clean --cfg config.yaml

# Generate features manually
python -m src.features --cfg config.yaml

# Run EDA manually
python -m src.EDA

# Train LightGBM manually
python -m src.forecast_lightgbm

# Train GRU4Rec manually
python -m src.GRU4REC_baseline

# Tune LightGBM manually
python -m src.tune_lightgbm

# Tune GRU4Rec manually
python -m src.tune_GRU4REC
```

### Generate Reports Manually

```bash
# Run quality analysis
python -m src.tuned_lightgbm_quality_diagram
```

### Launch Dashboard

```bash
# Start Streamlit dashboard
streamlit run app.py

# Access dashboard at: http://localhost:8501
```

## Pipeline Validation Commands

### Check Pipeline Status

```bash
# Check current pipeline status
python pipeline.py --status
```

## File Structure Verification

### Check Data Files

```bash
# List data files
dir data\raw
dir data\interim
dir data\processed

# Check file sizes
python -c "import pandas as pd; print(pd.read_parquet('data/interim/events_clean.parquet').shape)"
python -c "import pandas as pd; print(pd.read_parquet('data/processed/forecast_features.parquet').shape)"
```

### Check Model Files

```bash
# List model files
dir artefacts

# Check model file sizes
python -c "from pathlib import Path; print(f'LightGBM: {Path('artefacts/lightgbm_weighted.pkl').stat().st_size/1024/1024:.1f}MB')"
python -c "from pathlib import Path; print(f'GRU4Rec: {Path('artefacts/gru4rec_baseline.pt').stat().st_size/1024/1024:.1f}MB')"
```

### Check Report Files

```bash
# List report files
dir reports
dir reports\eda
dir reports\eda\figures
```

## Troubleshooting Commands

### Check Dependencies

```bash
# Install requirements
pip install -r requirements.txt

# Check specific packages
python -c "import lightgbm; print(lightgbm.__version__)"
python -c "import torch; print(torch.__version__)"
python -c "import streamlit; print(streamlit.__version__)"
```

### Debug Pipeline Issues

```bash
# Run pipeline with verbose output
python pipeline.py --verbose

# Run single stage for debugging
python pipeline.py --stage clean --verbose

# Force re-run specific stage
python pipeline.py --stage clean --force

# Check pipeline logs
type pipeline.log
```

## Environment Setup Commands

### Virtual Environment

```bash
# Create virtual environment
python -m venv .venv

# Activate virtual environment (Windows)
.venv\Scripts\activate

# Activate virtual environment (Linux/Mac)
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Git Commands

```bash
# Check git status
git status

# Add changes
git add .

# Commit changes
git commit -m "Updated pipeline"

# Push changes
git push origin main
```

## Performance Monitoring

### Check System Resources

```bash
# Monitor memory usage during pipeline execution
python -c "import psutil; print(f'Memory: {psutil.virtual_memory().percent}%')"

# Check disk space
python -c "import shutil; print(f'Disk space: {shutil.disk_usage('.').free / (1024**3):.1f}GB free')"
```

### Timing Pipeline Execution

```bash
# Time full pipeline execution
powershell "Measure-Command { python pipeline.py }"

# Time specific stage
powershell "Measure-Command { python pipeline.py --stage clean }"
```

## Notes

1. All commands assume you are in the project root directory
2. The pipeline automatically handles file existence checks
3. Use `--force` flag to override existing files
4. Check `pipeline.log` for detailed execution logs
5. Use `--verbose` flag for additional debugging information
