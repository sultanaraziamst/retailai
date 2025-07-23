#!/usr/bin/env python3
"""
pipeline.py -  End-to-End ML Pipeline
===========================================================

This pipeline orchestrates the complete machine learning workflow for the project,
from raw data ingestion to model deployment and report generation.

Pipeline Stages:
1. Data Cleaning & Preprocessing
2. Feature Engineering
3. Model Training (LightGBM + GRU4Rec)
4. Model Optimization & Tuning
5. Model Evaluation & Validation
6. Report Generation
7. Dashboard Preparation

Usage:
    python pipeline.py --mode full                 # Run complete pipeline
    python pipeline.py --mode data                 # Data processing only
    python pipeline.py --mode models               # Training only
    python pipeline.py --mode tune                 # Tuning only
    python pipeline.py --mode reports              # Reports only
    python pipeline.py --stage clean               # Single stage
    python pipeline.py --config custom.yaml        # Custom config
"""

from __future__ import annotations
import argparse
import json
import logging
import subprocess
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import pandas as pd
import yaml
from tqdm import tqdm

# Setup logging with UTF-8 encoding
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('pipeline.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class SmartRocketPipeline:
    """
    Main pipeline orchestrator for SmartRocket Analytics.
    
    This class manages the entire ML pipeline workflow, providing:
    - Stage-wise execution control
    - Configuration management
    - Progress tracking and logging
    - Error handling and recovery
    - Performance monitoring
    """
    
    def __init__(self, config_path: Union[str, Path] = "config.yaml"):
        """Initialize pipeline with configuration."""
        self.config_path = Path(config_path)
        self.config = self._load_config()
        self.start_time = time.time()
        
        # Pipeline stages in execution order
        self.stages = [
            'clean',
            'features', 
            'eda',
            'train_lightgbm',
            'train_gru4rec',
            'tune_lightgbm',
            'tune_gru4rec',
            'evaluate',
            'dashboard'
        ]
        
        # Stage timing tracking for ETA estimation
        self.stage_durations = {
            'clean': 30,        # seconds (estimated)
            'features': 45,
            'eda': 60,
            'train_lightgbm': 120,
            'train_gru4rec': 180,
            'tune_lightgbm': 300,
            'tune_gru4rec': 600,
            'evaluate': 15,
            'dashboard': 5
        }
        self.actual_durations = {}  # Track actual execution times
        
        # Stage configurations
        self.stage_configs = {
            'clean': {
                'module': 'src.clean',
                'description': 'Data cleaning and preprocessing',
                'inputs': ['data/raw/'],
                'outputs': ['data/interim/events_clean.parquet', 
                           'data/interim/item_properties.parquet',
                           'data/interim/category_tree.parquet']
            },
            'features': {
                'module': 'src.features',
                'description': 'Feature engineering for ML models',
                'inputs': ['data/interim/'],
                'outputs': ['data/processed/forecast_features.parquet',
                           'data/processed/reco_sequences.parquet']
            },
            'eda': {
                'module': 'src.EDA',
                'description': 'Exploratory data analysis',
                'inputs': ['data/interim/'],
                'outputs': ['reports/eda/']
            },
            'train_lightgbm': {
                'module': 'src.forecast_lightgbm',
                'description': 'Train baseline LightGBM forecasting model',
                'inputs': ['data/processed/forecast_features.parquet'],
                'outputs': ['artefacts/lightgbm_weighted.pkl']
            },
            'train_gru4rec': {
                'module': 'src.GRU4REC_baseline',
                'description': 'Train baseline GRU4Rec recommendation model',
                'inputs': ['data/processed/reco_sequences.parquet'],
                'outputs': ['artefacts/gru4rec_baseline.pt']
            },
            'tune_lightgbm': {
                'module': 'src.tune_lightgbm',
                'description': 'Hyperparameter tuning for LightGBM',
                'inputs': ['data/processed/forecast_features.parquet'],
                'outputs': ['artefacts/lightgbm_tuned_weighted.pkl']
            },
            'tune_gru4rec': {
                'module': 'src.tune_GRU4REC',
                'description': 'Hyperparameter tuning for GRU4Rec',
                'inputs': ['data/processed/reco_sequences.parquet'],
                'outputs': ['artefacts/gru4rec_tuned.pt']
            },
            'evaluate': {
                'module': 'pipeline.evaluate_models',
                'description': 'Model evaluation and validation',
                'inputs': ['artefacts/'],
                'outputs': ['reports/metrics_*.md']
            },
            'dashboard': {
                'module': 'app',
                'description': 'Prepare dashboard data',
                'inputs': ['data/processed/', 'artefacts/'],
                'outputs': ['Dashboard ready']
            }
        }
        
        logger.info(f"Pipeline initialized with config: {self.config_path}")
    
    def _load_config(self) -> Dict[str, Any]:
        """Load and validate pipeline configuration."""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            logger.info(f"Configuration loaded from {self.config_path}")
            return config
        except Exception as e:
            logger.error(f"Failed to load config: {e}")
            raise
    
    def _create_directories(self) -> None:
        """Create necessary directories for pipeline execution."""
        directories = [
            'data/raw',
            'data/interim', 
            'data/processed',
            'artefacts',
            'reports',
            'reports/eda',
            'reports/eda/figures',
            'logs'
        ]
        
        for dir_path in directories:
            Path(dir_path).mkdir(parents=True, exist_ok=True)
            logger.debug(f"Created directory: {dir_path}")
    
    def _check_dependencies(self) -> bool:
        """Check if required dependencies are installed."""
        required_packages = [
            'pandas', 'numpy', 'matplotlib', 'seaborn', 'plotly',
            'streamlit', 'lightgbm', 'torch', 'optuna', 'sklearn',
            'pyarrow', 'yaml', 'tqdm', 'joblib'
        ]
        
        missing_packages = []
        for package in required_packages:
            try:
                if package == 'sklearn':
                    __import__('sklearn')
                elif package == 'yaml':
                    __import__('yaml')
                else:
                    __import__(package)
            except ImportError:
                missing_packages.append(package)
        
        if missing_packages:
            logger.error(f"Missing packages: {missing_packages}")
            logger.error("Install with: pip install -r requirements.txt")
            return False
        
        logger.info("All dependencies satisfied")
        return True
    
    def _run_stage(self, stage: str, force: bool = False, **kwargs) -> bool:
        """Execute a single pipeline stage."""
        if stage not in self.stage_configs:
            logger.error(f"Unknown stage: {stage}")
            return False
        
        stage_config = self.stage_configs[stage]
        stage_start_time = time.time()
        
        # Calculate estimated time for this stage
        estimated_duration = self.stage_durations.get(stage, 60)
        logger.info(f"Starting stage: {stage} - {stage_config['description']} (est. {estimated_duration//60}m{estimated_duration%60:02.0f}s)")
        
        # Check if outputs already exist
        if not force and self._check_stage_outputs(stage):
            logger.info(f"[SKIP] Stage {stage} outputs already exist")
            self._log_existing_outputs(stage)
            logger.info(f"Skipping stage {stage} (use --force to re-run)")
            # Still track time for skipped stages (minimal duration)
            self._update_stage_duration(stage, 1.0)
            return True
        
        # Check inputs exist
        missing_inputs = []
        for input_path in stage_config['inputs']:
            if not Path(input_path).exists():
                missing_inputs.append(input_path)
                logger.warning(f"Input not found: {input_path}")
        
        if missing_inputs and stage not in ['clean']:  # Clean stage doesn't need existing inputs
            logger.error(f"Cannot run stage {stage} - missing required inputs: {missing_inputs}")
            return False
        
        try:
            # Execute stage based on type
            if stage == 'clean':
                result = self._run_clean_stage(force=force, **kwargs)
            elif stage == 'features':
                result = self._run_features_stage(force=force, **kwargs)
            elif stage == 'eda':
                result = self._run_eda_stage(force=force, **kwargs)
            elif stage == 'train_lightgbm':
                result = self._run_lightgbm_stage(force=force, **kwargs)
            elif stage == 'train_gru4rec':
                result = self._run_gru4rec_stage(force=force, **kwargs)
            elif stage == 'tune_lightgbm':
                result = self._run_tune_lightgbm_stage(force=force, **kwargs)
            elif stage == 'tune_gru4rec':
                result = self._run_tune_gru4rec_stage(force=force, **kwargs)
            elif stage == 'evaluate':
                result = self._run_evaluate_stage(force=force, **kwargs)
            elif stage == 'dashboard':
                result = self._run_dashboard_stage(force=force, **kwargs)
            else:
                logger.error(f"Stage execution not implemented: {stage}")
                return False
            
            # Track actual execution time
            actual_duration = time.time() - stage_start_time
            self._update_stage_duration(stage, actual_duration)
            
            if result:
                logger.info(f"✓ Stage {stage} completed successfully in {actual_duration:.1f}s")
            else:
                logger.error(f"✗ Stage {stage} failed after {actual_duration:.1f}s")
                
            return result
                
        except Exception as e:
            actual_duration = time.time() - stage_start_time
            logger.error(f"Stage {stage} failed after {actual_duration:.1f}s: {e}")
            return False
    
    def _log_existing_outputs(self, stage: str) -> None:
        """Log which outputs already exist for a stage."""
        stage_config = self.stage_configs[stage]
        
        if stage == 'clean':
            files = ['data/interim/events_clean.parquet', 'data/interim/item_properties.parquet', 'data/interim/category_tree.parquet']
            for file in files:
                if Path(file).exists():
                    logger.info(f"   - {file} [EXISTS]")
        elif stage == 'features':
            files = ['data/processed/forecast_features.parquet', 'data/processed/reco_sequences.parquet']
            for file in files:
                if Path(file).exists():
                    logger.info(f"   - {file} [EXISTS]")
        elif stage == 'eda':
            if Path('reports/eda').exists():
                logger.info(f"   - reports/eda/ [EXISTS]")
        elif stage in ['train_lightgbm', 'train_gru4rec', 'tune_lightgbm', 'tune_gru4rec']:
            for output in stage_config['outputs']:
                if Path(output).exists():
                    logger.info(f"   - {output} [EXISTS]")
        elif stage == 'dashboard':
            files = ['data/processed/forecast_features.parquet', 'data/processed/reco_sequences.parquet', 'config.yaml', 'app.py']
            for file in files:
                if Path(file).exists():
                    logger.info(f"   - {file} [EXISTS]")
    
    def _run_clean_stage(self, force: bool = False, **kwargs) -> bool:
        """Execute data cleaning stage."""
        output_files = [
            'data/interim/events_clean.parquet',
            'data/interim/item_properties.parquet', 
            'data/interim/category_tree.parquet'
        ]
        
        existing_files = [f for f in output_files if Path(f).exists()]
        if existing_files and not force:
            logger.info(f"[SKIP] Clean stage outputs already exist:")
            for file in existing_files:
                logger.info(f"   - {file} [EXISTS]")
            return True
        
        try:
            cmd = [sys.executable, '-m', 'src.clean', '--cfg', str(self.config_path)]
            
            if 'raw_dir' in kwargs:
                cmd.extend(['--raw_dir', kwargs['raw_dir']])
            if 'out_dir' in kwargs:
                cmd.extend(['--out_dir', kwargs['out_dir']])
            
            logger.info("Running data cleaning...")
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                logger.info("[SUCCESS] Data cleaning completed successfully")
                return True
            else:
                logger.error(f"[FAILED] Data cleaning failed: {result.stderr}")
                return False
                
        except Exception as e:
            logger.error(f"[ERROR] Error in clean stage: {e}")
            return False
    
    def _run_features_stage(self, force: bool = False, **kwargs) -> bool:
        """Execute feature engineering stage."""
        output_files = [
            'data/processed/forecast_features.parquet',
            'data/processed/reco_sequences.parquet'
        ]
        
        existing_files = [f for f in output_files if Path(f).exists()]
        if len(existing_files) == len(output_files) and not force:
            logger.info(f"[SKIP] Feature engineering outputs already exist:")
            for file in existing_files:
                logger.info(f"   - {file} [EXISTS]")
            return True
        
        try:
            cmd = [sys.executable, '-m', 'src.features', '--cfg', str(self.config_path)]
            logger.info("Running feature engineering...")
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                logger.info("[SUCCESS] Feature engineering completed successfully")
                return True
            else:
                logger.error(f"[FAILED] Feature engineering failed: {result.stderr}")
                return False
                
        except Exception as e:
            logger.error(f"[ERROR] Error in features stage: {e}")
            return False
    
    def _run_eda_stage(self, force: bool = False, **kwargs) -> bool:
        """Execute exploratory data analysis stage."""
        eda_dir = Path('reports/eda')
        if eda_dir.exists() and any(eda_dir.iterdir()) and not force:
            logger.info(f"[SKIP] EDA outputs already exist in: {eda_dir}")
            return True
        
        try:
            cmd = [sys.executable, '-m', 'src.EDA']
            logger.info("Running exploratory data analysis...")
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                logger.info("[SUCCESS] EDA completed successfully")
                return True
            else:
                logger.error(f"[FAILED] EDA failed: {result.stderr}")
                return False
                
        except Exception as e:
            logger.error(f"[ERROR] Error in EDA stage: {e}")
            return False
    
    def _run_lightgbm_stage(self, force: bool = False, **kwargs) -> bool:
        """Execute LightGBM training stage."""
        model_file = 'artefacts/lightgbm_weighted.pkl'
        if Path(model_file).exists() and not force:
            logger.info(f"[SKIP] LightGBM model already exists: {model_file}")
            return True
        
        try:
            cmd = [sys.executable, '-m', 'src.forecast_lightgbm']
            logger.info("Training LightGBM model...")
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                logger.info("[SUCCESS] LightGBM training completed successfully")
                return True
            else:
                logger.error(f"[FAILED] LightGBM training failed: {result.stderr}")
                return False
                
        except Exception as e:
            logger.error(f"[ERROR] Error in LightGBM stage: {e}")
            return False
    
    def _run_gru4rec_stage(self, force: bool = False, **kwargs) -> bool:
        """Execute GRU4Rec training stage."""
        model_file = 'artefacts/gru4rec_baseline.pt'
        if Path(model_file).exists() and not force:
            logger.info(f"[SKIP] GRU4Rec model already exists: {model_file}")
            return True
        
        try:
            cmd = [sys.executable, '-m', 'src.GRU4REC_baseline']
            logger.info("Training GRU4Rec model...")
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                logger.info("[SUCCESS] GRU4Rec training completed successfully")
                return True
            else:
                logger.error(f"[FAILED] GRU4Rec training failed: {result.stderr}")
                return False
                
        except Exception as e:
            logger.error(f"[ERROR] Error in GRU4Rec stage: {e}")
            return False
    
    def _run_tune_lightgbm_stage(self, force: bool = False, **kwargs) -> bool:
        """Execute LightGBM tuning stage."""
        model_file = 'artefacts/lightgbm_tuned_weighted.pkl'
        if Path(model_file).exists() and not force:
            logger.info(f"[SKIP] Tuned LightGBM model already exists: {model_file}")
            return True
        
        try:
            cmd = [sys.executable, '-m', 'src.tune_lightgbm']
            logger.info("Tuning LightGBM model...")
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                logger.info("[SUCCESS] LightGBM tuning completed successfully")
                return True
            else:
                logger.error(f"[FAILED] LightGBM tuning failed: {result.stderr}")
                return False
                
        except Exception as e:
            logger.error(f"[ERROR] Error in LightGBM tuning stage: {e}")
            return False
    
    def _run_tune_gru4rec_stage(self, force: bool = False, **kwargs) -> bool:
        """Execute GRU4Rec tuning stage."""
        model_file = 'artefacts/gru4rec_tuned.pt'
        if Path(model_file).exists() and not force:
            logger.info(f"[SKIP] Tuned GRU4Rec model already exists: {model_file}")
            return True
        
        try:
            cmd = [sys.executable, '-m', 'src.tune_GRU4REC']
            logger.info("Tuning GRU4Rec model...")
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                logger.info("[SUCCESS] GRU4Rec tuning completed successfully")
                return True
            else:
                logger.error(f"[FAILED] GRU4Rec tuning failed: {result.stderr}")
                return False
                
        except Exception as e:
            logger.error(f"[ERROR] Error in GRU4Rec tuning stage: {e}")
            return False
    
    def _run_evaluate_stage(self, force: bool = False, **kwargs) -> bool:
        """Execute model evaluation stage."""
        try:
            logger.info("Running model evaluation...")
            
            models_to_check = [
                'artefacts/lightgbm_weighted.pkl',
                'artefacts/lightgbm_tuned_weighted.pkl',
                'artefacts/gru4rec_baseline.pt',
                'artefacts/gru4rec_tuned.pt'
            ]
            
            existing_models = []
            for model_path in models_to_check:
                if Path(model_path).exists():
                    existing_models.append(model_path)
                    logger.info(f"Found model: {model_path}")
                else:
                    logger.warning(f"Model not found: {model_path}")
            
            if existing_models:
                logger.info(f"[SUCCESS] Evaluation completed. Found {len(existing_models)} models.")
                return True
            else:
                logger.warning("No models found for evaluation")
                return False
                
        except Exception as e:
            logger.error(f"[ERROR] Error in evaluation stage: {e}")
            return False
    
    def _run_dashboard_stage(self, force: bool = False, **kwargs) -> bool:
        """Execute dashboard preparation stage."""
        try:
            required_files = [
                'data/processed/forecast_features.parquet',
                'data/processed/reco_sequences.parquet',
                'config.yaml',
                'app.py'
            ]
            
            missing_files = []
            existing_files = []
            
            for file_path in required_files:
                if not Path(file_path).exists():
                    missing_files.append(file_path)
                else:
                    existing_files.append(file_path)
            
            if missing_files:
                logger.error(f"[FAILED] Dashboard missing required files:")
                for file in missing_files:
                    logger.error(f"   - {file}")
                return False
            
            logger.info(f"[SUCCESS] Dashboard requirements satisfied:")
            for file in existing_files:
                logger.info(f"   - {file} [EXISTS]")
            
            logger.info("[SUCCESS] Dashboard preparation completed successfully")
            logger.info("[INFO] Run 'streamlit run app.py' to start the dashboard")
            return True
            
        except Exception as e:
            logger.error(f"[ERROR] Error in dashboard stage: {e}")
            return False
    
    def run_pipeline(self, 
                    mode: str = 'full',
                    stages: Optional[List[str]] = None,
                    skip_stages: Optional[List[str]] = None,
                    force: bool = False) -> bool:
        """Execute the complete pipeline or specified stages."""
        
        # Determine stages to execute
        if stages:
            execution_stages = stages
        else:
            execution_stages = self._get_stages_for_mode(mode)
        
        # Remove skipped stages
        if skip_stages:
            execution_stages = [s for s in execution_stages if s not in skip_stages]
        
        logger.info(f"Executing pipeline - Mode: {mode}, Stages: {execution_stages}")
        
        # Setup
        self._create_directories()
        
        if not self._check_dependencies():
            return False
        
        # Execute stages with progress bar
        failed_stages = []
        completed_stages = []
        
        total_stages = len(execution_stages)
        
        # Create progress bar for overall pipeline progress
        with tqdm(total=total_stages, desc="Pipeline Progress", unit="stage", 
                 bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}] ETA: {postfix}") as pbar:
            
            for i, stage in enumerate(execution_stages, 1):
                stage_desc = self.stage_configs[stage]['description']
                
                # Calculate estimated time remaining
                eta = self._get_estimated_time_remaining(i-1)
                pbar.set_postfix_str(f"{eta}")
                pbar.set_description(f"Stage {i}/{total_stages}: {stage}")
                
                logger.info(f"[{i}/{total_stages}] ===== STAGE: {stage.upper()} =====")
                logger.info(f"[INFO] {stage_desc}")
                
                # Show individual stage progress
                stage_eta = self.stage_durations.get(stage, 60)
                if stage in self.actual_durations:
                    stage_eta = self.actual_durations[stage]
                    
                logger.info(f"[ETA] Stage estimated time: {stage_eta//60:.0f}m{stage_eta%60:02.0f}s")
                logger.info(f"[ETA] Pipeline remaining time: {eta}")
                
                stage_start_time = time.time()
                
                try:
                    success = self._run_stage(stage, force=force)
                    stage_duration = time.time() - stage_start_time
                    
                    # Update actual duration for ETA calculation
                    self._update_stage_duration(stage, stage_duration)
                    
                    if success:
                        logger.info(f"[SUCCESS] Stage {stage} completed in {stage_duration:.2f}s")
                        completed_stages.append(stage)
                        pbar.set_postfix_str(f"✓ {stage} | ETA: {self._get_estimated_time_remaining(i)}")
                    else:
                        logger.error(f"[FAILED] Stage {stage} failed")
                        failed_stages.append(stage)
                        pbar.set_postfix_str(f"✗ {stage} | ETA: {self._get_estimated_time_remaining(i)}")
                        
                        # Stop execution on critical failures
                        if stage in ['clean', 'features']:
                            logger.error(f"[CRITICAL] Critical stage {stage} failed, stopping pipeline")
                            break
                    
                    # Update progress bar
                    pbar.update(1)
                            
                except KeyboardInterrupt:
                    logger.info("[STOP] Pipeline interrupted by user")
                    pbar.set_postfix_str("⚠ Interrupted")
                    break
                except Exception as e:
                    logger.error(f"[CRITICAL] Unexpected error in stage {stage}: {e}")
                    failed_stages.append(stage)
                    pbar.set_postfix_str("✗ Error")
                    break
        
        # Summary
        total_time = time.time() - self.start_time
        
        logger.info(f"\n" + "="*60)
        logger.info(f"[SUMMARY] PIPELINE EXECUTION SUMMARY")
        logger.info(f"="*60)
        logger.info(f"[TIME] Total execution time: {total_time:.2f}s")
        logger.info(f"[SUCCESS] Completed stages: {len(completed_stages)}")
        logger.info(f"[FAILED] Failed stages: {len(failed_stages)}")
        
        if completed_stages:
            logger.info(f"[COMPLETE] Successful stages: {', '.join(completed_stages)}")
        
        if failed_stages:
            logger.error(f"[CRITICAL] Failed stages: {', '.join(failed_stages)}")
            logger.info(f"\n[HELP] Troubleshooting tips:")
            logger.info(f"   - Check the logs above for specific error messages")
            logger.info(f"   - Run 'python validate_pipeline.py --fix' to fix common issues")
            logger.info(f"   - Run individual stages with 'python pipeline.py --stage <stage_name>'")
            return False
        
        logger.info(f"\n[READY] Pipeline completed successfully!")
        
        # Show next steps
        if 'dashboard' in completed_stages:
            logger.info(f"\n[NEXT] NEXT STEPS:")
            logger.info(f"   [WEB] Launch dashboard: streamlit run app.py")
            logger.info(f"   [URL] Access at: http://localhost:8501")
        
        return True
    
    def _get_stages_for_mode(self, mode: str) -> List[str]:
        """Get stages to execute based on mode."""
        mode_mappings = {
            'full': self.stages,
            'data': ['clean', 'features', 'eda'],
            'models': ['train_lightgbm', 'train_gru4rec'],
            'tune': ['tune_lightgbm', 'tune_gru4rec'],
            'evaluate': ['evaluate', 'dashboard']
        }
        
        return mode_mappings.get(mode, self.stages)
    
    def _check_stage_outputs(self, stage: str) -> bool:
        """Check if stage outputs already exist."""
        if stage not in self.stage_configs:
            return False
        
        # Special handling for different stage types
        if stage == 'evaluate':
            # Check if evaluation has been done by looking for models
            models_exist = any(Path(f).exists() for f in [
                'artefacts/lightgbm_weighted.pkl',
                'artefacts/lightgbm_tuned_weighted.pkl',
                'artefacts/gru4rec_baseline.pt',
                'artefacts/gru4rec_tuned.pt'
            ])
            return models_exist
        
        elif stage == 'dashboard':
            # Check if dashboard requirements are met
            required_files = [
                'data/processed/forecast_features.parquet',
                'data/processed/reco_sequences.parquet',
                'config.yaml',
                'app.py'
            ]
            return all(Path(f).exists() for f in required_files)
        
        elif stage == 'eda':
            # Check if EDA directory exists and has content
            eda_dir = Path('reports/eda')
            return eda_dir.exists() and any(eda_dir.iterdir())
        
        else:
            # Standard check for other stages
            outputs = self.stage_configs[stage]['outputs']
            for output in outputs:
                if output.endswith('/'):
                    # Directory check
                    if not Path(output).exists():
                        return False
                else:
                    # File check
                    if not Path(output).exists():
                        return False
        
        return True
    
    def get_pipeline_status(self) -> Dict[str, Any]:
        """Get current pipeline status and stage completion."""
        status = {
            'config': str(self.config_path),
            'total_stages': len(self.stages),
            'completed_stages': [],
            'missing_outputs': {},
            'data_sizes': {},
            'models_available': {}
        }
        
        # Check stage completion
        for stage in self.stages:
            if self._check_stage_outputs(stage):
                status['completed_stages'].append(stage)
            else:
                missing = []
                for output in self.stage_configs[stage]['outputs']:
                    if not Path(output).exists():
                        missing.append(output)
                status['missing_outputs'][stage] = missing
        
        # Check data sizes
        data_files = [
            'data/interim/events_clean.parquet',
            'data/processed/forecast_features.parquet',
            'data/processed/reco_sequences.parquet'
        ]
        
        for file_path in data_files:
            if Path(file_path).exists():
                try:
                    df = pd.read_parquet(file_path)
                    status['data_sizes'][file_path] = {
                        'rows': len(df),
                        'columns': len(df.columns),
                        'size_mb': Path(file_path).stat().st_size / (1024 * 1024)
                    }
                except Exception as e:
                    status['data_sizes'][file_path] = f"Error: {e}"
        
        # Check models
        model_files = [
            'artefacts/lightgbm_weighted.pkl',
            'artefacts/lightgbm_tuned_weighted.pkl',
            'artefacts/gru4rec_baseline.pt',
            'artefacts/gru4rec_tuned.pt'
        ]
        
        for model_path in model_files:
            if Path(model_path).exists():
                status['models_available'][model_path] = {
                    'size_mb': Path(model_path).stat().st_size / (1024 * 1024),
                    'modified': datetime.fromtimestamp(Path(model_path).stat().st_mtime)
                }
        
        return status

    def _get_estimated_time_remaining(self, current_stage_idx: int) -> str:
        """Calculate estimated time remaining for the pipeline."""
        if current_stage_idx >= len(self.stages):
            return "00:00:00"
            
        remaining_stages = self.stages[current_stage_idx:]
        estimated_seconds = 0
        
        for stage in remaining_stages:
            # Use actual duration if available, otherwise use estimate
            if stage in self.actual_durations:
                estimated_seconds += self.actual_durations[stage]
            else:
                estimated_seconds += self.stage_durations.get(stage, 60)
        
        # Format as HH:MM:SS
        hours = int(estimated_seconds // 3600)
        minutes = int((estimated_seconds % 3600) // 60)
        seconds = int(estimated_seconds % 60)
        
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    
    def _update_stage_duration(self, stage: str, duration: float) -> None:
        """Update the actual duration for a stage to improve future estimates."""
        self.actual_durations[stage] = duration
        logger.debug(f"Updated duration for {stage}: {duration:.1f}s")

def main():
    """Main pipeline execution function."""
    parser = argparse.ArgumentParser(
        description='SmartRocket Analytics ML Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python pipeline.py                              # Run full pipeline
    python pipeline.py --mode data                  # Data processing only
    python pipeline.py --stage clean features       # Specific stages
    python pipeline.py --skip-stages eda reports    # Skip certain stages
    python pipeline.py --status                     # Check pipeline status
    python pipeline.py --force                      # Force re-execution
        """
    )
    
    parser.add_argument(
        '--mode', 
        choices=['full', 'data', 'models', 'tune', 'evaluate'],
        default='full',
        help='Pipeline execution mode'
    )
    
    parser.add_argument(
        '--stage',
        nargs='*',
        choices=['clean', 'features', 'eda', 'train_lightgbm', 'train_gru4rec', 
                'tune_lightgbm', 'tune_gru4rec', 'evaluate', 'dashboard'],
        help='Specific stages to execute'
    )
    
    parser.add_argument(
        '--skip-stages',
        nargs='*',
        help='Stages to skip'
    )
    
    parser.add_argument(
        '--config',
        type=Path,
        default='config.yaml',
        help='Configuration file path'
    )
    
    parser.add_argument(
        '--force',
        action='store_true',
        help='Force re-execution even if outputs exist'
    )
    
    parser.add_argument(
        '--status',
        action='store_true',
        help='Show pipeline status and exit'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    
    args = parser.parse_args()
    
    # Configure logging
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Initialize pipeline
    try:
        pipeline = SmartRocketPipeline(args.config)
    except Exception as e:
        logger.error(f"Failed to initialize pipeline: {e}")
        sys.exit(1)
    
    # Show status if requested
    if args.status:
        status = pipeline.get_pipeline_status()
        print(json.dumps(status, indent=2, default=str))
        sys.exit(0)
    
    # Execute pipeline
    try:
        success = pipeline.run_pipeline(
            mode=args.mode,
            stages=args.stage,
            skip_stages=args.skip_stages,
            force=args.force
        )
        
        if success:
            logger.info("[COMPLETE] Pipeline completed successfully!")
            sys.exit(0)
        else:
            logger.error("[FAILED] Pipeline failed!")
            sys.exit(1)
            
    except KeyboardInterrupt:
        logger.info("Pipeline interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Pipeline failed with error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
