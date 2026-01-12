"""
Script: train_xgb.py
Description:
    Trains an XGBoost baseline model for Landslide Susceptibility Mapping (LSM).
    Crucially, this script performs a "Static Bias" quantification analysis
    to verify if the model over-relies on static environmental factors (Slope, Lithology)
    while ignoring dynamic disturbance factors (dNDVI, dNBR) in post-fire scenarios.

    Configuration is loaded from: metadata/xgboost_config.yaml

Author: AI Assistant (Virgo Edition)
Date: 2026-01-05

python "experiments/01_ml_baselines/xgboost/train_xgb.py"
"""

import sys
import json
import yaml
import logging
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score, average_precision_score, classification_report

# Add project root to sys.path to allow imports
BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.append(str(BASE_DIR))

# Import Shared Utilities
sys.path.append(str(BASE_DIR / "experiments" / "01_ml_baselines"))
from ml_utils import (
    setup_logging,
    save_predictions
)

# Import custom path utility
SCRIPTS_DIR = BASE_DIR / "scripts" / "00_common"
sys.path.append(str(SCRIPTS_DIR))
import path_utils

# ==============================================================================
# EXPERIMENT CLASS
# ==============================================================================

class XGBoostBaseline:
    """
    Encapsulates the XGBoost training pipeline and Static Bias analysis.
    """

    def __init__(self, config_path: Path, mode: str = "dynamic"):
        """
        Args:
            config_path: Path to the specific YAML config for this experiment.
            mode: Experiment mode ('dynamic' or 'static').
        """
        self.config_path = config_path
        self.mode = mode
        self.config = self._load_config()
        
        # Load Dataset Config to resolve SU name
        self.dataset_config_path = BASE_DIR / "metadata" / f"dataset_config_{self.mode}.yaml"
        with open(self.dataset_config_path, "r", encoding="utf-8") as f:
            self.dataset_config = yaml.safe_load(f)
            
        self.su_name = path_utils.get_su_name(self.dataset_config)
        
        # Setup workspace with SU subdirectories
        base_output_dir = Path(self.config["experiment"]["output_dir"])
        self.output_dir = path_utils.resolve_su_path(base_output_dir, su_name=self.su_name)
        
        # Avoid double resolution: output_dir is already specific to SU
        self.results_dir = self.output_dir / "results"
        self.models_dir = self.output_dir / "models"
        
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger = setup_logging(f"{self.config['experiment']['name']}_{self.mode}", self.results_dir, f"train_xgb_{self.mode}.log")
        self.logger.info(f"Initialized Experiment: {self.config['experiment']['name']} | Mode: {self.mode}")
        
        # Placeholders
        self.df_merged: Optional[pd.DataFrame] = None
        self.model: Optional[xgb.XGBClassifier] = None
        self.features_static: List[str] = []
        self.features_dynamic: List[str] = []
        self.feature_names: List[str] = []

    def _load_config(self) -> Dict[str, Any]:
        """Loads YAML configuration."""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config not found: {self.config_path}")
        with open(self.config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    def load_and_align_data(self):
        """
        Loads features and labels, merges them, and aligns splits.
        """
        self.logger.info(">>> Loading Data...")
        
        # Resolve SU-specific data path
        base_tabular_dir = BASE_DIR / "04_tabular_SU"
        data_dir = path_utils.resolve_su_path(base_tabular_dir, su_name=self.su_name)
        
        feat_filename = f"su_features_{self.mode}.parquet"
        data_filename = f"tabular_dataset_{self.mode}.parquet"
        
        feat_path = data_dir / feat_filename
        data_path = data_dir / data_filename

        if not feat_path.exists() or not data_path.exists():
            self.logger.critical(f"Missing input files in {data_dir} for mode '{self.mode}'")
            sys.exit(1)

        # 1. Load Parquet Files
        self.logger.info(f"    Reading Features: {feat_path.name}")
        df_feats = pd.read_parquet(feat_path)
        
        self.logger.info(f"    Reading Dataset:  {data_path.name}")
        df_meta = pd.read_parquet(data_path)  # Should contain 'split', 'label', 'su_id' as index or col

        # 2. Merge
        # Ensure indices match (SU ID)
        if "su_id" in df_meta.columns:
            df_meta.set_index("su_id", inplace=True)
        
        # Inner Join: We need both features and valid labels
        self.df_merged = df_feats.join(df_meta, how="inner")
        self.logger.info(f"    Merged DataFrame Shape: {self.df_merged.shape}")
        
        # 3. Resolve Feature Columns based on Config
        all_cols = self.df_merged.columns.tolist()
        
        # Helper to find cols starting with prefix
        def get_cols(prefixes: List[str]) -> List[str]:
            found = []
            for pre in prefixes:
                # Matches "Slope_mean", "Slope", etc.
                matches = [c for c in all_cols if c.startswith(pre)]
                found.extend(matches)
            return sorted(list(set(found)))

        self.features_static = get_cols(self.config["features"]["static_prefixes"])
        self.features_dynamic = get_cols(self.config["features"]["dynamic_prefixes"])
        
        # Exclude blocked features (e.g., InSAR)
        excluded = get_cols(self.config["features"]["exclude_prefixes"])
        
        # Final Feature Set
        self.feature_names = [
            f for f in (self.features_static + self.features_dynamic) 
            if f not in excluded
        ]
        
        self.logger.info(f"    Feature Selection:")
        self.logger.info(f"      Static  ({len(self.features_static)}): {self.features_static[:3]} ...")
        self.logger.info(f"      Dynamic ({len(self.features_dynamic)}): {self.features_dynamic[:3]} ...")
        self.logger.info(f"      Total Input Features: {len(self.feature_names)}")

    def train(self):
        """
        Executes the XGBoost training pipeline.
        """
        if self.df_merged is None:
            raise RuntimeError("Data not loaded. Call load_and_align_data() first.")

        self.logger.info(">>> Preparing Training Sets...")
        
        # Filter by split
        # Also check for 'train_sample_mask' if available (Unified Framework Standard)
        
        if "train_sample_mask" in self.df_merged.columns:
            self.logger.info("    Using balanced 'train_sample_mask' for training.")
            train_mask = (self.df_merged["split"] == "train") & (self.df_merged["train_sample_mask"] == True)
        else:
            self.logger.warning("    'train_sample_mask' not found. Using full unbalanced train set.")
            train_mask = self.df_merged["split"] == "train"
            
        test_mask = self.df_merged["split"] == "test"
        
        X_train = self.df_merged.loc[train_mask, self.feature_names]
        y_train = self.df_merged.loc[train_mask, "label"]
        
        X_test = self.df_merged.loc[test_mask, self.feature_names]
        y_test = self.df_merged.loc[test_mask, "label"]
        
        self.logger.info(f"    Train Samples: {len(X_train)} | Pos Rate: {y_train.mean():.4f}")
        self.logger.info(f"    Test  Samples: {len(X_test)}  | Pos Rate: {y_test.mean():.4f}")

        # Hyperparameters
        params = self.config["model_params"].copy()
        
        # Auto-calculate scale_pos_weight if needed
        if params.get("scale_pos_weight") is None:
            neg, pos = np.bincount(y_train)
            scale_weight = neg / pos
            params["scale_pos_weight"] = scale_weight
            self.logger.info(f"    Auto-calculated scale_pos_weight: {scale_weight:.2f}")

        # Initialize Model
        self.model = xgb.XGBClassifier(
            **params,
            random_state=self.config["experiment"]["random_seed"]
        )

        # Train
        self.logger.info(">>> Starting Training...")
        start_time = time.time()
        
        self.model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            verbose=False 
        )
        
        duration = time.time() - start_time
        self.logger.info(f"    Training completed in {duration:.2f} seconds.")

    def evaluate(self):
        """
        Evaluates the model and saves predictions.
        """
        self.logger.info(">>> Evaluating Model...")
        
        # Predict on ALL data (for mapping)
        X_all = self.df_merged[self.feature_names]
        probs = self.model.predict_proba(X_all)[:, 1]
        
        # Use shared utility to save predictions
        # Dynamic Output Name
        out_csv = self.results_dir / f"xgb_predictions_{self.mode}.csv"
        save_predictions(self.df_merged, probs, out_csv)
        self.logger.info(f"    Predictions saved to: {out_csv}")

        # Metrics on Test Set
        mask_test = self.df_merged["split"] == "test"
        y_test = self.df_merged.loc[mask_test, "label"]
        p_test = probs[mask_test]
        
        auc = roc_auc_score(y_test, p_test)
        pr_auc = average_precision_score(y_test, p_test)
        
        self.logger.info(f"    [TEST METRICS]")
        self.logger.info(f"      ROC-AUC : {auc:.4f}")
        self.logger.info(f"      PR-AUC  : {pr_auc:.4f}")

    def analyze_static_bias(self):
        """
        Performs the core scientific validation:
        Calculates the ratio of Feature Importance (Gain) held by Static vs Dynamic features.
        """
        self.logger.info(">>> Analyzing Static Bias...")
        
        # Get Importance Dictionary (Type: Gain)
        importance_dict = self.model.get_booster().get_score(importance_type="gain")
        
        # Convert to DataFrame
        imp_df = pd.DataFrame(list(importance_dict.items()), columns=["Feature", "Gain"])
        imp_df["Gain"] = pd.to_numeric(imp_df["Gain"])
        
        # Grouping
        static_gain = imp_df[imp_df["Feature"].isin(self.features_static)]["Gain"].sum()
        dynamic_gain = imp_df[imp_df["Feature"].isin(self.features_dynamic)]["Gain"].sum()
        total_gain = static_gain + dynamic_gain + 1e-6 # Avoid div/0
        
        static_pct = (static_gain / total_gain) * 100
        dynamic_pct = (dynamic_gain / total_gain) * 100
        
        # Save Detailed Report (Dynamic Name)
        report_path = self.results_dir / f"bias_report_{self.mode}.txt"
        with open(report_path, "w", encoding="utf-8") as f:
            f.write("=== Static Bias Analysis Report ===\n")
            f.write(f"Model: XGBoost Baseline | Mode: {self.mode}\n")
            f.write("Metric: Total Information Gain\n\n")
            f.write(f"Static Importance  : {static_pct:.2f}%\n")
            f.write(f"Dynamic Importance : {dynamic_pct:.2f}%\n\n")
            f.write("Conclusion:\n")
            if static_pct > 70:
                f.write("[CONFIRMED] Model exhibits strong Static Bias (>70%).\n")
            else:
                f.write("[UNCERTAIN] Static Bias is not dominant.\n")
                
        self.logger.info(f"    Static Importance  : {static_pct:.2f}%")
        self.logger.info(f"    Dynamic Importance : {dynamic_pct:.2f}%")
        self.logger.info(f"    Bias Report saved to: {report_path}")
        
        # Save raw importance
        imp_path = self.results_dir / f"feature_importance_{self.mode}.csv"
        imp_df.sort_values("Gain", ascending=False).to_csv(imp_path, index=False)

    def save_model(self):
        # Dynamic Model Name
        model_path = self.models_dir / f"xgb_model_{self.mode}.json"
        self.model.save_model(model_path)
        self.logger.info(f"    Model saved to: {model_path}")

# ==============================================================================
# MAIN ENTRY POINT
# ==============================================================================

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, choices=["dynamic", "static"], default="dynamic", help="Experiment mode")
    args = parser.parse_args()

    # Locate Config Relative to Script
    # Script is in experiments/01_ml_baselines/xgboost/
    # Config is in metadata/
    BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent
    CONFIG_PATH = BASE_DIR / "metadata" / "xgboost_config.yaml"
    
    experiment = XGBoostBaseline(CONFIG_PATH, mode=args.mode)
    experiment.load_and_align_data()
    experiment.train()
    experiment.evaluate()
    experiment.analyze_static_bias()
    experiment.save_model()
    
    print(f"\n[SUCCESS] XGBoost Baseline Pipeline ({args.mode}) Completed.")
