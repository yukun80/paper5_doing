"""
Script: inference_xgb.py
Location: experiments/01_ml_baselines/xgboost/
Description:
    [Program 9-ML Inference] XGBoost Baseline Inference & Mapping.

    This script acts as the deployment module for the XGBoost model.
    It demonstrates how to:
    1.  Load the standalone JSON model (decoupling from training code).
    2.  Reload the configuration to ensure feature consistency.
    3.  Predict Landslide Susceptibility (Probability) for the entire study area.
    4.  Generate the final GeoTIFF Susceptibility Map (LSM).

    Modes:
    -   If 'xgb_predictions.csv' exists and is complete, it can optionally skip inference
        and just perform raster mapping (Fast Path).
    -   By default, it performs full inference to verify model integrity.

    Author: AI Assistant (Virgo Edition)
    Date: 2026-01-05

python "experiments/01_ml_baselines/xgboost/inference_xgb.py"
"""

import sys
import yaml
import logging
import time
from pathlib import Path
from typing import Dict, List, Any, Optional

import numpy as np
import pandas as pd
import xgboost as xgb
import rasterio

# ==============================================================================
# CONFIGURATION & CONSTANTS
# ==============================================================================

# Derive paths relative to this script
CURRENT_DIR = Path(__file__).resolve().parent
BASE_DIR = CURRENT_DIR.parent.parent.parent  # Up to datasets root
CONFIG_PATH = BASE_DIR / "metadata" / "xgboost_config.yaml"
SU_ID_RASTER = BASE_DIR / "02_aligned_grid" / "su_a50000_c03_geo.tif"

# Setup Logging
# Ensure results dir exists for logging
RESULTS_LOG_DIR = CURRENT_DIR / "results"
RESULTS_LOG_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(RESULTS_LOG_DIR / "inference.log", mode="w", encoding="utf-8"),
    ],
)
logger = logging.getLogger(__name__)


# ==============================================================================
# INFERENCE CLASS
# ==============================================================================


class XGBoostInference:
    """
    Handles the loading, prediction, and mapping phases for the XGBoost baseline.
    """

    def __init__(self, config_path: Path, mode: str = "dynamic"):
        self.config_path = config_path
        self.mode = mode
        self.config = self._load_config()
        self.output_dir = Path(self.config["experiment"]["output_dir"])
        self.results_dir = self.output_dir / "results"
        self.models_dir = self.output_dir / "models"
        
        self.results_dir.mkdir(parents=True, exist_ok=True)
        # Models dir should already exist from training, but checking is fine.

        # Paths (Resolved relative to project root)
        self.project_root = self.config_path.parent.parent
        
        # DYNAMIC PATHS
        self.model_path = self.models_dir / f"xgb_model_{self.mode}.json"
        
        # We need the features file to get the input data.
        # Following train_xgb.py logic, this is in 04_tabular_SU
        self.feature_path = self.project_root / "04_tabular_SU" / f"su_features_{self.mode}.parquet"

        # Verify Environment
        if not self.model_path.exists():
            logger.critical(f"Model file not found: {self.model_path}")
            logger.critical(f"Please run train_xgb.py --mode {self.mode} first.")
            sys.exit(1)
            
        if not self.feature_path.exists():
            logger.critical(f"Feature file not found: {self.feature_path}")
            sys.exit(1)

    def _load_config(self) -> Dict[str, Any]:
        with open(self.config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    def resolve_features(self, all_cols: List[str]) -> List[str]:
        """
        Reconstructs the exact feature list used during training based on config prefixes.
        Crucial for ensuring the input matrix X has the correct columns.
        """
        static_pre = self.config["features"]["static_prefixes"]
        dynamic_pre = self.config["features"]["dynamic_prefixes"]
        exclude_pre = self.config["features"]["exclude_prefixes"]

        # 1. Select
        selected = []
        for pre in static_pre + dynamic_pre:
            matches = [c for c in all_cols if c.startswith(pre)]
            selected.extend(matches)

        # 2. Exclude
        for pre in exclude_pre:
            selected = [c for c in selected if not c.startswith(pre)]

        # Deduplicate and Sort
        final_features = sorted(list(set(selected)))

        logger.info(f"    Resolved {len(final_features)} features from config.")
        return final_features

    def run_full_inference(self) -> pd.DataFrame:
        """
        Loads features, loads model, and predicts for ALL units.
        """
        logger.info(f">>> Starting Full Inference ({self.mode})...")

        # 1. Load Data
        logger.info(f"    Reading Features: {self.feature_path.name}")
        df = pd.read_parquet(self.feature_path)

        # Handle index
        if "su_id" not in df.columns:
            df["su_id"] = df.index

        # 2. Prepare Features
        feature_names = self.resolve_features(df.columns.tolist())

        # Check for missing columns
        missing = [f for f in feature_names if f not in df.columns]
        if missing:
            logger.critical(f"Dataset missing features required by config: {missing[:5]}...")
            sys.exit(1)

        X = df[feature_names]

        # 3. Load Model
        logger.info(f"    Loading Model: {self.model_path.name}")
        model = xgb.XGBClassifier()
        model.load_model(self.model_path)

        # 4. Predict
        logger.info(f"    Predicting for {len(X)} Slope Units...")
        probs = model.predict_proba(X)[:, 1]

        # 5. Pack Results
        df_res = pd.DataFrame({"su_id": df["su_id"].values, "xgb_prob": probs})

        return df_res

    def map_to_raster(self, df_preds: pd.DataFrame):
        """
        Maps the probability values back to the original SU raster geometry.
        """
        output_name = f"LSM_XGBoost_Prob_{self.mode.capitalize()}.tif" # e.g. LSM_XGBoost_Prob_Dynamic.tif
        logger.info(f">>> Generating Map: {output_name}")

        if not SU_ID_RASTER.exists():
            logger.critical(f"Reference Raster not found: {SU_ID_RASTER}")
            sys.exit(1)

        # 1. Read Reference Grid
        with rasterio.open(SU_ID_RASTER) as src:
            grid = src.read(1)
            profile = src.profile
            nodata_val = src.nodata

        # 2. Initialize Look-Up Table (LUT)
        # Find max ID in grid to size the LUT array
        valid_mask = grid != nodata_val
        if not valid_mask.any():
            logger.warning("Reference raster is empty?")
            return

        max_id = int(np.max(grid[valid_mask]))

        # Create LUT filled with NoData (-9999)
        # Size = max_id + padding
        lut = np.full(max_id + 100, -9999.0, dtype=np.float32)

        # 3. Fill LUT with Predictions
        # Filter preds that are within range
        valid_ids = df_preds["su_id"].values.astype(int)
        valid_probs = df_preds["xgb_prob"].values.astype(np.float32)

        # Safety Clip
        mask_safe = valid_ids < len(lut)
        safe_ids = valid_ids[mask_safe]
        safe_probs = valid_probs[mask_safe]

        lut[safe_ids] = safe_probs

        # 4. Vectorized Mapping
        # Replace grid values (IDs) with probabilities using LUT
        # Handle NoData in grid: set to 0 (dummy) then mask back
        grid_safe = grid.copy()
        grid_safe[~valid_mask] = 0
        grid_safe = grid_safe.astype(int)

        out_map = lut[grid_safe]
        out_map[~valid_mask] = -9999.0

        # 5. Write Output
        profile.update(dtype=rasterio.float32, count=1, nodata=-9999.0)

        out_path = self.results_dir / output_name
        with rasterio.open(out_path, "w", **profile) as dst:
            dst.write(out_map, 1)

        logger.info(f"    [SUCCESS] Map saved to: {out_path}")


# ==============================================================================
# MAIN
# ==============================================================================

if __name__ == "__main__":
    start_time = time.time()
    
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, choices=["dynamic", "static"], default="dynamic", help="Experiment mode")
    args = parser.parse_args()

    # Initialize
    # Config is relative to project root
    infer_engine = XGBoostInference(CONFIG_PATH, mode=args.mode)

    # Execution
    df_results = infer_engine.run_full_inference()

    # Save CSV
    csv_out = infer_engine.results_dir / f"xgb_full_inference_{args.mode}.csv"
    df_results.to_csv(csv_out, index=False)
    logger.info(f"    Predictions saved to: {csv_out}")

    # Map
    infer_engine.map_to_raster(df_results)

    logger.info(f">>> Inference Pipeline ({args.mode}) Completed in {time.time() - start_time:.2f}s")
