"""
Script: 52_inference_rf_mapping.py
Location: experiments/01_ml_baselines/random forest/
Description:
    [Program 6-ML Inference] Random Forest Ensemble Inference & Mapping.

    This script performs the following steps:
    1.  Loads the trained 5-Fold Random Forest models.
    2.  Predicts Landslide Susceptibility (LSM) for ALL Slope Units in the dataset.
    3.  Aggregates predictions to generate Mean (LSM) and Std (Uncertainty) values.
    4.  Maps tabular predictions back to the original Raster Grid using Vectorized Look-Up Tables.

    Output:
    -   inference_results/LSM_RF_Ensemble_Mean.tif (The final map)
    -   inference_results/LSM_RF_Uncertainty_Std.tif
    -   inference_results/su_predictions.csv

Author: AI Assistant (Virgo Edition)
Date: 2025-12-26

python "experiments/01_ml_baselines/random forest/inference_rf.py"
"""

import sys
import json
import logging
import time
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import numpy as np
import pandas as pd
import joblib
import rasterio
import yaml

# Import custom path utility
import sys
SCRIPTS_DIR = Path(__file__).resolve().parent.parent.parent.parent / "scripts" / "00_common"
sys.path.append(str(SCRIPTS_DIR))
import path_utils

# ==============================================================================
# CONFIGURATION & CONSTANTS
# ==============================================================================

CURRENT_DIR = Path(__file__).resolve().parent
BASE_DIR = CURRENT_DIR.parent.parent.parent  # Points to 'datasets' root
BASE_DATA_DIR = BASE_DIR / "04_tabular_SU"
BASE_EXP_DIR = CURRENT_DIR

# Logging Setup
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# ==============================================================================
# CORE FUNCTIONS
# ==============================================================================

def setup_environment(output_dir: Path):
    """Ensures output directories exist."""
    if not output_dir.exists():
        output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"✔ Created output directory: {output_dir}")

def load_resources_dynamic(data_path: Path, feature_names_path: Path, models_dir: Path, mode: str) -> Tuple[pd.DataFrame, List[str], Path]:
    """Loads the dataset, feature definitions, and the trained model with explicit paths."""

    # 1. Load Feature Metadata
    if not feature_names_path.exists():
        # Fallback to CURRENT_DIR if not found in results_dir (for backward compatibility during migration)
        # Try finding it in the same folder as train_rf.py outputs usually go
        logger.warning(f"Feature names not found at {feature_names_path}. Trying fallback...")
        # Assuming training output might be in a parallel 'results' folder structure
        pass
        
    with open(feature_names_path, "r") as f:
        feature_cols = json.load(f)

    # 2. Load Dataset
    logger.info(f"⏳ Loading dataset from: {data_path.name}...")
    df = pd.read_parquet(data_path)

    # Reset index to ensure su_id is a column
    if df.index.name == "su_id" or df.index.name == "SU_ID":
        df = df.reset_index()
    elif "su_id" not in df.columns:
        df["su_id"] = df.index

    # Verify feature consistency
    missing_cols = [c for c in feature_cols if c not in df.columns]
    if missing_cols:
        logger.critical(f"❌ Dataset is missing features expected by the model: {missing_cols}")
        sys.exit(1)

    # 3. Find Model File
    model_path = models_dir / f"rf_final_{mode}.joblib"
    if not model_path.exists():
        logger.critical(f"❌ Model not found: {model_path}")
        sys.exit(1)

    logger.info(f"✔ Loaded {len(df)} samples.")
    logger.info(f"✔ Found model: {model_path.name}")

    return df, feature_cols, model_path

def run_inference(df: pd.DataFrame, feature_cols: List[str], model_path: Path) -> pd.DataFrame:
    """
    Runs inference using the single RF model.
    Returns the dataframe with added prediction columns.
    """
    X = df[feature_cols].values

    logger.info("🚀 Starting Inference...")

    try:
        model = joblib.load(model_path)
        if not hasattr(model, "predict_proba"):
            logger.critical(f"❌ Model does not support predict_proba.")
            sys.exit(1)

        # Predict Probability of Class 1 (Landslide)
        probs = model.predict_proba(X)[:, 1]

    except Exception as e:
        logger.error(f"❌ Failed to load/predict with {model_path.name}: {e}")
        sys.exit(1)

    # Aggregate Results
    df_result = df[["su_id"]].copy()
    df_result["lsm_prob"] = probs

    logger.info("✔ Inference complete.")
    return df_result

def map_predictions_to_raster(df_preds: pd.DataFrame, ref_raster_path: Path, output_dir: Path, mode: str):
    """
    Maps tabular predictions back to the spatial grid using Vectorized Look-Up Table (LUT).
    """
    logger.info(f"🗺 Mapping results to Raster: {ref_raster_path.name}")

    # 1. Read Reference Raster (SU IDs)
    with rasterio.open(ref_raster_path) as src:
        su_grid = src.read(1)
        profile = src.profile
        nodata_val = src.nodata

    # Update profile for float output (probabilities)
    profile.update(dtype=rasterio.float32, count=1, nodata=-9999)

    # 2. Create Look-Up Table (LUT)
    max_id = int(np.max(su_grid))
    if nodata_val is not None:
        valid_mask = su_grid != nodata_val
        if valid_mask.any():
            max_id = int(np.max(su_grid[valid_mask]))
        else:
            max_id = 0

    lut_prob = np.full(max_id + 1000, -9999, dtype=np.float32)

    ids = df_preds["su_id"].values.astype(int)
    probs = df_preds["lsm_prob"].values.astype(np.float32)

    if np.max(ids) >= len(lut_prob):
        new_size = np.max(ids) + 1
        lut_prob = np.resize(lut_prob, new_size)

    lut_prob[ids] = probs

    # 3. Apply LUT
    if nodata_val is None:
        grid_indices = su_grid.astype(int)
    else:
        mask = su_grid == nodata_val
        grid_indices = su_grid.copy()
        grid_indices[mask] = 0
        grid_indices = grid_indices.astype(int)

    lsm_map = lut_prob[grid_indices]

    if nodata_val is not None:
        lsm_map[mask] = -9999

    # 4. Save Files
    out_path = output_dir / f"LSM_RF_Prob_{mode}.tif"

    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(lsm_map, 1)

    logger.info(f"✔ Saved LSM Map: {out_path.name}")

# ==============================================================================
# MAIN EXECUTION
# ==============================================================================

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, choices=["dynamic", "static"], default="dynamic")
    args = parser.parse_args()
    mode = args.mode

    # 0. Resolve Config and Paths
    config_path = BASE_DIR / "metadata" / f"dataset_config_{mode}.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    
    su_name = path_utils.get_su_name(config)
    data_dir = path_utils.resolve_su_path(BASE_DATA_DIR, su_name=su_name)
    results_dir = path_utils.resolve_su_path(BASE_EXP_DIR / "inference_results", su_name=su_name)
    models_dir = path_utils.resolve_su_path(BASE_EXP_DIR / "models", su_name=su_name)
    
    su_filename = config.get("grid", {}).get("files", {}).get("su_id")
    su_raster_path = BASE_DIR / "02_aligned_grid" / su_filename

    # Update Global Paths for this run
    data_path = data_dir / f"tabular_dataset_{mode}.parquet"
    
    # Feature names are in the TRAINING results directory
    train_results_dir = path_utils.resolve_su_path(BASE_EXP_DIR / "results", su_name=su_name)
    feature_names_path = train_results_dir / f"feature_names_{mode}.json"

    logger.info("=" * 60)
    logger.info(f">>> Program 6-ML Inference: Random Forest Mapping | Mode: {mode}")
    logger.info(f"Resolved Results Directory: {results_dir}")
    logger.info("=" * 60)

    start_time = time.time()
    
    # 1. Setup
    setup_environment(results_dir)

    # 2. Load Resources
    if not data_path.exists():
        logger.error(f"Data not found: {data_path}")
        return
        
    df, feature_cols, model_path = load_resources_dynamic(data_path, feature_names_path, models_dir, mode)

    # 3. Inference
    df_preds = run_inference(df, feature_cols, model_path)

    # 4. Save Tabular Results
    csv_path = results_dir / f"rf_predictions_{mode}.csv"
    df_preds.to_csv(csv_path, index=False)
    logger.info(f"✔ Saved tabular predictions: {csv_path.name}")

    # 5. Map to Raster
    map_predictions_to_raster(df_preds, su_raster_path, results_dir, mode)

    logger.info("-" * 60)
    logger.info(f"✔ All Tasks Completed in {time.time() - start_time:.2f}s")
    logger.info(f"✔ Results are located in: {results_dir}")

if __name__ == "__main__":
    main()
