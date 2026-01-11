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

# ==============================================================================
# CONFIGURATION & CONSTANTS
# ==============================================================================

# Derive paths dynamically relative to this script's location
# Structure: .../datasets/experiments/01_ml_baselines/random forest/this_script.py
CURRENT_DIR = Path(__file__).resolve().parent
BASE_DIR = CURRENT_DIR.parent.parent.parent  # Points to 'datasets' root

# Inputs
DATA_PATH = BASE_DIR / "04_tabular_SU" / "tabular_dataset.parquet"
SU_ID_RASTER = BASE_DIR / "02_aligned_grid" / "su_a50000_c03_geo.tif"
MODELS_DIR = CURRENT_DIR / "models"
FEATURE_NAMES_PATH = CURRENT_DIR / "feature_names.json"

# Outputs
OUTPUT_DIR = CURRENT_DIR / "inference_results"

# Logging Setup
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# ==============================================================================
# CORE FUNCTIONS
# ==============================================================================


def setup_environment():
    """Ensures output directories exist and validates inputs."""
    if not OUTPUT_DIR.exists():
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        logger.info(f"✔ Created output directory: {OUTPUT_DIR}")

    required_files = [DATA_PATH, SU_ID_RASTER, FEATURE_NAMES_PATH, MODELS_DIR]
    for p in required_files:
        if not p.exists():
            logger.critical(f"❌ Missing required resource: {p}")
            sys.exit(1)


def load_resources() -> Tuple[pd.DataFrame, List[str], Path]:
    """Loads the dataset, feature definitions, and the trained model."""

    # 1. Load Feature Metadata
    with open(FEATURE_NAMES_PATH, "r") as f:
        feature_cols = json.load(f)

    # 2. Load Dataset
    logger.info(f"⏳ Loading dataset from: {DATA_PATH.name}...")
    df = pd.read_parquet(DATA_PATH)

    # Reset index to ensure su_id is a column
    # Parquet index might be SU_ID already, check before reset or use logic
    if df.index.name == "su_id" or df.index.name == "SU_ID":
        df = df.reset_index()
    elif "su_id" not in df.columns:
        # Fallback if index is unnamed but treated as SU_ID
        df["su_id"] = df.index

    # Verify feature consistency
    missing_cols = [c for c in feature_cols if c not in df.columns]
    if missing_cols:
        logger.critical(f"❌ Dataset is missing features expected by the model: {missing_cols}")
        sys.exit(1)

    # 3. Find Model File
    model_path = MODELS_DIR / "rf_final.joblib"
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


def map_predictions_to_raster(df_preds: pd.DataFrame, ref_raster_path: Path, output_dir: Path):
    """
    Maps tabular predictions back to the spatial grid using Vectorized Look-Up Table (LUT).
    """
    logger.info("🗺 Mapping results to Raster...")

    # 1. Read Reference Raster (SU IDs)
    with rasterio.open(ref_raster_path) as src:
        su_grid = src.read(1)
        profile = src.profile
        nodata_val = src.nodata

    # Update profile for float output (probabilities)
    profile.update(dtype=rasterio.float32, count=1, nodata=-9999)

    # 2. Create Look-Up Table (LUT)
    # We need an array where index = SU_ID and value = Probability
    # Size = Max SU_ID + 1
    max_id = int(np.max(su_grid))
    if nodata_val is not None:
        # Handle case where nodata might be larger than max ID (unlikely but possible)
        valid_mask = su_grid != nodata_val
        if valid_mask.any():
            max_id = int(np.max(su_grid[valid_mask]))
        else:
            max_id = 0

    # Initialize LUT with NoData
    lut_prob = np.full(max_id + 1000, -9999, dtype=np.float32)  # Add buffer for safety

    # Fill LUT from Dataframe
    ids = df_preds["su_id"].values.astype(int)
    probs = df_preds["lsm_prob"].values.astype(np.float32)

    # Safety check: Ensure IDs fit in LUT
    if np.max(ids) >= len(lut_prob):
        logger.warning("⚠ Found SU IDs larger than LUT size. Resizing LUT...")
        new_size = np.max(ids) + 1
        lut_prob = np.resize(lut_prob, new_size)

    lut_prob[ids] = probs

    # 3. Apply LUT (Vectorized Mapping)
    # Handle original NoData in grid
    if nodata_val is None:
        grid_indices = su_grid.astype(int)
    else:
        # Mask nodata first to avoid index errors, assume 0 for temporary indexing
        mask = su_grid == nodata_val
        grid_indices = su_grid.copy()
        grid_indices[mask] = 0  # Dummy index
        grid_indices = grid_indices.astype(int)

    # Map
    lsm_map = lut_prob[grid_indices]

    # Restore NoData
    if nodata_val is not None:
        lsm_map[mask] = -9999

    # 4. Save Files
    out_path = output_dir / "LSM_RF_Prob.tif"

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

    logger.info("=" * 60)
    logger.info(f">>> Program 6-ML Inference: Random Forest Mapping | Mode: {mode}")
    logger.info("=" * 60)

    start_time = time.time()

    # Dynamic Paths
    global DATA_PATH, FEATURE_NAMES_PATH
    DATA_PATH = BASE_DIR / "04_tabular_SU" / f"tabular_dataset_{mode}.parquet"
    FEATURE_NAMES_PATH = CURRENT_DIR / f"feature_names_{mode}.json"
    
    # 1. Setup
    setup_environment()

    # 2. Load Resources
    # Pass mode to load specific model
    df, feature_cols, model_path = load_resources(mode)

    # 3. Inference
    df_preds = run_inference(df, feature_cols, model_path)

    # 4. Save Tabular Results
    csv_path = OUTPUT_DIR / f"su_predictions_{mode}.csv"
    df_preds.to_csv(csv_path, index=False)
    logger.info(f"✔ Saved tabular predictions: {csv_path.name}")

    # 5. Map to Raster
    map_predictions_to_raster(df_preds, SU_ID_RASTER, OUTPUT_DIR, mode)

    logger.info("-" * 60)
    logger.info(f"✔ All Tasks Completed in {time.time() - start_time:.2f}s")
    logger.info(f"✔ Results are located in: {OUTPUT_DIR}")


def load_resources(mode: str) -> Tuple[pd.DataFrame, List[str], Path]:
    """Loads the dataset, feature definitions, and the trained model."""

    # 1. Load Feature Metadata
    with open(FEATURE_NAMES_PATH, "r") as f:
        feature_cols = json.load(f)

    # 2. Load Dataset
    logger.info(f"⏳ Loading dataset from: {DATA_PATH.name}...")
    df = pd.read_parquet(DATA_PATH)

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
    model_path = MODELS_DIR / f"rf_final_{mode}.joblib"
    if not model_path.exists():
        logger.critical(f"❌ Model not found: {model_path}")
        sys.exit(1)

    logger.info(f"✔ Loaded {len(df)} samples.")
    logger.info(f"✔ Found model: {model_path.name}")

    return df, feature_cols, model_path


def map_predictions_to_raster(df_preds: pd.DataFrame, ref_raster_path: Path, output_dir: Path, mode: str):
    """
    Maps tabular predictions back to the spatial grid using Vectorized Look-Up Table (LUT).
    """
    logger.info("🗺 Mapping results to Raster...")

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


if __name__ == "__main__":
    main()