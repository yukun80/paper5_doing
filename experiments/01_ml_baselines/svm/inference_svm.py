"""
Script: inference_svm.py
Location: experiments/01_ml_baselines/svm/
Description:
    [Program 6-ML Inference] SVM Inference & Mapping.
    
    Generates the Landslide Susceptibility Map (LSM) for SVM.

python "experiments/01_ml_baselines/svm/inference_svm.py"
"""

import sys
import json
import logging
import time
from pathlib import Path
from typing import List, Tuple

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

# Paths
CURRENT_DIR = Path(__file__).resolve().parent
BASE_DIR = CURRENT_DIR.parent.parent.parent
BASE_DATA_DIR = BASE_DIR / "04_tabular_SU"
BASE_EXP_DIR = CURRENT_DIR

# Placeholders resolved in main()
DATA_PATH = None
SU_ID_RASTER = None
MODEL_PATH = None
OUTPUT_DIR = None

# Add project root for utilities
if str(BASE_DIR) not in sys.path:
    sys.path.append(str(BASE_DIR))
sys.path.append(str(BASE_DIR / "experiments" / "01_ml_baselines"))
from ml_utils import load_and_split_data, save_predictions

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

def map_to_raster(su_ids, probs, su_raster_path, output_path):
    logger.info(f"Mapping to raster: {output_path.name}")
    with rasterio.open(su_raster_path) as src:
        grid = src.read(1)
        profile = src.profile
        nodata = src.nodata

    max_id = int(np.max(grid))
    lut = np.full(max_id + 1000, -9999.0, dtype=np.float32)
    
    valid_ids = su_ids.astype(int)
    lut[valid_ids] = probs.astype(np.float32)

    mask = grid == nodata
    grid_safe = grid.copy()
    grid_safe[mask] = 0
    
    out_map = lut[grid_safe.astype(int)]
    out_map[mask] = -9999

    profile.update(dtype=rasterio.float32, nodata=-9999, count=1)
    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(out_map, 1)

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

    logger.info(f">>> SVM Inference Starting... | Mode: {mode}")
    logger.info(f"Resolved Results Directory: {results_dir}")
    
    # Dynamic Paths
    data_path = data_dir / f"tabular_dataset_{mode}.parquet"
    model_path = models_dir / f"svm_final_{mode}.joblib" # Use consistent naming from train_svm.py
    
    # 1. Load Data (Load ALL data for mapping)
    if not data_path.exists():
        logger.error(f"Data file not found: {data_path}")
        return
    df_all = pd.read_parquet(data_path)
    
    # Identify Features (Must match training prefixes)
    feat_cols = [c for c in df_all.columns if c.startswith("static_env_") or c.startswith("dynamic_forcing_")]
    
    # 2. Load Model
    if not model_path.exists():
        logger.error(f"Model not found at {model_path}")
        return
    pipeline = joblib.load(model_path)
    
    # 3. Predict
    X = df_all[feat_cols].values
    logger.info(f"Predicting for {len(X)} Slope Units...")
    
    # SVM might not have predict_proba depending on kernel/config, 
    # but our train script used decision_function or we can use decision_function here.
    # Actually train_svm.py uses predict_proba if probability=True, or decision_function.
    # Let's check train_svm.py again. (Wait, let's use decision_function for robustness)
    try:
        probs = pipeline.predict_proba(X)[:, 1]
    except:
        probs = pipeline.decision_function(X)
    
    # 4. Save CSV
    save_predictions(df_all, probs, results_dir / f"svm_predictions_all_{mode}.csv")
    
    # 5. Save Raster
    map_to_raster(df_all.index.values, probs, su_raster_path, results_dir / f"LSM_SVM_Prob_{mode}.tif")
    
    logger.info("✔ SVM Inference Complete.")

if __name__ == "__main__":
    main()
