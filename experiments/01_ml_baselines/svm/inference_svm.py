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

# Paths
CURRENT_DIR = Path(__file__).resolve().parent
BASE_DIR = CURRENT_DIR.parent.parent.parent
DATA_PATH = BASE_DIR / "04_tabular_SU" / "tabular_dataset.parquet"
SU_ID_RASTER = BASE_DIR / "02_aligned_grid" / "su_a50000_c03_geo.tif"
MODEL_PATH = CURRENT_DIR / "models" / "svm_pipeline.joblib"
OUTPUT_DIR = CURRENT_DIR / "inference_results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Add project root for utilities
if str(BASE_DIR) not in sys.path:
    sys.path.append(str(BASE_DIR))
sys.path.append(str(BASE_DIR / "experiments" / "01_ml_baselines"))
from ml_utils import load_and_split_data, save_predictions

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

def map_to_raster(su_ids, probs, output_path):
    logger.info(f"Mapping to raster: {output_path.name}")
    with rasterio.open(SU_ID_RASTER) as src:
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

    logger.info(f">>> SVM Inference Starting... | Mode: {mode}")
    
    # Dynamic Paths
    data_path = BASE_DIR / "04_tabular_SU" / f"tabular_dataset_{mode}.parquet"
    model_path = CURRENT_DIR / "models" / f"svm_pipeline_{mode}.joblib"
    
    # 1. Load Data (Load ALL data for mapping)
    df_all = pd.read_parquet(data_path)
    
    # Identify Features (Must match training prefixes)
    meta_cols = ["su_id", "label", "split", "ratio", "slide_pixels", "total_pixels", "centroid_x", "centroid_y", "geometry", "train_sample_mask"]
    feature_cols = [c for c in df_all.columns if c not in meta_cols]
    
    # 2. Load Model
    if not model_path.exists():
        logger.error(f"Model not found at {model_path}")
        return
    pipeline = joblib.load(model_path)
    
    # 3. Predict
    X = df_all[feature_cols].values
    logger.info(f"Predicting for {len(X)} Slope Units...")
    probs = pipeline.predict_proba(X)[:, 1]
    
    # 4. Save CSV
    save_predictions(df_all, probs, OUTPUT_DIR / f"svm_predictions_all_{mode}.csv")
    
    # 5. Save Raster
    map_to_raster(df_all.index.values, probs, OUTPUT_DIR / f"LSM_SVM_Prob_{mode}.tif")
    
    logger.info("✔ SVM Inference Complete.")

if __name__ == "__main__":
    main()
