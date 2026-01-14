"""
Module: insar_correction.py
Location: experiments/GNNExplainer/insar_correction.py
Description:
    Phase 3: InSAR-Based Physical Correction.
    
    This script implements the "Physics-Guided Posterior Correction" logic.
    It combines the Data-Driven GCN predictions (from inference_gcn.py) with the 
    Physical InSAR deformation evidence.
    
    Logic:
        Final_Risk = max(GNN_Prob, InSAR_Risk_Factor)
        
    Where InSAR_Risk_Factor is determined by deformation velocity thresholds.
    
    Input:
    -   CSV: 'inference_results/gcn_predictions_{mode}.csv'
    -   Raster: 'InSAR_desc_...tif'
    
    Output:
    -   GeoTIFF: 'final_maps/LSM_Final_InSAR_Corrected_{mode}.tif'

Author: AI Assistant (Virgo Edition)
Date: 2026-01-10
"""

import sys
import logging
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
import yaml

# Import custom path utility
SCRIPTS_DIR = Path(__file__).resolve().parent.parent.parent / "scripts" / "00_common"
sys.path.append(str(SCRIPTS_DIR))
import path_utils

# ==============================================================================
# CONFIGURATION
# ==============================================================================
BASE_DIR = Path(__file__).resolve().parent.parent.parent
BASE_INFERENCE_DIR = Path(__file__).resolve().parent / "inference_results"
BASE_OUTPUT_DIR = Path(__file__).resolve().parent / "final_maps"

# InSAR Configuration
INSAR_FILE_PATTERN = "InSAR_desc_2024_2025*nodata.tif"
DEFORMATION_THRESHOLD_M = 0.01 # m/year (Absolute value, equivalent to 10mm/yr)

# Resolved dynamically
INFERENCE_DIR = None
OUTPUT_DIR = None

def resolve_paths(mode):
    config_path = BASE_DIR / "metadata" / f"dataset_config_{mode}.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    
    global INFERENCE_DIR, OUTPUT_DIR
    INFERENCE_DIR = path_utils.resolve_su_path(BASE_INFERENCE_DIR, config=config)
    OUTPUT_DIR = path_utils.resolve_su_path(BASE_OUTPUT_DIR, config=config)
    return config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# ==============================================================================
# CORRECTION PIPELINE
# ==============================================================================

def calculate_su_stats(su_grid, insar_grid, su_nodata, insar_nodata):
    """
    Calculates the 'Mean of Bottom 20% Values' for each Slope Unit.
    Returns a Series indexed by SU_ID.
    """
    logger.info("Computing zonal statistics (Bottom 20% Mean) for InSAR...")
    
    # 1. Flatten and DataFrame
    # Mask valid areas only
    mask = (su_grid != su_nodata) & (insar_grid != insar_nodata)
    
    valid_su = su_grid[mask]
    valid_insar = insar_grid[mask]
    
    if len(valid_su) == 0:
        logger.warning("No valid overlapping pixels between SU and InSAR found.")
        return pd.Series(dtype=np.float32)

    df_pixels = pd.DataFrame({"su_id": valid_su, "val": valid_insar})
    
    # 2. Aggregation Function
    # Logic: Take smallest (most negative) 20% values, compute mean.
    def get_bottom20_mean(series):
        # n is 20% of count, at least 1
        n = int(len(series) * 0.2)
        if n < 1: n = 1
        
        # nsmallest is efficient
        return series.nsmallest(n).mean()

    # 3. GroupBy
    # This might take a moment for huge grids, but is robust
    stats = df_pixels.groupby("su_id")["val"].apply(get_bottom20_mean)
    return stats


def main(args):
    # 0. Resolve Paths
    config = resolve_paths(args.mode)
    logger.info(f"Resolved Final Maps Directory: {OUTPUT_DIR}")

    # 1. Load GCN Predictions (CSV)
    csv_path = INFERENCE_DIR / f"gcn_predictions_{args.mode}.csv"
    if not csv_path.exists():
        logger.critical(f"Prediction CSV not found: {csv_path}")
        logger.critical("Please run 'inference_gcn.py' first.")
        return

    logger.info(f"Loading predictions: {csv_path.name}")
    df_pred = pd.read_csv(csv_path)
    
    # 2. Load InSAR Data & SU Grid
    insar_dir = BASE_DIR / "02_aligned_grid"
    try:
        insar_path = list(insar_dir.glob(INSAR_FILE_PATTERN))[0]
    except IndexError:
        logger.critical(f"InSAR file matching '{INSAR_FILE_PATTERN}' not found in {insar_dir}")
        return

    su_filename = config.get("grid", {}).get("files", {}).get("su_id")
    su_raster_path = BASE_DIR / "02_aligned_grid" / su_filename
    
    logger.info(f"Loading InSAR: {insar_path.name}")
    logger.info(f"Loading SU Grid: {su_raster_path.name}")

    with rasterio.open(su_raster_path) as src_su, rasterio.open(insar_path) as src_insar:
        su_grid = src_su.read(1)
        insar_grid = src_insar.read(1)
        profile = src_su.profile
        su_nodata = src_su.nodata
        insar_nodata = src_insar.nodata
        
        if su_grid.shape != insar_grid.shape:
            logger.critical("Shape mismatch between SU raster and InSAR raster.")
            return

    # 3. Calculate SU InSAR Statistics
    # Metric: Mean of Bottom 20% (Most Negative) values -> Represents "Instability Rate"
    su_stats = calculate_su_stats(su_grid, insar_grid, su_nodata, insar_nodata)
    
    # 4. Merge Stats with Predictions
    # su_stats index is SU_ID (float/int). Map to df_pred['su_id']
    df_pred["insar_rate"] = df_pred["su_id"].map(su_stats)
    
    # Fill NaN InSAR rates (SUs with no InSAR data)
    # If no data, we assume 0 (Neutral) to avoid triggering rules
    df_pred["insar_rate"] = df_pred["insar_rate"].fillna(0.0)
    
    # 5. Apply Correction Logic
    # Initialize final_prob with original
    df_pred["final_prob"] = df_pred["prob"]
    df_pred["correction_type"] = "None"

    # Define Thresholds (m/year)
    # -15 mm/yr = -0.015
    # -10 mm/yr = -0.010
    # +20 mm/yr =  0.020
    
    THRESH_UNSTABLE = -0.015
    THRESH_STABLE_LOWER = -0.010
    THRESH_STABLE_UPPER = 0.020
    
    # Rule 1: Force Activation (InSAR Evidence of Failure)
    # Condition: Instability Rate < -15 mm/yr
    mask_activation = df_pred["insar_rate"] < THRESH_UNSTABLE
    df_pred.loc[mask_activation, "final_prob"] = 0.9
    df_pred.loc[mask_activation, "correction_type"] = "Activated"
    
    # Rule 2: Suppression (False Positive Correction)
    # Condition: Stable InSAR (-10mm < Rate < 20mm) AND High Model Risk (> 0.75)
    mask_suppression = (
        (df_pred["insar_rate"] > THRESH_STABLE_LOWER) & 
        (df_pred["insar_rate"] < THRESH_STABLE_UPPER) & 
        (df_pred["prob"] > 0.75)
    )
    # Set to 0.7 (Downgrade)
    df_pred.loc[mask_suppression, "final_prob"] = 0.7
    df_pred.loc[mask_suppression, "correction_type"] = "Suppressed"

    # Stats
    n_activated = mask_activation.sum()
    n_suppressed = mask_suppression.sum()
    logger.info(f"Correction Summary:")
    logger.info(f"  - Activated (Rate < -15mm/yr): {n_activated} SUs")
    logger.info(f"  - Suppressed (Stable & High Risk): {n_suppressed} SUs")
    
    # 6. Save Updated CSV
    out_csv = OUTPUT_DIR / f"predictions_corrected_{args.mode}.csv"
    df_pred.to_csv(out_csv, index=False)
    logger.info(f"Saved corrected tabular data: {out_csv}")

    # 7. Generate Final Raster
    logger.info("Generating Final Corrected Raster...")
    
    # Prepare LUT
    max_id = int(np.nanmax(su_grid)) if su_nodata is None else int(np.nanmax(su_grid[su_grid != su_nodata]))
    # Handle case where max_id is small or data is empty
    if max_id < 0: max_id = 0
    
    lut = np.full(max_id + 10000, -9999.0, dtype=np.float32)
    
    ids = df_pred["su_id"].values.astype(int)
    probs = df_pred["final_prob"].values.astype(np.float32)
    
    valid_ids = ids[ids < len(lut)]
    valid_probs = probs[ids < len(lut)]
    
    lut[valid_ids] = valid_probs
    
    # Map
    if su_nodata is not None:
        mask_nodata = su_grid == su_nodata
        grid_indices = su_grid.copy()
        grid_indices[mask_nodata] = 0
        grid_indices = grid_indices.astype(int)
    else:
        grid_indices = su_grid.astype(int)
        mask_nodata = np.zeros_like(su_grid, dtype=bool)

    # Safe lookup
    # Clip indices just in case
    grid_indices = np.clip(grid_indices, 0, len(lut) - 1)
    
    final_map = lut[grid_indices]
    final_map[mask_nodata] = -9999.0
    
    # Save
    profile.update(dtype=rasterio.float32, nodata=-9999.0)
    out_tif = OUTPUT_DIR / f"LSM_Final_InSAR_Corrected_{args.mode}.tif"
    
    with rasterio.open(out_tif, "w", **profile) as dst:
        dst.write(final_map, 1)
        
    logger.info(f"Saved Final Map: {out_tif}")

# ==============================================================================
# ENTRY POINT
# ==============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="dynamic")
    args = parser.parse_args()
    main(args)