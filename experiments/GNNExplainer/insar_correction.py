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
    
    # 2. Load InSAR Data
    insar_dir = BASE_DIR / "02_aligned_grid"
    try:
        insar_path = list(insar_dir.glob(INSAR_FILE_PATTERN))[0]
    except IndexError:
        logger.critical(f"InSAR file matching '{INSAR_FILE_PATTERN}' not found in {insar_dir}")
        return

    logger.info(f"Loading InSAR Data: {insar_path.name}")
    
    # 3. Spatial Mapping setup
    # DYNAMICALLY get SU grid filename from config
    su_filename = config.get("grid", {}).get("files", {}).get("su_id")
    su_raster_path = BASE_DIR / "02_aligned_grid" / su_filename
    
    with rasterio.open(su_raster_path) as src_su, rasterio.open(insar_path) as src_insar:
        su_grid = src_su.read(1)
        insar_grid = src_insar.read(1)
        profile = src_su.profile
        
        # Check alignment
        if su_grid.shape != insar_grid.shape:
            logger.critical("Shape mismatch between SU raster and InSAR raster.")
            return

    # 4. Raster-based Reconstruction (GCN Map)
    # Reconstruct the GCN probability map from CSV for pixel-wise comparison
    gcn_map = np.full_like(su_grid, -9999.0, dtype=np.float32)
    
    # Vectorized LUT mapping
    max_id = int(su_grid.max())
    # Safe LUT size
    lut = np.full(max_id + 1000, -9999.0, dtype=np.float32)
    
    ids = df_pred["su_id"].values.astype(int)
    # Column name fix: use 'prob' instead of 'gcn_prob'
    probs = df_pred["prob"].values.astype(np.float32)
    
    # Safety clip
    valid_mask = ids < len(lut)
    lut[ids[valid_mask]] = probs[valid_mask]
    
    # Apply LUT
    mask_nodata = su_grid == -9999
    # Use 0 for safe indexing of nodata (will be masked later)
    su_indices = su_grid.copy()
    su_indices[mask_nodata] = 0 
    su_indices = su_indices.astype(int)
    
    gcn_map = lut[su_indices]
    gcn_map[mask_nodata] = -9999.0
    
    # 5. Apply Correction Logic
    # InSAR Mask: Absolute velocity > threshold
    insar_nodata = -9999.0 
    insar_mask_valid = insar_grid != insar_nodata
    
    high_deform_mask = np.zeros_like(insar_grid, dtype=bool)
    high_deform_mask[insar_mask_valid] = np.abs(insar_grid[insar_mask_valid]) > DEFORMATION_THRESHOLD_M
    
    logger.info(f"Identified {np.sum(high_deform_mask)} pixels with high deformation (> {DEFORMATION_THRESHOLD_M} m/yr).")
    
    # Final Map
    final_map = gcn_map.copy()
    
    # Correction: Force High Risk (e.g., 0.95) where InSAR is high
    # Only apply where we have valid GCN prediction (to avoid filling background)
    correction_mask = high_deform_mask & (gcn_map != -9999.0)
    final_map[correction_mask] = np.maximum(final_map[correction_mask], 0.95)
    
    # 6. Save Outputs
    # Save GCN Map (Pre-Correction) - Optional, as inference_gcn.py already saves it
    # But saving a copy here confirms what was used
    profile.update(dtype=rasterio.float32, nodata=-9999.0)
    
    # Save Final Map (Post-Correction)
    out_tif = OUTPUT_DIR / f"LSM_Final_InSAR_Corrected_{args.mode}.tif"
    with rasterio.open(out_tif, "w", **profile) as dst:
        dst.write(final_map, 1)
        
    logger.info(f"Saved corrected map to: {out_tif}")

# ==============================================================================
# ENTRY POINT
# ==============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="dynamic")
    args = parser.parse_args()
    main(args)