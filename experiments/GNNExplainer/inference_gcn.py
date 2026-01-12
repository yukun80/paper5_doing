"""
Module: inference_gcn.py
Location: experiments/GNNExplainer/inference_gcn.py
Description:
    Full-Region Inference Script for the GNNExplainer GCN Model.
    
    This script performs the critical step between Training and Physical Correction.
    It takes the trained GCN model and generates a "Pure Data-Driven" Landslide Susceptibility Map (LSM)
    for the entire study area (all Slope Units).
    
    Output:
    -   CSV: 'inference_results/gcn_predictions_{mode}.csv' (Tabular probs)
    -   GeoTIFF: 'inference_results/LSM_GCN_Raw_{mode}.tif' (Spatial map)

Author: AI Assistant (Virgo Edition)
Date: 2026-01-10
"""

import sys
import logging
import argparse
from pathlib import Path

import torch
import numpy as np
import pandas as pd
import rasterio

# Add current directory to path
CURRENT_DIR = Path(__file__).resolve().parent
sys.path.append(str(CURRENT_DIR))

import models
from adapter import LandslideDataAdapter

# Import custom path utility
SCRIPTS_DIR = Path(__file__).resolve().parent.parent.parent / "scripts" / "00_common"
sys.path.append(str(SCRIPTS_DIR))
import path_utils
import yaml

# ==============================================================================
# CONFIGURATION
# ==============================================================================
BASE_DIR = Path(__file__).resolve().parent.parent.parent
BASE_INFERENCE_DIR = Path(__file__).resolve().parent / "inference_results"
BASE_CHECKPOINT_DIR = Path(__file__).resolve().parent / "checkpoints"
BASE_LOG_DIR = Path(__file__).resolve().parent / "logs"

# Resolved dynamically
INFERENCE_DIR = None
CHECKPOINT_DIR = None

def resolve_paths(mode):
    config_path = BASE_DIR / "metadata" / f"dataset_config_{mode}.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    
    global INFERENCE_DIR, CHECKPOINT_DIR
    INFERENCE_DIR = path_utils.resolve_su_path(BASE_INFERENCE_DIR, config=config)
    CHECKPOINT_DIR = path_utils.resolve_su_path(BASE_CHECKPOINT_DIR, config=config)
    return config_path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(BASE_LOG_DIR / "inference.log", mode="w", encoding="utf-8"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# ==============================================================================
# INFERENCE PIPELINE
# ==============================================================================

def map_predictions_to_raster(df_pred: pd.DataFrame, ref_raster_path: Path, output_path: Path):
    """
    Maps tabular predictions back to the spatial grid using Vectorized Look-Up Table (LUT).
    """
    logger.info(f"Mapping results to Raster: {ref_raster_path.name}")

    with rasterio.open(ref_raster_path) as src:
        su_grid = src.read(1)
        profile = src.profile
        nodata_val = src.nodata

    # Update profile for float output
    profile.update(dtype=rasterio.float32, count=1, nodata=-9999.0)

    # Create LUT
    max_id = int(np.max(su_grid))
    # Handle potential huge IDs by clipping or checking
    # Assuming IDs are reasonable (< 100,000)
    lut = np.full(max_id + 1000, -9999.0, dtype=np.float32)

    ids = df_pred["su_id"].values.astype(int)
    probs = df_pred["gcn_prob"].values.astype(np.float32)
    
    # Clip IDs to fit LUT
    valid_mask = ids < len(lut)
    lut[ids[valid_mask]] = probs[valid_mask]

    # Map
    if nodata_val is not None:
        mask = su_grid == nodata_val
        grid_indices = su_grid.copy()
        grid_indices[mask] = 0
        grid_indices = grid_indices.astype(int)
    else:
        grid_indices = su_grid.astype(int)
        mask = np.zeros_like(su_grid, dtype=bool)

    lsm_map = lut[grid_indices]
    lsm_map[mask] = -9999.0

    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(lsm_map, 1)
    
    logger.info(f"Saved GeoTIFF: {output_path}")


def main(args):
    # 0. Resolve Paths
    config_path = resolve_paths(args.mode)
    logger.info(f"Results will be saved to: {INFERENCE_DIR}")

    # 1. Load Data
    logger.info("Loading Data...")
    adapter = LandslideDataAdapter(base_dir=BASE_DIR, mode=args.mode, config_path=config_path)
    adapter.load_data()
    data = adapter.get_processed_data()
    
    # Extract data components
    feat = data["feat"]
    adj = data["adj"]
    label = data["label"]
    node_ids = data["node_ids"]
    test_mask = data["test_mask"]
    
    input_dim = feat.size(2)
    num_classes = 2

    # 2. Load Model
    checkpoint_path = CHECKPOINT_DIR / f"landslide_model_{args.mode}_best.pth.tar"
    if not checkpoint_path.exists():
        logger.error(f"Checkpoint not found: {checkpoint_path}")
        return

    logger.info(f"Loading checkpoint: {checkpoint_path}")
    
    # Robust device handling
    device = torch.device('cuda' if torch.cuda.is_available() and args.gpu else 'cpu')
    logger.info(f"Using device: {device}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # ... (Rest of model loading)
    model = models.GcnEncoderNode(
        input_dim=input_dim,
        hidden_dim=args.hidden_dim,
        embedding_dim=args.output_dim,
        label_dim=num_classes,
        num_layers=args.num_layers,
        bn=args.bn,
        dropout=0.0,
        args=args
    )
    model.load_state_dict(checkpoint["model_state"])
    
    model = model.to(device)
    feat = feat.to(device)
    adj = adj.to(device)
    model.eval()

    # 3. Predict
    with torch.no_grad():
        ypred, _ = model(feat, adj)
        probs = torch.softmax(ypred[0], dim=1)[:, 1].cpu().numpy()
        preds = torch.argmax(ypred[0], dim=1).cpu().numpy()

    # 4. Save CSV
    out_df = pd.DataFrame({
        "su_id": node_ids,
        "label": label[0].cpu().numpy(),
        "prob": probs, # Column name 'prob' matches convention
        "gcn_prob": probs, # Keep 'gcn_prob' for compatibility with map function if needed, or update map function
        "pred": preds,
        "split": ["test" if mask else "train" for mask in test_mask]
    })
    
    out_csv = INFERENCE_DIR / f"gcn_predictions_{args.mode}.csv"
    out_df.to_csv(out_csv, index=False)
    logger.info(f"Full predictions saved to: {out_csv}")

    # 5. Map to Raster
    # Resolve SU raster path dynamically
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    su_filename = config.get("grid", {}).get("files", {}).get("su_id")
    su_raster_path = BASE_DIR / "02_aligned_grid" / su_filename
    
    if not su_raster_path.exists():
        logger.warning(f"SU Raster not found at {su_raster_path}, skipping map generation.")
    else:
        tif_path = INFERENCE_DIR / f"LSM_GCN_Raw_{args.mode}.tif"
        map_predictions_to_raster(out_df, su_raster_path, tif_path)
    
    logger.info("Inference Complete.")

# ==============================================================================
# ENTRY POINT
# ==============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="dynamic")
    parser.add_argument("--gpu", action="store_true", default=True)
    
    # Model Args (Must match training)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--output-dim", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument("--bn", action="store_true", default=False)
    parser.add_argument("--method", type=str, default="base")
    parser.add_argument("--bias", action="store_true", default=True)

    args = parser.parse_args()
    main(args)
