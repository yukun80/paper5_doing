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

# ==============================================================================
# CONFIGURATION
# ==============================================================================
BASE_DIR = CURRENT_DIR.parent.parent
OUTPUT_DIR = CURRENT_DIR / "inference_results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(CURRENT_DIR / "logs" / "inference.log", mode="w", encoding="utf-8"),
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
    # 1. Load Data
    logger.info("Loading Data...")
    adapter = LandslideDataAdapter(base_dir=BASE_DIR, mode=args.mode)
    adapter.load_data()
    data = adapter.get_processed_data()
    
    # 2. Load Model
    ckpt_path = CURRENT_DIR / "checkpoints" / f"landslide_model_{args.mode}_best.pth.tar"
    if not ckpt_path.exists():
        logger.critical(f"Checkpoint not found: {ckpt_path}")
        return

    logger.info(f"Loading checkpoint: {ckpt_path.name}")
    # Fix for PyTorch 2.6+: explicit weights_only=False since we load full objects
    checkpoint = torch.load(ckpt_path, weights_only=False)
    
    input_dim = data["feat"].size(2)
    
    model = models.GcnEncoderNode(
        input_dim=input_dim,
        hidden_dim=args.hidden_dim,
        embedding_dim=args.output_dim,
        label_dim=2,
        num_layers=args.num_layers,
        bn=args.bn,
        dropout=0.0,
        args=args
    )
    model.load_state_dict(checkpoint["model_state"])
    
    if args.gpu and torch.cuda.is_available():
        model = model.cuda()
        feat = data["feat"].cuda()
        adj = data["adj"].cuda()
    else:
        feat = data["feat"]
        adj = data["adj"]

    # 3. Run Inference
    logger.info("Running inference on FULL dataset...")
    model.eval()
    with torch.no_grad():
        ypred, _ = model(feat, adj)
        # ypred: [1, N, 2]
        probs = torch.softmax(ypred[0], dim=1)[:, 1].cpu().numpy()
        preds = torch.argmax(ypred[0], dim=1).cpu().numpy()

    # 4. Save CSV Results
    df_res = pd.DataFrame({
        "su_id": data["node_ids"],
        "gcn_prob": probs,
        "gcn_pred": preds,
        "label": data["label"][0].cpu().numpy() # Add ground truth for reference
    })
    
    csv_path = OUTPUT_DIR / f"gcn_predictions_{args.mode}.csv"
    df_res.to_csv(csv_path, index=False)
    logger.info(f"Saved tabular predictions: {csv_path}")

    # 5. Map to Raster
    su_raster_path = BASE_DIR / "02_aligned_grid" / "su_a50000_c03_geo.tif"
    tif_path = OUTPUT_DIR / f"LSM_GCN_Raw_{args.mode}.tif"
    
    map_predictions_to_raster(df_res, su_raster_path, tif_path)
    
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
