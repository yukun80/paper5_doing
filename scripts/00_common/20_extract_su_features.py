"""
Script: 20_extract_su_features.py
Description:
    Extracts statistical features (Mean, Std, Min, Max, Mode) for each Slope Unit (SU)
    from the multi-band stacked raster. Implements a memory-efficient band-wise scanning
    approach and handles NoData (0) by dynamically masking invalid pixels.

python scripts/00_common/20_extract_su_features.py
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Any, Optional

import numpy as np
import pandas as pd
import rasterio
import scipy.ndimage
import yaml

# Import custom path utility
sys.path.append(str(Path(__file__).resolve().parent))
import path_utils

# ==============================================================================
# CONFIGURATION & CONSTANTS
# ==============================================================================

BASE_DIR = Path(__file__).resolve().parent.parent.parent
DEFAULT_CONFIG_PATH = BASE_DIR / "metadata" / "dataset_config_dynamic.yaml"
BASE_OUTPUT_DIR = BASE_DIR / "04_tabular_SU"
LOG_DIR = BASE_DIR / "logs"

# Ensure log directory exists
LOG_DIR.mkdir(exist_ok=True)

# Setup Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "extract_features.log", mode="w", encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)


# ==============================================================================
# UTILITY FUNCTIONS
# ==============================================================================
# ... (load_config, load_json, get_su_filename, fast_mode remain the same)


def load_config(path: Path) -> Dict[str, Any]:
    """Loads YAML configuration safely."""
    if not path.exists():
        logger.critical(f"Config file not found: {path}")
        sys.exit(1)
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_json(path: Path) -> List[Dict[str, Any]]:
    """Loads JSON metadata safely."""
    if not path.exists():
        logger.critical(f"Metadata file not found: {path}")
        sys.exit(1)
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_su_filename(config: Dict[str, Any]) -> str:
    """Extracts SU filename from config."""
    try:
        return config["grid"]["files"]["su_id"]
    except KeyError:
        logger.critical("Config missing 'grid.files.su_id'")
        sys.exit(1)


def fast_mode(labels: np.ndarray, values: np.ndarray, unique_labels: np.ndarray) -> np.ndarray:
    """Calculates categorical mode efficiently."""
    df = pd.DataFrame({"label": labels, "val": values})
    modes = df.groupby("label")["val"].agg(lambda x: pd.Series.mode(x).iloc[0])
    return modes.reindex(unique_labels, fill_value=0).values


# ==============================================================================
# MAIN PIPELINE
# ==============================================================================


def main():
    logger.info(">>> Starting Program 2: SU Feature Extraction")
    start_time = time.time()

    # 1. Parse Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--mode", type=str, choices=["dynamic", "static"], default="dynamic")
    args = parser.parse_args()

    mode = args.mode
    
    # Load Config early to resolve paths
    config = load_config(args.config)
    
    # 2. Resolve Dynamic Paths
    output_dir = path_utils.resolve_su_path(BASE_OUTPUT_DIR, config=config)
    output_parquet = output_dir / f"su_features_{mode}.parquet"
    
    logger.info(f"Resolved Output Directory: {output_dir}")

    # 3. Establish Input Path (Stacked Raster)
    # The stack is now also in an SU-specific subdirectory
    stack_dir = path_utils.resolve_su_path(BASE_DIR / "03_stacked_data", config=config)
    stack_path = stack_dir / f"Post_stack_{mode}.tif"
    meta_path = stack_dir / f"stack_metadata_post_{mode}.json"

    # Define SU Path
    su_filename = get_su_filename(config)
    su_path = BASE_DIR / "02_aligned_grid" / su_filename

    if not su_path.exists():
        logger.critical(f"SU ID file not found: {su_path}")
        sys.exit(1)
        
    if not stack_path.exists():
        logger.critical(f"Stacked raster not found: {stack_path}")
        sys.exit(1)

    # Load Stack Metadata
    stack_meta = load_json(meta_path)

    # 4. Load Reference SU Grid (Index Map)
    logger.info(f"Loading SU Grid: {su_path.name}")
    with rasterio.open(su_path) as src:
        su_grid = src.read(1)
        valid_mask = su_grid > 0
        all_su_ids = np.unique(su_grid[valid_mask])

    logger.info(f"Found {len(all_su_ids)} unique Slope Units.")

    # Initialize results container
    results_df = pd.DataFrame(index=all_su_ids)
    results_df.index.name = "su_id"

    # 5. Band-wise Processing Loop
    logger.info(f"--- Starting Extraction from {stack_path.name} ---")

    with rasterio.open(stack_path) as src:
        if src.count != len(stack_meta):
            logger.warning(f"Mismatch: Stack has {src.count} bands, but metadata has {len(stack_meta)} entries.")

        for meta in stack_meta:
            b_idx = meta["band_index"]
            b_name = meta["name"]
            b_type = meta["type"]

            logger.info(f"Processing Band {b_idx}: {b_name} ({b_type})")

            # Read Band Data
            data = src.read(b_idx)
            effective_labels = np.where(data != 0, su_grid, 0)

            # 3. Compute Statistics
            if b_type == "continuous":
                with np.errstate(divide="ignore", invalid="ignore"):
                    results_df[f"{b_name}_mean"] = scipy.ndimage.mean(
                        data, labels=effective_labels, index=all_su_ids
                    ).astype(np.float32)

            elif b_type == "categorical":
                valid_indices = effective_labels > 0
                if not np.any(valid_indices):
                    results_df[f"{b_name}_mode"] = 0
                else:
                    results_df[f"{b_name}_mode"] = fast_mode(
                        effective_labels[valid_indices], data[valid_indices], all_su_ids
                    ).astype(np.int32)

            del data, effective_labels

    # 6. Final Cleanup & Saving
    logger.info("--- Saving Results ---")
    if results_df.isna().values.any():
        results_df.fillna(0, inplace=True)

    results_df.to_parquet(output_parquet, compression="snappy")

    elapsed = time.time() - start_time
    logger.info(f"Saved {len(results_df)} SU features to: {output_parquet}")
    logger.info(f"Total Time: {elapsed:.2f} seconds")


if __name__ == "__main__":
    main()
