"""
Script: 30_generate_labels.py
Description:
    Generates binary labels (0/1) for each Slope Unit based on the Landslide Inventory.

    Policy: "Any Slide = Unstable"
    If an SU contains at least one pixel of landslide (Inventory > 0), it is labeled as 1.
    This strict approach prevents small landslides from being diluted by large SU areas.

python scripts/00_common/30_generate_labels.py
"""

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Dict, Any, Tuple

import numpy as np
import pandas as pd
import rasterio
import scipy.ndimage
import yaml

# ==============================================================================
# CONFIGURATION & CONSTANTS
# ==============================================================================

BASE_DIR = Path(__file__).resolve().parent.parent.parent
DEFAULT_CONFIG_PATH = BASE_DIR / "metadata" / "dataset_config_dynamic.yaml"
OUTPUT_DIR = BASE_DIR / "04_tabular_SU"
LOG_DIR = BASE_DIR / "logs"

# Setup Logging
LOG_DIR.mkdir(exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "generate_labels.log", mode="w", encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)


# ==============================================================================
# UTILITY FUNCTIONS
# ==============================================================================
# ... (load_config, load_raster, check_alignment remain the same)


def load_config(path: Path) -> Dict[str, Any]:
    """Loads YAML configuration safely."""
    if not path.exists():
        logger.critical(f"Config file not found: {path}")
        sys.exit(1)
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_raster(path: Path) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Loads a raster file and returns its data and profile."""
    if not path.exists():
        logger.critical(f"Raster file not found: {path}")
        sys.exit(1)
    with rasterio.open(path) as src:
        return src.read(1), src.profile


def check_alignment(su_shape: Tuple[int, int], inv_shape: Tuple[int, int]):
    """Strictly checks if SU and Inventory rasters have the same dimensions."""
    if su_shape != inv_shape:
        logger.critical(f"Shape Mismatch! SU: {su_shape} vs Inventory: {inv_shape}")
        logger.critical("Data must be perfectly aligned (Program 0's responsibility). Aborting.")
        sys.exit(1)


# ==============================================================================
# MAIN PIPELINE
# ==============================================================================


def main():
    logger.info(">>> Starting Program 3: Label Generation")
    start_time = time.time()

    # 1. Parse Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--mode", type=str, choices=["dynamic", "static"], default="dynamic")
    args = parser.parse_args()

    mode = args.mode
    output_parquet = OUTPUT_DIR / f"su_labels_{mode}.parquet"

    # 2. Load Config & Paths
    config = load_config(args.config)
    grid_files = config.get("grid", {}).get("files", {})

    su_filename = grid_files.get("su_id")
    inv_filename = grid_files.get("inventory")

    if not su_filename or not inv_filename:
        logger.critical("Config missing 'su_id' or 'inventory' filenames.")
        sys.exit(1)

    su_path = BASE_DIR / "02_aligned_grid" / su_filename
    inv_path = BASE_DIR / "02_aligned_grid" / inv_filename

    # 3. Load Data
    logger.info(f"Loading SU Grid: {su_filename}")
    su_grid, su_prof = load_raster(su_path)

    logger.info(f"Loading Inventory: {inv_filename}")
    inv_grid, inv_prof = load_raster(inv_path)

    # 4. Validation
    check_alignment(su_grid.shape, inv_grid.shape)

    # 5. Preprocessing
    inv_binary = (inv_grid > 0).astype(np.int32)
    su_ids = np.unique(su_grid[su_grid > 0])
    logger.info(f"Found {len(su_ids)} unique Slope Units.")

    # 6. Zonal Statistics
    logger.info("Calculating landslide pixel counts per SU...")
    ones_grid = np.ones_like(su_grid, dtype=np.int32)
    total_pixels = scipy.ndimage.sum(ones_grid, labels=su_grid, index=su_ids)
    slide_pixels = scipy.ndimage.sum(inv_binary, labels=su_grid, index=su_ids)

    # 7. Generate Labels
    labels = (slide_pixels > 0).astype(np.int32)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratios = slide_pixels / total_pixels
        ratios = np.nan_to_num(ratios, nan=0.0)

    # 8. Build DataFrame
    df = pd.DataFrame(
        {
            "label": labels,
            "ratio": ratios.astype(np.float32),
            "slide_pixels": slide_pixels.astype(np.int32),
            "total_pixels": total_pixels.astype(np.int32),
        },
        index=su_ids,
    )
    df.index.name = "su_id"

    # 9. Statistics & QC
    n_pos = df["label"].sum()
    n_neg = len(df) - n_pos
    pos_rate = (n_pos / len(df)) * 100

    logger.info("-" * 40)
    logger.info(f"Label Generation Statistics ({mode}):")
    logger.info(f"  Total SUs: {len(df)}")
    logger.info(f"  Positive (Unstable): {n_pos} ({pos_rate:.2f}%)")
    logger.info(f"  Negative (Stable)  : {n_neg}")
    logger.info("-" * 40)

    # 10. Save
    OUTPUT_DIR.mkdir(exist_ok=True)
    df.to_parquet(output_parquet, compression="snappy")

    elapsed = time.time() - start_time
    logger.info(f"Saved labels to: {output_parquet}")
    logger.info(f"Total Time: {elapsed:.2f} seconds")


if __name__ == "__main__":
    main()
