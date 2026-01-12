"""
Script: 10_build_raster_stack.py
Description:
    Reads the global configuration, validates the spatial alignment of all registered
    environmental factors, and stacks them into a single multi-band GeoTIFF.
    Also generates a metadata JSON file describing each band's content.

python scripts/00_common/10_build_raster_stack.py
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional

import numpy as np
import rasterio
import yaml
from rasterio.enums import Resampling

# Import custom path utility
sys.path.append(str(Path(__file__).resolve().parent))
import path_utils

# ==============================================================================
# CONFIGURATION & CONSTANTS
# ==============================================================================

BASE_DIR = Path(__file__).resolve().parent.parent.parent
DEFAULT_CONFIG_PATH = BASE_DIR / "metadata" / "dataset_config_dynamic.yaml"
INPUT_DIR = BASE_DIR / "02_aligned_grid"
BASE_OUTPUT_DIR = BASE_DIR / "03_stacked_data"
LOG_DIR = BASE_DIR / "logs"

# Output paths will be resolved dynamically in main()
OUTPUT_DIR = None
OUTPUT_TIF = None
OUTPUT_META = None

# Setup Logging
LOG_DIR.mkdir(exist_ok=True)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

if not logger.hasHandlers():
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")

    # File Handler
    fh = logging.FileHandler(LOG_DIR / "build_stack.log", mode="w", encoding="utf-8")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    # Stream Handler
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(formatter)
    logger.addHandler(sh)
    
    # Prevent propagation to root logger if it has its own handlers
    logger.propagate = False


# ==============================================================================
# UTILITY FUNCTIONS
# ==============================================================================


def load_config(path: Path) -> Dict[str, Any]:
    """Loads and returns the YAML configuration safely."""
    if not path.exists():
        logger.critical(f"Configuration file not found: {path}")
        sys.exit(1)
    try:
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    except Exception as e:
        logger.critical(f"Failed to parse config: {e}")
        sys.exit(1)


def is_spatially_aligned(src: rasterio.DatasetReader, ref_meta: Dict[str, Any], tol: float = 1e-5) -> bool:
    """
    Checks if the source raster aligns with the reference grid metadata.

    Args:
        src: The rasterio dataset reader for the file being checked.
        ref_meta: The metadata dictionary of the reference file (transform, width, height, crs).
        tol: Tolerance for floating point comparison of affine transform elements.

    Returns:
        True if aligned, False otherwise.
    """
    # 1. Check Dimensions
    if (src.width != ref_meta["width"]) or (src.height != ref_meta["height"]):
        logger.error(
            f"Dimension mismatch: Found ({src.width}, {src.height}), Expected ({ref_meta['width']}, {ref_meta['height']})"
        )
        return False

    # 2. Check CRS (Coordinate Reference System)
    # Note: src.crs might be None for some raw files, but aligned grid MUST have it.
    if src.crs != ref_meta["crs"]:
        logger.error(f"CRS mismatch: Found {src.crs}, Expected {ref_meta['crs']}")
        return False

    # 3. Check Transform (Affine Matrix)
    # We compare the 6 elements of the affine transform matrix with a tolerance.
    ref_transform = ref_meta["transform"]
    src_transform = src.transform

    # Compare element-wise
    for i in range(9):  # Affine is technically 3x3, rasterio usually exposes 6 elements but 'a' to 'f'
        # Using simple almost_equal for the relevant components
        if not np.isclose(src_transform[i], ref_transform[i], atol=tol):
            logger.error(f"Transform mismatch at index {i}: Found {src_transform[i]}, Expected {ref_transform[i]}")
            return False

    return True


# ==============================================================================
# MAIN PIPELINE
# ==============================================================================


def main():
    logger.info(">>> Starting Program 1: Dynamic Raster Stack Builder")

    # Parse Arguments
    parser = argparse.ArgumentParser(description="Build Raster Stack from Config")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH, help="Path to dataset config yaml")
    parser.add_argument("--mode", type=str, choices=["dynamic", "static"], default="dynamic", help="Experiment mode")
    args = parser.parse_args()
    
    config_path = args.config
    mode = args.mode
    logger.info(f"Using Configuration: {config_path} (Mode: {mode})")

    # 1. Load Configuration
    config = load_config(config_path)
    
    # 2. Update Output Paths based on SU Name and Mode
    global OUTPUT_DIR, OUTPUT_TIF, OUTPUT_META
    OUTPUT_DIR = path_utils.resolve_su_path(BASE_OUTPUT_DIR, config=config)
    OUTPUT_TIF = OUTPUT_DIR / f"Post_stack_{mode}.tif"
    # Metadata also moves to the SU-specific directory for consistency
    OUTPUT_META = OUTPUT_DIR / f"stack_metadata_post_{mode}.json"
    
    logger.info(f"Resolved Output Directory: {OUTPUT_DIR}")

    grid_cfg = config.get("grid", {})
    factors = config.get("factors", [])

    # 2. Establish Reference Grid
    # We use the Slope Unit ID raster as the spatial anchor.
    su_filename = grid_cfg.get("files", {}).get("su_id")
    if not su_filename:
        logger.critical("Config missing 'grid.files.su_id'. Cannot establish reference grid.")
        sys.exit(1)

    ref_path = INPUT_DIR / su_filename
    if not ref_path.exists():
        logger.critical(f"Reference file (SU ID) not found at: {ref_path}")
        sys.exit(1)

    logger.info(f"Reference Grid: {ref_path.name}")

    # Store reference metadata
    with rasterio.open(ref_path) as ref_ds:
        ref_meta = {
            "driver": "GTiff",
            "height": ref_ds.height,
            "width": ref_ds.width,
            "transform": ref_ds.transform,
            "crs": ref_ds.crs,
            "count": 0,  # Will be updated later
            "dtype": "float32",  # Unified dtype for the stack
            "nodata": grid_cfg.get("nodata", -9999),
            "compress": "lzw",
            "tiled": True,  # Enable tiling for faster access
            "blockxsize": 256,  # Standard tile size
            "blockysize": 256,
            "bigtiff": "YES",  # Enable BigTIFF support for files > 4GB
        }
        logger.info(f"Grid Size: {ref_ds.width} x {ref_ds.height}")
        logger.info(f"CRS: {ref_ds.crs}")

    # 3. Discovery & Validation Loop
    valid_layers = []  # List of tuples: (Path, factor_config)

    logger.info("--- Validating Factors ---")
    for factor in factors:
        fname = factor.get("filename")
        name = factor.get("name")
        required = factor.get("required", False)

        fpath = INPUT_DIR / fname

        if not fpath.exists():
            if required:
                logger.critical(f"MISSING REQUIRED FACTOR: {name} ({fname})")
                sys.exit(1)
            else:
                logger.warning(f"Skipping missing optional factor: {name} ({fname})")
                continue

        # Validate Spatial Alignment
        with rasterio.open(fpath) as src:
            if is_spatially_aligned(src, ref_meta):
                valid_layers.append((fpath, factor))
                logger.info(f"[OK] {name}")
            else:
                if required:
                    logger.critical(f"MISALIGNED REQUIRED FACTOR: {name} ({fname})")
                    sys.exit(1)
                else:
                    logger.warning(f"Skipping misaligned optional factor: {name} ({fname})")

    if not valid_layers:
        logger.critical("No valid factors found. Aborting.")
        sys.exit(1)

    # 4. Stacking & Writing
    # Update output metadata count
    ref_meta["count"] = len(valid_layers)
    target_nodata = ref_meta["nodata"]

    logger.info(f"--- Stacking {len(valid_layers)} Layers ---")
    logger.info(f"Output: {OUTPUT_TIF}")

    metadata_record = []

    try:
        with rasterio.open(OUTPUT_TIF, "w", **ref_meta) as dst:
            for idx, (fpath, factor) in enumerate(valid_layers, start=1):
                logger.info(f"Writing Band {idx}: {factor['name']}")

                with rasterio.open(fpath) as src:
                    # Read data (assume single band for now, usually band 1)
                    data = src.read(1)

                    # Handle NoData normalization
                    # If source has a nodata value different from target, we mask and fill
                    src_nodata = src.nodata
                    if src_nodata is not None and src_nodata != target_nodata:
                        # Create a mask where data matches source nodata
                        mask = (
                            np.isclose(data, src_nodata)
                            if np.issubdtype(data.dtype, np.floating)
                            else (data == src_nodata)
                        )
                        # Cast to float32 (target type)
                        data = data.astype(np.float32)
                        # Fill with target nodata
                        data[mask] = target_nodata
                    else:
                        data = data.astype(np.float32)

                    # Write to the specific band
                    dst.write(data, idx)
                    dst.set_band_description(idx, factor["name"])

                # Record Metadata for sidecar file
                metadata_record.append(
                    {
                        "band_index": idx,
                        "name": factor["name"],
                        "filename": factor["filename"],
                        "type": factor["type"],
                        "group": factor["group"],
                    }
                )

    except Exception as e:
        logger.critical(f"Failed during stacking: {e}")
        sys.exit(1)

    # 5. Write Sidecar Metadata
    with open(OUTPUT_META, "w", encoding="utf-8") as f:
        json.dump(metadata_record, f, indent=2)

    logger.info(f"Metadata written to: {OUTPUT_META}")
    logger.info(">>> Build Stack Completed Successfully.")


if __name__ == "__main__":
    main()
