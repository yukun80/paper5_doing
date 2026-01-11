"""
Script: 40_build_graph.py
Description:
    Constructs the spatial adjacency graph for Slope Units (SUs).

    Logic:
    1.  Reads the SU ID raster.
    2.  Uses vectorized numpy shifting to detect all pixel boundaries where SU ID changes.
    3.  Extracts unique pairs of adjacent SUs (Neighbors).
    4.  Generates an undirected graph edge list (A->B and B->A).

    This forms the "Skeleton" for Graph Neural Networks (GNNs).

python scripts/00_common/40_build_graph.py
"""

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Dict, Any, Tuple, Set

import numpy as np
import pandas as pd
import rasterio
import yaml

# ==============================================================================
# CONFIGURATION & CONSTANTS
# ==============================================================================

BASE_DIR = Path(__file__).resolve().parent.parent.parent
DEFAULT_CONFIG_PATH = BASE_DIR / "metadata" / "dataset_config_dynamic.yaml"
OUTPUT_DIR = BASE_DIR / "05_graph_SU"
LOG_DIR = BASE_DIR / "logs"

# Ensure output directory exists
OUTPUT_DIR.mkdir(exist_ok=True)
LOG_DIR.mkdir(exist_ok=True)

# Setup Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "build_graph.log", mode="w", encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)


# ==============================================================================
# UTILITY FUNCTIONS
# ==============================================================================
# ... (load_config, get_su_filename, extract_adjacency remain the same)

def load_config(path: Path) -> Dict[str, Any]:
    """Loads YAML configuration safely."""
    if not path.exists():
        logger.critical(f"Config file not found: {path}")
        sys.exit(1)
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_su_filename(config: Dict[str, Any]) -> str:
    """Extracts SU filename from config."""
    try:
        return config["grid"]["files"]["su_id"]
    except KeyError:
        logger.critical("Config missing 'grid.files.su_id'")
        sys.exit(1)


def extract_adjacency(grid: np.ndarray) -> Set[Tuple[int, int]]:
    """Extracts unique adjacent pairs."""
    left = grid[:, :-1]
    right = grid[:, 1:]
    h_mask = (left != right) & (left > 0) & (right > 0)
    h_pairs = np.column_stack((left[h_mask], right[h_mask]))

    up = grid[:-1, :]
    down = grid[1:, :]
    v_mask = (up != down) & (up > 0) & (down > 0)
    v_pairs = np.column_stack((up[v_mask], down[v_mask]))

    all_pairs = np.vstack((h_pairs, v_pairs))
    unique_pairs = np.unique(all_pairs, axis=0)
    return set(map(tuple, unique_pairs))


# ==============================================================================
# MAIN PIPELINE
# ==============================================================================


def main():
    logger.info(">>> Starting Program 4: Graph Construction (Topology)")
    start_time = time.time()

    # 1. Parse Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--mode", type=str, choices=["dynamic", "static"], default="dynamic")
    args = parser.parse_args()
    
    mode = args.mode
    output_parquet = OUTPUT_DIR / f"edges_{mode}.parquet"

    # 2. Load Config
    config = load_config(args.config)
    su_filename = get_su_filename(config)
    su_path = BASE_DIR / "02_aligned_grid" / su_filename

    if not su_path.exists():
        logger.critical(f"SU ID file not found: {su_path}")
        sys.exit(1)

    # 3. Load SU Grid
    logger.info(f"Loading SU Grid: {su_path.name}")
    with rasterio.open(su_path) as src:
        su_grid = src.read(1)

    # 4. Extract Neighbors
    logger.info("Scanning for spatial neighbors (Vectorized)...")
    raw_pairs = extract_adjacency(su_grid)
    logger.info(f"Detected {len(raw_pairs)} unique adjacency boundaries.")

    # 5. Symmetrize
    edge_list = []
    seen_edges = set()
    for src, dst in raw_pairs:
        if (src, dst) not in seen_edges:
            edge_list.append({"src": src, "dst": dst, "type": "spatial", "weight": 1.0})
            seen_edges.add((src, dst))
        if (dst, src) not in seen_edges:
            edge_list.append({"src": dst, "dst": src, "type": "spatial", "weight": 1.0})
            seen_edges.add((dst, src))

    # 6. Build DataFrame
    edges_df = pd.DataFrame(edge_list)
    edges_df.sort_values(by=["src", "dst"], inplace=True)

    # 7. Statistics
    n_edges = len(edges_df)
    n_nodes = len(np.unique(su_grid[su_grid > 0]))
    avg_degree = n_edges / n_nodes if n_nodes > 0 else 0

    logger.info("-" * 40)
    logger.info(f"Graph Statistics ({mode}):")
    logger.info(f"  Total Nodes (SUs): {n_nodes}")
    logger.info(f"  Total Edges      : {n_edges}")
    logger.info(f"  Average Degree   : {avg_degree:.2f}")
    logger.info("-" * 40)

    # 8. Save
    edges_df.to_parquet(output_parquet, compression="snappy")

    elapsed = time.time() - start_time
    logger.info(f"Saved graph edges to: {output_parquet}")
    logger.info(f"Total Time: {elapsed:.2f} seconds")


if __name__ == "__main__":
    main()
