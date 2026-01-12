"""
Module: adapter.py
Location: experiments/GNNExplainer/adapter.py
Description:
    Data Adapter for the GNNExplainer Landslide Susceptibility Pipeline.
    
    This module acts as the "Bridge" between the project's unified parquet datasets
    and the dense-matrix format required by the native GNNExplainer implementation.
    
    Design Philosophy (Perfectionist Edition):
    1.  Type Hints & Documentation: Every method is strictly typed and documented.
    2.  Immutability: Data loading should not unexpectedly mutate source files.
    3.  Traceability: Keeps track of feature names to ensure we can interpret the masks later.
    4.  Efficiency: Handles the sparse-to-dense conversion carefully.

Author: AI Assistant (Virgo Edition)
Date: 2026-01-10
"""

import sys
import logging
from pathlib import Path
from typing import Dict, Tuple, List, Optional, Union

import numpy as np
import pandas as pd
import torch
import yaml

# Import custom path utility
COMMON_UTILS_DIR = Path(__file__).resolve().parent.parent.parent / "scripts" / "00_common"
sys.path.append(str(COMMON_UTILS_DIR))
try:
    import path_utils
except ImportError:
    # Fallback if scripts/00_common is not in expected location
    path_utils = None

# ==============================================================================
# LOGGING SETUP
# ==============================================================================
logger = logging.getLogger(__name__)
if not logger.hasHandlers():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)]
    )

# ==============================================================================
# CLASS DEFINITION
# ==============================================================================

class LandslideDataAdapter:
    """
    Adapts Landslide Susceptibility data (Parquet) for GNNExplainer (Dense Tensors).
    """

    def __init__(self, base_dir: Union[str, Path], mode: str = "dynamic", config_path: Optional[Union[str, Path]] = None):
        """
        Initialize the adapter.

        Args:
            base_dir: Root directory of the dataset (e.g., .../datasets).
            mode: Experiment mode ('dynamic' or 'static'). Determines which files to load.
            config_path: Path to the dataset config YAML.
        """
        self.base_dir = Path(base_dir)
        self.mode = mode.lower()
        
        # Load Config to resolve SU-specific paths
        if config_path is None:
            config_path = self.base_dir / "metadata" / f"dataset_config_{self.mode}.yaml"
        
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
            
        su_name = "default_su"
        if path_utils:
            su_name = path_utils.get_su_name(config)
            
        # Define Paths with SU Subdirectories
        self.tabular_path = self.base_dir / "04_tabular_SU" / su_name / f"tabular_dataset_{self.mode}.parquet"
        self.edges_path = self.base_dir / "05_graph_SU" / su_name / f"edges_{self.mode}.parquet"
        
        # State placeholders
        self.df_features: Optional[pd.DataFrame] = None
        self.df_edges: Optional[pd.DataFrame] = None
        self.node_ids: Optional[np.ndarray] = None
        self.feature_names: List[str] = []
        
        # Tensor Cache
        self.adj_tensor: Optional[torch.Tensor] = None
        self.feat_tensor: Optional[torch.Tensor] = None
        self.label_tensor: Optional[torch.Tensor] = None
        self.train_mask: Optional[np.ndarray] = None
        self.test_mask: Optional[np.ndarray] = None

    def load_data(self) -> None:
        """
        Loads the raw parquet files into memory.
        Validates existence and consistency.
        """
        logger.info(f"Loading data for mode: '{self.mode}'...")
        
        if not self.tabular_path.exists():
            raise FileNotFoundError(f"Tabular dataset not found: {self.tabular_path}")
        if not self.edges_path.exists():
            raise FileNotFoundError(f"Edges dataset not found: {self.edges_path}")

        # 1. Load Tabular Data (Nodes)
        self.df_features = pd.read_parquet(self.tabular_path)
        
        # Ensure SU_ID is handled correctly (either index or column)
        if "su_id" not in self.df_features.columns and self.df_features.index.name == "su_id":
             self.df_features = self.df_features.reset_index()
        
        # Sort by SU_ID to ensure index alignment between Tensor and DataFrame
        self.df_features = self.df_features.sort_values("su_id").reset_index(drop=True)
        self.node_ids = self.df_features["su_id"].values
        
        # 2. Load Edges
        self.df_edges = pd.read_parquet(self.edges_path)
        
        logger.info(f"Loaded {len(self.df_features)} nodes and {len(self.df_edges)} edges.")

    def _build_adjacency(self) -> torch.Tensor:
        """
        Constructs a Dense Adjacency Matrix (N x N) from edge list.
        
        Returns:
            torch.Tensor: Float tensor of shape [1, N, N]. 
                          Note: GNNExplainer expects a batch dimension [Batch, N, N].
        """
        if self.df_features is None or self.df_edges is None:
            raise RuntimeError("Data not loaded. Call load_data() first.")

        num_nodes = len(self.df_features)
        
        # Create a mapping from SU_ID to Matrix Index (0 to N-1)
        # Since we sorted self.df_features by su_id, node_ids[i] corresponds to index i.
        su_to_idx = {su_id: i for i, su_id in enumerate(self.node_ids)}
        
        # Map edges to indices
        # Assuming edges columns are 'source', 'target' or similar. 
        # Let's verify standard naming from your project context (Phase 0).
        # Usually: 'source_su', 'target_su'
        src_col = "source" if "source" in self.df_edges.columns else self.df_edges.columns[0]
        dst_col = "target" if "target" in self.df_edges.columns else self.df_edges.columns[1]
        
        logger.debug(f"Mapping edges using columns: {src_col} -> {dst_col}")
        
        # Vectorized mapping using map is faster than iteration
        # Filter edges where nodes might be missing (data cleaning safety)
        valid_edges = self.df_edges[
            self.df_edges[src_col].isin(su_to_idx) & 
            self.df_edges[dst_col].isin(su_to_idx)
        ]
        
        src_indices = valid_edges[src_col].map(su_to_idx).values
        dst_indices = valid_edges[dst_col].map(su_to_idx).values
        
        # Build Sparse Tensor first to save memory during construction
        # Fix: Convert list of arrays to single array first to avoid PyTorch warning
        indices_np = np.array([src_indices, dst_indices])
        indices = torch.tensor(indices_np, dtype=torch.long)
        values = torch.ones(len(src_indices), dtype=torch.float)
        
        # Construct dense matrix
        # Warning: For very large graphs (e.g. >20k nodes), dense matrix might be huge (20k*20k*4 bytes ~ 1.6GB).
        # This is manageable for modern RAM, but close to the limit.
        adj = torch.sparse_coo_tensor(indices, values, (num_nodes, num_nodes)).to_dense()
        
        # Symmetrize (Landslide connectivity is usually bidirectional)
        adj = torch.max(adj, adj.t())
        
        # Add Self-loops (Standard GCN practice)
        adj.fill_diagonal_(1.0)
        
        # Add batch dimension [1, N, N]
        return adj.unsqueeze(0)

    def _build_features(self) -> torch.Tensor:
        """
        Extracts and normalizes features.
        
        Returns:
            torch.Tensor: Float tensor of shape [1, N, F].
        """
        # Identify feature columns
        # Exclude metadata columns (Strictly aligned with ml_utils.py)
        exclude = [
            "su_id", "label", "split", 
            "train_sample_mask", "ratio", "geometry",
            "slide_pixels", "total_pixels", 
            "centroid_x", "centroid_y",
            # Exclude InSAR constraints to avoid leakage
            "mean_vel", "top20_abs_mean", "is_stable"
        ]
        
        # Also exclude any column that starts with 'Unnamed' or is obviously metadata
        all_cols = self.df_features.columns
        feat_cols = [c for c in all_cols if c not in exclude and not c.startswith("Unnamed")]
        
        self.feature_names = feat_cols
        
        # Extract values
        X = self.df_features[feat_cols].values.astype(np.float32)
        
        # Convert to Tensor
        X_tensor = torch.tensor(X)
        
        # Row-normalize? Standard Scaler?
        # Ideally, features should be pre-scaled. Assuming Phase 1 did this.
        # If not, we can add simple normalization here.
        # For now, we assume input is ready-to-use.
        
        return X_tensor.unsqueeze(0)

    def get_processed_data(self) -> Dict[str, Union[torch.Tensor, List[int], List[str]]]:
        """
        Returns the dictionary required by the training/explanation scripts.
        
        Structure matches what GNNExplainer expects (mostly).
        """
        if self.adj_tensor is None:
            self.adj_tensor = self._build_adjacency()
            self.feat_tensor = self._build_features()
            
            # Labels: Shape [1, N] or [N] depending on loss function
            # models.py expects labels to be used with cross_entropy, usually [Batch, N]?
            # Actually GcnEncoderNode expects label to be [N] for CE loss in its forward/loss.
            # Let's provide [1, N] to match batch dim of others.
            labels = self.df_features["label"].values.astype(np.int64)
            self.label_tensor = torch.tensor(labels).unsqueeze(0)
            
            # Masks
            # Split logic: < 4100 is Test, >= 4100 is Train
            
            # Priority: Use 'train_sample_mask' if available (Balanced Sampling)
            if "train_sample_mask" in self.df_features.columns:
                # Ensure it's boolean
                mask_col = self.df_features["train_sample_mask"].astype(bool)
                self.train_mask = mask_col.values
                logger.info(f"Using Balanced Training Mask: {self.train_mask.sum()} samples.")
            else:
                # Fallback: Use all training data (Imbalanced)
                self.train_mask = (self.df_features["split"] == "train").values
                logger.warning("Balanced mask not found. Using full imbalanced training set.")

            self.test_mask = (self.df_features["split"] == "test").values
            
            # Train indices list
            self.train_idx = np.where(self.train_mask)[0].tolist()
            self.test_idx = np.where(self.test_mask)[0].tolist()

        return {
            "adj": self.adj_tensor,          # [1, N, N]
            "feat": self.feat_tensor,        # [1, N, F]
            "label": self.label_tensor,      # [1, N]
            "train_idx": self.train_idx,     # List[int]
            "test_idx": self.test_idx,       # List[int]
            "train_mask": self.train_mask,   # Array[bool]
            "test_mask": self.test_mask,     # Array[bool]
            "node_ids": self.node_ids,       # Array[int] (Original SU IDs)
            "feature_names": self.feature_names # List[str]
        }

if __name__ == "__main__":
    # Simple Test
    base_dir = Path(__file__).resolve().parent.parent.parent
    adapter = LandslideDataAdapter(base_dir, mode="dynamic")
    try:
        adapter.load_data()
        data = adapter.get_processed_data()
        print("Data loaded successfully.")
        print(f"Adj shape: {data['adj'].shape}")
        print(f"Feat shape: {data['feat'].shape}")
        print(f"Num Train: {len(data['train_idx'])}")
    except Exception as e:
        print(f"Adapter test failed: {e}")
