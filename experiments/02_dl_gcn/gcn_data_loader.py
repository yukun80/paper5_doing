# -*- coding: utf-8 -*-
"""
Module: GCN Data Loader
Location: experiments/02_dl_gcn/gcn_data_loader.py
Description: 
    Responsible for loading the unified tabular dataset and topological structure, 
    performing strict alignment, feature standardization, and sparse Laplacian matrix construction.
    
    Adheres to the "Single Source of Truth" principle by reading directly from:
    1. 04_tabular_SU/tabular_dataset.parquet (Features & Labels)
    2. 05_graph_SU/edges.parquet (Topology)

Author: AI Assistant (Virgo Edition)
Date: 2026-01-05
"""

import logging
import yaml
from pathlib import Path
from typing import Tuple, Dict, Optional, Union

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
from sklearn.preprocessing import StandardScaler

# Import custom path utility
import sys
SCRIPTS_DIR = Path(__file__).resolve().parent.parent.parent / "scripts" / "00_common"
sys.path.append(str(SCRIPTS_DIR))
try:
    import path_utils
except ImportError:
    path_utils = None

logger = logging.getLogger(__name__)

class GCNDataLoader:
    def __init__(self, base_dir: Path, device: str = "cpu", config_path: Optional[Union[str, Path]] = None):
        """
        Args:
            base_dir (Path): Project root directory.
            device (str): Computation device ('cpu' or 'cuda').
            config_path (Path, optional): Path to the dataset config YAML.
        """
        self.base_dir = base_dir
        self.device = device
        
        # Resolve SU-specific directories
        su_name = "default_su"
        if config_path and path_utils:
            with open(config_path, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f)
            su_name = path_utils.get_su_name(config)
            
        self.data_dir = self.base_dir / "04_tabular_SU"
        self.graph_dir = self.base_dir / "05_graph_SU"
        
        if path_utils:
            self.data_dir = path_utils.resolve_su_path(self.data_dir, su_name=su_name)
            self.graph_dir = path_utils.resolve_su_path(self.graph_dir, su_name=su_name)
        
        # Validation
        if not self.data_dir.exists() or not self.graph_dir.exists():
            logger.warning(f"Data directories missing in {base_dir} for SU {su_name}. Ensure Phase 0 is run.")

    def _load_raw_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Loads parquet files with error handling."""
        feat_path = self.data_dir / "tabular_dataset.parquet"
        edge_path = self.graph_dir / "edges.parquet"
        
        if not feat_path.exists():
            raise FileNotFoundError(f"Missing feature table: {feat_path}")
        if not edge_path.exists():
            raise FileNotFoundError(f"Missing edge table: {edge_path}")
            
        logger.info(f"Loading features from {feat_path.name}")
        df_feat = pd.read_parquet(feat_path)
        
        logger.info(f"Loading topology from {edge_path.name}")
        df_edges = pd.read_parquet(edge_path)
        
        return df_feat, df_edges

    def _normalize_adj(self, adj: sp.coo_matrix) -> torch.sparse.Tensor:
        """
        Computes the renormalized Laplacian: D^(-1/2) * (A + I) * D^(-1/2).
        
        Returns:
            torch.sparse.Tensor: The normalized adjacency matrix on the specified device.
        """
        # A_hat = A + I
        adj_hat = adj + sp.eye(adj.shape[0])
        
        # Degree Matrix D_hat
        row_sum = np.array(adj_hat.sum(1))
        d_inv_sqrt = np.power(row_sum, -0.5).flatten()
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
        d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
        
        # D^(-1/2) * A_hat * D^(-1/2)
        norm_adj = d_mat_inv_sqrt.dot(adj_hat).dot(d_mat_inv_sqrt).tocoo()
        
        # Convert to PyTorch Sparse Tensor
        indices = torch.from_numpy(
            np.vstack((norm_adj.row, norm_adj.col)).astype(np.int64)
        )
        values = torch.from_numpy(norm_adj.data.astype(np.float32))
        shape = torch.Size(norm_adj.shape)
        
        tensor_adj = torch.sparse_coo_tensor(indices, values, shape).to(self.device)
        return tensor_adj

    def load(self, mode: str = "dynamic") -> Dict[str, torch.Tensor]:
        """
        Main execution pipeline.
        
        Args:
            mode: 'dynamic' or 'static' to select dataset version.

        Returns:
            dict: {
                'features': Tensor (N, F),
                'adj': SparseTensor (N, N),
                'labels': Tensor (N,),
                'train_mask': BoolTensor (N,),
                'test_mask': BoolTensor (N,),
                'su_ids': Tensor (N,)
            }
        """
        # Load Raw Data with Mode
        feat_path = self.data_dir / f"tabular_dataset_{mode}.parquet"
        edge_path = self.graph_dir / f"edges_{mode}.parquet"
        
        if not feat_path.exists():
            raise FileNotFoundError(f"Missing feature table: {feat_path}")
        if not edge_path.exists():
            raise FileNotFoundError(f"Missing edge table: {edge_path}")
            
        logger.info(f"Loading features from {feat_path.name}")
        df = pd.read_parquet(feat_path)
        
        logger.info(f"Loading topology from {edge_path.name}")
        df_edges = pd.read_parquet(edge_path)
        
        # 1. Feature Selection (Exclude 'constraint_' which is for CXGNN/InSAR only)
        # We only use Static Env + Dynamic Forcing
        feat_cols = [c for c in df.columns if c.startswith("static_env_") or c.startswith("dynamic_forcing_")]
        
        if not feat_cols:
            raise ValueError("No valid features found. Check 'static_env_'/'dynamic_forcing_' prefixes.")
            
        logger.info(f"Selected {len(feat_cols)} features (Static + Dynamic).")
        
        # 2. Split Generation
        # Check for balanced mask
        if "train_sample_mask" in df.columns:
            logger.info("Applying balanced sampling mask for training.")
            train_mask = (df["split"] == "train") & (df["train_sample_mask"] == True)
            # Convert to numpy bool array
            train_mask = train_mask.values
        else:
            logger.warning("Balanced mask not found. Using full unbalanced train split.")
            train_mask = (df["split"] == "train").values
            
        test_mask = (df["split"] == "test").values
        
        # 3. Standardization (Fit on Balanced Train, Apply to All)
        # Prevents data leakage
        scaler = StandardScaler()
        features_raw = df[feat_cols].values.astype(np.float32)
        
        # Fit ONLY on the selected balanced training samples
        scaler.fit(features_raw[train_mask])
        features_norm = scaler.transform(features_raw)
        
        # 4. Topology Construction
        # Map SU_ID to 0..N index
        su_to_idx = {su_id: idx for idx, su_id in enumerate(df.index)}
        valid_su_set = set(df.index)
        
        # Identify Edge Columns Robustly
        cols = df_edges.columns.tolist()
        if "source" in cols and "target" in cols:
            src_col, tgt_col = "source", "target"
        elif "u" in cols and "v" in cols:
            src_col, tgt_col = "u", "v"
        else:
            # Fallback
            src_col, tgt_col = cols[0], cols[1]
            
        # Filter Edges (Keep only if both nodes exist in feature table)
        sources = []
        targets = []
        
        for _, row in df_edges.iterrows():
            u, v = int(row[src_col]), int(row[tgt_col])
            if u in valid_su_set and v in valid_su_set:
                idx_u, idx_v = su_to_idx[u], su_to_idx[v]
                # Undirected Graph
                sources.extend([idx_u, idx_v])
                targets.extend([idx_v, idx_u])
                
        # Build COO Matrix
        num_nodes = len(df)
        adj = sp.coo_matrix(
            (np.ones(len(sources)), (sources, targets)),
            shape=(num_nodes, num_nodes),
            dtype=np.float32
        )
        
        # 5. Convert to Tensors
        data = {
            "features": torch.tensor(features_norm, dtype=torch.float32).to(self.device),
            "adj": self._normalize_adj(adj),
            "labels": torch.tensor(df["label"].values, dtype=torch.long).to(self.device),
            "train_mask": torch.tensor(train_mask, dtype=torch.bool).to(self.device),
            "test_mask": torch.tensor(test_mask, dtype=torch.bool).to(self.device),
            "su_ids": df.index.values # Keep as numpy for export
        }
        
        logger.info(f"Data Loaded Successfully. Nodes: {num_nodes}, Edges: {len(sources)//2}")
        return data