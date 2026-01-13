"""
Module: visualize_results.py
Location: experiments/GNNExplainer/visualize_results.py
Description:
    The Visualization Engine for the GNNExplainer Framework.
    
    This module implements the "Micro" and "Macro" level interpretability visualizations
    as defined in the research proposal. It transforms raw explanation artifacts (.pkl)
    and summary statistics (.csv) into publication-ready figures and maps.
    
    Key Capabilities:
    1. Micro-Level: Feature Importance Radar Charts (Dynamic vs Static).
    2. Micro-Level: Top-k Subgraph Visualization (Risk Transmission).
    3. Macro-Level: Global Feature Importance Boxplots.
    4. Macro-Level: Spatial Dominance Mapping (Mechanism Partitioning).

Author: AI Assistant (Virgo Edition)
Date: 2026-01-12
"""

import sys
import os
import argparse
import logging
import pickle
import math
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
import rasterio
import yaml

# Add project root to path for imports
CURRENT_DIR = Path(__file__).resolve().parent
BASE_DIR = CURRENT_DIR.parent.parent
sys.path.append(str(BASE_DIR))

# Import Custom Modules
from experiments.GNNExplainer.utils.structs import ExplanationArtifact

# Add scripts/00_common to path for path_utils
COMMON_SCRIPTS_DIR = BASE_DIR / "scripts" / "00_common"
if str(COMMON_SCRIPTS_DIR) not in sys.path:
    sys.path.append(str(COMMON_SCRIPTS_DIR))
import path_utils

# Setup Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# Style Configuration for Publication
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 10,
    'axes.titlesize': 12,
    'axes.labelsize': 10,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.dpi': 300
})

class LandslideVisualizer:
    """
    Main class for generating interpretability visualizations.
    """
    
    def __init__(self, mode: str = "dynamic"):
        self.mode = mode
        self.config = self._load_config()
        self.results_dir = path_utils.resolve_su_path(CURRENT_DIR / "results", config=self.config)
        self.artifacts_dir = self.results_dir / "artifacts"
        self.output_dir = self.results_dir / "figures"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Define Feature Groups
        self.dynamic_feats = ['dNDVI', 'dNBR', 'dMNDWI', 'dNDVI_S2', 'dNBR_S2']
        self.static_feats = ['Slope', 'Aspect', 'Elevation', 'Curvature', 'TWI', 'Lithology']
        
    def _load_config(self) -> Dict:
        config_path = BASE_DIR / "metadata" / f"dataset_config_{self.mode}.yaml"
        if not config_path.exists():
            raise FileNotFoundError(f"Config not found: {config_path}")
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    def load_artifact(self, su_id: str) -> Optional[ExplanationArtifact]:
        """Loads a specific explanation artifact."""
        path = self.artifacts_dir / f"explanation_su_{su_id}.pkl"
        if not path.exists():
            logger.warning(f"Artifact not found for SU {su_id}")
            return None
        with open(path, "rb") as f:
            return pickle.load(f)

    # ==========================================================================
    # 1. MICRO-LEVEL: RADAR CHART
    # ==========================================================================
    def plot_radar_chart(self, su_id: str, save: bool = True):
        """
        Generates a Radar Chart comparing Dynamic vs Static feature contributions.
        """
        artifact = self.load_artifact(su_id)
        if not artifact: return

        # 1. Process Data
        feat_mask = artifact.feature_mask
        feat_names = artifact.feature_names
        
        # Normalize mask
        feat_mask = feat_mask / (np.sum(feat_mask) + 1e-9)
        
        # Filter and Group
        radar_data = {}
        for name, score in zip(feat_names, feat_mask):
            clean_name = name.replace("static_env_", "").replace("dynamic_forcing_", "")
            # Heuristic grouping
            is_dynamic = any(d in clean_name for d in self.dynamic_feats)
            key = f"{clean_name} {'(D)' if is_dynamic else '(S)'}"
            radar_data[key] = score
            
        # Select Top N features to avoid clutter
        top_k = 8
        sorted_items = sorted(radar_data.items(), key=lambda x: x[1], reverse=True)[:top_k]
        labels = [x[0] for x in sorted_items]
        values = [x[1] for x in sorted_items]
        
        # Close the loop
        values += values[:1]
        angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
        angles += angles[:1]
        
        # 2. Plot
        fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
        ax.fill(angles, values, color='red', alpha=0.25)
        ax.plot(angles, values, color='red', linewidth=2)
        
        # Styling
        ax.set_yticklabels([])
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(labels)
        ax.set_title(f"Feature Contribution Profile\nSU: {su_id}", pad=20)
        
        if save:
            out_path = self.output_dir / f"radar_su_{su_id}.png"
            plt.savefig(out_path, bbox_inches='tight')
            logger.info(f"Saved Radar Chart: {out_path}")
            plt.close()

    # ==========================================================================
    # 2. MICRO-LEVEL: SUBGRAPH
    # ==========================================================================
    def plot_subgraph(self, su_id: str, save: bool = True):
        """
        Visualizes the local explanatory subgraph using NetworkX.
        Nodes are colored by dNBR (Fire Severity) if available.
        """
        artifact = self.load_artifact(su_id)
        if not artifact: return

        G = nx.Graph()
        center_node = artifact.su_id
        
        # Add Central Node
        # Try to get dNBR from attributes
        center_attrs = artifact.node_attributes.get(center_node, {})
        center_dnbr = center_attrs.get('dNBR', 0) if 'dNBR' in center_attrs else 0
        G.add_node(center_node, type='Center', dnbr=center_dnbr)
        
        # Add Neighbors and Edges
        # Threshold edges for clarity
        edge_threshold = 0.05
        for (u, v), w in artifact.edge_weights.items():
            if w > edge_threshold:
                # Add nodes if not exist
                if u not in G: 
                    u_attrs = artifact.node_attributes.get(u, {})
                    G.add_node(u, type='Neighbor', dnbr=u_attrs.get('dNBR', 0))
                if v not in G:
                    v_attrs = artifact.node_attributes.get(v, {})
                    G.add_node(v, type='Neighbor', dnbr=v_attrs.get('dNBR', 0))
                
                G.add_edge(u, v, weight=w)
        
        # Layout
        pos = nx.spring_layout(G, seed=42, k=0.5)
        
        # Draw
        plt.figure(figsize=(8, 8))
        
        # Extract colors (dNBR based)
        node_colors = []
        for n in G.nodes():
            val = G.nodes[n].get('dnbr', 0)
            # Clip for better visualization (assuming dNBR range -0.5 to 1.0)
            node_colors.append(val)
            
        nodes = nx.draw_networkx_nodes(G, pos, node_size=500, node_color=node_colors, 
                                       cmap='YlOrRd', alpha=0.9, vmin=-0.1, vmax=0.8)
        
        # Draw Edges with width proportional to weight
        weights = [G[u][v]['weight'] * 5 for u, v in G.edges()]
        nx.draw_networkx_edges(G, pos, width=weights, alpha=0.6, edge_color='gray')
        
        # Labels
        nx.draw_networkx_labels(G, pos, font_size=8, font_color='black')
        
        plt.title(f"Risk Transmission Subgraph\nSU: {su_id} (Color: dNBR)", fontsize=14)
        plt.colorbar(nodes, label='Burn Severity (dNBR)', shrink=0.8)
        plt.axis('off')
        
        if save:
            out_path = self.output_dir / f"subgraph_su_{su_id}.png"
            plt.savefig(out_path, bbox_inches='tight')
            logger.info(f"Saved Subgraph: {out_path}")
            plt.close()

    # ==========================================================================
    # 3. MACRO-LEVEL: GLOBAL BOXPLOT
    # ==========================================================================
    def plot_global_importance(self, save: bool = True):
        """
        Aggregates feature importance from all explained nodes and plots a Boxplot.
        """
        csv_path = self.results_dir / f"feature_importance_{self.mode}.csv"
        if not csv_path.exists():
            logger.warning(f"Feature importance CSV not found: {csv_path}")
            return
            
        df = pd.read_csv(csv_path)
        # Drop su_id
        if 'su_id' in df.columns:
            df = df.drop(columns=['su_id'])
            
        # Clean column names
        df.columns = [c.replace("static_env_", "").replace("dynamic_forcing_", "") for c in df.columns]
        
        # Sort by median importance
        sorted_cols = df.median().sort_values(ascending=False).index
        df_sorted = df[sorted_cols]
        
        # Plot
        plt.figure(figsize=(12, 6))
        sns.boxplot(data=df_sorted, palette="viridis", flierprops={"marker": "x", "markersize": 3})
        plt.xticks(rotation=45, ha='right')
        plt.title(f"Global Feature Importance Distribution ({self.mode.capitalize()} Mode)")
        plt.ylabel("Importance Score (GNNExplainer)")
        plt.tight_layout()
        
        if save:
            out_path = self.output_dir / f"global_importance_boxplot_{self.mode}.png"
            plt.savefig(out_path)
            logger.info(f"Saved Global Boxplot: {out_path}")
            plt.close()

    # ==========================================================================
    # 4. MACRO-LEVEL: DOMINANCE MAP
    # ==========================================================================
    def generate_dominance_map(self):
        """
        Creates a GeoTIFF classifying each SU as 'Dynamic Dominant' or 'Static Dominant'.
        """
        csv_path = self.results_dir / f"feature_importance_{self.mode}.csv"
        if not csv_path.exists():
            logger.warning("Feature importance CSV not found.")
            return

        df = pd.read_csv(csv_path)
        
        # Load Template Raster (for spatial reference)
        su_filename = self.config.get("grid", {}).get("files", {}).get("su_id")
        template_path = BASE_DIR / "02_aligned_grid" / su_filename
        
        if not template_path.exists():
            logger.error(f"Template raster not found: {template_path}")
            return
            
        # Classification Logic
        results = {}
        for _, row in df.iterrows():
            su_id = int(row['su_id']) # Assuming numeric SU ID match
            
            # Calculate Sums
            dyn_sum = 0
            stat_sum = 0
            
            for col in df.columns:
                if col == 'su_id': continue
                val = row[col]
                clean_name = col.replace("static_env_", "").replace("dynamic_forcing_", "")
                
                if any(d in clean_name for d in self.dynamic_feats):
                    dyn_sum += val
                else:
                    stat_sum += val
            
            # 1 = Dynamic, 2 = Static
            # Simple logic: if dynamic sum > 30% of total (or just > static/2? adjust as needed)
            # Paper logic: "if sum_dyn > sum_stat" might be too strict if dynamic feats are fewer.
            # Let's use relative share.
            total = dyn_sum + stat_sum + 1e-9
            dyn_ratio = dyn_sum / total
            
            if dyn_ratio > 0.4: # Threshold: if dynamic explains > 40% (significant)
                results[su_id] = 1
            else:
                results[su_id] = 2

        # Write to Raster
        with rasterio.open(template_path) as src:
            su_grid = src.read(1)
            profile = src.profile
            
        out_grid = np.zeros_like(su_grid, dtype=np.uint8)
        
        # Vectorized Map
        # Create lookup table
        max_id = int(su_grid.max())
        lut = np.zeros(max_id + 1, dtype=np.uint8)
        
        for su_id, cls in results.items():
            if su_id <= max_id:
                lut[su_id] = cls
                
        # Apply LUT
        # 0 = Unexplained/Background, 1 = Dynamic, 2 = Static
        mask = su_grid != src.nodata
        valid_indices = su_grid[mask].astype(int)
        # Safety check
        valid_indices = valid_indices[valid_indices < len(lut)]
        
        # We need to be careful with indices mapping. 
        # Ideally, we map su_grid values -> class
        out_grid[mask] = lut[su_grid[mask].astype(int)]
        
        profile.update(dtype=rasterio.uint8, count=1, nodata=0)
        
        out_path = self.results_dir / "figures" / f"mechanism_dominance_map_{self.mode}.tif"
        with rasterio.open(out_path, "w", **profile) as dst:
            dst.write(out_grid, 1)
            dst.write_colormap(1, {
                1: (255, 0, 0, 255),    # Red for Dynamic
                2: (0, 0, 255, 255),    # Blue for Static
                0: (0, 0, 0, 0)         # Transparent
            })
            
        logger.info(f"Saved Dominance Map: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="GNNExplainer Visualization Tool")
    parser.add_argument("--mode", type=str, default="dynamic", help="Experiment mode (dynamic/static)")
    parser.add_argument("--task", type=str, default="all", choices=["all", "radar", "subgraph", "boxplot", "map"])
    parser.add_argument("--su", type=str, default=None, help="Specific SU ID for micro viz")
    
    args = parser.parse_args()
    
    viz = LandslideVisualizer(mode=args.mode)
    
    # Task Dispatcher
    if args.task in ["all", "boxplot"]:
        viz.plot_global_importance()
        
    if args.task in ["all", "map"]:
        viz.generate_dominance_map()
        
    if args.task in ["all", "radar", "subgraph"]:
        # Iterate over all artifacts if SU not specified
        if args.su:
            su_list = [args.su]
        else:
            # Find all pickles
            pkgs = list(viz.artifacts_dir.glob("explanation_su_*.pkl"))
            # Limit to first 5 for demo if not specific
            su_list = [p.stem.replace("explanation_su_", "") for p in pkgs[:5]]
            if not su_list:
                logger.warning("No artifacts found to plot.")
        
        for su in su_list:
            if args.task in ["all", "radar"]:
                viz.plot_radar_chart(su)
            if args.task in ["all", "subgraph"]:
                viz.plot_subgraph(su)

if __name__ == "__main__":
    main()
