"""
Module: explain_landslide.py
Location: experiments/GNNExplainer/explain_landslide.py
Description:
    GNNExplainer Interpretation Script for Landslide Susceptibility.
    
    This script loads the trained GCN model and applies the GNNExplainer algorithm
    to specific Target Nodes (Slope Units). It uncovers the "Why" behind the predictions.
    
    Key Features:
    -   Loads the best checkpoint from 'train_landslide.py'.
    -   Selects High-Risk nodes (True Positives).
    -   Optimizes Edge Masks and Feature Masks using Mutual Information maximization.
    -   Exports Feature Masks to CSV for quantitative "Static vs Dynamic" analysis.

Author: AI Assistant (Virgo Edition)
Date: 2026-01-10
"""

import sys
import os
import logging
import argparse
import pickle
import json
from pathlib import Path

import torch
import numpy as np
import pandas as pd

# Add current directory to path
CURRENT_DIR = Path(__file__).resolve().parent
sys.path.append(str(CURRENT_DIR))

import models
from explainer import explain
from adapter import LandslideDataAdapter

# ==============================================================================
# CONFIGURATION
# ==============================================================================
BASE_DIR = CURRENT_DIR.parent.parent
RESULTS_DIR = CURRENT_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(CURRENT_DIR / "logs" / "explanation.log", mode="w", encoding="utf-8"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# ==============================================================================
# EXPLANATION PIPELINE
# ==============================================================================

def main(args):
    # 1. Load Data & Metadata (For Feature Names)
    # We need the adapter just to get feature names, data is in checkpoint
    adapter = LandslideDataAdapter(base_dir=BASE_DIR, mode=args.mode)
    adapter.load_data()
    feature_names = adapter.get_processed_data()["feature_names"]
    
    # 2. Load Checkpoint
    ckpt_path = CURRENT_DIR / "checkpoints" / f"landslide_model_{args.mode}_best.pth.tar"
    if not ckpt_path.exists():
        logger.critical(f"Checkpoint not found: {ckpt_path}")
        return

    logger.info(f"Loading checkpoint: {ckpt_path.name}")
    # Fix for PyTorch 2.6+: explicit weights_only=False
    checkpoint = torch.load(ckpt_path, weights_only=False)
    
    cg_data = checkpoint["cg"]
    adj = cg_data["adj"]
    feat = cg_data["feat"]
    label = cg_data["label"]
    pred = cg_data["pred"]
    train_idx = cg_data["train_idx"]
    
    input_dim = feat.size(2)
    num_classes = 2
    
    # 3. Rebuild Model
    model = models.GcnEncoderNode(
        input_dim=input_dim,
        hidden_dim=args.hidden_dim,
        embedding_dim=args.output_dim,
        label_dim=num_classes,
        num_layers=args.num_layers,
        bn=args.bn,
        dropout=0.0, # No dropout during explanation
        args=args
    )
    model.load_state_dict(checkpoint["model_state"])
    if args.gpu:
        model = model.cuda()
    model.eval()

    # 4. Initialize Explainer
    # Note: args needs to have specific explainer params
    explainer_instance = explain.Explainer(
        model=model,
        adj=adj,
        feat=feat,
        label=label,
        pred=pred,
        train_idx=train_idx,
        args=args,
        print_training=True
    )
    
    # 5. Select Target Nodes
    # Strategy: Select True Positives (Label=1, Pred=1) in Test Set
    # We need to compute predictions first to be sure
    with torch.no_grad():
        if args.gpu:
            ypred, _ = model(feat.cuda(), adj.cuda())
        else:
            ypred, _ = model(feat, adj)
    
    pred_labels = torch.argmax(ypred[0], dim=1).cpu().numpy()
    true_labels = label[0].cpu().numpy()
    
    # Indices where True Label=1 and Pred Label=1
    tp_indices = np.where((true_labels == 1) & (pred_labels == 1))[0]
    
    logger.info(f"Found {len(tp_indices)} True Positive high-risk nodes.")
    
    # Explain top K nodes
    targets = tp_indices[:args.num_explain]
    logger.info(f"Explaining first {len(targets)} targets: {targets}")
    
    results = []
    
    for node_idx in targets:
        logger.info(f"--- Explaining Node {node_idx} ---")
        
        # Run Explainer
        # Returns masked_adj. Internally it optimizes feature mask too.
        # We need to access the feature mask from the explainer instance/module.
        # The 'explain' method returns masked_adj, but we want feature importance.
        # We need to modify 'explain' or access the internal state.
        # Looking at explain.py: explain() creates an ExplainModule and trains it.
        # It doesn't return feat_mask. 
        # Hack: We will modify explain.py later if needed, but for now let's assume
        # we can access it or we replicate the logic. 
        # Wait, I can't modify explain.py easily without re-writing it.
        # Actually, explain.py saves the adjacency mask to disk.
        # Feature mask is printed/logged to Tensorboard but not returned.
        # 
        # Perfectionist Fix: I will subclass or monkey-patch the ExplainModule? 
        # No, that's messy.
        # I will rely on the fact that ExplainModule is instantiated inside.
        # Let's write a custom explain loop here instead of calling explainer.explain()
        # This gives us full control over the output.
        
        # Custom Explain Logic (Adapted from explain.py)
        node_idx_new, sub_adj, sub_feat, sub_label, neighbors = explainer_instance.extract_neighborhood(node_idx)
        
        # Prepare inputs
        # Fix: Avoid copy warning by using as_tensor
        sub_adj_t = torch.as_tensor(sub_adj, dtype=torch.float).unsqueeze(0)
        sub_feat_t = torch.as_tensor(sub_feat, dtype=torch.float).unsqueeze(0)
        sub_label_t = torch.as_tensor(sub_label, dtype=torch.long).unsqueeze(0)
        
        if args.gpu:
            sub_adj_t = sub_adj_t.cuda()
            sub_feat_t = sub_feat_t.cuda()
            sub_label_t = sub_label_t.cuda()

        # Initialize Mask Module
        explainer_module = explain.ExplainModule(
            adj=sub_adj_t,
            x=sub_feat_t,
            model=model,
            label=sub_label_t,
            args=args
        )
        if args.gpu:
            explainer_module = explainer_module.cuda()
            
        # Optimization Loop
        explainer_module.train()
        optimizer = torch.optim.Adam([explainer_module.mask, explainer_module.feat_mask], lr=0.1)
        
        for epoch in range(args.num_epochs):
            explainer_module.zero_grad()
            optimizer.zero_grad()
            
            # Forward with masks
            ypred, _ = explainer_module(0, unconstrained=False)
            
            # Loss
            # Fix: Pass ALL subgraph predicted labels for Laplacian regularization
            # neighbors contains global indices of the subgraph nodes
            sub_pred_labels = pred_labels[neighbors]
            sub_pred_labels_t = torch.tensor(sub_pred_labels, dtype=torch.long)
            
            if args.gpu:
                sub_pred_labels_t = sub_pred_labels_t.cuda()
            
            # node_idx_new is the index of the target node within the subgraph (usually 0)
            loss = explainer_module.loss(ypred, sub_pred_labels_t, node_idx_new, epoch)
            
            loss.backward()
            optimizer.step()
            
        # Extract Results
        feat_mask = explainer_module.feat_mask.detach().sigmoid().cpu().numpy()
        
        # Store Feature Importance
        # Map back to feature names
        node_res = {"su_id": adapter.node_ids[node_idx]}
        for i, name in enumerate(feature_names):
            node_res[name] = feat_mask[i]
        results.append(node_res)
        
        logger.info(f"Node {node_idx} Explanation Done.")

    # Save Aggregate Results
    if not results:
        logger.warning("No nodes were explained (results list is empty). Saving empty CSV template.")
        df_res = pd.DataFrame(columns=["su_id"] + feature_names)
        df_res.to_csv(RESULTS_DIR / f"feature_importance_{args.mode}.csv", index=False)
        return

    df_res = pd.DataFrame(results)
    out_path = RESULTS_DIR / f"feature_importance_{args.mode}.csv"
    df_res.to_csv(out_path, index=False)
    logger.info(f"Feature importance saved to: {out_path}")
    
    # Calculate Mean Importance for Dynamic vs Static
    if "dynamic" in args.mode:
        dynamic_cols = [c for c in feature_names if "dynamic" in c or "dNDVI" in c or "dNBR" in c]
        static_cols = [c for c in feature_names if c not in dynamic_cols]
        
        mean_dyn = df_res[dynamic_cols].mean().mean()
        mean_stat = df_res[static_cols].mean().mean()
        
        logger.info(f"Mean Dynamic Importance: {mean_dyn:.4f}")
        logger.info(f"Mean Static Importance : {mean_stat:.4f}")

# ==============================================================================
# ENTRY POINT
# ==============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="dynamic")
    parser.add_argument("--num-explain", type=int, default=10, help="Number of nodes to explain")
    parser.add_argument("--num-epochs", type=int, default=100, help="Optimization epochs for explainer")
    parser.add_argument("--gpu", action="store_true", default=True)
    
    # Model Args (Must match training)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--output-dim", type=int, default=64)
    parser.add_argument("--num-gc-layers", type=int, default=3, dest="num_gc_layers") # Renamed from num-layers
    parser.add_argument("--bn", action="store_true", default=False)
    parser.add_argument("--method", type=str, default="base")
    parser.add_argument("--bias", action="store_true", default=True)
    
    # Explainer Args
    parser.add_argument("--mask-act", type=str, default="sigmoid")
    parser.add_argument("--mask-bias", action="store_true", default=True)
    parser.add_argument("--logdir", type=str, default="log") # Dummy for compatibility
    
    # Optimizer Args (Required by ExplainModule -> train_utils.build_optimizer)
    parser.add_argument("--opt", type=str, default="adam")
    parser.add_argument("--opt-scheduler", type=str, default="none")
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--opt-decay-step", type=int, default=50)
    parser.add_argument("--opt-decay-rate", type=float, default=0.5)
    parser.add_argument("--weight-decay", type=float, default=0.0)

    args = parser.parse_args()
    
    # Compatibility fix: Models.py expects num_layers, Explainer expects num_gc_layers
    # We set both to be safe
    args.num_layers = args.num_gc_layers
    
    main(args)
