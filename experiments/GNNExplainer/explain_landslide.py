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
import yaml

# Add current directory to path
CURRENT_DIR = Path(__file__).resolve().parent
sys.path.append(str(CURRENT_DIR))

# Add script directory for gnn_viz_kit
BASE_DIR = Path(__file__).resolve().parent.parent.parent
VIZ_KIT_DIR = BASE_DIR / "scripts" / "04_Fig"
sys.path.append(str(VIZ_KIT_DIR))

# Import GCN from the training module
GCN_MODULE_PATH = BASE_DIR / "experiments" / "02_dl_gcn"
if str(GCN_MODULE_PATH) not in sys.path:
    sys.path.append(str(GCN_MODULE_PATH))

try:
    from gcn_model import GCN
except ImportError:
    # We will handle this inside main or just let it fail if not found
    pass

try:
    from gnn_viz_kit.data_schema import ExplanationArtifact
    HAS_VIZ_KIT = True
except ImportError:
    HAS_VIZ_KIT = False

import models
from explainer import explain
from adapter import LandslideDataAdapter

# Import custom path utility
SCRIPTS_DIR = BASE_DIR / "scripts" / "00_common"
sys.path.append(str(SCRIPTS_DIR))
import path_utils

# ==============================================================================
# LOGGING SETUP
# ==============================================================================
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
# MODEL WRAPPER
# ==============================================================================

class GCNWrapper(torch.nn.Module):
    """
    Adapts the simple 2-layer GCN from experiments/02_dl_gcn to GNNExplainer's API.
    Handles dense adjacency matrices during explanation.
    """
    def __init__(self, original_model):
        super(GCNWrapper, self).__init__()
        self.model = original_model
        
    def forward(self, x, adj, **kwargs):
        # Flatten batch dim if present from GNNExplainer
        if x.dim() == 3: x = x.squeeze(0)
        if adj.dim() == 3: adj = adj.squeeze(0)
        
        # Layer 1
        support1 = torch.mm(x, self.model.gc1.weight)
        if adj.is_sparse:
            out1 = torch.sparse.mm(adj, support1)
        else:
            out1 = torch.mm(adj, support1)
             
        if self.model.gc1.bias is not None:
            out1 = out1 + self.model.gc1.bias
        x1 = torch.nn.functional.relu(out1)
        
        # Layer 2
        support2 = torch.mm(x1, self.model.gc2.weight)
        if adj.is_sparse:
            out2 = torch.sparse.mm(adj, support2)
        else:
            out2 = torch.mm(adj, support2)
             
        if self.model.gc2.bias is not None:
            out2 = out2 + self.model.gc2.bias
            
        logits = torch.nn.functional.log_softmax(out2, dim=1)
        
        # Return (logits, None) to match GNNExplainer expected return (pred, att)
        # We wrap logits in a batch dim for the explainer
        return logits.unsqueeze(0), None

    def load_state_dict(self, state_dict):
        self.model.load_state_dict(state_dict)
    
    def eval(self):
        self.model.eval()
        
    def cuda(self):
        self.model.cuda()
        return self

# ==============================================================================
# EXPLANATION PIPELINE
# ==============================================================================

def resolve_paths(mode):
    config_path = BASE_DIR / "metadata" / f"dataset_config_{mode}.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    
    results_dir = path_utils.resolve_su_path(CURRENT_DIR / "results", config=config)
    checkpoint_dir = path_utils.resolve_su_path(CURRENT_DIR / "checkpoints", config=config)
    return config_path, results_dir, checkpoint_dir

def main(args):
    # 0. Resolve Paths
    config_path, results_dir, checkpoint_dir = resolve_paths(args.mode)
    logger.info(f"Resolved Results Directory: {results_dir}")

    # 1. Load Data
    adapter = LandslideDataAdapter(base_dir=BASE_DIR, mode=args.mode, config_path=config_path)
    adapter.load_data()
    data = adapter.get_processed_data()
    feature_names = data["feature_names"]
    adj = data["adj"]
    feat = data["feat"]
    label = data["label"]
    train_idx = data["train_idx"]
    test_idx = data["test_idx"]

    # 2. Load Checkpoint
    train_results_base = BASE_DIR / "experiments" / "02_dl_gcn" / "results"
    train_model_dir = train_results_base / results_dir.name
    ckpt_path = train_model_dir / f"gcn_best_model_{args.mode}.pth"
    
    if not ckpt_path.exists():
        logger.warning(f"Model not found at {ckpt_path}. Checking local results...")
        ckpt_path = results_dir / f"gcn_best_model_{args.mode}.pth"
        
    if not ckpt_path.exists():
        logger.critical(f"Checkpoint not found at {ckpt_path}. Please run train_gcn.py first.")
        return

    logger.info(f"Loading checkpoint: {ckpt_path}")
    
    try:
        # PyTorch 2.6+ compatibility
        checkpoint = torch.load(ckpt_path, map_location='cuda' if args.gpu else 'cpu', weights_only=False)
        if isinstance(checkpoint, dict) and "model_state" in checkpoint:
            state_dict = checkpoint["model_state"]
        else:
            state_dict = checkpoint
    except Exception as e:
        logger.error(f"Failed to load checkpoint: {e}")
        return

    # 3. Rebuild Model
    input_dim = feat.size(2)
    from gcn_model import GCN
    raw_model = GCN(n_feat=input_dim, n_hidden=args.hidden_dim, n_class=2, dropout=0.0)
    model = GCNWrapper(raw_model)
    
    try:
        model.load_state_dict(state_dict)
        logger.info("Successfully loaded state_dict into GCNWrapper.")
    except RuntimeError as e:
        logger.error(f"State dict mismatch. Expected {input_dim} features. Error: {e}")
        return

    if args.gpu: model = model.cuda()
    model.eval()

    # 4. Initialize Explainer
    with torch.no_grad():
        if args.gpu:
            logits, _ = model(feat.cuda(), adj.cuda())
        else:
            logits, _ = model(feat, adj)
        pred = logits.argmax(dim=2).cpu() 
        
    explainer_instance = explain.Explainer(
        model=model, adj=adj, feat=feat, label=label, pred=pred,
        train_idx=train_idx, args=args, print_training=True
    )
    
    # 5. Select Target Nodes (True Positives in Test Set)
    pred_labels = pred[0].numpy()
    true_labels = label[0].numpy()
    test_indices = np.array(test_idx)
    tp_mask = (true_labels[test_indices] == 1) & (pred_labels[test_indices] == 1)
    tp_indices = test_indices[tp_mask]
    
    logger.info(f"Found {len(tp_indices)} True Positive high-risk nodes.")
    targets = tp_indices[:args.num_explain]
    
    results = []
    
    for node_idx in targets:
        logger.info(f"--- Explaining Node {node_idx} ---")
        node_idx_new, sub_adj, sub_feat, sub_label, neighbors = explainer_instance.extract_neighborhood(node_idx)
        
        sub_adj_t = torch.as_tensor(sub_adj, dtype=torch.float).unsqueeze(0)
        sub_feat_t = torch.as_tensor(sub_feat, dtype=torch.float).unsqueeze(0)
        sub_label_t = torch.as_tensor(sub_label, dtype=torch.long).unsqueeze(0)
        
        if args.gpu:
            sub_adj_t, sub_feat_t, sub_label_t = sub_adj_t.cuda(), sub_feat_t.cuda(), sub_label_t.cuda()

        explainer_module = explain.ExplainModule(
            adj=sub_adj_t, x=sub_feat_t, model=model, label=sub_label_t, args=args
        )
        if args.gpu: explainer_module = explainer_module.cuda()
            
        explainer_module.train()
        optimizer = torch.optim.Adam([explainer_module.mask, explainer_module.feat_mask], lr=0.1)
        
        for epoch in range(args.num_epochs):
            explainer_module.zero_grad()
            optimizer.zero_grad()
            ypred, _ = explainer_module(0, unconstrained=False)
            
            sub_pred_labels = pred_labels[neighbors]
            sub_pred_labels_t = torch.tensor(sub_pred_labels, dtype=torch.long)
            if args.gpu: sub_pred_labels_t = sub_pred_labels_t.cuda()
            
            loss = explainer_module.loss(ypred, sub_pred_labels_t, node_idx_new, epoch)
            loss.backward()
            optimizer.step()
            
        # Extract Results
        feat_mask = explainer_module.feat_mask.detach().sigmoid().cpu().numpy()
        masked_adj = explainer_module.mask.detach().sigmoid().cpu().numpy()
        
        local_to_su_id = {i: adapter.node_ids[global_idx] for i, global_idx in enumerate(neighbors)}
        edge_weights = {}
        rows, cols = np.where(masked_adj > 0.05)
        for r, c in zip(rows, cols):
            edge_weights[(local_to_su_id[r], local_to_su_id[c])] = float(masked_adj[r, c])

        node_attributes = {}
        dnbr_idx = next((i for i, n in enumerate(feature_names) if "dnbr" in n.lower()), -1)
        slope_idx = next((i for i, n in enumerate(feature_names) if "slope" in n.lower()), -1)
        sub_feat_cpu = sub_feat_t.squeeze(0).cpu().numpy()
        
        for i, global_idx in enumerate(neighbors):
            su_id = local_to_su_id[i]
            attrs = {}
            if dnbr_idx != -1: attrs['dNBR'] = float(sub_feat_cpu[i, dnbr_idx])
            if slope_idx != -1: attrs['Slope'] = float(sub_feat_cpu[i, slope_idx])
            node_attributes[su_id] = attrs

        su_id_target = adapter.node_ids[node_idx]
        results.append({"su_id": su_id_target, **dict(zip(feature_names, feat_mask))})
        
        if HAS_VIZ_KIT:
            artifact = ExplanationArtifact(
                su_id=su_id_target, node_idx=node_idx, dataset_split='test',
                prediction_prob=float(pred_labels[node_idx]), true_label=int(true_labels[node_idx]),
                neighbor_indices=[local_to_su_id[i] for i in range(len(neighbors))],
                edge_weights=edge_weights, feature_mask=feat_mask,
                feature_names=feature_names, node_attributes=node_attributes
            )
            artifact_dir = results_dir / "artifacts"
            artifact_dir.mkdir(exist_ok=True)
            with open(artifact_dir / f"explanation_su_{su_id_target}.pkl", "wb") as f:
                pickle.dump(artifact, f)
        
        logger.info(f"Node {node_idx} (SU {su_id_target}) Explanation Done.")

    if not results: return
    df_res = pd.DataFrame(results)
    out_path = results_dir / f"feature_importance_{args.mode}.csv"
    df_res.to_csv(out_path, index=False)
    logger.info(f"Feature importance saved to: {out_path}")

# ==============================================================================
# ENTRY POINT
# ==============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="dynamic")
    parser.add_argument("--num-explain", type=int, default=10)
    parser.add_argument("--num-epochs", type=int, default=100)
    parser.add_argument("--gpu", action="store_true", default=True)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--output-dim", type=int, default=64)
    parser.add_argument("--num-gc-layers", type=int, default=2) 
    parser.add_argument("--bn", action="store_true", default=False)
    parser.add_argument("--method", type=str, default="base")
    parser.add_argument("--bias", action="store_true", default=True)
    parser.add_argument("--mask-act", type=str, default="sigmoid")
    parser.add_argument("--mask-bias", action="store_true", default=True)
    parser.add_argument("--logdir", type=str, default="log")
    parser.add_argument("--opt", type=str, default="adam")
    parser.add_argument("--opt-scheduler", type=str, default="none")
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--opt-decay-step", type=int, default=50)
    parser.add_argument("--opt-decay-rate", type=float, default=0.5)
    parser.add_argument("--weight-decay", type=float, default=0.0)

    args = parser.parse_args()
    args.num_layers = args.num_gc_layers
    main(args)