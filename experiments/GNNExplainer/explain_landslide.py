"""
Module: explain_landslide.py
Location: experiments/GNNExplainer/explain_landslide.py
Description:
    GNNExplainer Interpretation Script for Landslide Susceptibility.

    This script uncovers the "Why" behind the predictions of the GCN model trained 
    by 'train_landslide.py'. It uses the 'GcnEncoderNode' architecture native to 
    this framework.

    Key Features:
    -   Multi-Run Averaging (MRA) for mask stability.
    -   Counterfactual Inference (CPD) to quantify absolute risk contribution.
    -   Dynamic Sensitivity Index (DSI) calculation.
    -   Zoning Classification (Static-Dominant vs Dynamic-Triggered).

python experiments/GNNExplainer/explain_landslide.py --mode dynamic --num-explain 10
"""

import sys
import os
import logging
import argparse
import pickle
import numpy as np
import pandas as pd
import yaml
from pathlib import Path

import torch

# ==============================================================================
# PATH SETUP
# ==============================================================================
CURRENT_DIR = Path(__file__).resolve().parent
sys.path.append(str(CURRENT_DIR))

BASE_DIR = Path(__file__).resolve().parent.parent.parent
SCRIPTS_DIR = BASE_DIR / "scripts" / "00_common"
sys.path.append(str(SCRIPTS_DIR))

import models
from explainer import explain
from adapter import LandslideDataAdapter
import path_utils
from utils.structs import ExplanationArtifact

# ==============================================================================
# LOGGING SETUP
# ==============================================================================
LOG_DIR = CURRENT_DIR / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "explanation.log", mode="w", encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)

# ==============================================================================
# UTILITIES
# ==============================================================================

def resolve_paths(mode):
    config_path = BASE_DIR / "metadata" / f"dataset_config_{mode}.yaml"
    if not config_path.exists():
        logger.error(f"Config file not found: {config_path}")
        sys.exit(1)
        
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    results_dir = path_utils.resolve_su_path(CURRENT_DIR / "results", config=config)
    checkpoint_dir = path_utils.resolve_su_path(CURRENT_DIR / "checkpoints", config=config)
    
    return config_path, results_dir, checkpoint_dir

def identify_dynamic_indices(feature_names):
    """Identifies indices of dynamic features (starting with 'd')."""
    indices = []
    for i, name in enumerate(feature_names):
        if name.lower().startswith("d") or name.lower().startswith("diff"):
            indices.append(i)
    return indices

# ==============================================================================
# MAIN PIPELINE
# ==============================================================================

def main(args):
    # 0. Resolve Paths & Config
    config_path, results_dir, checkpoint_dir = resolve_paths(args.mode)
    logger.info(f"--- Starting GNNExplainer ({args.mode}) ---")
    logger.info(f"Results Directory: {results_dir}")

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
    
    # Identify dynamic features for Counterfactual Analysis
    dynamic_feat_indices = identify_dynamic_indices(feature_names)
    logger.info(f"Identified {len(dynamic_feat_indices)} dynamic features: {[feature_names[i] for i in dynamic_feat_indices]}")

    input_dim = feat.size(2)
    num_classes = 2

    # 2. Initialize Model
    model = models.GcnEncoderNode(
        input_dim=input_dim,
        hidden_dim=args.hidden_dim,
        embedding_dim=args.output_dim,
        label_dim=num_classes,
        num_layers=args.num_layers,
        bn=args.bn,
        dropout=0.0,
        args=args
    )

    if args.gpu and torch.cuda.is_available():
        model = model.cuda()
        adj = adj.cuda()
        feat = feat.cuda()
        label = label.cuda()
    
    # 3. Load Checkpoint
    ckpt_filename = f"landslide_model_{args.mode}_best.pth.tar"
    ckpt_path = checkpoint_dir / ckpt_filename
    
    if not ckpt_path.exists():
        logger.critical(f"Checkpoint not found at: {ckpt_path}")
        return

    try:
        checkpoint = torch.load(ckpt_path, map_location="cuda" if args.gpu and torch.cuda.is_available() else "cpu", weights_only=False)
        model.load_state_dict(checkpoint["model_state"])
        model.eval()
        logger.info(f"Model loaded (Best AUC: {checkpoint.get('best_auc', 'N/A')}).")
    except Exception as e:
        logger.error(f"Failed to load checkpoint: {e}")
        return

    # 4. Generate Global Predictions
    with torch.no_grad():
        logits, _ = model(feat, adj)
        # Assuming logits [1, N, 2] or [N, 2]
        if logits.dim() == 3 and logits.size(0) == 1:
            logits = logits[0]
        
        probs = torch.softmax(logits, dim=1)
        pred = logits.argmax(dim=1).cpu()
        
        # Store global probabilities for Counterfactual Reference
        global_probs = probs[:, 1].cpu().numpy() # Probability of Landslide

    # 5. Initialize GNNExplainer
    explainer_instance = explain.Explainer(
        model=model,
        adj=adj,
        feat=feat,
        label=label,
        pred=pred,
        train_idx=train_idx,
        args=args,
        print_training=False
    )

    # 6. Select Target Nodes (TP + High Risk FN)
    pred_labels = pred.numpy()
    true_labels = label[0].cpu().numpy() if label.dim() == 2 else label.cpu().numpy()
    test_indices = np.array(test_idx)
    
    # TP: True Positive
    tp_mask = (true_labels[test_indices] == 1) & (pred_labels[test_indices] == 1)
    targets = test_indices[tp_mask]

    if args.explain_all:
        logger.info(f"Full Mode: Explaining {len(targets)} nodes.")
    else:
        sample_size = min(args.num_explain, len(targets))
        if sample_size > 0:
            targets = np.random.choice(targets, sample_size, replace=False)
            logger.info(f"Sample Mode: Explaining {len(targets)} random nodes.")
        else:
            logger.warning("No targets found.")
            return

    results = []
    artifact_dir = results_dir / "artifacts"
    artifact_dir.mkdir(parents=True, exist_ok=True)

    # 7. Explanation Loop
    for i, node_idx in enumerate(targets):
        su_id_target = adapter.node_ids[node_idx]
        
        # --- A. Neighborhood Extraction ---
        node_idx_new, sub_adj, sub_feat, sub_label, neighbors = explainer_instance.extract_neighborhood(node_idx)
        
        sub_adj_t = torch.as_tensor(sub_adj, dtype=torch.float).unsqueeze(0)
        sub_feat_t = torch.as_tensor(sub_feat, dtype=torch.float).unsqueeze(0)
        sub_label_t = torch.as_tensor(sub_label, dtype=torch.long).unsqueeze(0)
        
        if args.gpu and torch.cuda.is_available():
            sub_adj_t = sub_adj_t.cuda()
            sub_feat_t = sub_feat_t.cuda()
            sub_label_t = sub_label_t.cuda()

        # --- B. Counterfactual Inference (CPD) ---
        # Calculate prob drop when dynamic features are zeroed out
        orig_prob = float(global_probs[node_idx])
        
        # Clone features and zero out dynamic columns
        sub_feat_cf = sub_feat_t.clone()
        if dynamic_feat_indices:
            sub_feat_cf[:, :, dynamic_feat_indices] = 0.0 # Zeroing dynamic factors
        
        with torch.no_grad():
            logits_cf, _ = model(sub_feat_cf, sub_adj_t)
            probs_cf = torch.softmax(logits_cf, dim=2)
            # node_idx_new is the index of target in subgraph
            cf_prob = float(probs_cf[0, node_idx_new, 1].cpu().item())
        
        cpd = orig_prob - cf_prob # Counterfactual Probability Drop
        
        # --- C. Multi-Run Mask Optimization ---
        feat_mask_acc = None
        masked_adj_acc = None
        
        for run_i in range(args.num_runs):
            explainer_module = explain.ExplainModule(
                adj=sub_adj_t, 
                x=sub_feat_t, 
                model=model, 
                label=sub_label_t, 
                args=args
            )
            if args.gpu and torch.cuda.is_available():
                explainer_module = explainer_module.cuda()

            explainer_module.train()
            optimizer = torch.optim.Adam([explainer_module.mask, explainer_module.feat_mask], lr=args.lr)

            for epoch in range(args.num_epochs):
                explainer_module.zero_grad()
                optimizer.zero_grad()
                ypred, _ = explainer_module(0, unconstrained=False)
                sub_pred_labels = pred_labels[neighbors]
                sub_pred_labels_t = torch.tensor(sub_pred_labels, dtype=torch.long)
                if args.gpu and torch.cuda.is_available():
                    sub_pred_labels_t = sub_pred_labels_t.cuda()
                loss = explainer_module.loss(ypred, sub_pred_labels_t, node_idx_new, epoch)
                loss.backward()
                optimizer.step()

            curr_feat_mask = explainer_module.feat_mask.detach().sigmoid().cpu().numpy()
            curr_adj_mask = explainer_module.mask.detach().sigmoid().cpu().numpy()

            if feat_mask_acc is None:
                feat_mask_acc = curr_feat_mask
                masked_adj_acc = curr_adj_mask
            else:
                feat_mask_acc += curr_feat_mask
                masked_adj_acc += curr_adj_mask

        feat_mask = feat_mask_acc / args.num_runs
        masked_adj = masked_adj_acc / args.num_runs
        
        # --- D. Calculate DSI (Dynamic Sensitivity Index) ---
        sum_total_mask = np.sum(feat_mask)
        sum_dynamic_mask = np.sum(feat_mask[dynamic_feat_indices]) if dynamic_feat_indices else 0.0
        dsi = sum_dynamic_mask / (sum_total_mask + 1e-9)

        # --- E. Determine Zoning Class ---
        # Classification Logic
        # Static-Dominant: High Risk, Low DSI, Low CPD (Fire didn't matter)
        # Dynamic-Triggered: High Risk, Significant CPD (Fire pushed it over edge)
        zone_class = "Unknown"
        if orig_prob > 0.5:
            if cpd > 0.1 or dsi > 0.15: # Thresholds can be tuned
                zone_class = "Dynamic-Triggered"
            else:
                zone_class = "Static-Dominant"
        
        # --- F. Save Results ---
        result_row = {
            "su_id": su_id_target,
            "pred_prob": orig_prob,
            "pred_prob_cf": cf_prob,
            "cpd": cpd,
            "dsi": dsi,
            "zone_class": zone_class,
            **dict(zip(feature_names, feat_mask))
        }
        results.append(result_row)

        # Save detailed artifact
        local_to_su_id = {local_i: adapter.node_ids[global_idx] for local_i, global_idx in enumerate(neighbors)}
        edge_weights = {}
        rows, cols = np.where(masked_adj > 0.05)
        for r, c in zip(rows, cols):
            edge_weights[(local_to_su_id[r], local_to_su_id[c])] = float(masked_adj[r, c])
        
        # Node attrs for visualization
        node_attributes = {}
        sub_feat_cpu = sub_feat_t.squeeze(0).cpu().numpy()
        for local_i, global_idx in enumerate(neighbors):
            su_id = local_to_su_id[local_i]
            attrs = {name: float(sub_feat_cpu[local_i, idx]) for idx, name in enumerate(feature_names)}
            node_attributes[su_id] = attrs

        artifact = ExplanationArtifact(
            su_id=su_id_target,
            node_idx=node_idx,
            dataset_split="test",
            prediction_prob=orig_prob,
            true_label=int(true_labels[node_idx]),
            neighbor_indices=[local_to_su_id[i] for i in range(len(neighbors))],
            edge_weights=edge_weights,
            feature_mask=feat_mask,
            feature_names=feature_names,
            node_attributes=node_attributes,
        )
        # Inject extra metrics into artifact for specialized viz
        artifact.extra_metrics = {"cpd": cpd, "dsi": dsi, "zone_class": zone_class}
        
        with open(artifact_dir / f"explanation_su_{su_id_target}.pkl", "wb") as f:
            pickle.dump(artifact, f)
            
        if i % 10 == 0:
            logger.info(f"Processed {i+1} nodes...")

    # 8. Save Global CSV
    if results:
        df_res = pd.DataFrame(results)
        out_path = results_dir / f"explanation_summary_{args.mode}.csv"
        df_res.to_csv(out_path, index=False)
        logger.info(f"Summary saved to: {out_path}")
        
        # Basic stats
        logger.info("\n--- Zoning Distribution ---")
        logger.info(df_res["zone_class"].value_counts())
        logger.info(f"Mean DSI: {df_res['dsi'].mean():.4f}")
        logger.info(f"Mean CPD: {df_res['cpd'].mean():.4f}")

# ==============================================================================
# CLI
# ==============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="dynamic")
    parser.add_argument("--explain-all", action="store_true")
    parser.add_argument("--num-explain", type=int, default=50) # Increased default
    parser.add_argument("--num-epochs", type=int, default=50)
    parser.add_argument("--num-runs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=0.01)
    
    # Model Args
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--output-dim", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument("--bn", action="store_true", default=False)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--gpu", action="store_true", default=True)
    
    # Dummy
    parser.add_argument("--method", type=str, default="base")
    parser.add_argument("--bias", action="store_true", default=True)
    parser.add_argument("--mask-act", type=str, default="sigmoid")
    parser.add_argument("--mask-bias", action="store_true", default=True)
    parser.add_argument("--logdir", type=str, default="logs")
    parser.add_argument("--opt", type=str, default="adam")
    parser.add_argument("--opt-scheduler", type=str, default="none")
    parser.add_argument("--opt-decay-step", type=int, default=50)
    parser.add_argument("--opt-decay-rate", type=float, default=0.5)

    args = parser.parse_args()
    args.num_gc_layers = args.num_layers
    main(args)
