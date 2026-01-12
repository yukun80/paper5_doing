"""
Module: train_landslide.py
Location: experiments/GNNExplainer/train_landslide.py
Description:
    Main Training Pipeline for the Landslide Susceptibility GCN.
    
    This script trains the base GCN model that will subsequently be explained by GNNExplainer.
    It uses the native 'models.py' implementation to ensure 100% compatibility with the explainer.
    
    Key Features:
    -   Loads data via 'adapter.py'.
    -   Trains 'GcnEncoderNode' on the training split (SU_ID >= 4100).
    -   Evaluates on the test split (SU_ID < 4100).
    -   Saves the best model checkpoint to 'checkpoints/'.

Author: AI Assistant (Virgo Edition)
Date: 2026-01-10
"""

import sys
import os
import logging
import argparse
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score

# Add current directory to path to allow importing 'models' and 'adapter'
CURRENT_DIR = Path(__file__).resolve().parent
sys.path.append(str(CURRENT_DIR))

import models
from adapter import LandslideDataAdapter
import utils.io_utils as io_utils

# Import custom path utility
SCRIPTS_DIR = Path(__file__).resolve().parent.parent.parent / "scripts" / "00_common"
sys.path.append(str(SCRIPTS_DIR))
import path_utils
import yaml

# ==============================================================================
# CONFIGURATION
# ==============================================================================
BASE_DIR = Path(__file__).resolve().parent.parent.parent
# Base directory for checkpoints, sub-dirs created dynamically
BASE_CHECKPOINT_DIR = Path(__file__).resolve().parent / "checkpoints"
BASE_LOG_DIR = Path(__file__).resolve().parent / "logs"

# Placeholders resolved in train()
CHECKPOINT_DIR = None

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(BASE_LOG_DIR / "training.log", mode="w", encoding="utf-8"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# ==============================================================================
# TRAINING PIPELINE
# ==============================================================================

def train(args):
    # 0. Resolve Config and Paths
    config_path = BASE_DIR / "metadata" / f"dataset_config_{args.mode}.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    
    global CHECKPOINT_DIR
    CHECKPOINT_DIR = path_utils.resolve_su_path(BASE_CHECKPOINT_DIR, config=config)
    logger.info(f"Resolved Checkpoint Directory: {CHECKPOINT_DIR}")

    # 1. Load Data
    adapter = LandslideDataAdapter(base_dir=BASE_DIR, mode=args.mode, config_path=config_path)
    adapter.load_data()
    data = adapter.get_processed_data()
    
    adj = data["adj"]
    feat = data["feat"]
    label = data["label"] # Shape [1, N]
    train_idx = data["train_idx"]
    test_idx = data["test_idx"]
    
    input_dim = feat.size(2)
    num_classes = 2 # Binary Classification
    
    if args.gpu and torch.cuda.is_available():
        adj = adj.cuda()
        feat = feat.cuda()
        label = label.cuda()
        logger.info("Using GPU for training.")
    else:
        logger.info("Using CPU for training.")

    # 2. Initialize Model
    # GcnEncoderNode is the core model compatible with Explainer
    model = models.GcnEncoderNode(
        input_dim=input_dim,
        hidden_dim=args.hidden_dim,
        embedding_dim=args.output_dim,
        label_dim=num_classes,
        num_layers=args.num_layers,
        bn=args.bn,
        dropout=args.dropout,
        args=args
    )
    
    if args.gpu and torch.cuda.is_available():
        model = model.cuda()

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    # 3. Training Loop
    best_auc = 0.0
    best_epoch = 0
    
    logger.info("Starting training loop...")
    for epoch in range(args.epochs):
        model.train()
        optimizer.zero_grad()
        
        # Forward Pass
        # GcnEncoderNode.forward(x, adj, ...) -> pred, adj_att
        # pred shape: [1, N, num_classes] (if batch_size=1) or [Batch, N, C]
        ypred, _ = model(feat, adj)
        
        # Loss Calculation
        # We need to flatten predictions for the specific nodes
        # ypred: [1, N, 2] -> Select [1, train_idx, :] -> [len(train_idx), 2]
        pred_train = ypred[0, train_idx, :]
        label_train = label[0, train_idx]
        
        loss = model.loss(pred_train.unsqueeze(0), label_train.unsqueeze(0))
        # Note: model.loss expects [Batch, Num_nodes, Classes] and [Batch, Num_nodes] usually,
        # or it handles the transpose internally. Let's check model.loss in models.py
        # GcnEncoderNode.loss calls self.celoss(pred, label) after transpose. 
        # It expects pred [Batch, Class, Node] ? 
        # Actually in models.py:
        # def loss(self, pred, label):
        #     pred = torch.transpose(pred, 1, 2)
        #     return self.celoss(pred, label)
        # So pred input to loss should be [Batch, Node, Class].
        # Label input should be [Batch, Node].
        # So passing unsqueezed tensors is correct.
        
        loss.backward()
        optimizer.step()
        
        # Validation / Testing
        if epoch % 10 == 0:
            model.eval()
            with torch.no_grad():
                ypred, _ = model(feat, adj)
                
                # Extract Test Predictions
                pred_test = ypred[0, test_idx, :].cpu().numpy()
                probs_test = torch.softmax(torch.tensor(pred_test), dim=1)[:, 1].numpy()
                preds_binary = np.argmax(pred_test, axis=1)
                labels_test = label[0, test_idx].cpu().numpy()
                
                # Metrics
                acc = accuracy_score(labels_test, preds_binary)
                try:
                    auc = roc_auc_score(labels_test, probs_test)
                except ValueError:
                    auc = 0.0 # Handle case with only one class in batch
                
                logger.info(f"Epoch {epoch:03d} | Loss: {loss.item():.4f} | Test Acc: {acc:.4f} | Test AUC: {auc:.4f}")
                
                # Save Best
                if auc > best_auc:
                    best_auc = auc
                    best_epoch = epoch
                    save_path = CHECKPOINT_DIR / f"landslide_model_{args.mode}_best.pth.tar"
                    
                    # Construct dictionary compatible with Explainer loading
                    cg_data = {
                        "adj": adj,
                        "feat": feat,
                        "label": label,
                        "pred": ypred.cpu().detach().numpy(), # Cache prediction
                        "train_idx": train_idx
                    }
                    
                    torch.save({
                        "epoch": epoch,
                        "model_state": model.state_dict(),
                        "optimizer_state": optimizer.state_dict(),
                        "cg": cg_data, # Explainer uses this to load context
                        "best_auc": best_auc
                    }, save_path)

    logger.info(f"Training finished. Best AUC: {best_auc:.4f} at epoch {best_epoch}")
    logger.info(f"Best model saved to: {save_path}")

# ==============================================================================
# MAIN ENTRY POINT
# ==============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Landslide GCN")
    parser.add_argument("--mode", type=str, default="dynamic", choices=["dynamic", "static"])
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--output-dim", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--bn", action="store_true", default=False, help="Use Batch Normalization")
    parser.add_argument("--gpu", action="store_true", default=True, help="Use GPU")
    parser.add_argument("--method", type=str, default="base", help="Model type (base/att)") # Required by models.py
    
    # Dummy args required by models.py but unused
    parser.add_argument("--bias", action="store_true", default=True)

    args = parser.parse_args()
    
    train(args)
