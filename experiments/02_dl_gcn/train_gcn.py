# -*- coding: utf-8 -*-
"""
Script: train_gcn.py
Location: experiments/02_dl_gcn/train_gcn.py
Description:
    Main execution pipeline for the GCN Baseline.

    Workflow:
    1. Load unified data (gcn_data_loader).
    2. Initialize GCN (gcn_model).
    3. Train using CrossEntropy on training mask.
    4. Save Best Model (.pth) based on Test AUC.
    5. Evaluate on test mask (AUC, AUPRC).
    6. Export probability map (GeoTIFF) and CSV results.

Author: AI Assistant (Virgo Edition)
Date: 2026-01-05

python experiments/02_dl_gcn/train_gcn.py
"""

import sys
import time
import logging
import rasterio
import torch
import numpy as np
import pandas as pd
import torch.nn.functional as F
import torch.optim as optim
from pathlib import Path
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score, f1_score

# Add current directory to path for local imports
sys.path.append(str(Path(__file__).resolve().parent))
from gcn_data_loader import GCNDataLoader
from gcn_model import GCN

# --- Config ---
CONFIG = {
    "seed": 42,
    "hidden_dim": 64,
    "dropout": 0.5,
    "lr": 0.01,
    "weight_decay": 5e-4,
    "epochs": 200,
    "patience": 20,  # Early stopping
    "device": "cuda" if torch.cuda.is_available() else "cpu",
}

# --- Paths ---
BASE_DIR = Path(__file__).resolve().parent.parent.parent
OUTPUT_DIR = Path(__file__).resolve().parent / "results"
TEMPLATE_RASTER = BASE_DIR / "02_aligned_grid" / "su_a50000_c03_geo.tif"
MODEL_PATH = OUTPUT_DIR / "gcn_best_model.pth"

# --- Logging ---
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(OUTPUT_DIR / "training_gcn.log", mode="w", encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


def evaluate(model, features, adj, labels, mask):
    model.eval()
    with torch.no_grad():
        output = model(features, adj)
        loss = F.nll_loss(output[mask], labels[mask])

        # Metrics
        probs = torch.exp(output[mask])[:, 1].cpu().numpy()
        preds = output[mask].max(1)[1].cpu().numpy()
        y_true = labels[mask].cpu().numpy()

        auc = roc_auc_score(y_true, probs)
        auprc = average_precision_score(y_true, probs)
        acc = accuracy_score(y_true, preds)
        f1 = f1_score(y_true, preds)

        return loss.item(), acc, auc, auprc, f1, probs


def save_map_results(su_ids, probs_all, labels_all, split_mask, mode):
    """
    Saves CSV and generates GeoTIFF using the full probability map.
    """
    logger.info("Exporting results (CSV + GeoTIFF)...")

    # 1. Save CSV
    df_res = pd.DataFrame(
        {
            "su_id": su_ids,
            "label": labels_all.cpu().numpy(),
            "prob": probs_all,
            "split": ["train" if m else "test" for m in split_mask.cpu().numpy()],
        }
    )
    csv_path = OUTPUT_DIR / f"gcn_predictions_{mode}.csv"
    df_res.to_csv(csv_path, index=False)
    logger.info(f"Predictions saved to {csv_path}")

    # 2. Generate GeoTIFF
    if not TEMPLATE_RASTER.exists():
        logger.warning(f"Template raster not found at {TEMPLATE_RASTER}. Skipping map generation.")
        return

    # Map: SU_ID -> Prob
    prob_dict = dict(zip(df_res["su_id"], df_res["prob"]))

    with rasterio.open(TEMPLATE_RASTER) as src:
        meta = src.meta.copy()
        su_data = src.read(1)

        # Create output array
        out_data = np.full_like(su_data, -9999, dtype=np.float32)
        meta.update(dtype=rasterio.float32, nodata=-9999)

        # Vectorized mapping
        unique_sus = np.unique(su_data)
        count = 0
        for su_id in unique_sus:
            if su_id in prob_dict:
                out_data[su_data == su_id] = prob_dict[su_id]
                count += 1

        tif_path = OUTPUT_DIR / f"LSM_GCN_Prob_{mode}.tif"
        with rasterio.open(tif_path, "w", **meta) as dst:
            dst.write(out_data, 1)

        logger.info(f"LSM Map saved to {tif_path} (Mapped {count} SUs)")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, choices=["dynamic", "static"], default="dynamic")
    args = parser.parse_args()
    mode = args.mode

    set_seed(CONFIG["seed"])
    logger.info(f"Starting GCN Training on {CONFIG['device']} | Mode: {mode}")

    # Update Model Path
    global MODEL_PATH
    MODEL_PATH = OUTPUT_DIR / f"gcn_best_model_{mode}.pth"

    # 1. Load Data
    loader = GCNDataLoader(BASE_DIR, device=CONFIG["device"])
    data = loader.load(mode=mode)

    features = data["features"]
    adj = data["adj"]
    labels = data["labels"]
    idx_train = data["train_mask"]
    idx_test = data["test_mask"]

    # 2. Model
    model = GCN(n_feat=features.shape[1], n_hidden=CONFIG["hidden_dim"], n_class=2, dropout=CONFIG["dropout"]).to(
        CONFIG["device"]
    )

    optimizer = optim.Adam(model.parameters(), lr=CONFIG["lr"], weight_decay=CONFIG["weight_decay"])

    # 3. Training Loop
    best_auc = 0
    patience_counter = 0

    t_start = time.time()

    for epoch in range(CONFIG["epochs"]):
        model.train()
        optimizer.zero_grad()

        output = model(features, adj)
        loss_train = F.nll_loss(output[idx_train], labels[idx_train])

        loss_train.backward()
        optimizer.step()

        # Validation
        loss_val, acc, auc, auprc, f1, _ = evaluate(model, features, adj, labels, idx_test)

        if epoch % 10 == 0:
            logger.info(f"Epoch {epoch:03d}: Loss={loss_train.item():.4f} | Test AUC={auc:.4f} AUPRC={auprc:.4f}")

        # Early Stopping & Model Saving
        if auc > best_auc:
            best_auc = auc
            patience_counter = 0
            torch.save(model.state_dict(), MODEL_PATH)
        else:
            patience_counter += 1

        if patience_counter >= CONFIG["patience"]:
            logger.info("Early stopping triggered.")
            break

    logger.info(f"Training finished in {time.time() - t_start:.2f}s")

    # 4. Final Inference
    if MODEL_PATH.exists():
        logger.info(f"Loading best model from {MODEL_PATH}")
        model.load_state_dict(torch.load(MODEL_PATH))
    else:
        logger.warning("No model file saved. Using last state (performance might be suboptimal).")

    # Evaluation
    loss_test, acc, auc, auprc, f1, probs_test = evaluate(model, features, adj, labels, idx_test)
    logger.info(f"Best Test Performance: AUC={auc:.4f}, AUPRC={auprc:.4f}, Acc={acc:.4f}, F1={f1:.4f}")

    # Full Map Inference
    model.eval()
    with torch.no_grad():
        output_all = model(features, adj)
        probs_all = torch.exp(output_all)[:, 1].cpu().numpy()

    save_map_results(data["su_ids"], probs_all, labels, idx_train, mode)


if __name__ == "__main__":
    main()
