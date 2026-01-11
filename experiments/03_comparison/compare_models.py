"""
Script: compare_models.py
Description:
    [Program 9-Comparison] Model Evaluation & ROC Plotting

    Reads prediction CSVs from RF and CXGNN experiments, calculates comparative metrics,
    and generates high-quality ROC and PR curves.

Usage:
    python experiments/03_comparison/compare_models.py
"""

import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score

# ==============================================================================
# CONFIGURATION
# ==============================================================================

BASE_DIR = Path(__file__).resolve().parent.parent.parent
EXP_DIR = BASE_DIR / "experiments"
OUTPUT_DIR = EXP_DIR / "03_comparison"

RF_PRED_PATH = EXP_DIR / "01_ml_baselines" / "random forest" / "rf_predictions.csv"
CXGNN_PRED_PATH = EXP_DIR / "_CXGNN" / "results" / "cxgnn_predictions.csv"

# ==============================================================================
# MAIN LOGIC
# ==============================================================================

def main():
    print("=" * 60)
    print(">>> Program 9: Model Comparison & ROC Plotting")
    print("=" * 60)

    # 1. Load Data
    if not RF_PRED_PATH.exists():
        print(f"❌ RF predictions not found at: {RF_PRED_PATH}")
        return
    if not CXGNN_PRED_PATH.exists():
        print(f"❌ CXGNN predictions not found at: {CXGNN_PRED_PATH}")
        return

    df_rf = pd.read_csv(RF_PRED_PATH)
    df_cxgnn = pd.read_csv(CXGNN_PRED_PATH)

    print(f"✔ Loaded RF predictions: {len(df_rf)} samples")
    print(f"✔ Loaded CXGNN predictions: {len(df_cxgnn)} samples")

    # 2. Filter for Test Set (CXGNN might have all samples)
    # RF script only saved Test set predictions (split='test'), so we are good there.
    # CXGNN saved ALL predictions with a 'split' column. We need to filter.
    
    if 'split' in df_cxgnn.columns:
        df_cxgnn_test = df_cxgnn[df_cxgnn['split'] == 'test'].copy()
        print(f"✔ Filtered CXGNN to Test set: {len(df_cxgnn_test)} samples")
    else:
        print("⚠ CXGNN predictions missing 'split' column. Assuming all are Test (Risky).")
        df_cxgnn_test = df_cxgnn.copy()

    # Align Data (Optional but recommended: Intersection of SU_IDs)
    # We assume both tested on the same SU_ID < 4100 set.
    # Let's align by SU_ID just to be safe for paired plotting.
    
    # 2.5 Drop NaNs before alignment to avoid ValueError
    if df_cxgnn_test['cxgnn_prob'].isnull().any():
        n_nans = df_cxgnn_test['cxgnn_prob'].isnull().sum()
        print(f"⚠ Warning: Found {n_nans} NaN predictions in CXGNN output. Dropping them.")
        df_cxgnn_test = df_cxgnn_test.dropna(subset=['cxgnn_prob'])
        
    if df_rf['prob'].isnull().any():
        df_rf = df_rf.dropna(subset=['prob'])

    common_ids = set(df_rf['su_id']).intersection(set(df_cxgnn_test['su_id']))
    print(f"✔ Common Test Samples: {len(common_ids)}")
    
    if len(common_ids) == 0:
        print("❌ Error: No common SU IDs found between RF and CXGNN test sets. Check split logic.")
        return
    
    df_rf = df_rf[df_rf['su_id'].isin(common_ids)].sort_values('su_id')
    df_cxgnn_test = df_cxgnn_test[df_cxgnn_test['su_id'].isin(common_ids)].sort_values('su_id')
    
    # 3. Calculate ROC Curves
    fpr_rf, tpr_rf, _ = roc_curve(df_rf['label'], df_rf['prob'])
    auc_rf = auc(fpr_rf, tpr_rf)
    
    fpr_cx, tpr_cx, _ = roc_curve(df_cxgnn_test['label'], df_cxgnn_test['cxgnn_prob'])
    auc_cx = auc(fpr_cx, tpr_cx)

    # 4. Plot ROC
    plt.figure(figsize=(8, 6))
    plt.plot(fpr_rf, tpr_rf, label=f'Random Forest (AUC = {auc_rf:.4f})', color='gray', linestyle='--')
    plt.plot(fpr_cx, tpr_cx, label=f'CXGNN (AUC = {auc_cx:.4f})', color='red', linewidth=2)
    plt.plot([0, 1], [0, 1], 'k:', alpha=0.5)
    
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve Comparison: Dynamic Susceptibility')
    plt.legend(loc='lower right')
    plt.grid(alpha=0.3)
    
    roc_path = OUTPUT_DIR / "roc_comparison.png"
    plt.savefig(roc_path, dpi=300)
    print(f"✔ Saved ROC Plot: {roc_path}")
    plt.close()

    # 5. Plot PR Curves
    precision_rf, recall_rf, _ = precision_recall_curve(df_rf['label'], df_rf['prob'])
    pr_auc_rf = average_precision_score(df_rf['label'], df_rf['prob'])
    
    precision_cx, recall_cx, _ = precision_recall_curve(df_cxgnn_test['label'], df_cxgnn_test['cxgnn_prob'])
    pr_auc_cx = average_precision_score(df_cxgnn_test['label'], df_cxgnn_test['cxgnn_prob'])
    
    plt.figure(figsize=(8, 6))
    plt.plot(recall_rf, precision_rf, label=f'Random Forest (AP = {pr_auc_rf:.4f})', color='gray', linestyle='--')
    plt.plot(recall_cx, precision_cx, label=f'CXGNN (AP = {pr_auc_cx:.4f})', color='blue', linewidth=2)
    
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve Comparison')
    plt.legend(loc='lower left')
    plt.grid(alpha=0.3)
    
    pr_path = OUTPUT_DIR / "pr_comparison.png"
    plt.savefig(pr_path, dpi=300)
    print(f"✔ Saved PR Plot: {pr_path}")
    plt.close()
    
    print("[Done]")

if __name__ == "__main__":
    main()
