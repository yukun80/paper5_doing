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
import yaml

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(BASE_DIR / "scripts" / "00_common"))
import path_utils

# ==============================================================================
# CONFIGURATION
# ==============================================================================

EXP_DIR = BASE_DIR / "experiments"
BASE_COMPARISON_DIR = EXP_DIR / "03_comparison"

def get_predictions_path(model_name, su_name, mode):
    """Resolves the CSV path for a given model and SU scale."""
    if model_name == "rf":
        return EXP_DIR / "01_ml_baselines" / "random forest" / "results" / su_name / f"rf_predictions_{mode}.csv"
    elif model_name == "xgb":
        return EXP_DIR / "01_ml_baselines" / "xgboost" / "results" / su_name / f"xgb_predictions_{mode}.csv"
    elif model_name == "gcn":
        return EXP_DIR / "02_dl_gcn" / "results" / su_name / f"gcn_predictions_{mode}.csv"
    elif model_name == "gnn_explainer": # GNN predictions from the explainer directory
        return EXP_DIR / "GNNExplainer" / "inference_results" / su_name / f"gcn_predictions_{mode}.csv"
    return None

# ==============================================================================
# MAIN LOGIC
# ==============================================================================

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, choices=["dynamic", "static"], default="dynamic")
    parser.add_argument("--models", nargs="+", default=["rf", "xgb", "gcn"], help="Models to compare")
    args = parser.parse_args()
    mode = args.mode

    # 0. Resolve Paths
    config_path = BASE_DIR / "metadata" / f"dataset_config_{mode}.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    
    su_name = path_utils.get_su_name(config)
    output_dir = path_utils.resolve_su_path(BASE_COMPARISON_DIR, su_name=su_name)
    
    print("=" * 60)
    print(f">>> Program 9: Model Comparison | SU: {su_name} | Mode: {mode}")
    print("=" * 60)

    # 1. Load Data for each model
    model_data = {}
    for m in args.models:
        path = get_predictions_path(m, su_name, mode)
        if path and path.exists():
            df = pd.read_csv(path)
            # Ensure prob column name is consistent (GCN might use 'prob', RF uses 'prob')
            # If XGBoost used 'gcn_prob' by mistake, it's already fixed in previous step
            model_data[m] = df
            print(f"✔ Loaded {m.upper()} predictions: {len(df)} samples")
        else:
            print(f"⚠ Skipping {m.upper()}, path not found: {path}")

    if len(model_data) < 2:
        print("❌ Error: Need at least 2 models with valid predictions to compare.")
        return

    # 2. Plotting Setup
    plt.figure(figsize=(10, 8))
    colors = plt.cm.tab10(np.linspace(0, 1, len(model_data)))
    
    # 3. ROC and Metrics Calculation
    results_summary = []

    plt.figure(1) # ROC Plot
    for (m, df), color in zip(model_data.items(), colors):
        # Filter for Test set
        if 'split' in df.columns:
            df_test = df[df['split'] == 'test'].copy()
        else:
            df_test = df.copy()
            
        df_test = df_test.dropna(subset=['prob'])
        
        y_true = df_test['label']
        y_prob = df_test['prob']
        
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        roc_auc = auc(fpr, tpr)
        pr_auc = average_precision_score(y_true, y_prob)
        
        plt.plot(fpr, tpr, label=f'{m.upper()} (AUC = {roc_auc:.4f})', color=color, linewidth=2)
        results_summary.append({"Model": m.upper(), "ROC-AUC": roc_auc, "PR-AUC": pr_auc})

    plt.plot([0, 1], [0, 1], 'k:', alpha=0.5)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve Comparison (SU: {su_name})')
    plt.legend(loc='lower right')
    plt.grid(alpha=0.3)
    
    roc_path = output_dir / f"roc_comparison_{mode}.png"
    plt.savefig(roc_path, dpi=300)
    print(f"✔ Saved ROC Plot: {roc_path}")
    
    # ... (Similar for PR Curves if needed, or just print summary)
    df_sum = pd.DataFrame(results_summary)
    print("\n[Comparison Summary]")
    print(df_sum.to_string(index=False))
    df_sum.to_csv(output_dir / f"metrics_summary_{mode}.csv", index=False)
    
    print("\n[Done]")

if __name__ == "__main__":
    main()
