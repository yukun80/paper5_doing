"""
Script: train_svm.py
Description:
    [Program 6-ML] SVM Baseline Training
    
    Implements a Support Vector Machine classifier with a standard ML pipeline:
    1.  Data Loading via shared ml_utils.
    2.  Preprocessing: StandardScaler (Fit on Train, Transform on Test).
    3.  Model: SGDClassifier with 'hinge' loss (Linear SVM) or SVC (RBF).
        - Defaults to SGDClassifier for efficiency on larger datasets.
        - CalibratedClassifierCV is used to provide probability outputs.
    4.  Evaluation: Standard metrics (ROC, PR, etc.)

    Output Directory: experiments/01_ml_baselines/svm
"""

import joblib
import pandas as pd
from pathlib import Path
import yaml
import sys

# Add project root to sys.path to allow imports
BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.append(str(BASE_DIR))

# Import Shared Utilities
sys.path.append(str(BASE_DIR / "experiments" / "01_ml_baselines"))
from ml_utils import (
    setup_logging,
    load_and_split_data,
    calculate_metrics,
    save_predictions
)

# Import custom path utility
SCRIPTS_DIR = BASE_DIR / "scripts" / "00_common"
sys.path.append(str(SCRIPTS_DIR))
import path_utils

from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# ==============================================================================
# CONFIGURATION
# ==============================================================================

BASE_DATA_DIR = BASE_DIR / "04_tabular_SU"
BASE_EXP_DIR = BASE_DIR / "experiments" / "01_ml_baselines" / "svm"

# Model Params for SVC (RBF Kernel)
# Note: SVC does not support 'loss', 'alpha', 'n_jobs' directly in the same way as SGDClassifier
SVM_PARAMS = {
    "kernel": "rbf",
    "C": 1.0,
    "probability": True, # Required for predict_proba
    "random_state": 42,
    # "cache_size": 1000 # Optional optimization
}

# ==============================================================================
# MAIN PIPELINE
# ==============================================================================

def main():
    # Parse Args
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, choices=["dynamic", "static"], default="dynamic")
    args = parser.parse_args()
    mode = args.mode

    # 0. Resolve Config and Paths
    config_path = BASE_DIR / "metadata" / f"dataset_config_{mode}.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
        
    data_dir = path_utils.resolve_su_path(BASE_DATA_DIR, config=config)
    exp_dir = path_utils.resolve_su_path(BASE_EXP_DIR / "results", config=config)
    models_dir = path_utils.resolve_su_path(BASE_EXP_DIR / "models", config=config)
    
    local_data_path = data_dir / f"tabular_dataset_{mode}.parquet"
    
    logger = setup_logging(f"SVM_Baseline_{mode}", exp_dir, log_file=f"train_svm_{mode}.log")
    logger.info("=" * 60)
    logger.info(f">>> Program 6-ML: Support Vector Machine | Mode: {mode}")
    logger.info(f"Resolved Data Path: {local_data_path}")
    logger.info("=" * 60)

    # 1. Load & Split Data
    logger.info(f"Loading data from: {local_data_path.name}")
    try:
        df_train, df_test, feature_cols = load_and_split_data(local_data_path)
    except Exception as e:
        logger.critical(f"Data Load Failed: {e}")
        sys.exit(1)

    logger.info(f"  Features: {len(feature_cols)} columns")
    logger.info(f"  Train (Filtered): {len(df_train)} | Test: {len(df_test)}")

    X_train = df_train[feature_cols].values
    y_train = df_train["label"].values

    X_test = df_test[feature_cols].values
    y_test = df_test["label"].values

    # 3. Train Model
    logger.info("Training SVM (with RBF Kernel)...")
    svm_pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("svm", SVC(**SVM_PARAMS))
    ])
    svm_pipeline.fit(X_train, y_train)

    # Save Model
    joblib.dump(svm_pipeline, models_dir / f"svm_final_{mode}.joblib")
    logger.info(f"✔ Model saved to {models_dir}")

    # 4. Evaluate
    logger.info("Evaluating on Train & Test Set...")
    # With probability=True, we can use predict_proba
    y_prob_train = svm_pipeline.predict_proba(X_train)[:, 1]
    y_prob_test = svm_pipeline.predict_proba(X_test)[:, 1]

    metrics = calculate_metrics(y_test, y_prob_test)
    for k, v in metrics.items():
        logger.info(f"  {k:<15}: {v:.4f}")

    # 5. Save Results
    df_train['prob'] = y_prob_train
    df_train['split'] = 'train'
    
    df_test['prob'] = y_prob_test
    df_test['split'] = 'test'
    
    df_full = pd.concat([df_train, df_test], axis=0)
    
    # Ensure su_id is a column
    if df_full.index.name == 'su_id':
        df_full = df_full.reset_index()
    elif 'su_id' not in df_full.columns:
         df_full['su_id'] = df_full.index

    # Select only necessary columns
    out_cols = ['su_id', 'label', 'prob', 'split']
    df_full[out_cols].to_csv(exp_dir / f"svm_predictions_{mode}.csv", index=False)
    
    logger.info(f"✔ Full predictions saved to {exp_dir}")

    # 7. Coefficient Analysis (Only valid for Linear Kernel, RBF has no simple coefs)
    # Since we are using RBF, we skip coefficient extraction or use permutation importance if needed.
    # For now, we just log that we are skipping it for RBF.
    logger.info("Skipping feature coefficient extraction (Not applicable for RBF Kernel).")
    
    # Optional: If you really need importance, use permutation_importance
    # from sklearn.inspection import permutation_importance
    # r = permutation_importance(svm_pipeline, X_test, y_test, n_repeats=10, random_state=42)
    # ... (Implementation omitted for speed)

if __name__ == "__main__":
    main()
