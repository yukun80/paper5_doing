"""
Script: 51_train_rf_baseline.py (Refactored)
Description:
    [Program 6-ML] Random Forest Baseline Training

    Key Features:
    -   Uses standardized ml_utils for data loading & evaluation.
    -   Strict Spatial Split (Train vs Test).
    -   Dynamic Undersampling (1:1).

    Output Directory: experiments/01_ml_baselines/random_forest
"""

import sys
import json
import joblib
import pandas as pd
from pathlib import Path
import yaml

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

from sklearn.ensemble import RandomForestClassifier

# ==============================================================================
# CONFIGURATION
# ==============================================================================

BASE_DATA_DIR = BASE_DIR / "04_tabular_SU"
BASE_EXP_DIR = BASE_DIR / "experiments" / "01_ml_baselines" / "random forest"

# ... (Params)

RF_PARAMS = {
    "n_estimators": 200,
    "max_depth": None,
    "min_samples_leaf": 2,
    "max_features": "sqrt",
    "n_jobs": -1,
    "random_state": 42,
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
        
    # SU-specific directories
    data_dir = path_utils.resolve_su_path(BASE_DATA_DIR, config=config)
    exp_dir = path_utils.resolve_su_path(BASE_EXP_DIR / "results", config=config)
    models_dir = path_utils.resolve_su_path(BASE_EXP_DIR / "models", config=config)
    
    local_data_path = data_dir / f"tabular_dataset_{mode}.parquet"
    
    logger = setup_logging(f"RF_Baseline_{mode}", exp_dir)
    logger.info("=" * 60)
    logger.info(f">>> Program 6-ML: Random Forest (Refactored) | Mode: {mode}")
    logger.info(f"Resolved Data Path: {local_data_path}")
    logger.info("=" * 60)

    # 1. Load & Split Data (Standardized)
    logger.info(f"Loading data from: {local_data_path.name}")
    try:
        # Note: load_and_split_data now applies train_sample_mask automatically
        df_train, df_test, feature_cols = load_and_split_data(local_data_path)
    except Exception as e:
        logger.critical(f"Data Load Failed: {e}")
        sys.exit(1)

    logger.info(f"  Features: {len(feature_cols)} columns")
    logger.info(f"  Train (Filtered): {len(df_train)} | Test: {len(df_test)}")

    # Save feature names for reference
    with open(exp_dir / f"feature_names_{mode}.json", "w") as f:
        json.dump(feature_cols, f)

    X_train = df_train[feature_cols].values
    y_train = df_train["label"].values

    X_test = df_test[feature_cols].values
    y_test = df_test["label"].values

    # 3. Train Model
    logger.info("Training Random Forest...")
    rf = RandomForestClassifier(**RF_PARAMS)
    rf.fit(X_train, y_train)

    # Save Model
    joblib.dump(rf, models_dir / f"rf_final_{mode}.joblib")
    logger.info(f"✔ Model saved to {models_dir}")

    # 4. Evaluate
    logger.info("Evaluating on Train & Test Set...")
    y_prob_train = rf.predict_proba(X_train)[:, 1]
    y_prob_test = rf.predict_proba(X_test)[:, 1]
    
    metrics = calculate_metrics(y_test, y_prob_test)
    for k, v in metrics.items():
        logger.info(f"  {k:<15}: {v:.4f}")

    # 5. Save Results (Full Dataset)
    # Add 'split' column and merge
    df_train['prob'] = y_prob_train
    df_train['split'] = 'train'
    
    df_test['prob'] = y_prob_test
    df_test['split'] = 'test'
    
    df_full = pd.concat([df_train, df_test], axis=0)
    
    # Ensure su_id is a column (it might be the index)
    if df_full.index.name == 'su_id':
        df_full = df_full.reset_index()
    elif 'su_id' not in df_full.columns:
         # Fallback if index name is lost but it is the index
         df_full['su_id'] = df_full.index

    # Select only necessary columns
    out_cols = ['su_id', 'label', 'prob', 'split']
    df_full[out_cols].to_csv(exp_dir / f"rf_predictions_{mode}.csv", index=False)
    
    logger.info(f"✔ Full predictions saved to {exp_dir}")

    # 6. Feature Importance
    df_imp = pd.DataFrame({
        "feature": feature_cols,
        "importance": rf.feature_importances_
    }).sort_values(by="importance", ascending=False)
    
    df_imp.to_csv(exp_dir / f"feature_importance_{mode}.csv", index=False)
    logger.info("✔ Feature importance saved.")

if __name__ == "__main__":
    main()
