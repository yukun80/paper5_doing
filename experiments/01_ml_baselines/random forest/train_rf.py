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

# Add project root to sys.path to allow imports
BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.append(str(BASE_DIR))

from sklearn.ensemble import RandomForestClassifier

# Import Shared Utilities
sys.path.append(str(BASE_DIR / "experiments" / "01_ml_baselines"))
from ml_utils import (
    setup_logging,
    load_and_split_data,
    undersample_majority_class,
    calculate_metrics,
    save_predictions
)

# ==============================================================================
# CONFIGURATION
# ==============================================================================

DATA_PATH = BASE_DIR / "04_tabular_SU" / "tabular_dataset.parquet"
EXP_DIR = BASE_DIR / "experiments" / "01_ml_baselines" / "random forest"
MODELS_DIR = EXP_DIR / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

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

    # Dynamic Paths
    data_filename = f"tabular_dataset_{mode}.parquet"
    local_data_path = BASE_DIR / "04_tabular_SU" / data_filename
    
    logger = setup_logging(f"RF_Baseline_{mode}", EXP_DIR)
    logger.info("=" * 60)
    logger.info(f">>> Program 6-ML: Random Forest (Refactored) | Mode: {mode}")
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
    with open(EXP_DIR / f"feature_names_{mode}.json", "w") as f:
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
    joblib.dump(rf, MODELS_DIR / f"rf_final_{mode}.joblib")
    logger.info("✔ Model saved.")

    # 4. Evaluate
    logger.info("Evaluating on Test Set...")
    y_prob = rf.predict_proba(X_test)[:, 1]
    
    metrics = calculate_metrics(y_test, y_prob)
    for k, v in metrics.items():
        logger.info(f"  {k:<15}: {v:.4f}")

    # 5. Save Results
    save_predictions(df_test, y_prob, EXP_DIR / f"rf_predictions_{mode}.csv")
    logger.info("✔ Predictions saved.")

    # 6. Feature Importance
    df_imp = pd.DataFrame({
        "feature": feature_cols,
        "importance": rf.feature_importances_
    }).sort_values(by="importance", ascending=False)
    
    df_imp.to_csv(EXP_DIR / f"feature_importance_{mode}.csv", index=False)
    logger.info("✔ Feature importance saved.")

if __name__ == "__main__":
    main()
