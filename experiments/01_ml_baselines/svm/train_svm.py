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

import sys
import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import Pipeline

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.append(str(BASE_DIR))

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
EXP_DIR = BASE_DIR / "experiments" / "01_ml_baselines" / "svm"
MODELS_DIR = EXP_DIR / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# Model Params (SGD is much faster than SVC for >10k samples)
SVM_PARAMS = {
    "loss": "hinge",        # Linear SVM
    "penalty": "l2",
    "alpha": 0.0001,
    "max_iter": 1000,
    "tol": 1e-3,
    "random_state": 42,
    "n_jobs": -1
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

    logger = setup_logging(f"SVM_Baseline_{mode}", EXP_DIR, f"train_svm_{mode}.log")
    logger.info("=" * 60)
    logger.info(f">>> Program 6-ML: SVM (Linear SGD) Baseline | Mode: {mode}")
    logger.info("=" * 60)

    # 1. Load Data
    logger.info(f"Loading data from: {local_data_path.name}")
    try:
        # load_and_split_data handles mask-based balanced sampling now
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

    # 3. Build Pipeline
    # Crucial: SVM requires Standardization
    logger.info("Building Pipeline (Scaler -> LinearSVM)...")
    
    # Base SVM (Linear)
    base_svm = SGDClassifier(**SVM_PARAMS)
    
    # Calibration wrapper (to get predict_proba)
    # Method='sigmoid' is Platt Scaling
    calibrated_svm = CalibratedClassifierCV(base_svm, method='sigmoid', cv=3)

    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('svm', calibrated_svm)
    ])

    # 4. Train
    logger.info("Training...")
    pipeline.fit(X_train, y_train)

    # Save Model
    joblib.dump(pipeline, MODELS_DIR / f"svm_pipeline_{mode}.joblib")
    logger.info("✔ Model saved.")

    # 5. Evaluate
    logger.info("Evaluating on Test Set...")
    # predict_proba is now available via CalibratedClassifierCV
    y_prob = pipeline.predict_proba(X_test)[:, 1]
    
    metrics = calculate_metrics(y_test, y_prob)
    for k, v in metrics.items():
        logger.info(f"  {k:<15}: {v:.4f}")

    # 6. Save Predictions
    save_predictions(df_test, y_prob, EXP_DIR / f"svm_predictions_{mode}.csv")
    logger.info("✔ Predictions saved.")
    
    # 7. Coefficient Analysis (Feature Importance for Linear SVM)
    # Access inner base estimator
    try:
        # Quick refit for importance (no calibration)
        raw_svm = SGDClassifier(**SVM_PARAMS)
        raw_svm.fit(StandardScaler().fit_transform(X_train), y_train)
        
        coefs = raw_svm.coef_[0]
        df_imp = pd.DataFrame({
            "feature": feature_cols,
            "coefficient": np.abs(coefs), # Magnitude = Importance
            "raw_coef": coefs
        }).sort_values(by="coefficient", ascending=False)
        
        df_imp.to_csv(EXP_DIR / f"feature_coefficients_{mode}.csv", index=False)
        logger.info("✔ Feature coefficients saved.")
        
    except Exception as e:
        logger.warning(f"Could not extract coefficients: {e}")

if __name__ == "__main__":
    main()
