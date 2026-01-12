
import sys
import logging
from pathlib import Path
from typing import Tuple, List, Dict, Any, Optional

import pandas as pd
import numpy as np
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    f1_score,
    accuracy_score,
    recall_score,
    precision_score,
    confusion_matrix,
)

# ==============================================================================
# LOGGING
# ==============================================================================

def setup_logging(name: str, log_dir: Path, log_file: str = "training.log") -> logging.Logger:
    """Sets up a logger that writes to a file and stdout."""
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Check if handlers already exist to avoid duplicate logs
    logger = logging.getLogger(name)
    if logger.hasHandlers():
        return logger
        
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")

    # File Handler
    fh = logging.FileHandler(log_dir / log_file, mode="w", encoding="utf-8")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    # Stream Handler
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    return logger


# ==============================================================================
# DATA LOADING
# ==============================================================================

def load_and_split_data(
    data_path: Path, 
    target_col: str = "label",
    split_col: str = "split",
    meta_cols: Optional[List[str]] = None
) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    """
    Loads the parquet dataset and splits it into Train/Test sets based on the 'split' column.

    Args:
        data_path: Path to the .parquet file.
        target_col: Name of the label column.
        split_col: Name of the column defining 'train'/'test'.
        meta_cols: List of columns to exclude from features (e.g., ID, geometry). 
                   If None, uses a default list suitable for this project.

    Returns:
        df_train: Training DataFrame.
        df_test: Testing DataFrame.
        feature_cols: List of feature column names.
    """
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found at: {data_path}")

    df = pd.read_parquet(data_path)
    
    # Default metadata columns to exclude if not provided
    if meta_cols is None:
        meta_cols = [
            "su_id", "label", "split", 
            "ratio", "slide_pixels", "total_pixels", 
            "centroid_x", "centroid_y", "geometry",
            "train_sample_mask",
            # Exclude InSAR constraint columns (Physical validation only, not input features)
            "mean_vel", "top20_abs_mean", "is_stable"
        ]

    # Identify features
    feature_cols = [
        c for c in df.columns 
        if c not in meta_cols and c != target_col and c != split_col
    ]
    
    if split_col not in df.columns:
        raise ValueError(f"Split column '{split_col}' missing from dataset.")

    # Perform Split
    # Check for Balanced Mask
    if "train_sample_mask" in df.columns:
        # Use the mask to filter training data (Positive + Sampled Negative)
        # Note: split='train' check is redundant if mask is correct, but added for safety.
        df_train = df[(df[split_col] == "train") & (df["train_sample_mask"] == True)].copy()
        print(f"[Data Load] Applied 'train_sample_mask'. Training samples: {len(df_train)}")
    else:
        # Fallback to full unbalanced train
        print(f"[Data Load] 'train_sample_mask' not found. Using full unbalanced training set.")
        df_train = df[df[split_col] == "train"].copy()

    df_test = df[df[split_col] == "test"].copy()

    return df_train, df_test, feature_cols


def undersample_majority_class(
    df: pd.DataFrame, 
    target_col: str = "label", 
    ratio: float = 1.0, 
    random_state: int = 42
) -> pd.DataFrame:
    """
    Performs random undersampling of the majority class (0) to match the minority class (1).
    
    Args:
        df: Input DataFrame.
        ratio: Ratio of Negatives to Positives (default 1.0 = 1:1).
    """
    df_pos = df[df[target_col] == 1]
    df_neg = df[df[target_col] == 0]

    n_pos = len(df_pos)
    if n_pos == 0:
        return df

    n_neg_keep = int(n_pos * ratio)
    if n_neg_keep < len(df_neg):
        df_neg = df_neg.sample(n=n_neg_keep, random_state=random_state)
    
    df_balanced = pd.concat([df_pos, df_neg])
    df_balanced = df_balanced.sample(frac=1, random_state=random_state).reset_index(drop=True)
    
    return df_balanced


# ==============================================================================
# EVALUATION
# ==============================================================================

def calculate_metrics(
    y_true: np.ndarray, 
    y_prob: np.ndarray, 
    threshold: float = 0.5
) -> Dict[str, float]:
    """Calculates standard binary classification metrics."""
    y_pred = (y_prob >= threshold).astype(int)
    
    metrics = {
        "ROC-AUC": roc_auc_score(y_true, y_prob),
        "PR-AUC": average_precision_score(y_true, y_prob),
        "Accuracy": accuracy_score(y_true, y_pred),
        "Recall": recall_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "F1-Score": f1_score(y_true, y_pred)
    }
    return metrics


def save_predictions(
    df_source: pd.DataFrame, 
    y_prob: np.ndarray, 
    output_path: Path,
    target_col: str = "label"
):
    """
    Saves predictions to CSV with columns: su_id, label, prob, split.
    Assumes df_source has 'su_id' either as column or index.
    """
    out_df = pd.DataFrame()
    
    # Handle ID
    if "su_id" in df_source.columns:
        out_df["su_id"] = df_source["su_id"].values
    elif df_source.index.name == "su_id":
        out_df["su_id"] = df_source.index.values
    else:
        out_df["su_id"] = df_source.index.values

    out_df["label"] = df_source[target_col].values
    out_df["prob"] = y_prob
    
    if "split" in df_source.columns:
        out_df["split"] = df_source["split"].values
    else:
        out_df["split"] = "unknown"

    out_df.to_csv(output_path, index=False)
