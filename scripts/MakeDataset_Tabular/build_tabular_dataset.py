import os
import argparse
import yaml
import json
import numpy as np
import pandas as pd
from pathlib import Path

# Import custom path utility
import sys

COMMON_UTILS_DIR = Path(__file__).resolve().parent.parent / "00_common"
sys.path.append(str(COMMON_UTILS_DIR))
import path_utils

# --- Configuration ---
BASE_DIR = Path(__file__).resolve().parent.parent.parent
BASE_DATA_DIR = BASE_DIR / "04_tabular_SU"
METADATA_DIR = BASE_DIR / "metadata"
DEFAULT_CONFIG_PATH = METADATA_DIR / "dataset_config_dynamic.yaml"

# Output and input paths will be resolved dynamically in main()
DATA_DIR = None
OUTPUT_PATH = None

"""
python scripts/MakeDataset_RF/build_tabular_dataset.py --config metadata/dataset_config_dynamic.yaml
"""


def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    print(f"[Info] Starting Tabular Dataset Builder for RF/XGBoost...")

    # Parse Arguments
    parser = argparse.ArgumentParser(description="Build Tabular Dataset")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH, help="Path to dataset config yaml")
    parser.add_argument("--mode", type=str, choices=["dynamic", "static"], default="dynamic", help="Experiment mode")
    args = parser.parse_args()

    # 1. Load Config early
    config = load_config(args.config)

    # 2. Update Paths dynamically
    global DATA_DIR, OUTPUT_PATH
    DATA_DIR = path_utils.resolve_su_path(BASE_DATA_DIR, config=config)
    OUTPUT_PATH = DATA_DIR / f"tabular_dataset_{args.mode}.parquet"

    print(f"[Info] Resolved Data Directory: {DATA_DIR}")

    # 3. Load Data
    features_path = DATA_DIR / f"su_features_{args.mode}.parquet"
    labels_path = DATA_DIR / f"su_labels_{args.mode}.parquet"

    if not features_path.exists() or not labels_path.exists():
        raise FileNotFoundError(
            f"Missing input files for mode {args.mode}. Please run scripts/00_common/ with --mode {args.mode} first."
        )

    df_feats = pd.read_parquet(features_path)
    df_labels = pd.read_parquet(labels_path)

    # Merge on index (assuming index is SU_ID)
    print(f"[Info] Merging features ({df_feats.shape}) and labels ({df_labels.shape})...")
    # Ensure index alignment
    df = df_feats.join(df_labels, how="inner")
    print(f"[Info] Merged shape: {df.shape}")

    # --- InSAR-Guided Stability Analysis (Added) ---
    print(f"[Info] Loading InSAR for Stability Filtering...")
    import rasterio
    import scipy.ndimage

    # Locate InSAR File
    grid_dir = BASE_DIR / "02_aligned_grid"
    insar_files = list(grid_dir.glob("InSAR_desc_*nodata.tif"))

    # DYNAMICALLY get SU grid filename from config
    su_filename = config.get("grid", {}).get("files", {}).get("su_id")
    su_grid_path = grid_dir / su_filename

    if not insar_files or not su_grid_path.exists():
        print(f"[Warning] InSAR file or SU Grid not found. Skipping InSAR filtering. (Grid: {su_filename})")
        df["is_stable"] = True  # Default to all stable if data missing
    else:
        insar_path = insar_files[0]
        print(f"  - InSAR: {insar_path.name}")

        with rasterio.open(insar_path) as src_insar, rasterio.open(su_grid_path) as src_su:
            insar_data = src_insar.read(1)
            su_grid = src_su.read(1)

            # Mask valid area
            mask = (su_grid > 0) & (insar_data != -9999)  # Assuming -9999 is nodata

            valid_insar = insar_data[mask]
            valid_su = su_grid[mask]

            # We need to compute stats per SU
            # Since scipy.ndimage is slow for custom functions (like percentile),
            # and we need Top 20% mean, we might need a loop or pandas groupby.
            # Pandas is often faster for this than repeated ndimage calls with custom funcs.

            print(f"  - Computing zonal statistics (Mean & Top20%)...")
            df_insar = pd.DataFrame({"su_id": valid_su, "val": valid_insar})

            # Define aggregations
            def get_top20_mean(x):
                # Mean of the top 20% largest absolute values?
                # Or simply top 20% values (positive)?
                # Requirement: "Average deformation rate ... within +/- 0.02"
                # Usually stability means absolute velocity is low.
                # So we check the mean of the absolute velocities?
                # Or the mean of the raw velocities?
                # "正负0.02以内" implies raw value interval [-0.02, 0.02].
                # "Top 20%" usually refers to the most significant deformation.
                # If deformation is negative (subsidence), "Top 20%" might mean the most negative?
                # Let's use Absolute Value for robustness to check "Activity".
                # Activity = Abs(Velocity).
                # We want Activity < 0.02.
                return np.mean(np.sort(np.abs(x))[-int(len(x) * 0.2 + 1) :])

            stats = df_insar.groupby("su_id")["val"].agg(
                mean_vel=lambda x: np.mean(x), top20_abs_mean=lambda x: get_top20_mean(x)
            )

            # Join back to main df
            df = df.join(stats, how="left")

            # Fill NaN (SUs with no valid InSAR pixels) with "Unstable" or "Stable"?
            # Safer to assume Unstable to avoid bad negatives, or Stable?
            # Let's fill with 0.0 (Stable) if missing, assuming no info = no deformation observed?
            # Or better: if missing, we can't judge. Let's assume Stable to avoid discarding data.
            df["mean_vel"] = df["mean_vel"].fillna(0.0)
            df["top20_abs_mean"] = df["top20_abs_mean"].fillna(0.0)

            # Stability Criterion: Mean within +/- 0.02 OR Top20_Abs within 0.02
            # Note: User said "Mean ... OR Top20 ... within +/- 0.02".
            # Top20_Abs mean is always positive. So < 0.02 checks the magnitude.
            threshold = 0.02
            df["is_stable"] = (df["mean_vel"].abs() < threshold) | (df["top20_abs_mean"] < threshold)

            stable_count = df["is_stable"].sum()
            print(f"  - Stability Analysis: {stable_count}/{len(df)} SUs are considered stable (Thresh={threshold}).")

    # 2. Rename Columns based on Roles (Optional but good for analysis)
    print(f"[Info] Using Config: {args.config}")
    config = load_config(args.config)
    factor_map = {f["name"]: f["role"] for f in config["factors"]}

    # Simple logic: if column starts with factor name, prepend role
    # e.g., "Slope_mean" -> "static_env_Slope_mean"
    new_cols = {}
    for col in df.columns:
        if col in ["label", "ratio", "geometry", "su_id"]:
            continue

        # Try to match prefix
        for factor_name, role in factor_map.items():
            if col.startswith(factor_name):
                new_cols[col] = f"{role}_{col}"
                break

    if new_cols:
        print(f"[Info] Renaming {len(new_cols)} columns with role prefixes...")
        df.rename(columns=new_cols, inplace=True)

    # 3. Deterministic Train/Test Split (Percentage Based)
    # Requirement:
    #   - Test Set: First 40% of sorted SU_IDs
    #   - Train Set: Remaining 60%

    print(f"[Info] Performing Deterministic Split (First 40% Test, Last 60% Train)...")

    # Ensure index is treated as SU_ID (integer) and sorted
    su_ids = df.index.to_series().astype(int).sort_values()

    # Calculate Split Index
    n_total = len(su_ids)
    split_idx = int(n_total * 0.45)  # 40% mark

    # Determine Threshold SU_ID (The ID at the cut-off point)
    # IDs are 1-based usually, but we use the sorted array to find the value.
    # The split_idx points to the first element of the TRAIN set (if we slice [:split_idx] for test)
    # or the first element AFTER test set.

    # Let's verify:
    # [0, 1, ..., split_idx-1] -> Test (Length = split_idx)
    # [split_idx, ..., n-1]    -> Train

    if split_idx >= n_total:
        raise ValueError("Dataset too small for 40% split.")

    split_threshold_id = su_ids.iloc[split_idx]

    print(f"  - Total SUs: {n_total}")
    print(f"  - Split Index: {split_idx} (Top 40%)")
    print(f"  - Dynamic Split Threshold SU_ID: < {split_threshold_id}")

    # Create 'split' column
    # Initialize with 'train'
    df["split"] = "train"

    # Mark test set (IDs strictly less than the threshold value found at 40% mark)
    # Note: Since su_ids is sorted, using the value at split_idx as upper bound works.
    test_mask = df.index < split_threshold_id
    df.loc[test_mask, "split"] = "test"

    train_count = (df["split"] == "train").sum()
    test_count = (df["split"] == "test").sum()

    print(f"[Success] Split Completed:")
    print(f"  - Train (>= {split_threshold_id}): {train_count} samples ({train_count/n_total:.1%})")
    print(f"  - Test  (< {split_threshold_id}) : {test_count} samples ({test_count/n_total:.1%})")

    # 4. Balanced Sampling Mask for Training
    # We create a mask that selects all positives and an equal number of negatives from the TRAIN split.
    # This ensures models can easily filter for a balanced training set while keeping the full data available.
    print(f"[Info] Generating Balanced Training Mask (1:1 Ratio) with InSAR Filter...")

    np.random.seed(42)
    df["train_sample_mask"] = False

    # Filter Train Split
    train_indices = df[df["split"] == "train"].index

    # Separate Pos/Neg in Train
    pos_train_indices = df.loc[train_indices][df.loc[train_indices, "label"] == 1].index
    neg_train_indices = df.loc[train_indices][df.loc[train_indices, "label"] == 0].index

    # Further Filter Negatives by Stability
    # We want negatives that are PHYSICALLY STABLE (is_stable=True)
    if "is_stable" in df.columns:
        stable_neg_indices = df.loc[neg_train_indices][df.loc[neg_train_indices, "is_stable"] == True].index
        unstable_neg_indices = df.loc[neg_train_indices][df.loc[neg_train_indices, "is_stable"] == False].index
    else:
        # Fallback if logic failed upstream
        stable_neg_indices = neg_train_indices
        unstable_neg_indices = []

    n_pos = len(pos_train_indices)
    n_neg_stable = len(stable_neg_indices)

    selected_neg_indices = []

    # Sampling Logic
    if n_neg_stable >= n_pos:
        # Ideal case: We have enough stable negatives
        print(f"  - Sufficient stable negatives found ({n_neg_stable} >= {n_pos}). using pure stable subset.")
        selected_neg_indices = np.random.choice(stable_neg_indices, size=n_pos, replace=False)
    else:
        # Deficit case: Take all stable, fill with unstable
        print(f"  - [Warning] Insufficient stable negatives ({n_neg_stable} < {n_pos}). Mixing with unstable.")
        selected_neg_indices.extend(stable_neg_indices)

        n_needed = n_pos - n_neg_stable
        # Sample from unstable
        if len(unstable_neg_indices) >= n_needed:
            fill_indices = np.random.choice(unstable_neg_indices, size=n_needed, replace=False)
            selected_neg_indices.extend(fill_indices)
        else:
            # Extreme deficit (should not happen usually)
            selected_neg_indices.extend(unstable_neg_indices)

    selected_neg_indices = np.array(selected_neg_indices)

    print(f"  - Train Positives: {n_pos}")
    print(f"  - Train Negatives: {len(selected_neg_indices)}")
    print(f"    - Stable  : {len(set(selected_neg_indices) & set(stable_neg_indices))}")
    print(f"    - Unstable: {len(set(selected_neg_indices) & set(unstable_neg_indices))}")

    # Update Mask
    df.loc[pos_train_indices, "train_sample_mask"] = True
    df.loc[selected_neg_indices, "train_sample_mask"] = True

    # 4.5 Feature Normalization (Z-Score) - ADDED
    # Standardize features to have mean=0 and std=1.
    # This is crucial for Neural Networks (GCN) and distance-based models.
    # RF/XGBoost are robust to this.
    print(f"[Info] Applying Z-Score Normalization to features...")

    exclude_cols = [
        "label",
        "split",
        "train_sample_mask",
        "is_stable",
        "mean_vel",
        "top20_abs_mean",
        "ratio",
        "geometry",
        "slide_pixels",
        "total_pixels",
        "centroid_x",
        "centroid_y",
    ]

    feature_cols = [c for c in df.columns if c not in exclude_cols and not c.startswith("Unnamed")]

    # Calculate stats and normalize
    # We use global stats here for simplicity and to ensure the map is consistent.
    # (Strict ML would use train stats only, but for spatial mapping global is often preferred to avoid boundary artifacts)
    stats_log = {}
    for col in feature_cols:
        # Check if numeric
        if pd.api.types.is_numeric_dtype(df[col]):
            mean_val = df[col].mean()
            std_val = df[col].std()
            if std_val == 0:
                std_val = 1.0  # Avoid division by zero

            df[col] = (df[col] - mean_val) / (std_val + 1e-9)
            stats_log[col] = {"mean": mean_val, "std": std_val}

    print(f"  - Normalized {len(feature_cols)} feature columns.")
    # Optional: Save stats to metadata if needed later

    # 5. Save
    print(f"[Info] Saving Tabular Dataset to {OUTPUT_PATH}...")
    df.to_parquet(OUTPUT_PATH, index=True)
    print("[Done]")


if __name__ == "__main__":
    main()
