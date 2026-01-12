# Project Context: InSAR-Gated GNNExplainer for Dynamic Landslide Susceptibility

## 1. Project Overview
**Title:** Dynamic Activation of Post-Fire Landslide Susceptibility: A GNNExplainer Approach with InSAR Constraints.
**Goal:** To overcome the "Static Bias" of traditional ML models in post-fire environments by using **GNNExplainer** to reveal the "Dynamic Activation" mechanism (dNBR, dNDVI) and using **InSAR** as a physical post-correction constraint.

**Core Architecture:**
*   **Paradigm:** Config-Driven Data Pipeline -> Common Foundation -> **Unified Tabular Master** -> Model Baselines & GNNExplainer.
*   **Multi-Scale Support:** Fully decoupled data/result directories using SU-specific subfolders resolved via `path_utils.py`.
*   **Key Model:** **GCN (Base Model)** + **GNNExplainer (Interpretation Engine)**.

## 2. Directory Structure & Standards (Updated)

### Root Directory: `E:\Document\paper_library\5th_paper_InSAR\datasets`

*   **`02_aligned_grid/`**: **[Input Source]** (Aligned rasters, 10m).
*   **`03_stacked_data/<SU_NAME>/`**: SU-specific multi-band stacks.
*   **`04_tabular_SU/<SU_NAME>/`**: SU-specific features, labels, and `tabular_dataset.parquet`.
*   **`05_graph_SU/<SU_NAME>/`**: SU-specific graph edges.
*   **`experiments/`**:
    *   `01_ml_baselines/`: RF, SVM, XGBoost results saved in `<MODEL>/results/<SU_NAME>/` or `inference_results/<SU_NAME>/`.
    *   `02_dl_gcn/`: GCN Baseline results in `results/<SU_NAME>/`.
    *   `GNNExplainer/`: Checkpoints, Inference, and Explanations in SU-specific subfolders.
*   **`scripts/00_common/path_utils.py`**: Centralized path resolver for all scripts.

## 3. Workflow & Pipeline (Status: Multi-Scale Framework Ready)

### Phase 0: Preprocessing (SU-Agnostic)
*   Standardized scripts 10-40 to auto-generate SU-specific subdirectories.
*   Fixed `20_extract_su_features.py` variable scope and metadata loading bugs.

### Phase 1: Unified Dataset (Percentage-Based Split)
*   **Dynamic Split**: Replaced hardcoded threshold with **40% Test / 60% Train** split based on sorted SU IDs.
*   **Physical Integrity**: Updated `ml_utils.py` to exclude InSAR columns (`is_stable`, `mean_vel`) from training features to prevent data leakage.

### Phase 2: Baselines & Comparison
*   **SVM Upgrade**: Switched to RBF Kernel with probability calibration; fixed parameter incompatibility errors.
*   **Inference Optimization**: All ML/DL inference scripts now generate GeoTIFFs using config-driven SU rasters and dynamic output paths.
*   **Efficiency**: Created `run_models_only_*.bat` to bypass time-consuming preprocessing when iterating on models.

## 4. Development Conventions
*   **Isolation**: NEVER save results directly in the root output folders. Always use `path_utils.resolve_su_path`.
*   **Feature Consistency**: Use `static_env_` and `dynamic_forcing_` prefixes for feature selection; ensure InSAR data remains as a "constraint" only.
*   **Naming**: Standardized probability column name as `prob` across all CSVs.

## 5. Recent "Memories" & Decisions
*   **2026-01-11: Framework Robustness & Bug Squash**:
    *   **Pathing**: Implemented `path_utils.py` to manage multi-scale SU data isolation.
    *   **Data Unpacking**: Fixed `inference_gcn.py` regressions where `input_dim` and GPU transfers were missing.
    *   **Logic Alignment**: Synchronized feature selection between `ml_utils.py` (exclusion-based) and inference scripts (prefix-based) by adding InSAR columns to the exclusion list.
    *   **Code Cleanliness**: Refactored `inference_rf.py` to eliminate redundant function definitions and fix path resolution for feature metadata.
*   **2026-01-11: Scale Transition**: Successfully switched study scale from 50k to 10k SU (`su_a10000_c01_10m.tif`) via configuration.

## 6. Pending Tasks
*   **Validation**: Execute the Models-Only pipeline for the 10k scale and verify ROC/PR metrics.
*   **Explanation**: Analyze Feature Mask shifts in GNNExplainer for the new SU scale.
