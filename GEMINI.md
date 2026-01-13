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

## 3. Workflow & Pipeline (Status: Explanation Phase Ready)

### Phase 0: Preprocessing (SU-Agnostic)
*   Standardized scripts 10-40 to auto-generate SU-specific subdirectories.
*   Fixed `20_extract_su_features.py` variable scope and metadata loading bugs.

### Phase 1: Unified Dataset (Percentage-Based Split)
*   **Dynamic Split**: Replaced hardcoded threshold with **40% Test / 60% Train** split based on sorted SU IDs.
*   **Physical Integrity**: Updated `ml_utils.py` to exclude InSAR columns (`is_stable`, `mean_vel`) from training features to prevent data leakage.

### Phase 2: Baselines & Comparison
*   **SVM Upgrade**: Switched to RBF Kernel with probability calibration; fixed parameter incompatibility errors.
*   **Inference Optimization**: All ML/DL inference scripts now generate GeoTIFFs using config-driven SU rasters and dynamic output paths.

### Phase 3: Mechanism Explanation & Correction (New)
*   **Stability**: GNNExplainer now uses **Multi-Run Averaging** to output robust feature importance.
*   **Context**: Enhanced logic to capture full neighborhood context (Dynamic factors + Topography).
*   **Constraint**: InSAR post-correction logic (`insar_correction.py`) ready for deployment.

## 4. Development Conventions
*   **Isolation**: NEVER save results directly in the root output folders. Always use `path_utils.resolve_su_path`.
*   **Feature Consistency**: Use `static_env_` and `dynamic_forcing_` prefixes for feature selection; ensure InSAR data remains as a "constraint" only.
*   **Naming**: Standardized probability column name as `prob` across all CSVs.

## 5. Recent "Memories" & Decisions
*   **2026-01-12: GNNExplainer Optimization & Stabilization**:
    *   **Stability**: Implemented **Multi-Run Averaging (MRA)** in `explain_landslide.py`. The explainer now runs optimization 5 times (configurable) per node and averages the masks to eliminate random initialization noise.
    *   **Interpretability**: Enhanced neighbor attribute extraction to dynamically capture all `dynamic_` factors and key terrain metrics (Slope, Elev, Aspect) instead of hardcoded lists.
    *   **Robustness**: Added a local fallback definition for `ExplanationArtifact` to handle missing `gnn_viz_kit` dependencies, ensuring `.pkl` artifacts are always generated.
*   **2026-01-11: Framework Robustness & Bug Squash**:
    *   **Pathing**: Implemented `path_utils.py` to manage multi-scale SU data isolation.
    *   **Logic Alignment**: Synchronized feature selection between `ml_utils.py` (exclusion-based) and inference scripts (prefix-based) by adding InSAR columns to the exclusion list.
    *   **Scale Transition**: Successfully switched study scale from 50k to 10k SU (`su_a10000_c01_10m.tif`) via configuration.

## 6. Pending Tasks
*   **Execution**: Run `explain_landslide.py` (with MRA) to generate high-confidence, quantified feature importance for the 10k scale.
*   **Correction**: Execute `insar_correction.py` to produce the final hybrid susceptibility maps.
*   **Visualization**: (Future) Re-integrate `gnn_viz_kit` for publication-quality rendering of the explanation artifacts.