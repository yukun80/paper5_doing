# Project Context: InSAR-Gated GNNExplainer for Dynamic Landslide Susceptibility

## 1. Project Overview
**Title:** Dynamic Activation of Post-Fire Landslide Susceptibility: A GNNExplainer Approach with InSAR Constraints.
**Goal:** To overcome the "Static Bias" of traditional ML models in post-fire environments by using **GNNExplainer** to reveal the "Dynamic Activation" mechanism (dNBR, dNDVI) and using **InSAR** as a physical post-correction constraint.

**Core Architecture:**
*   **Paradigm:** Config-Driven Data Pipeline -> Common Foundation -> **Unified Tabular Master** -> Model Baselines & GNNExplainer.
*   **Key Model:** **GCN (Base Model)** + **GNNExplainer (Interpretation Engine)**.
*   **Tech Stack:** Python, Rasterio, Geopandas, Scikit-learn, PyTorch (Geometric), XGBoost, Tensorboard.

## 2. Directory Structure & Standards

### Root Directory: `E:\Document\paper_library\5th_paper_InSAR\datasets`

*   **`02_aligned_grid/`**: **[Input Source]**
    *   Aligned rasters (10m Resolution).
    *   **Dynamic Factors:** `S2_dNDVI...`, `S2_dNBR...`, `S2_dMNDWI...` (Difference Indices).
    *   **Correction Source:** `InSAR_desc_2024_2025...` (Deformation Velocity).
*   **`04_tabular_SU/`**: **[Processing Output]**
    *   `su_features.parquet`: Raw statistical features.
    *   `tabular_dataset.parquet`: **[Unified Master]** Merged features, labels, and Deterministic Splits.
*   **`05_graph_SU/`**: **[Processing Output]**
    *   `edges.parquet`: Adjacency list.
*   **`metadata/`**:
    *   `dataset_config.yaml`: Global dataset configuration.
*   **`experiments/`**:
    *   `01_ml_baselines/`: RF, SVM, XGBoost.
    *   `02_dl_gcn/`: Standard GCN Baseline.
    *   `GNNExplainer/`: **[Core Innovation]**
        *   `train_landslide.py`: Main GCN training.
        *   `inference_gcn.py`: Full-region probability mapping.
        *   `explain_landslide.py`: Feature/Edge Mask extraction for scientific interpretation.
        *   `insar_correction.py`: Post-processing using InSAR logic.
        *   `adapter.py`: Bridge between Parquet/NetworkX and Dense Tensors.
*   **`scripts/`**:
    *   `00_common/`: Shared raster/graph processing.
    *   `MakeDataset_Tabular/`: Builds the unified tabular dataset.

## 3. Workflow & Pipeline (Status: GNNExplainer Active)

### Phase 0: Common Foundation (Completed)
1.  **Raster Stacking**: `10_build_raster_stack.py`
2.  **Feature Extraction**: `20_extract_su_features.py`
3.  **Labeling**: `30_generate_labels.py`
4.  **Graph Topology**: `40_build_graph.py`

### Phase 1: Unified Dataset (Refactored)
5.  **Tabular Master Builder**: `scripts/MakeDataset_Tabular/build_tabular_dataset.py`
    *   Output: `tabular_dataset.parquet`.

### Phase 2: Baselines & Static Bias Verification
6.  **ML Pipelines**: RF, SVM, and XGBoost scripts in `experiments/01_ml_baselines/`.
    *   **Verification**: Static Bias Confirmed (71.27% Static Importance).

### Phase 3: GNNExplainer & Physics Correction (Active)
7.  **Training**: `experiments/GNNExplainer/train_landslide.py`
    *   Trains a `GcnEncoderNode` compatible with the explainer.
8.  **Inference**: `experiments/GNNExplainer/inference_gcn.py`
    *   Generates `LSM_GCN_Raw_Prob.tif`.
9.  **Explanation**: `experiments/GNNExplainer/explain_landslide.py`
    *   Optimizes masks for High-Risk nodes to reveal Dynamic Factor contribution.
10. **Correction**: `experiments/GNNExplainer/insar_correction.py`
    *   Logic: `Final_Risk = max(Model_Prob, InSAR_High_Deformation)`.

## 4. Development Conventions
*   **Data Split**: Strict deterministic split (SU_ID 4100).
*   **Model Input**: InSAR is **NOT** an input feature; it is only used for post-correction.
*   **Artifacts**: All experiments must save outputs to dedicated subdirectories (`results/`, `models/`, `logs/`) to avoid root directory clutter.

## 5. Recent "Memories" & Decisions
*   **2026-01-10: Advanced Collinearity Analysis & Visualization**:
    *   **Modular Analyzer**: Implemented `collinearity_analyzer.py` with Config-Driven logic and smart column matching (handling `static_env_` prefixes and `_mean` suffixes).
    *   **Feature Optimization**: Replaced `TRI` with `Aspect` in both Static/Dynamic configs to resolve severe multicollinearity with `Slope`.
    *   **High-Fidelity Viz**: 
        *   `plot_correlation_matrix.py`: Full Pearson matrix with tilted labels and perfectly aligned colorbar height.
        *   `plot_vif_tol_combo.py`: Dual-axis (VIF Bars + TOL Line) combo chart with multi-tier risk coloring (10/20/30) and offset-axis layout to prevent legend overlap.
    *   **Aesthetics**: Adhered to "Academic Minimalism" by removing figure titles and streamlining legends for publication.
*   **2026-01-10: Pipeline Finalization & Stabilization**:
    *   **Solved IndexError/RuntimeError in GNNExplainer**: Fixed the loss function by passing full subgraph prediction labels, enabling proper Laplacian regularization.
    *   **Code Quality**: Eliminated PyTorch UserWarnings by using recommended tensor construction patterns (`as_tensor`, `clone().detach()`). Fixed `IndentationError` in `explain.py`.
    *   **Scientific Improvement**: Implemented **InSAR-guided Negative Sampling** in `build_tabular_dataset.py`. Negative samples (Non-landslides) are now physically verified for stability (Velocity < 0.02), ensuring a higher quality training signal.
    *   **Output Management**: Standardized output structure across XGBoost and GNNExplainer (`results/`, `models/`, `logs/`).
    *   **Cleanup**: Removed duplicate code blocks in `20_extract_su_features.py` and archived deprecated `CXGNN` components.

## 6. Pending Tasks
*   **Final Run**: Execute `run_all_experiments_dynamic.bat` for final results.
*   **Analysis**: Evaluate the "Feature Importance" shift between Dynamic and Static models.
*   **Correction**: Verify InSAR units (m vs mm) in `insar_correction.py` if high-deformation zones are still 0.
