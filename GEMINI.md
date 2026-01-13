# Project Context: InSAR-Gated GNNExplainer for Dynamic Landslide Susceptibility

## 1. Project Overview
**Title:** Dynamic Activation of Post-Fire Landslide Susceptibility: A GNNExplainer Approach with InSAR Constraints.
**Goal:** To quantify the "Dynamic Activation" mechanism of post-fire landslides using **GNNExplainer 2.0** (with Counterfactual Inference) and validate it with **InSAR** physical constraints.

## 2. Recent Major Updates (2026-01-13)

### A. Algorithm & Interpretation (GNNExplainer 2.0)
*   **Architecture**: Fully decoupled from `02_dl_gcn` baseline. Now uses a native `GcnEncoderNode` integrated with `train_landslide.py`.
*   **Counterfactual Inference (CPD)**: Implemented `CPD = P_orig - P_cf` to quantify the absolute risk contribution of fire disturbances.
*   **Dynamic Sensitivity (DSI)**: Implemented `DSI` index to measure relative model attention to dynamic factors.
*   **Zoning**: Automated classification of "Dynamic-Triggered" vs "Static-Dominant" landslides based on CPD/DSI.
*   **Optimization**: Tuned default parameters (`epoch=50`, `run=1`) for high-efficiency debugging without sacrificing core insights.

### B. Data Engineering
*   **Feature Simplification**: **Removed** `Std`, `Min`, `Max` statistics. Model input is now strictly **18 features** (Mean/Mode only) to prevent information dilution and overfitting.
*   **InSAR-Guided Sampling**: Implemented strict negative sample selection. Training negatives are now prioritized from physically stable areas (`velocity < 0.02`).
*   **Sampling Strategy**: Introduced `random_balanced` (7:3 split + 1:1 undersampling) vs `block_split` (spatial cut) configuration in YAML.

### C. Workflow Automation
*   **Dedicated Pipelines**: Created `run_data_pipeline_*.bat` for pure data processing and `run_gnnexplainer_*.bat` for model iteration.
*   **Path Management**: Multi-scale support (e.g., `su_a500000_c06`) is fully config-driven via `path_utils.py`.

## 3. Directory Structure (Key Components)

*   `experiments/GNNExplainer/`
    *   `explain_landslide.py`: **Core Engine**. Handles interpretation, CPD calculation, and artifact generation.
    *   `adapter.py`: **Data Bridge**. Dynamic sampling strategy implementation.
    *   `models.py`: **Native Model**. GCN implementation compatible with masking.
    *   `inference_gcn.py`: **Mapping**. Generates full-region susceptibility maps.
*   `scripts/00_common/`
    *   `20_extract_su_features.py`: **Feature Extractor**. Now optimized for Mean/Mode only.
*   `metadata/`
    *   `dataset_config_dynamic.yaml`: **Control Center**. Manages file paths, factors, and sampling strategies.

## 4. Pending Tasks
1.  **Consistency Fix**: Update `build_tabular_dataset.py` to respect the `sampling.strategy` from YAML config (currently hardcoded to Block Split).
2.  **Visualization**: Develop `visualize_results.py` to plot the "Mechanism Zoning Map" and "DSI vs Slope" scatter plots using the new Artifacts.
3.  **Execution**: Run the full pipeline on `su_a500000_c06` to generate final publication figures.
