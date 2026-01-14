# Project Context: InSAR-Gated GNNExplainer for Dynamic Landslide Susceptibility

## 1. Project Overview
**Title:** Dynamic Activation of Post-Fire Landslide Susceptibility: A GNNExplainer Approach with InSAR Constraints.
**Goal:** To quantify the "Dynamic Activation" mechanism of post-fire landslides using **GNNExplainer 2.0** (with Counterfactual Inference) and validate it with **InSAR** physical constraints.

## 2. Recent Major Updates (2026-01-14)

### A. InSAR Physical Correction (Major Upgrade)
*   **Object-Based Logic**: Refactored `insar_correction.py` to move from pixel-based to **Slope Unit (SU) based** correction.
*   **Statistical Aggregation**: Now calculates the **Mean of the Bottom 20%** (most significant subsidence) pixels within each SU to define its instability rate ($V_{su}$). 
*   **Bi-Directional Correction**:
    *   **Force Activation**: If $V_{su} < -15 \text{ mm/yr}$, risk is forced to **0.9** (Physical Failure Evidence).
    *   **False Positive Suppression**: If $V_{su} \in [-10, 20] \text{ mm/yr}$ (Stable) AND Model Prob > 0.75, risk is downgraded to **0.7** (Model Hallucination Check).

### B. Algorithm & Interpretation (GNNExplainer 2.0)
*   **Architecture**: Fully decoupled from `02_dl_gcn` baseline. Now uses a native `GcnEncoderNode` integrated with `train_landslide.py`.
*   **Counterfactual Inference (CPD)**: Implemented `CPD = P_orig - P_cf` to quantify the absolute risk contribution of fire disturbances.
*   **Dynamic Sensitivity (DSI)**: Implemented `DSI` index to measure relative model attention to dynamic factors.
*   **Zoning**: Automated classification of "Dynamic-Triggered" vs "Static-Dominant" landslides based on CPD/DSI.

### C. Data Engineering
*   **Feature Simplification**: **Removed** `Std`, `Min`, `Max` statistics. Model input is now strictly **18 features** (Mean/Mode only).
*   **Sampling Strategy**: Introduced `random_balanced` (7:3 split + 1:1 undersampling) vs `block_split` (spatial cut) configuration in YAML.

## 3. Directory Structure (Key Components)

*   `experiments/GNNExplainer/`
    *   `insar_correction.py`: **Physical Validator**. Implements the new Object-Based Bi-Directional Correction logic.
    *   `explain_landslide.py`: **Core Engine**. Handles interpretation, CPD calculation, and artifact generation.
    *   `adapter.py`: **Data Bridge**. Dynamic sampling strategy implementation.
    *   `models.py`: **Native Model**. GCN implementation compatible with masking.
*   `scripts/00_common/`
    *   `20_extract_su_features.py`: **Feature Extractor**. Optimized for Mean/Mode only.
*   `metadata/`
    *   `dataset_config_dynamic.yaml`: **Control Center**. Manages file paths, factors, and sampling strategies.

## 4. Pending Tasks
1.  **CRITICAL FIX**: Refactor `build_tabular_dataset.py` to correctly implement `random_balanced` sampling (currently hardcoded to Block Split). This is blocking the correct execution of the randomized experiment.
2.  **Visualization**: Develop `visualize_results.py` to plot the "Mechanism Zoning Map" and "DSI vs Slope" scatter plots using the new Artifacts.
3.  **Execution**: Run the full pipeline on `su_a500000_c06` to generate final publication figures.