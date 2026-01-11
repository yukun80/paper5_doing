# 数据处理与数据集构建流程方案 (v5.1 - Unified ML Framework)

## 1. 设计理念：统一数据源与因果解耦

为了确保科学对比的严谨性，所有模型（从简单的 SVM 到复杂的 CXGNN）必须基于**完全相同**的特征提取结果和样本划分。

**核心原则：**
1.  **单源真理 (Single Source of Truth)**：所有模型统一从 `tabular_dataset.parquet` 读取数据。
2.  **角色前缀 (Role Prefixing)**：在数据构建阶段，为特征列名增加 `static_env_`, `dynamic_forcing_`, `constraint_` 前缀，实现自动化的特征角色识别。
3.  **确定性划分 (Deterministic Split)**：严格按 SU_ID 4100 阈值进行空间分割，禁止随机 Shuffle 以防空间信息泄露。

---

## 2. 目录结构与关键文件

*   `04_tabular_SU/`
    *   `tabular_dataset.parquet`: **核心主表**。包含所有 SU 的均值/标准差特征、二值化标签以及 `split` 标记。
*   `experiments/01_ml_baselines/`
    *   `ml_utils.py`: **通用逻辑库**。封装了数据加载、下采样、指标计算和结果保存，确保 RF/XGB/SVM 的评估口径完全一致。

---

## 3. 自动化处理流程 (Run via run_all_experiments.bat)

### Phase 0: 基础特征处理
*   运行 `scripts/00_common/` 下的 10-40 脚本，完成栅格堆叠、统计特征提取、标签生成和拓扑构建。
*   **注意**：`10_build_raster_stack.py` 现支持 `--config` 参数，可指定使用动态或静态配置。

### Phase 1: 统一表格构建 (`scripts/MakeDataset_Tabular/`)
*   **核心动作**：
    1.  合并 `su_features` 和 `su_labels`。
    2.  根据 `--config` 指定的配置文件（如 `metadata/dataset_config_dynamic.yaml`），重命名特征列。
    3.  注入 `split` 列。
*   **产出**：`tabular_dataset.parquet`。

### Phase 2: 图数据集构建 (`scripts/MakeDataset_CXGNN/`)
*   **核心动作**：
    1.  从 `tabular_dataset.parquet` 读取节点属性和划分。
    2.  从 `edges.parquet` 读取拓扑。
    3.  预计算 1-hop 和 2-hop 邻域。
*   **产出**：`causal_graph_data.pkl` (通用图数据格式，供 GCN 和 GNNExplainer 使用)。

### Phase 3: GCN 训练与 GNNExplainer 解释 (`experiments/GNNExplainer/`)
*   **模型训练**：使用标准 GCN 模型训练滑坡易发性预测器（注意：InSAR 数据**不**作为模型输入）。
*   **机理解释**：使用 `GNNExplainer` 对高风险节点（TP/FN）进行解释，提取关键边（Edge Mask）和关键特征（Feature Mask）。
*   **InSAR 物理校正**：
    1.  读取 `InSAR_desc_...` 形变速率栅格。
    2.  按阈值（如 >10mm/yr）划分为形变等级。
    3.  对 GCN 生成的易发性图进行逻辑运算：若 `InSAR_Level == High`，则强制设为 `High Risk`。
*   **验证**：对比动态配置与静态配置下的解释结果，证明动态因子的激活作用。

---

## 4. 特征角色定义与双重配置

项目现在支持两套实验配置，通过 `metadata/` 下的 yaml 文件控制：

### A. Dynamic Configuration (`dataset_config_dynamic.yaml`) - **Main**
*   **Static Environment**: 地形、岩性、降雨。
*   **Dynamic Forcing**: `dNDVI`, `dNBR`, `dMNDWI` (差分指数)。
*   **目的**: 捕捉火后环境的动态变化。

### B. Static Configuration (`dataset_config_static.yaml`) - **Control**
*   **Static Environment**: 地形、岩性 + **Pre-fire Indices** (S2_NBR_Pre, S2_NDVI_Pre...)。
*   **Dynamic Forcing**: **Empty / None**.
*   **目的**: 模拟传统方法（仅关注静态背景），作为对照组以证明 Dynamic 特征的必要性。

---

## 5. 预期产出标准

所有模型运行后必须生成：
1.  **CSV 预测表**: 包含 `su_id`, `label`, `prob`, `split`。
2.  **GeoTIFF 易发性图**: 10m 分辨率，命名为 `LSM_[Model]_[Type].tif`。
3.  **解释报告**: 
    *   **Feature Mask**: 关键特征的重要性权重可视化。
    *   **Explanation Subgraph**: 关键致灾子图的可视化。
