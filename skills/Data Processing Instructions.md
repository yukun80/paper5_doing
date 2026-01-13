# 数据处理与数据集构建流程 (v6.0 - Simplified Features)

## 1. 核心变更
*   **特征精简**: 放弃了 `Std`, `Min`, `Max` 等统计量，连续变量仅保留 **均值 (Mean)**，分类变量仅保留 **众数 (Mode)**。这使得每个 Slope Unit 仅有 18 个输入特征，极大降低了冗余。
*   **采样策略**: 引入了 `sampling.strategy` 配置，支持 `block_split` 和 `random_balanced` 的无缝切换。
*   **InSAR 筛选**: 在构建训练集时，利用 InSAR `mean_vel` 和 `top20_abs_mean` 指标优先筛选物理稳定的负样本。

## 2. 目录结构规范

*   **`metadata/`**: 存放 `dataset_config_*.yaml`，控制所有参数（包括采样策略）。
*   **`04_tabular_SU/<SU_ID>/`**:
    *   `su_features_dynamic.parquet`: 原始特征表（无标签）。
    *   `su_labels_dynamic.parquet`: 标签表。
    *   `tabular_dataset_dynamic.parquet`: **最终训练数据集**（包含 Features, Label, Split, InSAR_Stability）。
*   **`experiments/GNNExplainer/results/<SU_ID>/`**: 存放解释结果 CSV 和 Artifacts。

## 3. 脚本执行指南

### 3.1 仅处理数据 (不训练)
*   **动态模式**: `.\run_data_pipeline_dynamic.bat`
*   **静态模式**: `.\run_data_pipeline_static.bat`
*   **作用**: 重新生成特征提取、构建图拓扑、生成 Parquet 数据集。当修改了 `20_extract_su_features.py` 或 `build_tabular_dataset.py` 后需运行此步。

### 3.2 全流程 (含训练与解释)
*   **动态模式**: `.\run_gnnexplainer_dynamic.bat`
*   **静态模式**: `.\run_gnnexplainer_static.bat`
*   **作用**: 依次执行 训练 -> 推理 -> 解释 -> 校正。

### 3.3 切换采样策略
1.  打开 `metadata/dataset_config_dynamic.yaml`。
2.  修改 `sampling` 块：
    ```yaml
    sampling:
      strategy: "random_balanced" # 或 "block_split"
      train_ratio: 0.7
      pos_neg_ratio: 1.0
    ```
3.  **注意**: 目前 `build_tabular_dataset.py` 尚未完全动态读取此配置（待修复），GNN Adapter 已支持。建议在修复 `build_tabular_dataset.py` 前，GNN 实验结果以 `adapter.py` 的日志输出为准。

## 4. 特征列名约定
*   **静态环境**: `static_env_<FeatureName>_mean` (e.g., `static_env_Slope_mean`)
*   **动态扰动**: `dynamic_forcing_<FeatureName>_mean` (e.g., `dynamic_forcing_dNBR_mean`)
*   这种前缀命名法是自动化分析脚本识别特征角色的基础，**严禁修改**。