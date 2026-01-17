@echo off
SETLOCAL EnableDelayedExpansion

:: ==============================================================================
:: InSAR-Gated GNNExplainer Full Pipeline Runner (Windows Batch Version)
:: CONFIGURATION: DYNAMIC (Post-fire)
:: ==============================================================================

echo.
echo ============================================================
echo [PHASE 0] Raw Data Processing (Common Foundation)
echo ============================================================
echo 1. Building Raster Stack...
python scripts/00_common/10_build_raster_stack.py --config metadata/dataset_config_dynamic.yaml --mode dynamic
if %ERRORLEVEL% NEQ 0 (echo [ERROR] Failed in Raster Stack Builder && exit /b %ERRORLEVEL%)

echo 2. Extracting Slope Unit Features...
python scripts/00_common/20_extract_su_features.py --config metadata/dataset_config_dynamic.yaml --mode dynamic
if %ERRORLEVEL% NEQ 0 (echo [ERROR] Failed in Feature Extraction && exit /b %ERRORLEVEL%)

echo 3. Generating Labels...
python scripts/00_common/30_generate_labels.py --config metadata/dataset_config_dynamic.yaml --mode dynamic
if %ERRORLEVEL% NEQ 0 (echo [ERROR] Failed in Label Generation && exit /b %ERRORLEVEL%)

echo 4. Building Graph Topology...
python scripts/00_common/40_build_graph.py --config metadata/dataset_config_dynamic.yaml --mode dynamic
if %ERRORLEVEL% NEQ 0 (echo [ERROR] Failed in Graph Builder && exit /b %ERRORLEVEL%)

echo.
echo ============================================================
echo [PHASE 1] Data Standardization
echo ============================================================
echo Building Unified Tabular Dataset...
python scripts/MakeDataset_Tabular/build_tabular_dataset.py --config metadata/dataset_config_dynamic.yaml --mode dynamic
if %ERRORLEVEL% NEQ 0 (echo [ERROR] Failed in Tabular Dataset Builder && exit /b %ERRORLEVEL%)


echo.
echo ============================================================
echo [PHASE 2] Random Forest Pipeline
echo ============================================================
echo Training RF...
python "experiments/01_ml_baselines/random forest/train_rf.py" --mode dynamic
echo Inference and Mapping RF...
python "experiments/01_ml_baselines/random forest/inference_rf.py" --mode dynamic

echo.
echo ============================================================
echo [PHASE 3] SVM Pipeline
echo ============================================================
echo Training SVM...
python experiments/01_ml_baselines/svm/train_svm.py --mode dynamic
echo Inference and Mapping SVM...
python experiments/01_ml_baselines/svm/inference_svm.py --mode dynamic

echo.
echo ============================================================
echo [PHASE 4] XGBoost Pipeline
echo ============================================================
python -c "import xgboost" 2>NUL
if %ERRORLEVEL% EQU 0 (
    echo Training XGBoost...
    python "experiments/01_ml_baselines/xgboost/train_xgb.py" --mode dynamic
    echo Inference and Mapping XGBoost...
    python "experiments/01_ml_baselines/xgboost/inference_xgb.py" --mode dynamic
) else (
    echo [SKIP] XGBoost library not found. Skipping Phase 4.
)

echo.
echo ============================================================
echo [PHASE 5] GCN (Deep Learning Baseline) Pipeline
echo ============================================================
echo Training and Mapping GCN...
python experiments/02_dl_gcn/train_gcn.py --mode dynamic
if %ERRORLEVEL% NEQ 0 (echo [ERROR] Failed in GCN Pipeline && exit /b %ERRORLEVEL%)

echo.
echo ============================================================
echo [PHASE 6] GNNExplainer Pipeline (Core Model)
echo ============================================================
echo 1. Training Base GCN...
python experiments/GNNExplainer/train_landslide.py --mode dynamic
if %ERRORLEVEL% NEQ 0 (echo [ERROR] Failed in GNN Training && exit /b %ERRORLEVEL%)

echo 2. Running Full Inference...
python experiments/GNNExplainer/inference_gcn.py --mode dynamic
if %ERRORLEVEL% NEQ 0 (echo [ERROR] Failed in Inference && exit /b %ERRORLEVEL%)

echo.
echo ============================================================
echo [SUCCESS] All Phases Completed Successfully!
echo ============================================================
