@echo off
SETLOCAL EnableDelayedExpansion

:: ==============================================================================
:: InSAR-Gated GNNExplainer Models-Only Runner (Windows Batch Version)
:: CONFIGURATION: DYNAMIC (Post-fire)
::
:: SKIPS Data Processing (Phase 0 & 1). Starts from Model Training.
:: ==============================================================================

echo.
echo ============================================================
echo [PHASE 2] Random Forest Pipeline
echo ============================================================
echo Training RF...
python "experiments/01_ml_baselines/random forest/train_rf.py" --mode dynamic
if %ERRORLEVEL% NEQ 0 (echo [ERROR] Failed in RF Training && exit /b %ERRORLEVEL%)

echo Inference and Mapping RF...
python "experiments/01_ml_baselines/random forest/inference_rf.py" --mode dynamic
if %ERRORLEVEL% NEQ 0 (echo [ERROR] Failed in RF Inference && exit /b %ERRORLEVEL%)

echo.
echo ============================================================
echo [PHASE 3] SVM Pipeline
echo ============================================================
echo Training SVM...
python experiments/01_ml_baselines/svm/train_svm.py --mode dynamic
if %ERRORLEVEL% NEQ 0 (echo [ERROR] Failed in SVM Training && exit /b %ERRORLEVEL%)

echo Inference and Mapping SVM...
python experiments/01_ml_baselines/svm/inference_svm.py --mode dynamic
if %ERRORLEVEL% NEQ 0 (echo [ERROR] Failed in SVM Inference && exit /b %ERRORLEVEL%)

echo.
echo ============================================================
echo [PHASE 4] XGBoost Pipeline
echo ============================================================
python -c "import xgboost" 2>NUL
if %ERRORLEVEL% EQU 0 (
    echo Training XGBoost...
    python "experiments/01_ml_baselines/xgboost/train_xgb.py" --mode dynamic
    if %ERRORLEVEL% NEQ 0 (echo [ERROR] Failed in XGBoost Training && exit /b %ERRORLEVEL%)
    
    echo Inference and Mapping XGBoost...
    python "experiments/01_ml_baselines/xgboost/inference_xgb.py" --mode dynamic
    if %ERRORLEVEL% NEQ 0 (echo [ERROR] Failed in XGBoost Inference && exit /b %ERRORLEVEL%)
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

echo 3. Explaining High-Risk Nodes...
python experiments/GNNExplainer/explain_landslide.py --mode dynamic
if %ERRORLEVEL% NEQ 0 (echo [ERROR] Failed in Explanation && exit /b %ERRORLEVEL%)

echo 4. Applying InSAR Correction...
python experiments/GNNExplainer/insar_correction.py --mode dynamic
if %ERRORLEVEL% NEQ 0 (echo [ERROR] Failed in Correction && exit /b %ERRORLEVEL%)

echo.
echo ============================================================
echo [PHASE 7] Model Comparison
echo ============================================================
echo Generating ROC/PR Comparison Plots...
python experiments/03_comparison/compare_models.py --mode dynamic
if %ERRORLEVEL% NEQ 0 (echo [ERROR] Failed in Comparison && exit /b %ERRORLEVEL%)

echo.
echo ============================================================
echo [SUCCESS] Models-Only Pipeline Completed Successfully!
echo ============================================================
pause
