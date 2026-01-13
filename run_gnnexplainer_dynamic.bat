@echo off
SETLOCAL EnableDelayedExpansion

:: ==============================================================================
:: GNNExplainer Pipeline Runner - DYNAMIC MODE
:: Description: Runs the full GNNExplainer analysis for Post-Fire Dynamic Factors.
:: ==============================================================================

set MODE=dynamic

echo.
echo ============================================================
echo [GNNExplainer] Starting Pipeline - Mode: %MODE%
echo ============================================================

echo 1. Training Base GCN...
python experiments/GNNExplainer/train_landslide.py --mode %MODE%
if %ERRORLEVEL% NEQ 0 (echo [ERROR] Failed in GNN Training && pause && exit /b %ERRORLEVEL%)

echo 2. Running Full Inference...
python experiments/GNNExplainer/inference_gcn.py --mode %MODE%
if %ERRORLEVEL% NEQ 0 (echo [ERROR] Failed in Inference && pause && exit /b %ERRORLEVEL%)

echo 3. Explaining High-Risk Nodes...
python experiments/GNNExplainer/explain_landslide.py --mode %MODE% --num-explain 10
if %ERRORLEVEL% NEQ 0 (echo [ERROR] Failed in Explanation && pause && exit /b %ERRORLEVEL%)

echo 4. Applying InSAR Correction...
python experiments/GNNExplainer/insar_correction.py --mode %MODE%
if %ERRORLEVEL% NEQ 0 (echo [ERROR] Failed in Correction && pause && exit /b %ERRORLEVEL%)

echo.
echo ============================================================
echo [SUCCESS] GNNExplainer Dynamic Pipeline Completed!
echo ============================================================
pause