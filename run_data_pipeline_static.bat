@echo off
SETLOCAL EnableDelayedExpansion

:: ==============================================================================
:: InSAR-Gated GNNExplainer Data Pipeline - STATIC MODE
:: Description: Processes raw data and builds datasets for Static Control analysis.
::              Does NOT include model training.
:: ==============================================================================

set MODE=static
set CONFIG=metadata/dataset_config_static.yaml

echo.
echo ============================================================
echo [Data Pipeline] Starting - Mode: %MODE%
echo ============================================================

echo 1. Building Raster Stack...
python scripts/00_common/10_build_raster_stack.py --config %CONFIG% --mode %MODE%
if %ERRORLEVEL% NEQ 0 (echo [ERROR] Failed in Raster Stack Builder && pause && exit /b %ERRORLEVEL%)

echo 2. Extracting Slope Unit Features...
python scripts/00_common/20_extract_su_features.py --config %CONFIG% --mode %MODE%
if %ERRORLEVEL% NEQ 0 (echo [ERROR] Failed in Feature Extraction && pause && exit /b %ERRORLEVEL%)

echo 3. Generating Labels...
python scripts/00_common/30_generate_labels.py --config %CONFIG% --mode %MODE%
if %ERRORLEVEL% NEQ 0 (echo [ERROR] Failed in Label Generation && pause && exit /b %ERRORLEVEL%)

echo 4. Building Graph Topology...
python scripts/00_common/40_build_graph.py --config %CONFIG% --mode %MODE%
if %ERRORLEVEL% NEQ 0 (echo [ERROR] Failed in Graph Builder && pause && exit /b %ERRORLEVEL%)

echo 5. Building Unified Tabular Dataset...
python scripts/MakeDataset_Tabular/build_tabular_dataset.py --config %CONFIG% --mode %MODE%
if %ERRORLEVEL% NEQ 0 (echo [ERROR] Failed in Tabular Dataset Builder && pause && exit /b %ERRORLEVEL%)

echo.
echo ============================================================
echo [SUCCESS] Data Processing Completed for %MODE% Mode!
echo ============================================================
pause
