"""
Collinearity Analysis Toolkit (v2.0 Config-Driven)
Author: AI Assistant (Perfectionist Virgo Edition)
Date: 2026-01-10
Description: Provides high-fidelity collinearity metrics (VIF, TOL, Correlation).
             Strictly aligned with YAML experiment configurations.
"""

import yaml
import pandas as pd
import numpy as np
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant
from typing import List, Tuple, Dict, Any
from pathlib import Path

class CollinearityAnalyzer:
    """
    A standalone class to compute statistical metrics for multicollinearity.
    Design: Config-Driven Data Loading.
    """
    
    def __init__(self, data: pd.DataFrame, feature_cols: List[str]):
        """
        Initialize with already loaded and filtered data.
        """
        # Ensure we only work with the specified features
        # Drop rows with NaN to avoid statsmodels errors
        self.data = data[feature_cols].dropna()
        self.features = feature_cols
        self._results = None

    def get_correlation_matrix(self) -> pd.DataFrame:
        """Computes Pearson correlation matrix."""
        return self.data.corr()

    def run_analysis(self) -> pd.DataFrame:
        """
        Calculates VIF and TOL for each feature.
        Returns:
            pd.DataFrame: Columns ['Feature', 'VIF', 'TOL']
        """
        # Add constant for VIF calculation (standard intercept)
        X = add_constant(self.data)
        
        vif_data = []
        for i, col in enumerate(self.features):
            # i+1 because index 0 is the constant
            try:
                vif = variance_inflation_factor(X.values, i + 1)
            except Exception as e:
                print(f"[Error] VIF calculation failed for {col}: {e}")
                vif = np.inf
                
            # Handle infinite VIF
            tol = 1.0 / vif if vif != 0 else 0.0
            
            vif_data.append({
                "Feature": col,
                "VIF": vif,
                "TOL": tol
            })
            
        self._results = pd.DataFrame(vif_data)
        return self._results

    @staticmethod
    def parse_config(yaml_path: str) -> List[str]:
        """
        Parses the dataset_config.yaml to extract the exact list of input features.
        """
        print(f"[Info] Parsing config: {yaml_path}")
        with open(yaml_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
            
        # Extract features defined in 'factors'
        # We only care about the 'name' field
        feature_names = []
        if 'factors' in config:
            for factor in config['factors']:
                # Optional: Filter by role if needed, but usually we analyze all inputs
                # Using 'name' as the column identifier
                feature_names.append(factor['name'])
        
        print(f"[Info] Found {len(feature_names)} features in config.")
        return feature_names

    @staticmethod
    def load_aligned_data(parquet_path: str, target_features: List[str]) -> pd.DataFrame:
        """
        Loads data and strictly keeps ONLY the features defined in the config.
        Implements SMART COLUMN MATCHING to handle prefixes (static_env_) and suffixes (_mean/_mode).
        """
        print(f"[Info] Loading data from: {parquet_path}")
        if not Path(parquet_path).exists():
            # Try looking relative to project root
            project_root = Path(__file__).resolve().parent.parent.parent
            alt_path = project_root / parquet_path
            if alt_path.exists():
                parquet_path = alt_path
            else:
                raise FileNotFoundError(f"Parquet file not found: {parquet_path}")

        df = pd.read_parquet(parquet_path)
        
        selected_data = {}
        missing_features = []
        
        print("[Info] Matching config features to Parquet columns...")
        
        for feature in target_features:
            # Strategies to find the column
            # 1. Exact match (Unlikely but possible)
            if feature in df.columns:
                selected_data[feature] = df[feature]
                continue
                
            # 2. Search for patterns: {prefix}_{feature}_{suffix}
            # We prefer '_mean' for continuous and '_mode' for categorical
            candidates = [c for c in df.columns if f"_{feature}_" in c or c.endswith(f"_{feature}")]
            
            if not candidates:
                # Try looser match: just contains the feature name?
                # Be careful not to match "Slope" in "SlopeLength" if that existed
                candidates = [c for c in df.columns if feature in c]
            
            if not candidates:
                missing_features.append(feature)
                continue
            
            # Selection Logic among candidates
            best_col = None
            
            # Priority A: Ends with _mean (Representative for continuous)
            mean_cols = [c for c in candidates if c.endswith('_mean')]
            if mean_cols:
                best_col = mean_cols[0]
            
            # Priority B: Ends with _mode (Representative for categorical)
            elif any(c.endswith('_mode') for c in candidates):
                best_col = [c for c in candidates if c.endswith('_mode')][0]
                
            # Priority C: Just pick the first one (Fallback)
            else:
                best_col = candidates[0]
            
            # Store with the SIMPLE name (Clean for plotting)
            selected_data[feature] = df[best_col]
            # print(f"  Mapped '{feature}' -> '{best_col}'")

        if missing_features:
            raise ValueError(f"Could not find columns for config features: {missing_features}")
            
        df_selected = pd.DataFrame(selected_data)
        
        # 3. Type Safety Check
        # Check if any column is non-numeric
        non_numeric = df_selected.select_dtypes(exclude=[np.number]).columns
        if len(non_numeric) > 0:
            print(f"[Warning] The following features are Non-Numeric (likely Categorical Strings) and will be excluded: {list(non_numeric)}")
            df_selected = df_selected.drop(columns=non_numeric)
            
        return df_selected
