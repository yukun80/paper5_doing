"""
Module: path_utils.py
Location: scripts/00_common/path_utils.py
Description:
    Core path management utility for the Landslide Susceptibility Framework.
    Implements dynamic subdirectory resolution based on the Slope Unit (SU) identifier.
    
    Ensures structural isolation of experiments across different SU scales 
    (e.g., 50000 vs 10000) by creating sub-folders within existing output directories.

Author: AI Assistant (Virgo Edition - Perfectionist)
Date: 2026-01-11
"""

import logging
from pathlib import Path
from typing import Dict, Any, Union, Optional

def get_su_name(config: Dict[str, Any]) -> str:
    """
    Extracts a clean identifier name from the Slope Unit filename in config.
    
    Example:
        'su_a50000_c03_geo.tif' -> 'su_a50000_c03_geo'
        
    Args:
        config (Dict): The loaded YAML configuration.
        
    Returns:
        str: The SU stem name used for subdirectory creation.
    """
    try:
        su_filename = config.get("grid", {}).get("files", {}).get("su_id")
        if not su_filename:
            logging.warning("Config missing 'grid.files.su_id'. Using 'default_su'.")
            return "default_su"
        
        # Use .stem to get filename without extension
        return Path(su_filename).stem
    except Exception as e:
        logging.error(f"Failed to resolve SU name from config: {e}")
        return "default_su"

def resolve_su_path(base_dir: Union[str, Path], config: Optional[Dict[str, Any]] = None, su_name: Optional[str] = None) -> Path:
    """
    Resolves the final output directory by appending the SU identifier as a subdirectory.
    Automatically handles directory creation (mkdir -p).
    
    Args:
        base_dir (Union[str, Path]): The root output directory (e.g., '04_tabular_SU').
        config (Dict, optional): If provided, extracts SU name from it.
        su_name (str, optional): If provided, uses this name directly.
        
    Returns:
        Path: The absolute path to the SU-specific subdirectory.
        
    Note:
        Either 'config' or 'su_name' must be provided.
    """
    base_path = Path(base_dir).resolve()
    
    # 1. Determine the SU name
    final_su_name = su_name
    if not final_su_name and config:
        final_su_name = get_su_name(config)
    
    if not final_su_name:
        final_su_name = "unknown_su"
        
    # 2. Construct final path
    target_path = base_path / final_su_name
    
    # 3. Ensure existence (Perfectionist's safety check)
    if not target_path.exists():
        target_path.mkdir(parents=True, exist_ok=True)
        # We don't log here to keep stdout clean, but the dir is guaranteed to exist
        
    return target_path
