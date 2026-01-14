"""
Script: plot_comparison_radar.py
Location: scripts/04_Fig/03_Interpretation/plot_comparison_radar.py
Description:
    A standalone tool to plot overlaid radar charts comparing two different
    experiments (e.g., Dynamic vs Static). 
    
    Usage:
    Manually input the feature importance scores into the dictionaries below
    and run the script. It produces a publication-quality comparison figure.

python scripts/04_Fig/03_Interpretation/plot_comparison_radar.py
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ==============================================================================
# 1. MANUAL DATA INPUT AREA
# ==============================================================================
# Instructions: 
# Paste the scores from your explanation_summary_*.csv files here.
# Features not present in an experiment will automatically be treated as 0.

# Example SU ID for filename
SU_ID = "2050"

# Data from Dynamic Experiment (RED)
DATA_DYNAMIC = {
    "Slope": 0.25, "Aspect": 0.05, "Elevation": 0.12, "Curvature": 0.03, "TWI": 0.08,
    "Lithology": 0.15, "Soil": 0.04, "LULC": 0.02, "Dis2River": 0.06, "Dis2Road": 0.04,
    "Dis2Fault": 0.03, "Precipitation": 0.05, "NBR_Pre": 0.02, "NDVI_Pre": 0.03, "NDWI_Pre": 0.01,
    "dNDVI": 0.18, "dNBR": 0.22, "dMNDWI": 0.05  # Dynamic factors
}

# Data from Static Experiment (BLUE)
DATA_STATIC = {
    "Slope": 0.35, "Aspect": 0.08, "Elevation": 0.15, "Curvature": 0.05, "TWI": 0.10,
    "Lithology": 0.18, "Soil": 0.05, "LULC": 0.03, "Dis2River": 0.08, "Dis2Road": 0.05,
    "Dis2Fault": 0.04, "Precipitation": 0.07, "NBR_Pre": 0.03, "NDVI_Pre": 0.04, "NDWI_Pre": 0.02
    # Dynamic factors are missing in static experiment, will be 0.0
}

# ==============================================================================
# 2. CONFIGURATION
# ==============================================================================
OUTPUT_DIR = Path(__file__).parent / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

# Define feature groups for alphabetical sorting and consistent layout
DYNAMIC_FEATS_LIST = ["dNDVI", "dNBR", "dMNDWI"]

def plot_comparison():
    # A. Get All Feature Names (Union) and Sort
    all_keys = set(DATA_DYNAMIC.keys()) | set(DATA_STATIC.keys())
    
    # Split into Static and Dynamic groups
    static_keys = sorted([k for k in all_keys if k not in DYNAMIC_FEATS_LIST])
    dynamic_keys = sorted([k for k in all_keys if k in DYNAMIC_FEATS_LIST])
    
    # Final Sequence: Static (Alpha) then Dynamic (Alpha)
    # This puts Dynamic factors on the "right side" of the polar plot
    labels = static_keys + dynamic_keys
    
    # B. Extract Values and Pad with 0
    vals_dyn = [DATA_DYNAMIC.get(k, 0.0) for k in labels]
    vals_stat = [DATA_STATIC.get(k, 0.0) for k in labels]
    
    # C. Prepare Polar Plot Data
    num_vars = len(labels)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    
    # Close the loops
    vals_dyn += vals_dyn[:1]
    vals_stat += vals_stat[:1]
    angles += angles[:1]
    
    # D. Drawing
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    
    # --- Plot Series 1: Dynamic (Red) ---
    ax.fill(angles, vals_dyn, color="red", alpha=0.4, label="Dynamic Exp")
    ax.plot(angles, vals_dyn, color="red", linewidth=1.5)
    
    # --- Plot Series 2: Static (Blue) ---
    ax.fill(angles, vals_stat, color="blue", alpha=0.4, label="Static Exp")
    ax.plot(angles, vals_stat, color="blue", linewidth=1.5, linestyle="-")
    
    # E. Visual Refinement
    # 1. Bold the outermost circular frame
    ax.spines['polar'].set_linewidth(3.0)
    
    # 2. X-Axis (Feature Names)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=10, fontweight='bold')
    ax.tick_params(axis='x', pad=25) 
    
    # 3. Y-Axis (Scales) - STRICT LINEAR
    all_vals = vals_dyn + vals_stat
    max_val = max(all_vals) if all_vals else 0.2
    limit = max_val * 1.3 # 30% margin for visual clarity
    ax.set_ylim(0, limit)
    
    r_ticks = np.linspace(0, limit, 6)[1:] # 5 levels
    ax.set_rticks(r_ticks)
    
    # Style tick labels: Pure Black, Bold, with Background Bbox
    ax.set_yticklabels([f"{t:.2f}" for t in r_ticks], fontsize=9, color="black", fontweight="bold")
    for tick in ax.get_yticklabels():
        tick.set_bbox(dict(facecolor='white', edgecolor='none', alpha=0.8, pad=1))
    
    ax.set_rlabel_position(90) # Vertical upward scales
    
    # 4. Grid and Legend
    ax.grid(True, linestyle='--', alpha=0.6, color='gray')
    ax.legend(loc='upper right', bbox_to_anchor=(1.15, 1.15))
    
    # F. Save
    out_path = OUTPUT_DIR / f"comparison_radar_su_{SU_ID}.png"
    plt.savefig(out_path, bbox_inches='tight', dpi=300)
    print(f"[Success] Comparison Radar Chart saved to: {out_path}")
    plt.show()

if __name__ == "__main__":
    plot_comparison()
