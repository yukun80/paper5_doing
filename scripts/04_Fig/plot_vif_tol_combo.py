"""
Script: plot_vif_tol_combo.py
Description: Generates a Dual-Axis Combo Chart (Bar + Line).
             Primary Axis (Bottom): VIF (Bar Chart).
             Secondary Axis (Top): TOL (Line Chart).
python scripts/04_Fig/plot_vif_tol_combo.py
"""

import sys
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path

# Add current directory to path
sys.path.append(str(Path(__file__).parent))
from collinearity_analyzer import CollinearityAnalyzer

# --- Publication Style Setup ---
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman"],
    "font.size": 11,
    "savefig.dpi": 300
})

EXPERIMENTS = [
    {
        "name": "Dynamic",
        "config": "metadata/dataset_config_dynamic.yaml",
        "data": "04_tabular_SU/tabular_dataset_dynamic.parquet"
    }
]

def plot_experiment(exp_meta):
    name = exp_meta["name"]
    print(f"\n=== Processing Combo Chart: {name} ===")
    
    # 1. Load Data
    try:
        target_features = CollinearityAnalyzer.parse_config(exp_meta["config"])
        df = CollinearityAnalyzer.load_aligned_data(exp_meta["data"], target_features)
    except Exception as e:
        print(f"[Skip] {name}: {e}")
        return

    # 2. Compute Metrics
    analyzer = CollinearityAnalyzer(df, df.columns.tolist())
    vif_df = analyzer.run_analysis()
    
    # Sort by VIF Descending (Highest Risk at Top)
    vif_df = vif_df.sort_values("VIF", ascending=True).reset_index(drop=True)
    
    # 3. Setup Dual Axis Plot
    fig, ax1 = plt.subplots(figsize=(10, max(6, len(vif_df)*0.5)))
    ax2 = ax1.twiny() # Create secondary x-axis sharing the same y-axis
    
    y_pos = np.arange(len(vif_df))
    
    # --- Layer 1: VIF Bars (Bottom Axis) ---
    # Multi-tier Color Mapping Logic (Thresholds: 10, 20, 30)
    colors = []
    for val in vif_df['VIF']:
        if val < 10: colors.append('#1a9850')     # Green (Base)
        elif val < 20: colors.append('#fee08b')   # Yellow (Moderate)
        elif val < 30: colors.append('#f46d43')   # Orange (High)
        else: colors.append('#d73027')            # Red (Extreme)
        
    bars = ax1.barh(y_pos, vif_df['VIF'], color=colors, alpha=0.8, height=0.6, label='VIF')
    
    # Add VIF Value Labels on Bars
    for i, bar in enumerate(bars):
        width = bar.get_width()
        ax1.text(width + 0.1, bar.get_y() + bar.get_height()/2, 
                 f"{width:.2f}", va='center', fontsize=9, color='black')

    ax1.set_xlabel('Variance Inflation Factor (VIF)', fontweight='bold', color='#2c3e50')
    ax1.tick_params(axis='x', labelcolor='#2c3e50')
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(vif_df['Feature'])
    
    # Increase X-limit for VIF to provide space for legend on the right
    max_vif = vif_df['VIF'].max()
    ax1.set_xlim(0, max_vif * 1.4) 

    # --- Layer 2: TOL Line (Top Axis) ---
    # We set xlim to (-0.4, 1.5):
    # - The negative -0.4 creates a "safe lane" on the left so TOL points don't hit the Y-axis/VIF labels.
    # - The 1.5 keeps the right-side "safe lane" for the legend.
    line = ax2.plot(vif_df['TOL'], y_pos, color='#34495e', linestyle='--', marker='o', 
                    linewidth=1.2, markersize=6, label='TOL', alpha=0.9)
    
    ax2.set_xlabel('Tolerance (TOL)', fontweight='bold', color='#34495e')
    ax2.tick_params(axis='x', labelcolor='#34495e')
    ax2.set_xlim(-0.4, 1.5) 
    # Manually set ticks to ensure we only show 0 to 1.0
    ax2.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    
    # Add Threshold Lines (10, 20, 30)
    for thresh in [10, 20, 30]:
        ax1.axvline(thresh, color='#95a5a6', linestyle=':', linewidth=0.8, alpha=0.5)
    
    # --- Simplified Legend ---
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#1a9850', label='VIF < 10'),
        Patch(facecolor='#fee08b', label='VIF 10-20'),
        Patch(facecolor='#f46d43', label='VIF 20-30'),
        Patch(facecolor='#d73027', label='VIF $\geq$ 30'),
        plt.Line2D([0], [0], color='#34495e', linestyle='--', marker='o', label='TOL')
    ]
    ax1.legend(handles=legend_elements, loc='lower right', frameon=True, 
               fontsize=9, title="Diagnostics", title_fontsize=10)

    # Title removed per user request
    # plt.title(...)
    
    # Save
    out_file = Path(f"scripts/04_Fig/Fig_Collinearity_{name}_Combo.png")
    plt.savefig(out_file, bbox_inches='tight')
    print(f"[Success] Saved: {out_file}")
    plt.close()

def main():
    for exp in EXPERIMENTS:
        plot_experiment(exp)

if __name__ == "__main__":
    main()
