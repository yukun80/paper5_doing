"""
Script: plot_vif_tol_combo.py
Description: Generates a Dual-Axis Combo Chart (Bar + Line).
             Primary Axis (Bottom): VIF (Bar Chart).
             Secondary Axis (Top): TOL (Line Chart).
python scripts/04_Fig/01_Feature_Analysis/plot_vif_tol_combo.py
"""

import sys
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from pathlib import Path

# Add current directory to path
current_dir = Path(__file__).resolve().parent
sys.path.append(str(current_dir))

# Add project root specific script path for common utils
# Structure: scripts/04_Fig/01_Feature_Analysis/ -> scripts/00_common
common_utils_dir = current_dir.parent.parent / "00_common"
sys.path.append(str(common_utils_dir))

from collinearity_analyzer import CollinearityAnalyzer
import path_utils

# --- Publication Style Setup ---
plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman"], "font.size": 14, "savefig.dpi": 300})

EXPERIMENTS = [{"name": "Dynamic", "config": "metadata/dataset_config_dynamic.yaml", "mode": "dynamic"}]


def plot_experiment(exp_meta):
    name = exp_meta["name"]
    print(f"\n=== Processing Combo Chart: {name} ===")

    # 1. Load Config & Resolve Path Dynamically
    try:
        # Load yaml to get SU ID
        with open(exp_meta["config"], "r", encoding="utf-8") as f:
            config_data = yaml.safe_load(f)

        # Resolve Base Data Directory (Project Root / 04_tabular_SU)
        # We assume the script is run from Project Root, but let's be safe
        project_root = Path.cwd()
        if not (project_root / "04_tabular_SU").exists():
            # Fallback if running from script dir
            project_root = current_dir.parents[2]

        base_data_dir = project_root / "04_tabular_SU"

        # Use path_utils to find the correct subdirectory (e.g., su_a10000_c01)
        su_data_dir = path_utils.resolve_su_path(base_data_dir, config=config_data)

        # Construct full path
        data_path = su_data_dir / f"tabular_dataset_{exp_meta['mode']}.parquet"
        print(f"[Info] Resolved Data Path: {data_path}")

        target_features = CollinearityAnalyzer.parse_config(exp_meta["config"])
        df = CollinearityAnalyzer.load_aligned_data(str(data_path), target_features)
    except Exception as e:
        print(f"[Skip] {name}: {e}")
        import traceback

        traceback.print_exc()
        return

    # 2. Compute Metrics
    analyzer = CollinearityAnalyzer(df, df.columns.tolist())
    vif_df = analyzer.run_analysis()

    # Sort by VIF Descending (Highest Risk at Top)
    vif_df = vif_df.sort_values("VIF", ascending=True).reset_index(drop=True)

    # 3. Setup Dual Axis Plot
    fig, ax1 = plt.subplots(figsize=(12, max(8, len(vif_df) * 0.6)))
    ax2 = ax1.twiny()  # Create secondary x-axis sharing the same y-axis

    y_pos = np.arange(len(vif_df))

    # --- Layer 1: VIF Bars (Bottom Axis) ---
    # Multi-tier Color Mapping Logic (Thresholds: 5, 10)
    colors = []
    for val in vif_df["VIF"]:
        if val < 5:
            colors.append("#1a9850")  # Green (Safe)
        elif val < 10:
            colors.append("#fee08b")  # Yellow (Warning)
        else:
            colors.append("#d73027")  # Orange-Red (High Collinearity)

    bars = ax1.barh(y_pos, vif_df["VIF"], color=colors, alpha=0.8, height=0.6, label="VIF")

    # Add VIF Value Labels on Bars
    for i, bar in enumerate(bars):
        width = bar.get_width()
        ax1.text(
            width + 0.2, bar.get_y() + bar.get_height() / 2, f"{width:.2f}", va="center", fontsize=14, color="black"
        )

    ax1.set_xlabel("Variance Inflation Factor (VIF)", fontweight="bold", color="#2c3e50", fontsize=16)
    ax1.tick_params(axis="x", labelcolor="#2c3e50", labelsize=14)
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(vif_df["Feature"], fontsize=14)

    # Increase X-limit for VIF to provide space for legend on the right
    max_vif = vif_df["VIF"].max()
    ax1.set_xlim(0, max(12, max_vif * 1.5))

    # --- Layer 2: TOL Line (Top Axis) ---
    # We set xlim to (-0.4, 1.5):
    # - The negative -0.4 creates a "safe lane" on the left so TOL points don't hit the Y-axis/VIF labels.
    # - The 1.5 keeps the right-side "safe lane" for the legend.
    line = ax2.plot(
        vif_df["TOL"],
        y_pos,
        color="darkblue",
        linestyle="--",
        marker="o",
        linewidth=1.5,
        markersize=8,
        label="TOL",
        alpha=0.9,
    )

    ax2.set_xlabel("Tolerance (TOL)", fontweight="bold", color="darkblue", fontsize=14)
    ax2.tick_params(axis="x", colors="darkblue", labelcolor="darkblue", labelsize=14)
    ax2.spines['top'].set_color('darkblue')
    ax2.set_xlim(0, 1.42857)  # Starts from 0, keeps 1.0 at 70% width (1.0/0.7)
    # Manually set ticks to ensure we only show 0 to 1.0
    ax2.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])

    # Remove any background grids
    ax1.grid(False)
    ax2.grid(False)

    # --- Simplified Legend ---
    from matplotlib.patches import Patch

    legend_elements = [
        Patch(facecolor="#1a9850", label="VIF < 5 (Ideal)"),
        Patch(facecolor="#fee08b", label="VIF 5-10"),
        Patch(facecolor="#d73027", label="VIF $\geq$ 10"),
        plt.Line2D([0], [0], color="darkblue", linestyle="--", marker="o", label="TOL"),
    ]
    ax1.legend(handles=legend_elements, loc="lower right", frameon=True, fontsize=16)

    # Title removed per user request
    # plt.title(...)

    # Save
    out_file = Path(f"scripts/04_Fig/01_Feature_Analysis/output/Fig_Collinearity_{name}_Combo.png")
    plt.savefig(out_file, bbox_inches="tight")
    print(f"[Success] Saved: {out_file}")
    plt.close()


def main():
    for exp in EXPERIMENTS:
        plot_experiment(exp)


if __name__ == "__main__":
    main()
