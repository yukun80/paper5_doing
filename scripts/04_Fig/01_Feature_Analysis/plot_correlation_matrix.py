"""
Script: plot_correlation_matrix.py
Description: Generates a Clean Lower-Triangle Correlation Heatmap.
             Focuses on high data-ink ratio and academic aesthetics.

python scripts/04_Fig/01_Feature_Analysis/plot_correlation_matrix.py
"""

import sys
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from pathlib import Path

# Add current directory to path
sys.path.append(str(Path(__file__).parent))
from collinearity_analyzer import CollinearityAnalyzer

# --- Publication Style Setup ---
plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman"],
        "font.size": 11,
        "axes.labelsize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "savefig.dpi": 300,
        "figure.autolayout": True,
    }
)

EXPERIMENTS = [
    {
        "name": "Dynamic",
        "config": "metadata/dataset_config_dynamic.yaml",
        "data": "04_tabular_SU/tabular_dataset_dynamic.parquet",
    }
]


def plot_experiment(exp_meta):
    name = exp_meta["name"]
    print(f"\n=== Processing Heatmap: {name} ===")

    # 1. Load Data
    try:
        target_features = CollinearityAnalyzer.parse_config(exp_meta["config"])
        df = CollinearityAnalyzer.load_aligned_data(exp_meta["data"], target_features)
    except Exception as e:
        print(f"[Skip] {name}: {e}")
        return

    # 2. Compute Correlation
    analyzer = CollinearityAnalyzer(df, df.columns.tolist())
    corr = analyzer.get_correlation_matrix()

    # 3. Setup Plot
    # Dynamic sizing based on number of features
    size = max(8, len(df.columns) * 0.6)
    fig, ax = plt.subplots(figsize=(size, size))

    # 4. Align colorbar height with matrix height
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)

    # 5. Draw Heatmap (Full Matrix)
    sns.heatmap(
        corr,
        cmap="RdBu_r",
        center=0,
        vmax=1,
        vmin=-1,
        square=True,
        linewidths=0.5,
        cbar_ax=cax,
        cbar_kws={"label": "Pearson Correlation Coefficient ($r$)"},
        annot=True,
        fmt=".2f",
        annot_kws={"size": 9},
        ax=ax,
    )

    # 6. Final Aesthetic Polish
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    plt.setp(ax.get_yticklabels(), rotation=0)

    # Title removed per user request for publication-ready style
    # plt.title(...)

    # Save
    out_file = Path(f"scripts/04_Fig/01_Feature_Analysis/output/Fig_Correlation_{name}.png")
    plt.savefig(out_file, bbox_inches="tight")
    print(f"[Success] Saved: {out_file}")
    plt.close()


def main():
    for exp in EXPERIMENTS:
        plot_experiment(exp)


if __name__ == "__main__":
    main()
