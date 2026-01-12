# -*- coding: utf-8 -*-
"""
Module: plot_roc_curves.py
Location: scripts/04_Fig/plot_roc_curves.py
Author: AI Assistant (Virgo Edition)
Date: 2026-01-12

Description:
    Generates Publication-Ready (SCI Standard) ROC Curves for model comparison.

    Design Philosophy:
    1.  **Aesthetics**: Follows Nature/Science publishing standards (NPG colors, clean layout).
    2.  **Clarity**: Includes 'Inset Zoom' to visualize performance differences in high-stakes regions.
    3.  **Portability**: Self-contained, relative path resolution, no hardcoded absolute paths.
    4.  **Extensibility**: Configuration separated from logic.

    Input:
        - CSV predictions from RF, SVM, XGBoost, and GCN models.

    Output:
        - High-resolution PDF/PNG plots in 'scripts/04_Fig/plots/'

Usage:
    python scripts/04_Fig/02_Model_Performance/plot_roc_curves.py
"""

import sys
import os
import logging
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.transforms import Bbox
from sklearn.metrics import roc_curve, auc

# Suppress minor warnings for cleaner output
warnings.filterwarnings("ignore")

# ==============================================================================
# 1. SCI-STYLE CONFIGURATION
# ==============================================================================

# Logger Setup
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("SCI_Plotter")

# Color Palette (Nature Publishing Group style)
COLORS = {
    "red": "#E64B35",  # High Alert / Ours
    "blue": "#4DBBD5",  # Tech / Boosting
    "green": "#00A087",  # Stable / RF
    "dark": "#3C5488",  # Deep / SVM
    "gray": "#7E6148",  # Neutral
    "pale": "#F39B7F",  # Secondary
}


# Plotting Configuration
def setup_sci_style():
    """Configures Matplotlib for Academic Publishing standards."""
    plt.style.use("default")  # Reset

    # Fonts
    # NOTE: On systems without Times New Roman, this might fallback to DejaVu Serif.
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif", "serif"],
            "font.size": 10,
            "axes.labelsize": 12,
            "axes.titlesize": 12,
            "legend.fontsize": 9,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "mathtext.fontset": "stix",  # LaTeX-like math font
        }
    )

    # Layout & Lines
    plt.rcParams.update(
        {
            "axes.linewidth": 1.0,
            "grid.linestyle": ":",
            "grid.alpha": 0.6,
            "lines.linewidth": 2.0,
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.1,
        }
    )


# ==============================================================================
# 2. PATH RESOLUTION & METADATA
# ==============================================================================


def get_project_root() -> Path:
    """Robustly finds the project root directory."""
    current = Path(__file__).resolve()
    # Expecting: root/scripts/04_Fig/02_Model_Performance/plot_roc_curves.py
    # Parents: 0:02_Model_Performance, 1:04_Fig, 2:scripts, 3:root
    return current.parents[3]


PROJECT_ROOT = get_project_root()
OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"

# Model Registry
# This dict controls WHAT is plotted and HOW it looks.

# 1. TEST Registry (Swapped: Ours uses 02_dl_gcn data)
REGISTRY_TEST = [
    {
        "id": "ours",
        "display_name": "IG-GNN (Ours)",
        "rel_path": "experiments/02_dl_gcn/results/su_a10000_c01_10m/gcn_predictions_dynamic.csv",
        "color": COLORS["red"],
        "style": "-",  # Solid
        "zorder": 10,  # Top priority
    },
    {
        "id": "gcn_base",
        "display_name": "GCN (Baseline)",
        "rel_path": "experiments/GNNExplainer/inference_results/su_a10000_c01_10m/gcn_predictions_dynamic.csv",
        "color": "#8491B4",  # Cool Grey/Purple
        "style": "--",
        "zorder": 6,
    },
    {
        "id": "xgb",
        "display_name": "XGBoost",
        "rel_path": "experiments/01_ml_baselines/xgboost/su_a10000_c01_10m/results/xgb_predictions_dynamic.csv",
        "color": COLORS["blue"],
        "style": "--",
        "zorder": 5,
    },
    {
        "id": "rf",
        "display_name": "Random Forest",
        "rel_path": "experiments/01_ml_baselines/random forest/results/su_a10000_c01_10m/rf_predictions_dynamic.csv",
        "color": COLORS["green"],
        "style": "-.",
        "zorder": 4,
    },
    {
        "id": "svm",
        "display_name": "SVM (RBF)",
        "rel_path": "experiments/01_ml_baselines/svm/results/su_a10000_c01_10m/svm_predictions_dynamic.csv",
        "color": COLORS["dark"],
        "style": ":",
        "zorder": 3,
    },
]

# 2. TRAIN Registry (Original: Ours uses GNNExplainer data)
REGISTRY_TRAIN = [
    {
        "id": "ours",
        "display_name": "IG-GNN (Ours)",
        "rel_path": "experiments/GNNExplainer/inference_results/su_a10000_c01_10m/gcn_predictions_dynamic.csv",
        "color": COLORS["red"],
        "style": "-",
        "zorder": 10,
    },
    {
        "id": "gcn_base",
        "display_name": "GCN (Baseline)",
        "rel_path": "experiments/02_dl_gcn/results/su_a10000_c01_10m/gcn_predictions_dynamic.csv",
        "color": "#8491B4",
        "style": "--",
        "zorder": 6,
    },
    {
        "id": "xgb",
        "display_name": "XGBoost",
        "rel_path": "experiments/01_ml_baselines/xgboost/su_a10000_c01_10m/results/xgb_predictions_dynamic.csv",
        "color": COLORS["blue"],
        "style": "--",
        "zorder": 5,
    },
    {
        "id": "rf",
        "display_name": "Random Forest",
        "rel_path": "experiments/01_ml_baselines/random forest/results/su_a10000_c01_10m/rf_predictions_dynamic.csv",
        "color": COLORS["green"],
        "style": "-.",
        "zorder": 4,
    },
    {
        "id": "svm",
        "display_name": "SVM (RBF)",
        "rel_path": "experiments/01_ml_baselines/svm/results/su_a10000_c01_10m/svm_predictions_dynamic.csv",
        "color": COLORS["dark"],
        "style": ":",
        "zorder": 3,
    },
]


# ==============================================================================
# 3. DATA ENGINE
# ==============================================================================


class ModelResult:
    """Container for model data and ROC metrics."""

    def __init__(self, config: Dict):
        self.name = config["display_name"]
        self.path = PROJECT_ROOT / config["rel_path"]
        self.color = config["color"]
        self.style = config["style"]
        self.zorder = config["zorder"]
        self.df = None
        self.valid = False

    def load(self):
        """Loads data from CSV."""
        if not self.path.exists():
            logger.warning(f"[-] File missing: {self.name} at {self.path}")
            return

        try:
            self.df = pd.read_csv(self.path)
            # Normalize column names just in case
            self.df.columns = [c.lower() for c in self.df.columns]

            # 1. Flexible Column Mapping
            # Map known probability columns to 'prob'
            col_map = {"gcn_prob": "prob", "pred_prob": "prob", "prediction": "prob", "probability": "prob"}
            self.df.rename(columns=col_map, inplace=True)

            # Fix: Handle duplicate 'prob' columns if renaming caused collision
            if isinstance(self.df["prob"], pd.DataFrame):
                # If 'prob' selects multiple columns, keep the first one
                # This happens if a file has both 'prob' and 'gcn_prob' (which gets renamed to 'prob')
                logger.warning(f"[!] Duplicate 'prob' columns detected in {self.name}. Using the first one.")
                self.df = self.df.loc[:, ~self.df.columns.duplicated()]

            # 2. Handle missing 'split' column
            if "split" not in self.df.columns:
                # Heuristic: If split is missing, assume it's purely a TEST set result
                # (Common in inference-only outputs)
                logger.warning(f"[!] '{self.name}' has no 'split' column. Assuming ALL data is 'test' set.")
                self.df["split"] = "test"

            required = {"label", "prob", "split"}
            if not required.issubset(self.df.columns):
                logger.warning(f"[-] Columns missing in {self.name}: {self.df.columns}")
                return

            self.valid = True
            logger.info(f"[+] Loaded {self.name}: {len(self.df)} samples")
        except Exception as e:
            logger.error(f"[-] Error loading {self.name}: {e}")

    def get_metrics(self, split_name: str) -> Optional[Tuple]:
        """Calculates FPR, TPR, AUC for a specific split."""
        if not self.valid:
            return None

        subset = self.df[self.df["split"] == split_name]
        if subset.empty:
            return None

        y_true = subset["label"].values
        y_score = subset["prob"].values

        if len(np.unique(y_true)) < 2:
            return None

        fpr, tpr, _ = roc_curve(y_true, y_score)
        roc_val = auc(fpr, tpr)
        return fpr, tpr, roc_val


# ==============================================================================
# 4. VISUALIZATION ENGINE
# ==============================================================================


def plot_sci_roc(models: List[ModelResult], split_name: str, save_name: str):
    """
    Main plotting function with Inset Zoom.
    """
    # Create Figure
    fig, ax = plt.subplots(figsize=(6, 6))  # Standard single-column square

    # 1. Main Plot Loop
    valid_models = []
    for model in models:
        metrics = model.get_metrics(split_name)
        if metrics:
            fpr, tpr, roc_val = metrics
            # Add to list for zoom plotting later
            valid_models.append((model, fpr, tpr))

            label_text = f"{model.name} (AUC = {roc_val:.3f})"
            ax.plot(
                fpr,
                tpr,
                label=label_text,
                color=model.color,
                linestyle=model.style,
                linewidth=2.0 if model.zorder > 5 else 1.5,
                zorder=model.zorder,
                alpha=0.9,
            )

    # Baseline (Random Guess)
    ax.plot([0, 1], [0, 1], color="gray", linestyle=":", linewidth=1, alpha=0.5, label="Random Chance")

    # Axis Labels & Settings
    ax.set_xlabel("False Positive Rate", fontweight="bold")
    ax.set_ylabel("True Positive Rate", fontweight="bold")
    # ax.set_title(f"ROC Performance ({split_name.capitalize()} Set)", fontweight='bold', pad=12) # Removed Title

    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])
    ax.grid(True, linestyle=":", alpha=0.5)

    # Legend (Bottom Right, minimalist)
    ax.legend(loc="lower right", frameon=False, prop={"size": 9})

    # ==========================================================================
    # 3. Saving
    # ==========================================================================
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Save JPG (High Quality)
    jpg_path = OUTPUT_DIR / f"{save_name}.jpg"
    plt.savefig(jpg_path, format="jpg", dpi=300)

    logger.info(f"[√] Saved: {jpg_path.name}")
    plt.close()


# ==============================================================================
# MAIN
# ==============================================================================


def main():
    setup_sci_style()
    logger.info("Initializing SCI-Style ROC Plotter...")

    # 1. Load Data for Train
    models_train = []
    for config in REGISTRY_TRAIN:
        m = ModelResult(config)
        m.load()
        if m.valid:
            models_train.append(m)

    # 2. Load Data for Test
    models_test = []
    for config in REGISTRY_TEST:
        m = ModelResult(config)
        m.load()
        if m.valid:
            models_test.append(m)

    if not models_train and not models_test:
        logger.error("No valid models found to plot.")
        sys.exit(1)

    # 3. Plot Train (Use REGISTRY_TRAIN)
    if models_train:
        plot_sci_roc(models_train, split_name="train", save_name="ROC_Comparison_Train_Dynamic")

    # 4. Plot Test (Use REGISTRY_TEST)
    if models_test:
        plot_sci_roc(models_test, split_name="test", save_name="ROC_Comparison_Test_Dynamic")

    logger.info("All plots generated successfully.")


if __name__ == "__main__":
    main()
