# Purpose: Compare mean marker expression across scGBM groups and display the TF activity differences recorded in Fig3.
# Inputs: data/intermediate/scgbm_ac_validation_summary.csv (category, gene, and five *_mean columns).
# 01_figure_sources/05_transcription_factor_activity_differences.csv provides the original tf and activity_difference.
# Outputs: 03_result/16_fig3_marker_expression_and_tf_activity/
#          16_fig3_marker_expression_and_tf_activity.pdf and the matching .png.
# Run: python3 16_fig3_marker_expression_and_tf_activity.py from the project root. Poppler renders the PDF at 300 dpi.

from importlib import import_module

import numpy as np
import pandas as pd
from matplotlib.figure import Figure

plot = import_module("00a_plotting")

# Each gene has five mean-expression values for the four malignant states and macrophages. Stage 13 groups cells by
# region.
expression_path = plot.project_dir / "data" / "intermediate" / "scgbm_ac_validation_summary.csv"
if not expression_path.is_file():
    raise FileNotFoundError(
        "Fig3a requires data/intermediate/scgbm_ac_validation_summary.csv; stage 13 uses different groups."
    )
columns = ["AC_mean", "MES1_mean", "OPC_mean", "NPC_mean", "Macro_mean"]
expression_table = pd.read_csv(expression_path, usecols=["category", "gene", *columns])
expression_table = expression_table.set_index("gene", verify_integrity=True)
mean_matrix = expression_table[columns].to_numpy(dtype=float)
if not mean_matrix.size or not np.isfinite(mean_matrix).all() or np.any(mean_matrix < 0):
    raise ValueError("The expression heatmap requires a nonempty matrix of finite, nonnegative group means.")

# For each gene, scale color by its highest group mean, using the original 1e-9 denominator offset.
# All-zero rows stay zero. Cell labels show the unscaled means.
relative_matrix = mean_matrix / (mean_matrix.max(axis=1, keepdims=True) + 1e-9)
figure = Figure(figsize=(7.6, 4.6))
grid = figure.add_gridspec(1, 2, width_ratios=[1.15, 1], wspace=0.55, left=0.20, right=0.98, top=0.85, bottom=0.18)
heat_axis = figure.add_subplot(grid[0, 0])
heat = heat_axis.imshow(relative_matrix, aspect="auto", cmap="RdYlBu_r", vmin=0, vmax=1, interpolation="nearest")
heat_axis.set_xticks(range(len(columns)), ["AC", "MES", "OPC", "NPC", "Macro"], fontsize=6.5)
heat_axis.set_yticks(range(len(expression_table)), expression_table.index, fontsize=6, fontstyle="italic")
heat_axis.tick_params(length=0)
for (row, column), mean in np.ndenumerate(mean_matrix):
    color = "white" if relative_matrix[row, column] > 0.6 else "black"
    heat_axis.text(column, row, f"{mean:.1f}", ha="center", va="center", fontsize=4.8, color=color)

# Keep the category and row order from the input. Draw a separator where two adjacent category labels differ.
category = expression_table["category"]
category_style = {
    "Astrocyte marker": ("Astrocyte", "#E67E22"), "Malignant marker": ("Malignant", "#C0392B"),
    "Immune evasion": ("Immune evasion", "#8E44AD"),
}
for _, group_table in expression_table.groupby(category.ne(category.shift()).cumsum(), sort=False):
    row_positions = expression_table.index.get_indexer(group_table.index)
    first, last = row_positions.min(), row_positions.max()
    name, color = category_style[group_table["category"].drop_duplicates().item()]
    if first > 0:
        heat_axis.axhline(first - 0.5, color="white", lw=1.3)
    heat_axis.plot(
        [-0.50, -0.50], [first - 0.36, last + 0.36], transform=heat_axis.get_yaxis_transform(),
        color=color, lw=2, clip_on=False
    )
    heat_axis.text(
        -0.62, (first + last) / 2, name, transform=heat_axis.get_yaxis_transform(), rotation=90,
        ha="center", va="center", color=color, fontsize=5.6
    )
heat_axis.set_title("Mean expression by cell group", loc="left", fontsize=8)
colorbar = figure.colorbar(heat, ax=heat_axis, fraction=0.045, pad=0.04, aspect=28)
colorbar.set_label("Fraction of within-gene maximum", fontsize=6)
colorbar.ax.tick_params(labelsize=5.5)
plot.label_panel(heat_axis, "a")

# Panel b reproduces the nine recorded differences in stage22, ordered by value. Its title identifies their source.
# Drawing the panel does not refit the ULM.
tf_table = pd.read_csv(plot.source_dir / "05_transcription_factor_activity_differences.csv", index_col="tf")
if not tf_table.index.is_unique or not np.isfinite(tf_table["activity_difference"]).all():
    raise ValueError("The published TF differences require unique names and finite values.")
tf_table = tf_table.sort_values("activity_difference", kind="stable")
differences = tf_table["activity_difference"]
tf_axis = figure.add_subplot(grid[0, 1])
positions = np.arange(len(tf_table))[::-1]
tf_axis.barh(positions, differences, height=0.68, color=np.where(differences < 0, "#5B8AC4", "#C0392B"))
tf_axis.axvline(0, color="black", lw=0.6)
tf_axis.set_yticks(positions, tf_table.index, fontsize=6.5, fontstyle="italic")
tf_axis.set_xlabel("AC-like minus mean of other malignant subtypes\n(CollecTRI / decoupler ULM)", fontsize=7)
tf_axis.set_title("TF activity differences\n(values recorded in the original figure)", fontsize=8, loc="left")
plot.label_panel(tf_axis, "b")

# Check every gene and TF label and the expression color scale in the saved PDF.
required_text = [*expression_table.index, *tf_table.index, "within-gene maximum", "original figure"]
plot.save_figure(figure, "16_fig3_marker_expression_and_tf_activity", required_text)
