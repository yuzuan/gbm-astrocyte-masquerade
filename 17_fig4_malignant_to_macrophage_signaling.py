# Purpose: Show malignant-subtype to macrophage expression-product scores by pathway and the proposed links in Fig4.
# Inputs: Stage 07 tam_scores (ligand–receptor pairs by AC, MES1, OPC, and NPC) or archived
#         communication_analysis_results.json.
#         Source table 07 lists the schematic edges specified in the original plotting script.
# Outputs: 03_result/17_fig4_malignant_to_macrophage_signaling/
#          17_fig4_malignant_to_macrophage_signaling.pdf and the matching .png.
# Run: python3 17_fig4_malignant_to_macrophage_signaling.py from the project root. Poppler renders the PDF at 300 dpi.

from importlib import import_module

import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from matplotlib.patches import PathPatch
from matplotlib.path import Path

plot = import_module("00a_plotting")

# Each score multiplies mean ligand expression in a sender subtype by mean receptor expression in macrophages.
# Read stage 07 or its original archive.
result = plot.read_analysis_json("communication_analysis_results.json")
score_table = pd.DataFrame.from_dict(result["tam_scores"], orient="index")
subtypes = ["AC", "MES1", "OPC", "NPC"]
score_table = score_table.loc[:, subtypes]
if not score_table.index.is_unique or not np.isfinite(score_table).all().all() or (score_table < 0).any().any():
    raise ValueError("Ligand–receptor pairs are duplicated or have missing, negative, or nonfinite subtype scores.")

# Panel a scales each edge width by the sum over all included pairs for that subtype. The circles' positions and
# sizes are schematic and do not represent cell counts or spatial distances.
totals = score_table.sum(axis=0)
figure = Figure(figsize=(7.6, 3.2))
grid = figure.add_gridspec(1, 3, width_ratios=[1, 1, 1.28], wspace=0.50, left=0.05, right=0.985, top=0.79, bottom=0.18)
network_axis = figure.add_subplot(grid[0, 0])
network_axis.axis("off")
network_axis.set(xlim=(-1.3, 1.7), ylim=(-1.3, 1.3))
colors = {"AC": "#C0392B", "MES1": "#E67E22", "OPC": "#7FB77E", "NPC": "#5B8AC4"}
for subtype, angle in zip(subtypes, np.linspace(120, 240, len(subtypes))):
    x = 0.9 * np.cos(np.radians(angle))
    y = 0.9 * np.sin(np.radians(angle))
    network_axis.scatter(x, y, s=120, color=colors[subtype], zorder=3)
    network_axis.text(x * 1.35, y * 1.35, subtype, ha="center", va="center", fontsize=6, color=colors[subtype])
    if totals[subtype] > 0:
        curve = Path([(x, y), (0, y * 0.4), (0.9, 0)], [Path.MOVETO, Path.CURVE3, Path.CURVE3])
        width = 0.5 + 5 * totals[subtype] / totals.max()
        network_axis.add_patch(PathPatch(curve, fill=False, color=colors[subtype], lw=width, alpha=0.6))
network_axis.scatter(0.9, 0, s=200, color="#34495E", zorder=3)
network_axis.text(0.9, 0.31, "Macrophage", ha="center", fontsize=6.3)
network_axis.set_title("Summed expression-product\nscore to macrophages", loc="left", fontsize=7.5)
plot.label_panel(network_axis, "a")

# Sum the original ligand–receptor pairs within each pathway and subtype. A missing required pair raises an error.
pathways = {
    "TGF-β": ["TGFB2-TGFBR1", "TGFB2-TGFBR2"], "CSF1": ["CSF1-CSF1R"], "Chemokine": ["CCL2-CCR2", "CXCL12-CXCR4"],
}
pathway_axis = figure.add_subplot(grid[0, 1])
positions = np.arange(len(pathways))
for index, subtype in enumerate(subtypes):
    values = [score_table.loc[pairs, subtype].sum() for pairs in pathways.values()]
    pathway_axis.bar(positions + (index - 1.5) * 0.20, values, width=0.20, color=colors[subtype], label=subtype)
pathway_axis.set_xticks(positions, pathways.keys(), fontsize=6)
pathway_axis.set_ylabel("Expression-product score", fontsize=7)
pathway_axis.legend(fontsize=5.5, ncol=2, frameon=False)
pathway_axis.set_title("Scores by pathway", loc="left", fontsize=7.5)
plot.label_panel(pathway_axis, "b")

# Panel c reproduces the original hypothetical model. Dashed edges denote proposed links; these analyses did not
# test receptor activation, TF regulation, or target-gene responses.
edge_table = pd.read_csv(plot.source_dir / "07_hypothesized_communication_edges.csv")
model_axis = figure.add_subplot(grid[0, 2])
model_axis.axis("off")
model_axis.set(xlim=(-0.35, 3.35), ylim=(-0.95, 2.75))
layers = [
    ("Ligand", ["TGFB2", "SPP1", "CSF1"], "#C0392B"), ("Receptor", ["TGFBR", "CD44", "CSF1R"], "#E67E22"),
    ("TF", ["SMAD3", "NF-κB", "STAT3"], "#8E44AD"), ("Target", ["IL10", "ARG1", "VEGFA"], "#34495E"),
]
node_positions = {}
for x, (layer, nodes, color) in enumerate(layers):
    model_axis.text(x, 2.54, layer, ha="center", fontsize=5.7, fontweight="bold", color=color)
    for y, node in zip([2, 1, 0], nodes):
        node_positions[layer, node] = (x, y)
        model_axis.scatter(x, y, s=65, color=color, zorder=3)
        model_axis.text(x, y - 0.21, node, ha="center", va="top", fontsize=5.1, color=color)
for edge in edge_table.itertuples(index=False):
    start = node_positions[edge.source_layer, edge.source_node]
    end = node_positions[edge.target_layer, edge.target_node]
    model_axis.plot([start[0], end[0]], [start[1], end[1]], ls="--", color="#929A9D", lw=0.65, zorder=1)
model_axis.set_title("Hypothesized links\n(original schematic)", loc="left", fontsize=7.5)
plot.label_panel(model_axis, "c")

# Check the receiver, pathway names, and model endpoints from the plotted data across all three panels.
required_text = ["Macrophage", *subtypes, "Chemokine", "original schematic", *edge_table["target_node"].unique()]
plot.save_figure(figure, "17_fig4_malignant_to_macrophage_signaling", required_text)
