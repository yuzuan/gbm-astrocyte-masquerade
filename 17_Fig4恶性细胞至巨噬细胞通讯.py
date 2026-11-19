# 目的：绘制早期 Fig4，展示恶性亚型至巨噬细胞的表达乘积评分、通路分组及原图提出的候选联系。
# 输入：07 的 tam_scores（配体受体对 × AC、MES1、OPC、NPC），并兼容历史 communication_analysis_results.json。
#       01_图件来源表 07 为旧绘图脚本明写的示意连线。
# 输出：03_result/17_Fig4恶性细胞至巨噬细胞通讯/17_Fig4恶性细胞至巨噬细胞通讯.pdf 和 17_Fig4恶性细胞至巨噬细胞通讯.png。
# 运行方式：在仓库根目录运行 python3 17_Fig4恶性细胞至巨噬细胞通讯.py。沿用 Python 和 Poppler 300 dpi 渲染。

from importlib import import_module

import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from matplotlib.patches import PathPatch
from matplotlib.path import Path

plot = import_module("00a_绘图")

# 每个评分是发送亚型的配体均值与巨噬细胞受体均值之积。从 07 或其同名历史归档中读取。
result = plot.read_analysis_json("communication_analysis_results.json")
score_table = pd.DataFrame.from_dict(result["tam_scores"], orient="index")
subtypes = ["AC", "MES1", "OPC", "NPC"]
score_table = score_table.loc[:, subtypes]
if not score_table.index.is_unique or not np.isfinite(score_table).all().all() or (score_table < 0).any().any():
    raise ValueError("配体受体对重复或其亚型评分缺失、为负")

# a 的连线粗细取自所有入表配体受体对的列和。圆的位置和尺寸是示意坐标，没有细胞数或空间距离含义。
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

# b 的三个通路沿用原图所取的基因对。分组内逐亚型相加，缺少某个必要基因对会报出其名称。
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

# c 的全部节点和线来自旧图的假设模型。虚线均为示意联系，此处没有对受体激活、TF 调控或靶基因响应作检验。
edge_table = pd.read_csv(plot.source_dir / "07_原图假设性通讯连线.csv")
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

# 核对三面板中的接收细胞、通路和模型端点，标签来自实际用于绘制的行列。
required_text = ["Macrophage", *subtypes, "Chemokine", "original schematic", *edge_table["target_node"].unique()]
plot.save_figure(figure, "17_Fig4恶性细胞至巨噬细胞通讯", required_text)
