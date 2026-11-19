# 目的：绘制早期 Fig2，以各队列免疫参照阈值比较星形胶质细胞群、作者标签、参照细胞和组织部位的 CNV 阳性率。
# 输入：01、03、05 的 CNV 结果（strict_CNVpos_pct、cnv_pos_pct、reference_FPR 和发现队列作者标签）。
#       01_图件来源表 01、02、03 的已公开总体、部分参照及分部位百分比，scGBM 少突参照在原图缺报。
# 输出：03_result/15_Fig2星形胶质细胞群CNV阳性率/15_Fig2星形胶质细胞群CNV阳性率.pdf 和 15_Fig2星形胶质细胞群CNV阳性率.png。
# 运行方式：在仓库根目录运行 python3 15_Fig2星形胶质细胞群CNV阳性率.py。沿用 Python 和 Poppler 300 dpi 渲染。

from importlib import import_module

import numpy as np
import pandas as pd
from matplotlib.figure import Figure

plot = import_module("00a_绘图")

# a 的每根柱是一个队列的总体比例。百分比只以该队列目标群中的细胞为分母。
cnv_table_total = plot.read_cnv_table(require_references=True)
figure = Figure(figsize=(7.4, 6.6))
axes = figure.subplots(2, 2)
figure.subplots_adjust(hspace=0.58, wspace=0.37, left=0.10, right=0.98, top=0.91, bottom=0.16)
overall_axis, annotation_axis = axes[0]
reference_axis, tissue_axis = axes[1]
required_text = plot.draw_cnv_rates(overall_axis, cnv_table_total, decimals=1)
overall_axis.set_title("CNV signature across cohorts", fontsize=8, loc="left")

# b 使用同一次标志基因分析中的作者标签和髓系参照，四个分母各自独立。
annotation_rates = [
    cnv_table_total.at["GSE211376", "astro_pct"], cnv_table_total.at["GSE211376", "author_ac_pct"],
    cnv_table_total.at["GSE211376", "author_astro_pct"], cnv_table_total.at["GSE211376", "myeloid_pct"],
]
if not all(np.isfinite(value) and 0 <= value <= 100 for value in annotation_rates):
    raise ValueError("发现队列分组缺少有效 CNV 百分比")
bars = annotation_axis.bar(
    range(len(annotation_rates)), annotation_rates, color=["#27AE60", "#C0392B", "#7D3C98", "#95A5A6"]
)
annotation_labels = [f"{value:.1f}%" for value in annotation_rates]
annotation_axis.bar_label(bars, labels=annotation_labels, fontsize=6.8, padding=2)
annotation_axis.set_xticks(
    range(len(annotation_rates)), ["Marker-defined\nastrocyte", "AC-like", "Refined\nastrocyte", "Myeloid\nreference"],
    fontsize=6
)
annotation_axis.set_ylim(0, 107)
annotation_axis.set_ylabel("CNV+ (%)", fontsize=7.5)
annotation_axis.set_title("GSE211376 annotation groups", fontsize=8, loc="left")
required_text.extend(annotation_labels)

# c 的参照率是实测读数。旧图对 scGBM 少突参照标 n/a；保留位置和缺报标记，仅为实报值画柱。
positions = np.arange(len(cnv_table_total))
for column, offset, color, label in [
    ("myeloid_pct", -0.19, "#2E86C1", "Myeloid"), ("oligo_pct", 0.19, "#AAB7B8", "Oligodendrocyte")
]:
    available = cnv_table_total[column].notna().to_numpy()
    values = cnv_table_total.loc[available, column]
    bars = reference_axis.bar(positions[available] + offset, values, width=0.38, color=color, label=label)
    labels = [f"{value:.1f}%" for value in values]
    reference_axis.bar_label(bars, labels=labels, padding=2, fontsize=5.7)
    required_text.extend(labels)
    for missing_position in positions[~available]:
        reference_axis.text(missing_position + offset, 0.3, "n/a", ha="center", fontsize=6)
reference_axis.set_xticks(positions, [name.replace("+", "\n+") for name in cnv_table_total.index], fontsize=6)
reference_axis.set_ylim(0, cnv_table_total[["myeloid_pct", "oligo_pct"]].max().max() * 1.32)
reference_axis.set_ylabel("Reference cells above threshold (%)", fontsize=7)
reference_axis.set_title("Observed reference-cell positivity", fontsize=8, loc="left")
reference_axis.legend(fontsize=6, frameon=False, loc="upper left")

# d 的肿瘤与瘤旁条形取自公开旧图，不回推病人数或未发表分母。旧图未在这段代码中给出重新分层定阈值的步骤。
tissue_table = pd.read_csv(plot.source_dir / "03_肿瘤与瘤旁组织CNV阳性率.csv", index_col="cohort")
tissue_cohorts = tissue_table.index
for column, offset, color, label in [
    ("tumor_pct", -0.18, "#C0392B", "Tumor"), ("peritumoral_pct", 0.18, "#5DADE2", "Peritumoral")
]:
    values = tissue_table[column]
    if not values.between(0, 100).all():
        raise ValueError(f"冻结组织分层表的 {column} 有无效比例")
    bars = tissue_axis.bar(np.arange(len(values)) + offset, values, width=0.36, color=color, label=label)
    labels = [f"{rate:.1f}%" for rate in values]
    tissue_axis.bar_label(bars, labels=labels, fontsize=6.3, padding=2)
    required_text.extend(labels)

# 位置比较是组织内合并细胞的比例。虚线是阈值设计的 5% 名义上尾，病人配对未用于估计此比例。
tissue_axis.axhline(100 - 95, color="#697780", ls="--", lw=0.8)
tissue_axis.set_xticks(range(len(tissue_cohorts)), [name.replace("+", "\n+") for name in tissue_cohorts], fontsize=6)
tissue_axis.set_ylim(0, 119)
tissue_axis.set_ylabel("Astrocyte-group CNV+ (%)", fontsize=7.4)
tissue_axis.set_title("Tissue strata\n(original figure values)", fontsize=8, loc="left")
tissue_axis.legend(fontsize=6, frameon=False, ncol=2, loc="upper right")

# 核对四个面板标注的百分比及组织分层的来源标注，防止数据正确却贴错图。
for axis, letter in zip(axes.flat, "abcd"):
    plot.label_panel(axis, letter)
required_text.extend(["GSE211376", "Oligodendrocyte", "n/a", "original figure values"])
plot.save_figure(figure, "15_Fig2星形胶质细胞群CNV阳性率", required_text)
