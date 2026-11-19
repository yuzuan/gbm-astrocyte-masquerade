# 目的：绘制早期研究 Fig1，展示标志基因与 CNV 评分的研究设计、分队列阳性率和队列规模。
# 输入：01、03、05 的 CNV 汇总和 04 的 leiden；01_图件来源表 01、02 提供队列规模、平台及原图百分比。
# 输出：03_result/14_Fig1队列与研究设计/14_Fig1队列与研究设计.pdf 和 14_Fig1队列与研究设计.png。
# 运行方式：在仓库根目录运行 python3 14_Fig1队列与研究设计.py。沿用 Python 绘图，PNG 由 Poppler 按 300 dpi 转换。

from importlib import import_module

import numpy as np
from matplotlib.figure import Figure

plot = import_module("00a_绘图")

# a、b 都是研究设计示意。方框坐标和箭头表示阅读顺序，没有细胞计数量纲。
figure = Figure(figsize=(7.4, 6.9))
grid = figure.add_gridspec(
    3, 2, height_ratios=[0.89, 0.86, 1.15], width_ratios=[1, 1.17], hspace=0.55, wspace=0.31,
    left=0.10, right=0.985, top=0.93, bottom=0.09
)
concept_axis = figure.add_subplot(grid[0, :])
concept_axis.axis("off")
concept_axis.set_title("Astrocyte markers and the GBM copy-number signature", fontsize=9, loc="left")
concept_boxes = [
    (0.14, "AC-like cell\nS100B, SLC1A3, ALDOC", "#FADBD8"),
    (0.51, "Astrocyte-marker\nassignment", "#D5F5E3"),
    (0.87, "Evaluate the\n+7/−10 signature", "#D6EAF8"),
]
for position, label, color in concept_boxes:
    concept_axis.text(
        position, 0.53, label, ha="center", va="center", fontsize=8,
        bbox={"boxstyle": "round,pad=0.65", "fc": color, "ec": "#47606F"}
    )
for start, end in [(0.28, 0.40), (0.64, 0.77)]:
    concept_axis.annotate("", xy=(end, 0.53), xytext=(start, 0.53), arrowprops={"arrowstyle": "->", "color": "#546772"})
plot.label_panel(concept_axis, "a")

# b 的评分逻辑适用于各队列。标志基因筛选、Leiden 簇选择和免疫参照细胞的具体定义随队列而异。
design_axis = figure.add_subplot(grid[1, :])
design_axis.axis("off")
design_axis.set_title("Astrocyte-group selection and independent +7/−10 scoring", fontsize=9, loc="left")
steps = [
    ("Public GBM\ncohort", "#D6EAF8"), ("Identify astrocyte-\nmarker groups", "#D5F5E3"),
    ("Score +7/−10 vs\nimmune reference", "#FADBD8"), ("Report fraction\nabove threshold", "#E8DAEF"),
]
centers = np.linspace(0.10, 0.90, len(steps))
for position, (label, color) in zip(centers, steps):
    design_axis.text(
        position, 0.57, label, ha="center", va="center", fontsize=7.2,
        bbox={"boxstyle": "round,pad=0.55", "fc": color, "ec": "#47606F"}
    )
for left, right in zip(centers[:-1], centers[1:]):
    design_axis.annotate("", xy=(right - 0.088, 0.57), xytext=(left + 0.088, 0.57), arrowprops={"arrowstyle": "->"})
design_axis.text(0.5, 0.14, "Group definitions and immune references are cohort-specific.", ha="center", fontsize=7)
plot.label_panel(design_axis, "b")

# c 的分母是各队列星形胶质细胞群。显示范围取自同一组柱高，取整仅用于标注。
cnv_table_total = plot.read_cnv_table()
rates_axis = figure.add_subplot(grid[2, 0])
rate_labels = plot.draw_cnv_rates(rates_axis, cnv_table_total, decimals=0)
rates = cnv_table_total["astro_pct"]
rates_axis.set_title(f"CNV-positive fraction: {rates.min():.0f}–{rates.max():.0f}%", fontsize=8, loc="left")
plot.label_panel(rates_axis, "c")

# d 的分母是整队列。表的单元格直接来自上面按 accession 取出的各行。
cohort_axis = figure.add_subplot(grid[2, 1])
cohort_axis.axis("off")
cohort_axis.set_title("Public GBM cohorts", fontsize=8, loc="center")

# 每行是一个队列。患者数是早期公开图表的口径，细胞/细胞核数包含全队列，均与 CNV 目标群分母有别。
display_rows = []
for row in cnv_table_total.itertuples():
    display_rows.append([row.Index, f"{row.cells:,}", str(row.patients), row.platform])
cohort_grid = cohort_axis.table(
    cellText=display_rows, colLabels=["GEO accession", "Cells/nuclei", "Patients", "Platform"],
    colWidths=[0.37, 0.23, 0.18, 0.22], cellLoc="center", bbox=[0, 0.23, 1, 0.68]
)
cohort_grid.auto_set_font_size(False)
cohort_grid.set_fontsize(5.8)
for (row, _), cell in cohort_grid.get_celld().items():
    cell.set_edgecolor("#D5DBDB")
    if row == 0:
        cell.set_facecolor("#2C3E50")
        cell.set_text_props(color="white", fontweight="bold")
    else:
        cell.set_facecolor("#EBF5FB" if row % 2 else "white")
total_label = f"{cnv_table_total['cells'].sum():,} cells and nuclei"
cohort_axis.text(0.5, 0.10, total_label, ha="center", fontsize=7)
plot.label_panel(cohort_axis, "d")

# 回读 PDF 时核对两段设计图的文字、柱顶比例、表中各行的细胞数及全队列合计。
required_text = ["SLC1A3", "immune reference", total_label, *rate_labels]
required_text.extend(row[1] for row in display_rows)
plot.save_figure(figure, "14_Fig1队列与研究设计", required_text)
