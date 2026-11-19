# 目的：绘制早期 Fig5，比较星形胶质细胞发送群移除 CNV 阳性细胞前后，髓系接收群的表达乘积评分。
# 输入：09、10 的 pairs 字段以及兼容的两队列旧 JSON，历史图5使用冻结分区链所得的 10。
# 输出：03_result/18_Fig5移除CNV阳性细胞前后的通讯评分/18_Fig5移除CNV阳性细胞前后的通讯评分.pdf 和 18_Fig5移除CNV阳性细胞前后的通讯评分.png。
# 运行方式：在仓库根目录运行 python3 18_Fig5移除CNV阳性细胞前后的通讯评分.py。沿用 Python 和 Poppler 300 dpi 渲染。

from importlib import import_module

import numpy as np
from matplotlib.figure import Figure

plot = import_module("00a_绘图")

# a 使用同一接收群的受体均值。两根柱之间比较的是计算筛选前后的分数，不能由此推断体内信号被激活或抑制。
scores_table_total = plot.read_decontamination_table()
figure = Figure(figsize=(7.4, 3.2))
score_axis, change_axis = figure.subplots(1, 2)
figure.subplots_adjust(wspace=0.47, left=0.105, right=0.98, top=0.79, bottom=0.29)
required_text = plot.draw_tgfb2_scores(score_axis, scores_table_total)
score_axis.set_title("TGFB2–TGFBR2 before and after\nsender-cell filtering", fontsize=8, loc="left")
plot.label_panel(score_axis, "a")

# b 保留原图选定的基因对与横轴顺序。百分比取上游字段，不用四舍五入后的柱高再计算。
pair_order = ["TGFB2->TGFBR2", "TGFB2->TGFBR1", "CSF1->CSF1R", "TGFB1->TGFBR1", "ANXA1->FPR1"]
change_table = scores_table_total["pct_change"].unstack("cohort").loc[pair_order]
positions = np.arange(len(change_table))
for cohort, offset, color in [("GSE211376", -0.19, "#27AE60"), ("GSE174554", 0.19, "#C0392B")]:
    bars = change_axis.bar(positions + offset, change_table[cohort], width=0.38, color=color, label=cohort)
    labels = [f"{value:+.1f}" for value in change_table[cohort]]
    change_axis.bar_label(bars, labels=labels, fontsize=5.1, padding=1)
    required_text.extend(labels)
change_axis.axhline(0, color="#555555", lw=0.8)
change_axis.margins(y=0.23)
change_axis.set_xticks(positions, [name.replace("->", "→\n") for name in pair_order], fontsize=6)
change_axis.set_ylabel("Change after sender-cell filtering (%)", fontsize=7.2)
change_axis.set_title("Changes for selected ligand–receptor pairs", loc="left", fontsize=7.8)
change_axis.legend(fontsize=6.2, frameon=False, loc="upper left", ncol=2)
plot.label_panel(change_axis, "b")

# 源表每个队列的变化值都应在成品中可读，同时核对两个坐标轴使用的指标。
required_text.extend(["TGFB2", "TGFBR2", "ANXA1", "expression-product score", "GSE174554"])
plot.save_figure(figure, "18_Fig5移除CNV阳性细胞前后的通讯评分", required_text)
