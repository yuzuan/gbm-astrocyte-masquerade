# 目的：绘制早期研究图形摘要，汇总四个队列的 CNV 阳性率和两队列发送群筛选前后的 TGFB2–TGFBR2 表达乘积。
# 输入：01、03、04、05 的 CNV 与细胞数，09、10 中 TGFB2–TGFBR2 的 before、after；仓库内旧代码明写的表 01、02。
# 输出：03_result/20_图形摘要/20_图形摘要.pdf 和 20_图形摘要.png。
# 运行方式：在仓库根目录运行 python3 20_图形摘要.py。沿用 Python，PNG 由同一 PDF 通过 Poppler 按 300 dpi 转换。

from importlib import import_module

from matplotlib.figure import Figure

plot = import_module("00a_绘图")

# a 与 Fig1、Fig2 使用同一队列表，柱顶取整仅用于显示。该比例的分母是目标星形胶质细胞群。
cnv_table_total = plot.read_cnv_table()
figure = Figure(figsize=(11, 4.2))
rates_axis, scores_axis, summary_axis = figure.subplots(1, 3)
figure.subplots_adjust(left=0.065, right=0.98, wspace=0.38, top=0.83, bottom=0.17)
required_text = plot.draw_cnv_rates(rates_axis, cnv_table_total, decimals=0)
rates_axis.set_title("CNV signature in astrocyte groups", fontsize=8.8, loc="left")
plot.label_panel(rates_axis, "a")

# b 沿用旧摘要 stage26 的公式，从两根实画柱的已发布评分计算变化。各队列的筛选前评分须为正。
scores_table_total = plot.read_decontamination_table(for_abstract=True)
before = scores_table_total["before"]
after = scores_table_total["after"]
if not before.gt(0).all():
    raise ValueError("图形摘要的 TGFB2–TGFBR2 筛选前评分须为正，才能计算变化率")
scores_table_total["pct_change"] = (after - before) / before * 100
change_labels = plot.draw_tgfb2_scores(scores_axis, scores_table_total)
scores_axis.set_title("TGFB2–TGFBR2 sender-cell comparison", fontsize=8.6, loc="left")
plot.label_panel(scores_axis, "b")

# c 的队列数、细胞总数、比例范围和两个变化值取自左侧面板。三个框仅排列摘要内容。
rates = cnv_table_total["astro_pct"]
summary_axis.axis("off")
summary_axis.set_title("Cohort-level summary", fontsize=8.8, loc="center")
count_label = f"{cnv_table_total['cells'].sum():,} cells and nuclei"
rate_summary = f"{rates.min():.0f}–{rates.max():.0f}% CNV-positive\nin {len(rates)} astrocyte groups"
summary_axis.text(
    0.5, 0.78, f"{rate_summary}\n{count_label} across full cohorts", ha="center", va="center", fontsize=8,
    bbox={"boxstyle": "round,pad=0.65", "fc": "#FADBD8", "ec": "#C0392B"}
)
change_summary = "\n".join(f"{cohort}: {label}" for cohort, label in zip(["GSE211376", "GSE174554"], change_labels))
summary_axis.text(
    0.5, 0.47, f"TGFB2–TGFBR2 score change\n{change_summary}", ha="center", va="center", fontsize=8,
    bbox={"boxstyle": "round,pad=0.65", "fc": "#E8DAEF", "ec": "#884EA0"}
)
summary_axis.text(
    0.5, 0.16, "Expression-product scores compare\nfull and CNV-negative senders.", ha="center", va="center",
    fontsize=7.5,
    bbox={"boxstyle": "round,pad=0.65", "fc": "#D5F5E3", "ec": "#27AE60"}
)
plot.label_panel(summary_axis, "c")

# 从 PDF 核对完整队列总数、两队列变化读数及汇总框文字。
required_text.extend([count_label, *change_labels, "Cohort-level summary", "CNV-negative senders"])
plot.save_figure(figure, "20_图形摘要", required_text)
