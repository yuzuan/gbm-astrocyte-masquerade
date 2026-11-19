# Purpose: Summarize CNV rates across four cohorts and TGFB2–TGFBR2 scores before/after sender filtering in two cohorts.
# Inputs: CNV results and cell counts from 01, 03, 04, and 05. Stages 09 and 10 supply TGFB2–TGFBR2 before and after
#         scores. Source tables 01 and 02 contain the figures reported in the original code.
# Outputs: 03_result/20_graphical_abstract/
#          20_graphical_abstract.pdf and the matching .png.
# Run: python3 20_graphical_abstract.py from the project root. Poppler renders the PDF at 300 dpi.

from importlib import import_module

from matplotlib.figure import Figure

plot = import_module("00a_plotting")

# Use the same cohort table as Fig1 and Fig2. The target astrocyte group is the denominator. Round labels for display.
cnv_table_total = plot.read_cnv_table()
figure = Figure(figsize=(11, 4.2))
rates_axis, scores_axis, summary_axis = figure.subplots(1, 3)
figure.subplots_adjust(left=0.065, right=0.98, wspace=0.38, top=0.83, bottom=0.17)
required_text = plot.draw_cnv_rates(rates_axis, cnv_table_total, decimals=0)
rates_axis.set_title("CNV signature in astrocyte groups", fontsize=8.8, loc="left")
plot.label_panel(rates_axis, "a")

# Use the formula from the original stage26 abstract and the two score values actually plotted.
# The prefiltering score in each cohort must be positive.
scores_table_total = plot.read_decontamination_table(for_abstract=True)
before = scores_table_total["before"]
after = scores_table_total["after"]
if not before.gt(0).all():
    raise ValueError("TGFB2–TGFBR2 prefiltering scores must be positive to calculate changes in the abstract.")
scores_table_total["pct_change"] = (after - before) / before * 100
change_labels = plot.draw_tgfb2_scores(scores_axis, scores_table_total)
scores_axis.set_title("TGFB2–TGFBR2 sender-cell comparison", fontsize=8.6, loc="left")
plot.label_panel(scores_axis, "b")

# Read the cohort count, total cell count, range, and two changes from the data in the first two panels.
# The boxes organize the summary text.
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

# Verify the full-cohort cell count, both cohort changes, and the summary text in the PDF.
required_text.extend([count_label, *change_labels, "Cohort-level summary", "CNV-negative senders"])
plot.save_figure(figure, "20_graphical_abstract", required_text)
