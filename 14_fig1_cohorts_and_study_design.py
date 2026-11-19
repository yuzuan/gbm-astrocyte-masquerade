# Purpose: Show the marker and CNV study design, cohort-specific positivity, and cohort sizes in the original Fig1.
# Inputs: Stage 01, 03, and 05 CNV summaries and stage 04 Leiden labels. Source tables 01 and 02 supply cohort sizes,
#         platforms, and percentages recorded in the original figure.
# Outputs: 03_result/14_fig1_cohorts_and_study_design/
#          14_fig1_cohorts_and_study_design.pdf and the matching .png.
# Run: python3 14_fig1_cohorts_and_study_design.py from the project root. Poppler renders the PDF at 300 dpi.

from importlib import import_module

import numpy as np
from matplotlib.figure import Figure

plot = import_module("00a_plotting")

# Panels a and b illustrate the study design. Box positions and arrows show the reading order and have no cell-count
# units.
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

# The scoring sequence applies to each cohort. Marker filters, Leiden cluster selection, and immune references vary.
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

# Each cohort's astrocyte group is the denominator. Derive the range from the bar heights and round only the labels.
cnv_table_total = plot.read_cnv_table()
rates_axis = figure.add_subplot(grid[2, 0])
rate_labels = plot.draw_cnv_rates(rates_axis, cnv_table_total, decimals=0)
rates = cnv_table_total["astro_pct"]
rates_axis.set_title(f"CNV-positive fraction: {rates.min():.0f}–{rates.max():.0f}%", fontsize=8, loc="left")
plot.label_panel(rates_axis, "c")

# The table describes entire cohorts. Use the rows indexed by GEO accession.
cohort_axis = figure.add_subplot(grid[2, 1])
cohort_axis.axis("off")
cohort_axis.set_title("Public GBM cohorts", fontsize=8, loc="center")

# Each row describes one cohort. Patient counts follow the original figure. Cell and nucleus counts cover the full
# cohort, whereas the CNV percentage uses only the target group.
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

# Verify both design diagrams, the bar labels, the four cohort cell counts, and their combined total in the PDF.
required_text = ["SLC1A3", "immune reference", total_label, *rate_labels]
required_text.extend(row[1] for row in display_rows)
plot.save_figure(figure, "14_fig1_cohorts_and_study_design", required_text)
