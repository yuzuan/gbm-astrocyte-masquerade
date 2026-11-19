# Purpose: Use cohort-specific immune thresholds to compare CNV positivity across astrocyte groups, author annotations,
#          reference cells, and tissue sites in Fig2.
# Inputs: CNV results from 01, 03, and 05 (strict_CNVpos_pct, cnv_pos_pct, reference_FPR, and discovery annotations).
# Public tables 01–03 provide overall, selected reference, and tissue rates. The scGBM oligodendrocyte rate is missing.
# Outputs: 03_result/15_fig2_astrocyte_cnv_rates/
#          15_fig2_astrocyte_cnv_rates.pdf and the matching .png.
# Run: python3 15_fig2_astrocyte_cnv_rates.py from the project root. Poppler renders the PDF at 300 dpi.

from importlib import import_module

import numpy as np
import pandas as pd
from matplotlib.figure import Figure

plot = import_module("00a_plotting")

# Each bar shows the overall percentage within the target astrocyte group of one cohort.
cnv_table_total = plot.read_cnv_table(require_references=True)
figure = Figure(figsize=(7.4, 6.6))
axes = figure.subplots(2, 2)
figure.subplots_adjust(hspace=0.58, wspace=0.37, left=0.10, right=0.98, top=0.91, bottom=0.16)
overall_axis, annotation_axis = axes[0]
reference_axis, tissue_axis = axes[1]
required_text = plot.draw_cnv_rates(overall_axis, cnv_table_total, decimals=1)
overall_axis.set_title("CNV signature across cohorts", fontsize=8, loc="left")

# Panel b uses four separate denominators from the same marker-based analysis, including the author and myeloid groups.
annotation_rates = [
    cnv_table_total.at["GSE211376", "astro_pct"], cnv_table_total.at["GSE211376", "author_ac_pct"],
    cnv_table_total.at["GSE211376", "author_astro_pct"], cnv_table_total.at["GSE211376", "myeloid_pct"],
]
if not all(np.isfinite(value) and 0 <= value <= 100 for value in annotation_rates):
    raise ValueError("The discovery-cohort groups require valid CNV percentages.")
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

# Panel c shows observed reference rates. Keep the scGBM oligodendrocyte position labeled n/a and draw only reported
# bars.
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

# Panel d uses the published tumor and peritumoral percentages. The original code gives neither patient counts nor
# tissue-specific denominators or procedures for estimating a new threshold in each tissue.
tissue_table = pd.read_csv(plot.source_dir / "03_tumor_peritumor_cnv_rates.csv", index_col="cohort")
tissue_cohorts = tissue_table.index
for column, offset, color, label in [
    ("tumor_pct", -0.18, "#C0392B", "Tumor"), ("peritumoral_pct", 0.18, "#5DADE2", "Peritumoral")
]:
    values = tissue_table[column]
    if not values.between(0, 100).all():
        raise ValueError(f"The published tissue-stratified table contains invalid percentages in {column}.")
    bars = tissue_axis.bar(np.arange(len(values)) + offset, values, width=0.36, color=color, label=label)
    labels = [f"{rate:.1f}%" for rate in values]
    tissue_axis.bar_label(bars, labels=labels, fontsize=6.3, padding=2)
    required_text.extend(labels)

# These proportions pool cells within tissue. The dashed line marks the nominal 5% tail of the threshold.
# Patient pairing was not used to estimate these proportions.
tissue_axis.axhline(100 - 95, color="#697780", ls="--", lw=0.8)
tissue_axis.set_xticks(range(len(tissue_cohorts)), [name.replace("+", "\n+") for name in tissue_cohorts], fontsize=6)
tissue_axis.set_ylim(0, 119)
tissue_axis.set_ylabel("Astrocyte-group CNV+ (%)", fontsize=7.4)
tissue_axis.set_title("Tissue strata\n(original figure values)", fontsize=8, loc="left")
tissue_axis.legend(fontsize=6, frameon=False, ncol=2, loc="upper right")

# Check the percentages and the tissue panel's source note in the exported figure.
for axis, letter in zip(axes.flat, "abcd"):
    plot.label_panel(axis, letter)
required_text.extend(["GSE211376", "Oligodendrocyte", "n/a", "original figure values"])
plot.save_figure(figure, "15_fig2_astrocyte_cnv_rates", required_text)
