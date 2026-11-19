# Purpose: Compare myeloid-receiver expression-product scores before and after filtering CNV-positive astrocyte senders.
# Inputs: The pairs field from stages 09 and 10 or the two archived JSON files. Historical Fig5 uses stage 10 output
#         from the frozen partition.
# Outputs: 03_result/18_fig5_cnv_sender_filtering/
#          18_fig5_cnv_sender_filtering.pdf and the matching .png.
# Run: python3 18_fig5_cnv_sender_filtering.py from the project root. Poppler renders the PDF at 300 dpi.

from importlib import import_module

import numpy as np
from matplotlib.figure import Figure

plot = import_module("00a_plotting")

# Both bars use the same receiver-group receptor mean. The comparison measures a change in the computed score after
# sender filtering and does not establish activation or inhibition of signaling in vivo.
scores_table_total = plot.read_decontamination_table()
figure = Figure(figsize=(7.4, 3.2))
score_axis, change_axis = figure.subplots(1, 2)
figure.subplots_adjust(wspace=0.47, left=0.105, right=0.98, top=0.79, bottom=0.29)
required_text = plot.draw_tgfb2_scores(score_axis, scores_table_total)
score_axis.set_title("TGFB2–TGFBR2 before and after\nsender-cell filtering", fontsize=8, loc="left")
plot.label_panel(score_axis, "a")

# Keep the original gene pairs and axis order. Read changes calculated upstream from unrounded scores.
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

# Verify the displayed changes in each cohort and the score definitions on both axes.
required_text.extend(["TGFB2", "TGFBR2", "ANXA1", "expression-product score", "GSE174554"])
plot.save_figure(figure, "18_fig5_cnv_sender_filtering", required_text)
