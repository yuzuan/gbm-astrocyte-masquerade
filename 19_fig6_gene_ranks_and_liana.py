# Purpose: Compare AC-like expression ranks across two cohorts and show independent snGBM LIANA ligand ranks in Fig6.
# Inputs: The stage 12 ranking table (category, gene, both cohort ranks, both_AC_top2) and stage 08
#         AC_like_rank_key_ligands.
#         The original rank table is also supported. Source table 11 lists the five ligand ranks in the original code.
# Outputs: 03_result/19_fig6_gene_ranks_and_liana/
#          19_fig6_gene_ranks_and_liana.pdf and the matching .png.
# Run: python3 19_fig6_gene_ranks_and_liana.py from the project root. Poppler renders the PDF at 300 dpi.

from importlib import import_module

import numpy as np
import pandas as pd
from matplotlib.figure import Figure

plot = import_module("00a_plotting")

# Each row is one gene ranked in both cohorts. Preserve upstream row order. The original figure circles TGFB2.
rank_path = plot.analysis_path("fig6_concordance_gse211376_vs_scgbm.csv")
rank_columns = ["category", "scGBM_AC_rank", "gse211376_AC_rank", "both_AC_top2"]
rank_table = pd.read_csv(rank_path, usecols=[*rank_columns, "gene"], dtype={"both_AC_top2": "boolean"})
if rank_table["gene"].isna().any() or not rank_table["gene"].is_unique:
    raise ValueError("The cross-cohort ranking table requires one row per gene.")
is_tgfb2 = rank_table["gene"].eq("TGFB2")

# The denominator is the number of genes with comparable ranks. Confirm the joint top-two flag from both ranks
# so the string 'False' cannot be counted as true.
ranks = [1, 2, 3, 4]
if rank_table[rank_columns].isna().any().any() or is_tgfb2.sum() != 1:
    raise ValueError("The joint ranks contain missing values or do not identify a single TGFB2 row.")
scgbm_ranks = rank_table["scGBM_AC_rank"]
discovery_ranks = rank_table["gse211376_AC_rank"]
if not scgbm_ranks.isin(ranks).all() or not discovery_ranks.isin(ranks).all():
    raise ValueError("Expression ranks must correspond to the four malignant subtypes.")
both_top = rank_table["both_AC_top2"].to_numpy(dtype=bool)
if not np.array_equal(both_top, scgbm_ranks.le(2) & discovery_ranks.le(2)):
    raise ValueError("The joint top-two flag disagrees with the cohort ranks.")

# For each category, calculate the percentage ranked first in the discovery cohort. The original 50% cutoff affects
# only the color; the upstream table supplies the ranks.
first_pct = discovery_ranks.eq(1).groupby(rank_table["category"]).mean().mul(100).sort_values()
figure = Figure(figsize=(7.6, 3.0))
category_axis, scatter_axis, ligand_axis = figure.subplots(1, 3)
figure.subplots_adjust(wspace=0.68, left=0.12, right=0.985, top=0.77, bottom=0.25)
positions = np.arange(len(first_pct))
category_axis.barh(positions, first_pct, color=np.where(first_pct >= 50, "#C0392B", "#95A5A6"), height=0.70)
short_labels = {
    "Checkpoint ligands": "Checkpoint", "Complement regulators": "Complement", "Dont-eat-me": "Don't-eat-me",
    "IFN response": "IFN resp.", "Immunosuppressive": "Immunosupp.", "Immunosuppressive chemokines": "Immun. chemok.",
}
category_labels = [short_labels.get(name, name) for name in first_pct.index]
category_axis.set_yticks(positions, category_labels, fontsize=6.1)
category_axis.set(xlim=(0, 105), xlabel="AC-like ranked first (%)")
category_axis.xaxis.label.set_size(7)
category_axis.set_title("GSE211376 expression ranks\nby immune-gene category", fontsize=7.2, loc="left")
plot.label_panel(category_axis, "a")

# Jitter display coordinates only. Preserve seed 1 and the two uniform draws over ±0.12, x before y.
# Place the TGFB2 circle at its unjittered ranks.
rng = np.random.default_rng(1)
x = scgbm_ranks.to_numpy() + rng.uniform(-0.12, 0.12, len(rank_table))
y = discovery_ranks.to_numpy() + rng.uniform(-0.12, 0.12, len(rank_table))
scatter_axis.scatter(x[~both_top], y[~both_top], s=16, c="#95A5A6", alpha=0.5, edgecolors="none", label="Other")
scatter_axis.scatter(x[both_top], y[both_top], s=18, c="#2E86C1", alpha=0.8, edgecolors="none", label="Both top two")
gene_x = scgbm_ranks.loc[is_tgfb2].item()
gene_y = discovery_ranks.loc[is_tgfb2].item()
scatter_axis.scatter(gene_x, gene_y, s=60, facecolors="none", edgecolors="#C0392B", lw=1.6)
scatter_axis.annotate(
    "TGFB2", (gene_x, gene_y), xytext=(-6, -13), textcoords="offset points", ha="right", color="#C0392B", fontsize=6.6
)
scatter_axis.plot([0.5, 4.5], [0.5, 4.5], ls=":", color="#B0B0B0", lw=1, zorder=0)
scatter_axis.set(xlim=(4.6, 0.4), ylim=(4.6, 0.4), xticks=ranks, yticks=ranks)
scatter_axis.set_xlabel("AC rank: GSE162631+273274", fontsize=6.5)
scatter_axis.set_ylabel("AC rank: GSE211376", fontsize=6.5)
agreement_label = f"{both_top.sum()}/{len(rank_table)}"
scatter_axis.set_title(f"Cross-cohort concordance\n(both AC top two: {agreement_label})", loc="left", fontsize=7.2)
scatter_axis.legend(fontsize=5.6, frameon=False, loc="lower left")
plot.label_panel(scatter_axis, "b")

# Panel c reads LIANA ligand ranks from 08 or the five ranks in the original plotting code. Each ligand is ranked
# across the subtypes with comparable results.
liana = plot.read_analysis_json("liana_validation.json", required=False)
if liana is None:
    ligand_table = pd.read_csv(plot.source_dir / "11_liana_ligand_ranks.csv")
else:
    ligand_table = pd.DataFrame(liana["AC_like_rank_key_ligands"])
ligand_table = ligand_table.set_index("ligand", verify_integrity=True)
ligand_ranks = ligand_table.loc[["TGFB2", "NECTIN2", "SPP1", "TGFB1", "MIF"], "AC_rank"]
if not ligand_ranks.isin(ranks).all():
    raise ValueError("A displayed LIANA ligand has an invalid rank.")
positions = np.arange(len(ligand_ranks))
ligand_axis.hlines(positions, 0, ligand_ranks, color="#CCCCCC", lw=1.2)
ligand_axis.scatter(ligand_ranks, positions, s=48, c=np.where(ligand_ranks.eq(1), "#C0392B", "#8FA5B0"))
for position, rank in zip(positions, ligand_ranks):
    ligand_axis.text(rank + 0.14, position, f"#{rank}", fontsize=6.1, va="center")
ligand_axis.set_yticks(positions, ligand_ranks.index, fontsize=6.4, fontstyle="italic")
ligand_axis.set(xlim=(0.4, 4.6), xticks=ranks)
ligand_axis.invert_yaxis()
ligand_axis.set_xlabel("AC rank among scored subtypes", fontsize=6.5)
ligand_axis.set_title("LIANA to macrophages\nGSE276177+286419", loc="left", fontsize=7.1)
plot.label_panel(ligand_axis, "c")

# Verify the plotted categories, the numerator and denominator of the joint top-two count, and all LIANA ligands.
required_text = [*category_labels, agreement_label, "Both top two", "LIANA", *ligand_ranks.index]
plot.save_figure(figure, "19_fig6_gene_ranks_and_liana", required_text)
