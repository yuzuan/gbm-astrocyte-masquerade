# Purpose: Average GSE211376 immune panel expression in four tumor states and compare within-cohort AC ranks with
# scGBM.
# Inputs: 03_result/11_cross_cohort_immune_gene_ranks/01_immune_gene_expression_and_ranks.csv (gene, scGBM_AC_rank and
# other published fields).
# Alternatively use data/intermediate/immune_evasion_cross_cohort.csv.
# data/intermediate/GSE211376_metadata.csv.gz provides cell IDs (first column) and predicted.high_hierarchy labels.
# In data/intermediate/GSE211376_raw_counts.tsv.gz, each count row represents one gene.
# Outputs: 03_result/12_gse211376_subtype_expression_and_ranks/01_mean_expression_by_subtype.csv and
# 02_cross_cohort_ac_ranks.csv in the same directory.
# From the repository root with the project NumPy and pandas environment, run:
# python 12_gse211376_subtype_expression_and_ranks.py

import gzip
from importlib import import_module
from pathlib import Path

import numpy as np
import pandas as pd

communication = import_module("00a_cell_communication")
upstream = import_module("00a_expression_and_clustering")

# Keep the gene order and scGBM ranks from stage 11. The externally supplied frozen panel has the same published
# fields.
panel_path = upstream.result_or_frozen(
    Path("03_result/11_cross_cohort_immune_gene_ranks/01_immune_gene_expression_and_ranks.csv"),
    "immune_evasion_cross_cohort.csv",
)
panel_table = pd.read_csv(panel_path)
genes = panel_table["gene"].tolist()
gene_set = set(genes)

# The count header lists cell IDs and the metadata uses its first column as row names. Accumulate library totals for
# all cells before selecting each state by label.
with gzip.open("data/intermediate/GSE211376_raw_counts.tsv.gz", "rt", encoding="utf-8") as stream:
    cells = communication.count_matrix_cells(stream)
    metadata_table = upstream.gse211376_metadata(cells)
    labels = metadata_table["predicted.high_hierarchy"]
    total_counts = np.zeros(len(cells))
    panel_counts = {}
    for line in stream:
        gene, counts = communication.read_gene_counts(line, len(cells))
        total_counts += counts
        if gene in gene_set:
            panel_counts[gene] = counts

# Include off-panel genes in library sizes. Apply log1p to each gene's per-cell counts per 10,000, then average within
# each state. Leave all four means missing for an unmeasured gene.
size_factors = communication.library_size_factors(total_counts)
states = {"AC": "AC-like", "MES": "MES-like", "OPC": "OPC-like", "NPC": "NPC-like"}
masks = {state: labels.eq(label).to_numpy() for state, label in states.items()}
expression_table = pd.DataFrame(np.nan, index=pd.Index(genes, name="gene"), columns=list(states))
for gene, counts in panel_counts.items():
    values = np.log1p(counts / size_factors)
    for state, mask in masks.items():
        if mask.any():
            expression_table.loc[gene, state] = values[mask].mean()

# Rank only genes with means for all four states. Use the original AC, MES, OPC, NPC column order and descending
# quicksort to find AC's position.
ac_ranks = pd.Series(np.nan, index=expression_table.index)
for gene, state_means in expression_table.dropna().iterrows():
    ordered = state_means.sort_values(ascending=False, kind="quicksort")
    ac_ranks.loc[gene] = ordered.index.get_loc("AC") + 1

# Merge one-to-one by gene, keeping every field from the published panel. The two concordance fields here compare
# GSE211376 with scGBM.
gse_table = pd.DataFrame({"gse211376_AC": expression_table["AC"], "gse211376_AC_rank": ac_ranks})
comparison_table_total = panel_table.merge(
    gse_table, left_on="gene", right_index=True, how="left", validate="one_to_one"
)
rank_columns = ["gse211376_AC_rank", "scGBM_AC_rank"]
valid_table = comparison_table_total.dropna(subset=rank_columns).copy()
valid_table[rank_columns] = valid_table[rank_columns].astype(int)
valid_table["rank_concordant"] = valid_table["gse211376_AC_rank"].eq(valid_table["scGBM_AC_rank"])
valid_table["both_AC_top2"] = valid_table[rank_columns].le(2).all(axis=1)

# Derive the eligible denominator from the panel and coverage across all four states. Verify every column in both
# files on readback in original gene order.
complete = expression_table.notna().all(axis=1)
expected = panel_table["scGBM_AC_rank"].notna() & panel_table["gene"].isin(expression_table.index[complete])
if not valid_table[rank_columns].isin(range(1, len(states) + 1)).all().all():
    raise ValueError("A rank falls outside the range of the four states.")
output_dir = Path("03_result/12_gse211376_subtype_expression_and_ranks")
communication.write_gene_table(output_dir / "01_mean_expression_by_subtype.csv", expression_table.reset_index(), genes)
communication.write_gene_table(
    output_dir / "02_cross_cohort_ac_ranks.csv", valid_table, panel_table.loc[expected, "gene"]
)
print(f"Panel genes: {len(genes)}; comparable AC ranks in both cohorts: {len(valid_table)}.")
