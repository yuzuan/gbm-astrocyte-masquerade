# Purpose: Select GSE211376 astrocytes by lineage-marker expression alone, estimate their +7/-10 exceedance rate,
#          and compare it with the rates for the author labels.
# Inputs: data/intermediate/GSE211376_raw_counts.tsv.gz (genes by barcode), GSE211376_metadata.csv.gz in the same
#         directory (barcode, predicted.high_hierarchy), and var/chromosome in data/external/scGBM/adata2.h5ad.
# Output: 03_result/03_gse211376_marker_defined_astrocytes/01_gse211376_marker_defined_astrocyte_cnv.json
# Usage: From the repository root, run python 03_gse211376_marker_defined_astrocytes.py with the original count reader.

from importlib import import_module
from pathlib import Path

import numpy as np

common = import_module("00a_expression_and_clustering")

# The original myeloid panel adds C1QA and C1QB to the GSE174554 panel. Read counts as float64 and sum separately
# for each of the three lineage panels and each chromosome.
gene_sets = common.chromosome_gene_sets()
gene_sets.update(common.lineage_genes)
gene_sets["imm"] = common.lineage_genes["imm"] + ["C1QA", "C1QB"]
expression_table = common.gse211376_expression(gene_sets, np.float64)

# The historical astrocyte definition requires an astrocyte score above the cohort's upper quartile, an immune score
# at or below the median, and an oligodendrocyte score below 1.0.
astro_score = expression_table["astro"].to_numpy()
immune_score = expression_table["imm"].to_numpy()
oligo_score = expression_table["oligo"].to_numpy()
astrocytes = (astro_score > np.quantile(astro_score, 0.75)) & (immune_score <= np.quantile(immune_score, 0.50))
astrocytes &= oligo_score < 1.0

# Cells above the immune-score 90th percentile are the myeloid reference for centering and thresholding.
# Oligodendrocytes provide a comparison group.
myeloid = immune_score > np.quantile(immune_score, 0.90)
oligodendrocytes = (oligo_score > np.quantile(oligo_score, 0.75)) & (astro_score <= np.quantile(astro_score, 0.50))
cnv_scores, threshold, _, _ = common.centered_cnv(
    expression_table["chr7"].to_numpy(), expression_table["chr10"].to_numpy(), myeloid, myeloid
)
positive = cnv_scores > threshold
marker_astro = common.positive_summary(positive, astrocytes)
reference_myeloid = common.positive_summary(positive, myeloid)
reference_oligo = common.positive_summary(positive, oligodendrocytes)

# Author labels enter only the two post-selection comparisons. Align by barcode to preserve cell order and annotation
# coverage even when the metadata contain additional rows.
metadata_table = common.gse211376_metadata(expression_table.index)
cell_types = metadata_table["predicted.high_hierarchy"]
result = {
    "dataset": "GSE211376",
    "n_cells": len(expression_table),
    "strict_threshold": float(threshold),
    "marker_defined_astrocyte": marker_astro,
    "reference_FPR": {"myeloid": reference_myeloid["cnv_pos_pct"], "oligodendrocyte": reference_oligo["cnv_pos_pct"]},
    "author_astrocyte_only": common.positive_summary(positive, cell_types.eq("Astrocyte")),
    "author_AClike": common.positive_summary(positive, cell_types.eq("AC-like")),
}

# Marker-defined astrocytes must form a nonempty group disjoint from the myeloid cells that set the cutoff.
if marker_astro["n"] == 0 or np.any(astrocytes & myeloid):
    raise ValueError("GSE211376 marker-defined astrocytes are absent or overlap the myeloid threshold reference.")
common.write_checked_json(
    Path("03_result/03_gse211376_marker_defined_astrocytes/01_gse211376_marker_defined_astrocyte_cnv.json"), result
)
print(f"GSE211376: {marker_astro['n']} marker-defined astrocytes, with {marker_astro['cnv_pos_pct']}% above cutoff.")
