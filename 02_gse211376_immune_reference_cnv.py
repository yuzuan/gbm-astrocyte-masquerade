# Purpose: Estimate the immune-reference +7/-10 exceedance rate for the GSE211376 author labels Astrocyte and AC-like,
#          both pooled and separately.
# Inputs: data/intermediate/GSE211376_raw_counts.tsv.gz (genes by barcode), GSE211376_metadata.csv.gz in the same
#         directory (barcode, predicted.high_hierarchy, patient), and var/chromosome in data/external/scGBM/adata2.h5ad.
# Output: 03_result/02_gse211376_immune_reference_cnv/01_gse211376_immune_reference_cnv.json
# Usage: From the repository root, run python 02_gse211376_immune_reference_cnv.py with the original numeric precision.

from importlib import import_module
from pathlib import Path

import numpy as np

common = import_module("00a_expression_and_clustering")

# Read integer counts row by row as float32 and accumulate in float64. Each chromosome uses its matched gene count.
expression_table = common.gse211376_expression(common.chromosome_gene_sets(), np.float32)
metadata_table = common.gse211376_metadata(expression_table.index)
if metadata_table["patient"].isna().any():
    raise ValueError("Some GSE211376 count barcodes lack patient IDs, so the patient denominator cannot be determined.")
cell_types = metadata_table["predicted.high_hierarchy"]

# The two TAM types center the scores. They also set the 95th percentile together with CD4/CD8 cells. Classify each
# cell against this unrounded strict cutoff.
tam = cell_types.isin(["TAM-MG", "TAM-BDM"])
immune = tam | cell_types.eq("CD4/CD8")
cnv_scores, threshold, center7, center10 = common.centered_cnv(
    expression_table["chr7"].to_numpy(), expression_table["chr10"].to_numpy(), tam, immune
)
positive = cnv_scores > threshold

# Keep the original cell count, percentage above threshold, and median score for each observed author annotation.
by_annotation = {}
for label in (
    "Astrocyte", "AC-like", "MES-like", "NPC-like", "OPC-like", "OPC", "Oligodendrocyte",
    "TAM-MG", "TAM-BDM", "CD4/CD8", "Endothelial", "Pericyte", "Neuron",
):
    selected = cell_types.eq(label)
    if selected.any():
        group = common.positive_summary(positive, selected)
        group["cnv_score_median"] = float(np.median(cnv_scores[selected]))
        by_annotation[label] = group

# Pool the two mutually exclusive author labels by cell. The pooled TAM rate likewise uses the total cell count.
coarse = common.positive_summary(positive, cell_types.isin(["Astrocyte", "AC-like"]))
refined = common.positive_summary(positive, cell_types.eq("Astrocyte"))
ac_like = common.positive_summary(positive, cell_types.eq("AC-like"))
reference_rates = {}
for label in ("TAM-MG", "TAM-BDM", "Oligodendrocyte", "CD4/CD8"):
    group = common.positive_summary(positive, cell_types.eq(label))
    reference_rates[label] = group["cnv_pos_pct"]
combined_tam = common.positive_summary(positive, tam)
reference_rates["TAM_combined"] = combined_tam["cnv_pos_pct"]

# IDH status is a historical cohort description. The GEO barcode table supplies cell labels and patient IDs but
# does not establish individual patients' IDH status.
result = {
    "dataset": "GSE211376",
    "source": "GEO GSE211376 snRNA-seq counts and author cell annotations.",
    "n_cells": len(expression_table),
    "n_patients": int(metadata_table["patient"].nunique()),
    "IDH": "wildtype",
    "idh_source": "Historical cohort description; individual IDH status is absent from the cell metadata.",
    "strict_threshold": float(threshold),
    "tam_ref_center": {"chr7": float(center7), "chr10": float(center10)},
    "coarse_astro_cluster": {"definition": "Author labels Astrocyte + AC-like", **coarse},
    "refined_astrocyte_only": refined,
    "AC_like": ac_like,
    "reference_FPR": reference_rates,
    "by_annotation": by_annotation,
}

# Check that the pooled denominator equals the sum of both subgroups. Read back every JSON field and the exact cutoff.
if coarse["n"] != refined["n"] + ac_like["n"] or coarse["n"] == 0:
    raise ValueError("The pooled GSE211376 astrocyte count is zero or differs from the sum of the two author groups.")
common.write_checked_json(
    Path("03_result/02_gse211376_immune_reference_cnv/01_gse211376_immune_reference_cnv.json"), result
)
print(
    f"The pooled GSE211376 author labels cover {coarse['n']} cells, "
    f"with {coarse['cnv_pos_pct']}% above the immune-reference cutoff."
)
