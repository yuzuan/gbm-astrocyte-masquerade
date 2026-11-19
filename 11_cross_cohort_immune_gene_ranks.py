# Purpose: Compare the rank of AC-like mean immune gene expression among four tumor states in scGBM and the in-house
# cohort.
# Inputs: X and Minor_cluster in data/external/scGBM/adata2.h5ad; X and state in data/malignant_gbm_cnv.h5ad.
# Output: 03_result/11_cross_cohort_immune_gene_ranks/01_immune_gene_expression_and_ranks.csv
# Run from the repository root with the project Python, AnnData and NumPy: python 11_cross_cohort_immune_gene_ranks.py

from contextlib import closing
from importlib import import_module

import anndata as ad
import numpy as np
import pandas as pd

communication = import_module("00a_cell_communication")

# Keep the published gene categories and order. Each output row represents a gene measured in both expression matrices.
panel = {
    "MHC-I": ["HLA-A", "HLA-B", "HLA-C", "B2M", "TAP1", "TAP2"],
    "MHC-II": ["HLA-DRA", "HLA-DRB1", "HLA-DPA1", "HLA-DPB1"],
    "Checkpoint ligands": ["CD274", "PDCD1LG2", "CD80", "CD86", "LGALS9", "NECTIN2", "PVR", "VTCN1"],
    "Immunosuppressive": ["TGFB1", "TGFB2", "IL10", "IDO1", "PTGS2", "ARG1", "NOS2"],
    "Complement regulators": ["CD46", "CD55", "CD59"],
    "Dont-eat-me": ["CD47", "CD24"],
    "Immunosuppressive chemokines": ["CCL2", "CSF1", "CXCL12", "CCL22"],
    "IFN response": ["IFITM1", "IFITM2", "IFITM3", "ISG15", "MX1", "STAT1", "IRF1"],
}
category_by_gene = {}
for category, genes in panel.items():
    for gene in genes:
        category_by_gene[gene] = category

# Match genes by name. Sample up to 5,000 cells per scGBM state and retain every cell in each in-house state.
scgbm = ad.read_h5ad("data/external/scGBM/adata2.h5ad", backed="r")
with closing(scgbm.file):
    own_dataset = ad.read_h5ad("data/malignant_gbm_cnv.h5ad", backed="r")
    with closing(own_dataset.file):
        shared_genes = [gene for gene in category_by_gene if gene in scgbm.var_names and gene in own_dataset.var_names]
        scgbm_means = {}
        for state, mask in communication.scgbm_state_masks(scgbm.obs["Minor_cluster"]).items():
            positions = communication.sampled_scgbm_positions(mask)
            expression_table = communication.cell_expression_table(scgbm, positions, shared_genes)
            scgbm_means[state] = {gene: np.mean(expression_table[gene].to_numpy()) for gene in shared_genes}

        # Use the original in-house labels, including AClike. Rank states within each cohort, then compare ranks
        # across cohorts.
        own_means = {}
        for state in ("AClike", "MESlike", "OPClike", "NPClike"):
            positions = np.flatnonzero(own_dataset.obs["state"].eq(state))
            expression_table = communication.cell_expression_table(own_dataset, positions, shared_genes)
            own_means[state] = {gene: np.mean(expression_table[gene].to_numpy()) for gene in shared_genes}

# Rows are genes and columns are states. AC's rank is one plus the number of states with a strictly higher mean. Ties
# share a rank. Round the means to three decimals after ranking.
scgbm_table = pd.DataFrame(scgbm_means, index=shared_genes)
own_table = pd.DataFrame(own_means, index=shared_genes)
if not np.isfinite(scgbm_table.to_numpy()).all() or not np.isfinite(own_table.to_numpy()).all():
    raise ValueError("Nonfinite state means prevent ranking all four states.")
scgbm_ranks = scgbm_table.rank(axis=1, ascending=False, method="min")["AC"].astype(int)
own_ranks = own_table.rank(axis=1, ascending=False, method="min")["AClike"].astype(int)
result_table = pd.DataFrame(
    {
        "category": [category_by_gene[gene] for gene in shared_genes],
        "gene": shared_genes,
        "scGBM_AC": scgbm_table["AC"].round(3),
        "scGBM_AC_rank": scgbm_ranks,
        "own_AC": own_table["AClike"].round(3),
        "own_AC_rank": own_ranks,
        "rank_concordant": scgbm_ranks == own_ranks,
        "both_AC_top2": (scgbm_ranks <= 2) & (own_ranks <= 2),
    }
)

# Check the rank direction against the unrounded cohort means: AC ranks first if it equals the maximum for that gene.
if not scgbm_ranks.eq(1).equals(scgbm_table["AC"].eq(scgbm_table.max(axis=1))):
    raise ValueError("First-place AC ranks in scGBM disagree with the gene-wise expression maxima.")
if not own_ranks.eq(1).equals(own_table["AClike"].eq(own_table.max(axis=1))):
    raise ValueError("First-place AC ranks in the in-house cohort disagree with the gene-wise expression maxima.")
communication.write_gene_table(
    "03_result/11_cross_cohort_immune_gene_ranks/01_immune_gene_expression_and_ranks.csv", result_table, shared_genes
)
print(f"Shared panel genes: {len(shared_genes)}; equal AC ranks: {int(result_table['rank_concordant'].sum())}.")
