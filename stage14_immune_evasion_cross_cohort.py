#!/usr/bin/env python3
"""
Stage 14: Immune evasion cross-cohort
Source artifact: immune_evasion_cross_cohort.csv (version: 081c02eb-ac10-4b22-8d61-630ee819c8c9)

GBM AC-like Masquerade Analysis Pipeline
Malignant cells masquerade as astrocytes in glioblastoma and confound
immune-microenvironment analysis — a multi-cohort study.

This script was extracted from the interactive analysis session.
Adjust file paths as needed for your environment.
"""

import os
os.environ["NUMBA_CACHE_DIR"] = "/tmp/numba_cache"
os.environ["NUMBA_DISABLE_JIT"] = "1"
os.makedirs("/tmp/numba_cache", exist_ok=True)

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp
import warnings
warnings.simplefilter("ignore")

# Load scGBM data
adata = ad.read_h5ad("data/external/scGBM/adata2.h5ad", backed='r')
var_names = adata.var_names.tolist()

# Load own malignant data
own_malig = ad.read_h5ad("data/malignant_gbm_cnv.h5ad")
own_var = own_malig.var_names.tolist()

# Define immune evasion panel
immune_evasion_panel = {
    'MHC-I': ['HLA-A', 'HLA-B', 'HLA-C', 'B2M', 'TAP1', 'TAP2'],
    'MHC-II': ['HLA-DRA', 'HLA-DRB1', 'HLA-DPA1', 'HLA-DPB1'],
    'Checkpoint ligands': ['CD274', 'PDCD1LG2', 'CD80', 'CD86', 'LGALS9', 'NECTIN2', 'PVR', 'VTCN1'],
    'Immunosuppressive': ['TGFB1', 'TGFB2', 'IL10', 'IDO1', 'PTGS2', 'ARG1', 'NOS2'],
    'Complement regulators': ['CD46', 'CD55', 'CD59'],
    'Dont-eat-me': ['CD47', 'CD24'],
    'Immunosuppressive chemokines': ['CCL2', 'CSF1', 'CXCL12', 'CCL22'],
    'IFN response': ['IFITM1', 'IFITM2', 'IFITM3', 'ISG15', 'MX1', 'STAT1', 'IRF1'],
}

all_genes = []
for cat, genes in immune_evasion_panel.items():
    avail = [g for g in genes if g in var_names]
    all_genes.extend(avail)

def get_mean_expr(mask, gene_list):
    idx = np.where(mask.values)[0]
    if len(idx) > 5000:
        np.random.seed(42)
        idx = np.random.choice(idx, 5000, replace=False)
    means = {}
    pcts = {}
    for gn in gene_list:
        gi = var_names.index(gn)
        col = adata.X[idx, gi]
        if sp.issparse(col):
            col = col.toarray().flatten()
        else:
            col = np.array(col).flatten()
        means[gn] = np.mean(col)
        pcts[gn] = (col > 0).mean() * 100
    return means, pcts

masks = {
    'AC': adata.obs['Minor_cluster'] == 'AC',
    'MES1': adata.obs['Minor_cluster'] == 'MES1',
    'OPC': adata.obs['Minor_cluster'] == 'OPC',
    'NPC': adata.obs['Minor_cluster'].isin(['NPC1', 'NPC2']),
    'Macrophage': adata.obs['cell_type'] == 'Macrophage',
    'Microglia': adata.obs['cell_type'] == 'Microglia',
    'T/NK': adata.obs['cell_type'] == 'NK/T',
}

results = {}
for name, mask in masks.items():
    results[name] = get_mean_expr(mask, all_genes)

# Extract own data results
own_states = ['AClike', 'MESlike', 'OPClike', 'NPClike']
own_results = {}
for state in own_states:
    mask = own_malig.obs['state'] == state
    idx = np.where(mask.values)[0]
    means = {}
    for g in all_genes:
        if g in own_var:
            gi = own_var.index(g)
            col = own_malig.X[idx, gi]
            if sp.issparse(col):
                col = col.toarray().flatten()
            means[g] = np.mean(col)
        else:
            means[g] = np.nan
    own_results[state] = means

malignant_types = ['AC', 'MES1', 'OPC', 'NPC']

concordance = []

for cat, genes in immune_evasion_panel.items():
    avail_both = [g for g in genes if g in var_names and g in own_var]
    for g in avail_both:
        sc_vals = {st.replace('like', ''): results[st][0][g]
                   for st in ['AC', 'MES1', 'OPC', 'NPC']}
        sc_rank = 1 + sum(1 for v in sc_vals.values() if v > sc_vals['AC'])

        own_vals = {st.replace('like', ''): own_results[st][g] for st in own_states}
        own_rank = 1 + sum(1 for v in own_vals.values() if v > own_vals['AC'])

        concordance.append({
            'category': cat,
            'gene': g,
            'scGBM_AC': round(sc_vals['AC'], 3),
            'scGBM_AC_rank': sc_rank,
            'own_AC': round(own_vals['AC'], 3),
            'own_AC_rank': own_rank,
            'rank_concordant': sc_rank == own_rank,
            'both_AC_top2': sc_rank <= 2 and own_rank <= 2,
        })

df_conc = pd.DataFrame(concordance)
df_conc.to_csv("immune_evasion_cross_cohort.csv", index=False)

adata.file.close()
