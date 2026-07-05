#!/usr/bin/env python3
"""
Stage 16: scGBM AC validation
Source artifact: scgbm_ac_masquerade_validation.csv (version: a82d1458-7091-4e3d-bf80-e9fb82d3b269)

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

adata = ad.read_h5ad("{{artifact:MISSING:f76ec3e5-640d-42c0-9744-af7788ab771e}}", backed='r')

masquerade_genes = ['GFAP', 'AQP4', 'S100B', 'SLC1A3', 'CLU', 'ALDOC', 'CD44', 'VIM',
                    'CHI3L1', 'CHI3L2', 'EGFR', 'PDGFRA', 'SOX2', 'NES', 'HIF1A',
                    'VEGFA', 'LGALS1', 'LGALS3', 'SERPINE1', 'ANXA1']

var_names = adata.var_names.tolist()
available = [g for g in masquerade_genes if g in var_names]
gene_idx = [var_names.index(g) for g in available]

ac_tumor = (adata.obs['Minor_cluster'] == 'AC') & (adata.obs['tissue_type'] == 'Tumor')
ac_peri = (adata.obs['Minor_cluster'] == 'AC') & (adata.obs['tissue_type'] == 'Peritumoral')

def get_mean_expr(mask, gene_indices, gene_names):
    idx = np.where(mask.values)[0]
    if len(idx) > 5000:
        np.random.seed(42)
        idx = np.random.choice(idx, 5000, replace=False)
    means = {}
    pcts = {}
    for gi, gn in zip(gene_indices, gene_names):
        col = adata.X[idx, gi]
        if sp.issparse(col):
            col = col.toarray().flatten()
        else:
            col = np.array(col).flatten()
        means[gn] = np.mean(col)
        pcts[gn] = (col > 0).mean() * 100
    return means, pcts

ac_t_means, ac_t_pcts = get_mean_expr(ac_tumor, gene_idx, available)
ac_p_means, ac_p_pcts = get_mean_expr(ac_peri, gene_idx, available)

comparison = pd.DataFrame({
    'gene': available,
    'AC_Tumor_mean': [ac_t_means[g] for g in available],
    'AC_Peri_mean': [ac_p_means[g] for g in available],
    'AC_Tumor_pct': [ac_t_pcts[g] for g in available],
    'AC_Peri_pct': [ac_p_pcts[g] for g in available],
})

comparison['fold_T_vs_P'] = comparison['AC_Tumor_mean'] / comparison['AC_Peri_mean'].replace(0, np.nan)

own_fold = {
    'GFAP': 0.47, 'AQP4': 1.36, 'S100B': 1.00, 'SLC1A3': 1.27, 'CLU': 1.23,
    'ALDOC': 0.95, 'CD44': 0.44, 'VIM': 1.28, 'CHI3L1': 0.94, 'CHI3L2': 0.07,
    'EGFR': 2.13, 'PDGFRA': 0.20, 'SOX2': 1.63, 'NES': 2.66, 'HIF1A': 1.17,
    'VEGFA': 0.70, 'LGALS1': 1.08, 'LGALS3': 2.10, 'SERPINE1': 1.13, 'ANXA1': 0.61
}
comparison['own_fold'] = comparison['gene'].map(own_fold)

def classify(f):
    if pd.isna(f): return 'NA'
    if f > 1.3: return 'malignant↑'
    if f < 0.8: return 'normal↑'
    return 'shared'

comparison['own_interpret'] = comparison['own_fold'].apply(classify)
comparison['scgbm_interpret'] = comparison['fold_T_vs_P'].apply(
    lambda f: 'Tumor↑' if f > 1.3 else ('Peri↑' if f < 0.8 else 'shared') if pd.notna(f) else 'NA'
)

comparison.to_csv("scgbm_ac_masquerade_validation.csv", index=False)

df = pd.read_csv("scgbm_ac_masquerade_validation.csv")
df['CAVEAT'] = ("Both AC-Tumor and AC-Peri are CNV+ malignant (mean cnv_score=0.0145); "
                "fold_T_vs_P reflects location difference NOT malignant-vs-normal. "
                "20% concordance with own_fold is expected because own data compares "
                "CNV+ malignant vs CNV- normal astrocytes (different baseline).")
df.to_csv("scgbm_ac_masquerade_validation.csv", index=False)
