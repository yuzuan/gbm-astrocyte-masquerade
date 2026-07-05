#!/usr/bin/env python3
"""
Stage 11: LIANA cross-method validation
Source artifact: liana_validation.json (version: 24890dca-ff5a-490f-b7b9-58f738f14a74)

GBM AC-like Masquerade Analysis Pipeline
Malignant cells masquerade as astrocytes in glioblastoma and confound
immune-microenvironment analysis — a multi-cohort study.

This script was extracted from the interactive analysis session.
Adjust file paths as needed for your environment.
"""

import os
os.environ['NUMBA_DISABLE_JIT'] = '1'

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse
import json

# Load scGBM to get chromosome map
sc_a = ad.read_h5ad("data/external/scGBM/adata2.h5ad", backed='r')
chrom_map = dict(zip(sc_a.var.index.astype(str), sc_a.var['chromosome'].astype(str)))
chr7_genes = {g for g, c in chrom_map.items() if c == 'chr7'}
chr10_genes = {g for g, c in chrom_map.items() if c == 'chr10'}

# Load snGBM
sn = ad.read_h5ad("data/external/snGBM/results/1_snRNA_processing/adata1.h5ad")
sn_genes = set(sn.var_names)
c7 = sorted(chr7_genes & sn_genes)
c10 = sorted(chr10_genes & sn_genes)

def mean_expr(adata, genes):
    idx = [adata.var_names.get_loc(g) for g in genes if g in adata.var_names]
    X = adata[:, idx].X
    X = X.toarray() if sparse.issparse(X) else np.asarray(X)
    return X.mean(1)

# Compute CNV scores
sn.obs['chr7_raw'] = mean_expr(sn, c7)
sn.obs['chr10_raw'] = mean_expr(sn, c10)

ref2 = sn.obs['leiden_0.5'].isin(['5', '10']).values
r7 = sn.obs.loc[ref2, 'chr7_raw'].mean()
r10 = sn.obs.loc[ref2, 'chr10_raw'].mean()
sn.obs['cnv7'] = sn.obs['chr7_raw'] - r7
sn.obs['cnv10'] = sn.obs['chr10_raw'] - r10
sn.obs['cnv_score'] = sn.obs['cnv7'] - sn.obs['cnv10']

thr = np.quantile(sn.obs.loc[ref2, 'cnv_score'], 0.95)
sn.obs['malig_strict'] = sn.obs['cnv_score'] > thr

# Load LIANA results
la = ad.read_h5ad("data/external/snGBM/results/6_CCC_liana/adata_with_liana.h5ad", backed='r')
lr = la.uns['liana_res'].copy()

malig = ['AC-like', 'MES-like', 'OPC-like', 'NPC-like']
tam = lr[(lr['source'].isin(malig)) & (lr['target'] == 'Macrophage')].copy()

key_ligs = ['TGFB1', 'TGFB2', 'TGFB3', 'CSF1', 'CCL2', 'CD274', 'PDCD1LG2', 'NECTIN2', 'PVR', 'SPP1', 'IL10', 'LGALS9', 'MIF']
sub = tam[tam['ligand_complex'].isin(key_ligs)].copy()

results = []
for lig in key_ligs:
    d = sub[sub['ligand_complex'] == lig]
    if len(d) == 0:
        continue
    agg = d.groupby('source')['lr_means'].max().reindex(malig).dropna()
    if 'AC-like' not in agg.index or len(agg) < 2:
        continue
    rank = (agg.rank(ascending=False)['AC-like'])
    results.append((lig, int(rank), len(agg), round(agg['AC-like'], 2),
                    agg.idxmax(), round(agg.max(), 2)))

rdf = pd.DataFrame(results, columns=['ligand', 'AC_rank', 'n_subtypes', 'AC_lr_mean', 'top_subtype', 'top_val'])
ac1 = (rdf['AC_rank'] == 1).sum()

liana_val = {
    "tool": "LIANA (real ligand-receptor framework, CellPhoneDB magnitude + specificity ranks) run on independent snGBM 69k",
    "malig_to_macrophage_pairs": int(len(tam)),
    "AC_like_rank_key_ligands": rdf.to_dict('records'),
    "AC_like_top1": f"{ac1}/{len(rdf)} key immunosuppressive ligands",
    "robust_across_toolchange": ["TGFB2 (AC-like #1, only expressing subtype)", "NECTIN2 checkpoint (AC-like #1)", "SPP1 (AC-like #1)"],
    "NOT_recovered_by_real_tool": ["CSF1 and CCL2 did not pass LIANA expression/specificity filters (contradicts our product-statistic claim of AC-like dominance)", "TGFB1 and MIF higher in NPC-like"],
    "interpretation": "Real LIANA supports TGFB2 as the robust AC-like immunosuppressive ligand; the broader 'AC #1 in 8/9 pairs' from our in-house product statistic is NOT fully reproduced and should be narrowed."
}
json.dump(liana_val, open("liana_validation.json", "w"), indent=2)
print("saved liana_validation.json")
print(json.dumps({k: liana_val[k] for k in ['AC_like_top1', 'robust_across_toolchange', 'NOT_recovered_by_real_tool']}, indent=1, ensure_ascii=False))
