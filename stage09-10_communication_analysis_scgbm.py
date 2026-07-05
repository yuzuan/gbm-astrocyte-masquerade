#!/usr/bin/env python3
"""
Stage 09-10: Communication analysis (scGBM)
Source artifact: communication_analysis_results.json (version: 792df217-b898-42ea-b0c1-68590eec417e)

GBM AC-like Masquerade Analysis Pipeline
Malignant cells masquerade as astrocytes in glioblastoma and confound
immune-microenvironment analysis — a multi-cohort study.

This script was extracted from the interactive analysis session.
Adjust file paths as needed for your environment.
"""

import os
os.environ["NUMBA_CACHE_DIR"] = "/tmp/numba_cache"
os.environ["NUMBA_DISABLE_JIT"] = "1"

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp
import json
import warnings
warnings.simplefilter("ignore")

# Load own malignant data
own = ad.read_h5ad("data/malignant_gbm_cnv.h5ad")
own_var = own.var_names.tolist()

# Load scGBM backed
scgbm = ad.read_h5ad("data/external/scGBM/adata2.h5ad", backed='r')
sc_var = scgbm.var_names.tolist()

def extract_expr(adata, var_list, mask, genes, max_cells=5000):
    idx = np.where(mask.values)[0]
    if len(idx) > max_cells:
        np.random.seed(42)
        idx = np.random.choice(idx, max_cells, replace=False)
    means, pcts = {}, {}
    for g in genes:
        if g in var_list:
            gi = var_list.index(g)
            col = adata.X[idx, gi]
            if sp.issparse(col):
                col = col.toarray().flatten()
            else:
                col = np.array(col).flatten()
            means[g] = float(np.mean(col))
            pcts[g] = float((col > 0).mean() * 100)
        else:
            means[g] = np.nan
            pcts[g] = np.nan
    return means, pcts

# Define L-R pairs for immunosuppressive communication
lr_pairs_tam = {
    'TGFB2-TGFBR1': ('TGFB2', 'TGFBR1'),
    'TGFB2-TGFBR2': ('TGFB2', 'TGFBR2'),
    'CCL2-CCR2': ('CCL2', 'CCR2'),
    'CSF1-CSF1R': ('CSF1', 'CSF1R'),
    'CXCL12-CXCR4': ('CXCL12', 'CXCR4'),
}

lr_pairs_tnk = {
    'CD274-PDCD1': ('CD274', 'PDCD1'),
    'NECTIN2-TIGIT': ('NECTIN2', 'TIGIT'),
    'PVR-TIGIT': ('PVR', 'TIGIT'),
    'PDCD1LG2-PDCD1': ('PDCD1LG2', 'PDCD1'),
}

all_lr_genes = set()
for pairs in [lr_pairs_tam, lr_pairs_tnk]:
    for l, r in pairs.values():
        all_lr_genes.add(l)
        all_lr_genes.add(r)

all_lr_genes = list(all_lr_genes)
avail = [g for g in all_lr_genes if g in sc_var]

sc_cell_masks = {
    'AC': scgbm.obs['Minor_cluster'] == 'AC',
    'MES1': scgbm.obs['Minor_cluster'] == 'MES1',
    'OPC': scgbm.obs['Minor_cluster'] == 'OPC',
    'NPC': scgbm.obs['Minor_cluster'].isin(['NPC1', 'NPC2']),
    'TAM': scgbm.obs['Major_category'] == 'TAM/Macrophage',
    'T/NK': scgbm.obs['Major_category'] == 'T/NK',
}

lr_expr = {}
for name, mask in sc_cell_masks.items():
    m, p = extract_expr(scgbm, sc_var, mask, avail)
    lr_expr[name] = m

malignant_types = ['AC', 'MES1', 'OPC', 'NPC']

tam_scores = {}
for pair_name, (lig, rec) in lr_pairs_tam.items():
    if lig not in avail or rec not in avail:
        continue
    scores = {}
    for st in malignant_types:
        scores[st] = lr_expr[st][lig] * lr_expr['TAM'][rec]
    tam_scores[pair_name] = scores

tnk_scores = {}
for pair_name, (lig, rec) in lr_pairs_tnk.items():
    if lig not in avail or rec not in avail:
        continue
    scores = {}
    for st in malignant_types:
        scores[st] = lr_expr[st][lig] * lr_expr['T/NK'][rec]
    tnk_scores[pair_name] = scores

scgbm.file.close()

# Load communication CSV
comm = pd.read_csv("data/cellcell_communication.csv")
gbm_am = comm[(comm['sender'] == 'Astrocyte') & (comm['receiver'] == 'Myeloid') & (comm['condition'] == 'GBM')]

# AClike expression for key ligands from own malignant data
key_ligands = ['LGALS1', 'ANXA1', 'TGFB1', 'TGFB2', 'CCL2', 'CSF1',
               'CD274', 'VEGFA', 'EGFR', 'SERPINE1']

ac_mask = own.obs['state'] == 'AClike'
ac_idx = np.where(ac_mask.values)[0]

ac_expr = {}
for g in key_ligands:
    if g in own_var:
        gi = own_var.index(g)
        col = own.X[ac_idx, gi]
        if sp.issparse(col):
            col = col.toarray().flatten()
        ac_expr[g] = float(np.mean(col))

# Use GBM-specific contamination rate
f_malig = 0.741
f_normal = 1 - f_malig

results_pollution = []
for _, row in gbm_am.iterrows():
    lig, rec = row['ligand'], row['receptor']
    mean_before, score_before, p_before = row['mean_L_sender'], row['score'], row['perm_p']
    if lig in ac_expr:
        mean_malig = ac_expr[lig]
        mean_after = max(0, (mean_before - f_malig * mean_malig) / f_normal)
        score_after = score_before * (mean_after / mean_before) if mean_before > 0 else 0
        pct_change = ((score_after - score_before) / score_before * 100) if score_before > 0 else 0
        results_pollution.append({
            'ligand': lig, 'receptor': rec,
            'mean_L_before': round(mean_before, 4), 'mean_L_after': round(mean_after, 4),
            'mean_L_malig': round(mean_malig, 4),
            'score_before': round(score_before, 4), 'score_after': round(score_after, 4),
            'pct_change': round(pct_change, 1), 'sig_before': 'yes' if p_before < 0.05 else 'no',
        })

comm_data = {
    'tam_scores': tam_scores,
    'tnk_scores': tnk_scores,
    'pollution_results': results_pollution,
    'contamination_rates': {
        'Astrocyte_cnv_pos': 72.3,
        'Astrocyte_total': 3074,
        'Astrocyte_cnv_pos_n': 2222,
        'Astrocyte_cleaned_n': 852,
        'Myeloid_cnv_pos': 2.1,
        'Oligodendrocyte_cnv_pos': 11.7,
        'Astrocyte_cnv_pos_GBM': 74.1,
        'Astrocyte_cnv_pos_Normal': 20.2,
        'f_malig_used': 0.741,
        'note': 'GBM-specific rate (74.1%) used for GBM communication decontamination',
    }
}

with open('communication_analysis_results.json', 'w') as f:
    json.dump(comm_data, f, indent=2, default=str)
print("Saved communication_analysis_results.json")
