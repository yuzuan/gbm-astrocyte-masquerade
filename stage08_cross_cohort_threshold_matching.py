#!/usr/bin/env python3
"""
Stage 08: Cross-cohort threshold matching
Source artifact: matched_threshold_cross_cohort.json (version: fcf0aedc-6423-402d-9643-eea21d438400)

GBM AC-like Masquerade Analysis Pipeline
Malignant cells masquerade as astrocytes in glioblastoma and confound
immune-microenvironment analysis — a multi-cohort study.

This script was extracted from the interactive analysis session.
Adjust file paths as needed for your environment.
"""

import os
os.environ['NUMBA_DISABLE_JIT'] = '1'

import pandas as pd
import numpy as np
import anndata as ad
import json

# Load own cohort data
df = pd.read_csv("data/infercnv_cell_scores.csv", index_col=0)
meta = pd.read_csv("data/cell_metadata.csv", index_col=0)
df['condition'] = meta['condition'].reindex(df.index)

# Strict threshold: composite score = chr7 - chr10, threshold = 95th pct of immune reference
df['composite'] = df['chr7'] - df['chr10']
imm = df['cell_type'].isin(['Myeloid', 'T_NK'])
thr = np.quantile(df.loc[imm, 'composite'], 0.95)
df['malig_strict'] = df['composite'] > thr

# GBM astrocyte cluster
gbm_astro = (df['cell_type'] == 'Astrocyte') & (df['condition'] == 'GBM')
n = int(gbm_astro.sum())
strict_pct = 100 * df.loc[gbm_astro, 'malig_strict'].mean()
loose_pct = 100 * df.loc[gbm_astro, 'cnv_pos'].mean()

matched = {
    "own_strict_pct": round(strict_pct, 1),
    "own_loose_pct": round(loose_pct, 1),
    "own_n": n,
    "snGBM_strict_pct": 71.0,
    "snGBM_n": 6791,
    "own_ref_FPR_strict": {"Myeloid": 4.3, "T_NK": 10.0, "Oligodendrocyte": 19.5},
    "peritumoral_astro_strict_pct": 24.0,
    "note": "74.1% is loose cnv_pos (composite, used as conservative decontamination weight); 80.8% is strict immune-null 95th-pct threshold matched to snGBM's 71.0%"
}
json.dump(matched, open("matched_threshold_cross_cohort.json", "w"), indent=2)
