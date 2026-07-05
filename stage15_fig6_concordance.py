#!/usr/bin/env python3
"""
Stage 15: Fig6 concordance
Source artifact: fig6_concordance_gse211376_vs_scgbm.csv (version: 01dee4c7-55b6-4ad4-955a-14d559d8e42c)

GBM AC-like Masquerade Analysis Pipeline
Malignant cells masquerade as astrocytes in glioblastoma and confound
immune-microenvironment analysis — a multi-cohort study.

This script was extracted from the interactive analysis session.
Adjust file paths as needed for your environment.
"""

import gzip, numpy as np, pandas as pd, h5py

# Load immune evasion panel
panel = pd.read_csv('data/intermediate/immune_evasion_cross_cohort.csv')
genes41 = panel['gene'].tolist()
gene_set = set(genes41)

# Load GSE211376 metadata
meta = pd.read_csv('data/intermediate/GSE211376_metadata.csv.gz', index_col=0)
ann = meta['predicted.high_hierarchy']
subtypes = {'AC': 'AC-like', 'MES': 'MES-like', 'OPC': 'OPC-like', 'NPC': 'NPC-like'}

# Stream genes x cells TSV
f = gzip.open('data/intermediate/GSE211376_raw_counts.tsv.gz', 'rt')
header = f.readline().rstrip('\n').split('\t')
cells = np.array(header)
ncell = len(cells)

mmap = dict(zip(meta.index, ann))
cell_lab = np.array([mmap.get(c, 'NA') for c in cells])
masks = {k: (cell_lab == v) for k, v in subtypes.items()}

tot = np.zeros(ncell)
gene_rows = {}
for line in f:
    i = line.find('\t')
    g = line[:i]
    vals = np.fromstring(line[i+1:], sep='\t')
    if len(vals) != ncell:
        continue
    tot += vals
    if g in gene_set:
        gene_rows[g] = vals
f.close()

sf = tot / 1e4
sf[sf == 0] = 1e-9

rows = []
for g in genes41:
    if g not in gene_rows:
        rows.append({'gene': g, **{k: np.nan for k in subtypes}})
        continue
    ln = np.log1p(gene_rows[g] / sf)
    rows.append({'gene': g, **{k: float(ln[masks[k]].mean()) for k in subtypes}})
expr = pd.DataFrame(rows).set_index('gene')
expr.to_csv('gse211376_subtype_expr_41genes.csv')

# Compute AC-like rank among 4 subtypes
def ac_rank(row):
    if row.isna().any():
        return np.nan
    order = row.sort_values(ascending=False).index.tolist()
    return order.index('AC') + 1

expr['gse211376_AC'] = expr['AC']
expr['gse211376_AC_rank'] = expr.apply(lambda r: ac_rank(r[['AC', 'MES', 'OPC', 'NPC']]), axis=1)

m = panel.merge(expr[['gse211376_AC', 'gse211376_AC_rank']], left_on='gene', right_index=True, how='left')
valid = m.dropna(subset=['gse211376_AC_rank', 'scGBM_AC_rank']).copy()
valid['gse211376_AC_rank'] = valid['gse211376_AC_rank'].astype(int)
valid['scGBM_AC_rank'] = valid['scGBM_AC_rank'].astype(int)

valid['rank_concordant'] = valid['gse211376_AC_rank'] == valid['scGBM_AC_rank']
valid['both_AC_top2'] = (valid['gse211376_AC_rank'] <= 2) & (valid['scGBM_AC_rank'] <= 2)

valid.to_csv('fig6_concordance_gse211376_vs_scgbm.csv', index=False)
