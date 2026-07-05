#!/usr/bin/env python3
"""
Stage 12: GSE211376 communication decontamination
Source artifact: gse211376_communication_decontam.json (version: 3712d2c7-ac44-476f-86aa-6db125437e2f)

GBM AC-like Masquerade Analysis Pipeline
Malignant cells masquerade as astrocytes in glioblastoma and confound
immune-microenvironment analysis — a multi-cohort study.

This script was extracted from the interactive analysis session.
Adjust file paths as needed for your environment.
"""

import gzip, numpy as np, pandas as pd, h5py, json

# chr sets from scGBM
fs = h5py.File('data/external/scGBM/adata2.h5ad', 'r')
v = fs['var']
scg = np.array([g.decode() for g in v['_index'][:]])
ch = v['chromosome']
cc = [c.decode() for c in ch['categories'][:]]
cd = ch['codes'][:]
scchrom = np.array([cc[c] if 0 <= c < len(cc) else 'NA' for c in cd])
fs.close()
chr7 = set(scg[scchrom == 'chr7'])
chr10 = set(scg[scchrom == 'chr10'])

lig_rec = ['TGFB2', 'TGFBR1', 'TGFBR2', 'TGFB1', 'ANXA1', 'FPR1', 'CSF1', 'CSF1R']
astro_g = ['GFAP', 'AQP4', 'SLC1A3', 'S100B', 'ALDOC', 'CLU']
imm_g = ['PTPRC', 'CD68', 'CD14', 'AIF1', 'ITGAM', 'CSF1R', 'C1QA', 'C1QB']
oli_g = ['MOG', 'PLP1', 'MBP', 'MOBP']
want = set(lig_rec) | set(astro_g) | set(imm_g) | set(oli_g) | chr7 | chr10

f = gzip.open('data/intermediate/GSE211376_raw_counts.tsv.gz', 'rt')
cells = np.array(f.readline().rstrip('\n').split('\t'))
n = len(cells)
c7 = np.zeros(n); c10 = np.zeros(n); tot = np.zeros(n)
astro = np.zeros(n); imm = np.zeros(n); oli = np.zeros(n)
genev = {}
n7 = n10 = na = ni = no = 0
for line in f:
    i = line.find('\t')
    g = line[:i]
    vals = np.fromstring(line[i+1:], sep='\t')
    if len(vals) != n:
        continue
    tot += vals
    if g in chr7:
        c7 += vals; n7 += 1
    if g in chr10:
        c10 += vals; n10 += 1
    if g in astro_g:
        astro += vals; na += 1
    if g in imm_g:
        imm += vals; ni += 1
    if g in oli_g:
        oli += vals; no += 1
    if g in set(lig_rec):
        genev[g] = vals
f.close()

sf = tot / 1e4; sf[sf == 0] = 1e-9
c7n = np.log1p(c7 / sf / max(n7, 1))
c10n = np.log1p(c10 / sf / max(n10, 1))
astn = np.log1p(astro / sf / max(na, 1))
immn = np.log1p(imm / sf / max(ni, 1))
olin = np.log1p(oli / sf / max(no, 1))

# marker-defined astrocyte cluster (non-circular)
qa = np.quantile(astn, 0.75)
astro_mask = (astn > qa) & (immn <= np.quantile(immn, 0.5)) & (olin < 1.0)
myeloid = immn > np.quantile(immn, 0.90)
ref7 = c7n[myeloid].mean(); ref10 = c10n[myeloid].mean()
cnv = (c7n - ref7) - (c10n - ref10)
strict = np.percentile(cnv[myeloid], 95)

clean = astro_mask & (cnv <= strict)

def lnmean(g, mask):
    return float(np.log1p(genev[g] / sf)[mask].mean()) if g in genev else np.nan

def comm(lig, rec, sender):
    return lnmean(lig, sender) * lnmean(rec, myeloid)

pairs = {}
for lig, rec in [('TGFB2', 'TGFBR1'), ('TGFB2', 'TGFBR2'), ('TGFB1', 'TGFBR1'), ('ANXA1', 'FPR1'), ('CSF1', 'CSF1R')]:
    b = comm(lig, rec, astro_mask)
    a = comm(lig, rec, clean)
    pairs[f"{lig}->{rec}"] = {'before': round(b, 4), 'after': round(a, 4), 'pct_change': round(100 * (a - b) / b, 1) if b else 0.0}

print(f"astro cluster n={int(astro_mask.sum())} clean n={int(clean.sum())}")
for k, vv in pairs.items():
    print(f"  {k}: {vv['before']}→{vv['after']} ({vv['pct_change']:+}%)")

json.dump({'dataset': 'GSE211376', 'n_sender': int(astro_mask.sum()), 'n_clean': int(clean.sum()), 'pairs': pairs},
          open('gse211376_communication_decontam.json', 'w'), indent=2)
print("saved")
