#!/usr/bin/env python3
"""
Stage 04: GSE211376 marker-defined noncircular
Source artifact: gse211376_marker_noncircular.json (version: d44b8e7e-265f-48bb-8b5d-35d6d1e2c19a)

GBM AC-like Masquerade Analysis Pipeline
Malignant cells masquerade as astrocytes in glioblastoma and confound
immune-microenvironment analysis — a multi-cohort study.

This script was extracted from the interactive analysis session.
Adjust file paths as needed for your environment.
"""

import gzip, numpy as np, pandas as pd, h5py, json

# chr sets from scGBM
fs=h5py.File('data/external/scGBM/adata2.h5ad','r')
v=fs['var']; scg=np.array([g.decode() for g in v['_index'][:]])
ch=v['chromosome']; cc=[c.decode() for c in ch['categories'][:]]; cd=ch['codes'][:]
scchrom=np.array([cc[c] if 0<=c<len(cc) else 'NA' for c in cd]); fs.close()
chr7_set=set(scg[scchrom=='chr7']); chr10_set=set(scg[scchrom=='chr10'])

astro_g=['GFAP','AQP4','SLC1A3','S100B','ALDOC','CLU']
imm_g=['PTPRC','CD68','CD14','AIF1','ITGAM','CSF1R','C1QA','C1QB']
oli_g=['MOG','PLP1','MBP','MOBP']
want=set(astro_g)|set(imm_g)|set(oli_g)|chr7_set|chr10_set

# stream genes×cells TSV; first line = cell barcodes
f=gzip.open('data/intermediate/GSE211376_raw_counts.tsv.gz','rt')
header=f.readline().rstrip('\n').split('\t')
cells=np.array(header)  # first token may be empty or a gene-col header
ncell=len(cells)
# accumulators
c7=np.zeros(ncell); c10=np.zeros(ncell); tot=np.zeros(ncell)
astro=np.zeros(ncell); imm=np.zeros(ncell); oli=np.zeros(ncell)
n7=n10=na=ni=no=0
for line in f:
    i=line.find('\t')
    g=line[:i]
    if g in want or True:  # need total from all genes
        vals=np.fromstring(line[i+1:], sep='\t')
        if len(vals)!=ncell: continue
        tot+=vals
        if g in chr7_set: c7+=vals; n7+=1
        if g in chr10_set: c10+=vals; n10+=1
        if g in astro_g: astro+=vals; na+=1
        if g in imm_g: imm+=vals; ni+=1
        if g in oli_g: oli+=vals; no+=1
f.close()
sf=tot/1e4; sf[sf==0]=1e-9
c7n=np.log1p(c7/sf/max(n7,1)); c10n=np.log1p(c10/sf/max(n10,1))
astn=np.log1p(astro/sf/max(na,1)); immn=np.log1p(imm/sf/max(ni,1)); olin=np.log1p(oli/sf/max(no,1))
print(f"cells={ncell} chr7={n7} chr10={n10} astro_g={na} imm_g={ni} oli_g={no}")

# marker-defined astrocyte cluster (non-circular)
qa=np.quantile(astn,0.75)
astro_mask=(astn>qa)&(immn<=np.quantile(immn,0.5))&(olin<1.0)
myeloid=immn>np.quantile(immn,0.90)
oligo=(olin>np.quantile(olin,0.75))&(astn<=np.quantile(astn,0.5))
ref7=c7n[myeloid].mean(); ref10=c10n[myeloid].mean()
cnv=(c7n-ref7)-(c10n-ref10)
strict=np.percentile(cnv[myeloid],95)
def pct(m): return round(float(100*(cnv[m]>strict).mean()),1)
print(f"strict thr={strict:.4f}")
print(f"marker-defined astrocyte cluster n={int(astro_mask.sum())}: CNV+ {pct(astro_mask)}%")
print(f"myeloid ref n={int(myeloid.sum())}: {pct(myeloid)}%")
print(f"oligo ref n={int(oligo.sum())}: {pct(oligo)}%")

# also report by author annotation for the killer-punch (refined Astrocyte-only)
meta=pd.read_csv('data/intermediate/GSE211376_metadata.csv.gz', index_col=0)
# align meta to cells order
cell_ids=cells
if len(meta)==ncell:
    ann=meta['predicted.high_hierarchy'].values
    # try align by barcode
    mmap=dict(zip(meta.index, meta['predicted.high_hierarchy']))
    ann=np.array([mmap.get(c,'NA') for c in cell_ids])
    for lab in ['Astrocyte','AC-like']:
        m=ann==lab
        if m.sum()>0: print(f"author {lab} n={int(m.sum())}: CNV+ {pct(m)}%")
out={'dataset':'GSE211376','n_cells':int(ncell),'strict_threshold':float(strict),
     'marker_defined_astrocyte':{'n':int(astro_mask.sum()),'cnv_pos_pct':pct(astro_mask)},
     'reference_FPR':{'myeloid':pct(myeloid),'oligodendrocyte':pct(oligo)}}
try:
    out['author_astrocyte_only']={'n':int((ann=='Astrocyte').sum()),'cnv_pos_pct':pct(ann=='Astrocyte')}
    out['author_AClike']={'n':int((ann=='AC-like').sum()),'cnv_pos_pct':pct(ann=='AC-like')}
except: pass
json.dump(out, open('gse211376_marker_noncircular.json','w'), indent=2)
print("saved")
