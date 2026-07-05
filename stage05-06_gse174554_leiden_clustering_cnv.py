#!/usr/bin/env python3
"""
Stage 05-06: GSE174554 Leiden clustering + CNV
Source artifact: gse174554_leiden_cnv.json (version: 67eaaa9b-e974-4775-8ee5-753be257f971)

GBM AC-like Masquerade Analysis Pipeline
Malignant cells masquerade as astrocytes in glioblastoma and confound
immune-microenvironment analysis — a multi-cohort study.

This script was extracted from the interactive analysis session.
Adjust file paths as needed for your environment.
"""

import gzip, glob, os, gc, numpy as np, scipy.sparse as sp
from scipy.io import mmread
import pandas as pd, h5py, json

D='data/GSE174554'
SKIP={'GSM5319529_SF8963'}

# chr sets from scGBM
fs=h5py.File('data/external/scGBM/adata2.h5ad','r')
v=fs['var']; scg=np.array([g.decode() for g in v['_index'][:]])
ch=v['chromosome']; cc=[c.decode() for c in ch['categories'][:]]; cd=ch['codes'][:]
scchrom=np.array([cc[c] if 0<=c<len(cc) else 'NA' for c in cd]); fs.close()
chr7_set=set(scg[scchrom=='chr7']); chr10_set=set(scg[scchrom=='chr10'])

ref_feat=sorted(glob.glob(f'{D}/*_features.tsv.gz'))[0]
genes=np.array([l.split('\t')[0].strip() for l in gzip.open(ref_feat,'rt')])
is7=np.isin(genes,list(chr7_set)); is10=np.isin(genes,list(chr10_set)); n7=is7.sum(); n10=is10.sum()

# load cluster labels
npz=np.load('gse174554_cluster.npz', allow_pickle=True)
leiden=npz['leiden']; TN=npz['TN']; SAMP=npz['SAMP']

# rebuild per-cell chr7/chr10 in SAME order as clustering
mtx_files=sorted(glob.glob(f'{D}/*_matrix.mtx.gz'))
mtx_files=[m for m in mtx_files if not any(s in m for s in SKIP)]
c7=[]; c10=[]; imm_all=[]
imm_g=['PTPRC','CD68','CD14','AIF1','ITGAM','CSF1R']; gidx={g:i for i,g in enumerate(genes)}
imm_idx=[gidx[g] for g in imm_g if g in gidx]
for mf in mtx_files:
    base=mf.replace('_matrix.mtx.gz',''); bf=base+'_barcodes.tsv.gz'
    if not os.path.exists(bf): continue
    M=mmread(mf).tocsr().T.astype(np.float32)  # cells x genes
    if M.shape[1]!=len(genes): continue
    tot=np.asarray(M.sum(1)).ravel(); sf=tot/1e4; sf[sf==0]=1e-9
    c7.append(np.log1p(np.asarray(M[:,is7].sum(1)).ravel()/sf/n7))
    c10.append(np.log1p(np.asarray(M[:,is10].sum(1)).ravel()/sf/n10))
    imm_all.append(np.log1p(np.asarray(M[:,imm_idx].sum(1)).ravel()/sf/len(imm_idx)))
    del M; gc.collect()
c7=np.concatenate(c7); c10=np.concatenate(c10); imm=np.concatenate(imm_all)
assert len(c7)==len(leiden), f"align fail {len(c7)} vs {len(leiden)}"

# myeloid reference = immune-marker-high cells
mac = imm>np.quantile(imm,0.90)
ref7=c7[mac].mean(); ref10=c10[mac].mean()
cnv=(c7-ref7)-(c10-ref10)
strict=np.percentile(cnv[mac],95)

astro=leiden==5  # astrocyte cluster
print(f"strict thr={strict:.4f} | n_mac_ref={mac.sum()}")
print(f"Leiden astrocyte cluster (cl5) n={astro.sum()}")
print(f"  overall CNV+%: {100*(cnv[astro]>strict).mean():.1f}")
for t in ['Tumor','Normal','NA']:
    m=astro&(TN==t)
    if m.sum()>0: print(f"  {t:8}: n={m.sum():>6}, CNV+%={100*(cnv[m]>strict).mean():.1f}")
# also cluster 10 (secondary astro)
a2=leiden==10
print(f"\nsecondary astro (cl10) n={a2.sum()}: CNV+%={100*(cnv[a2]>strict).mean():.1f}")
# reference floors
print(f"\nmyeloid ref CNV+%: {100*(cnv[mac]>strict).mean():.1f}")
oli=leiden==1
print(f"oligodendrocyte (cl1) CNV+%: {100*(cnv[oli]>strict).mean():.1f}")

np.savez_compressed('gse174554_cnv_leiden.npz', cnv=cnv, leiden=leiden, TN=TN, strict=strict)
out={'method':'Leiden clustering (res=0.5, harmony batch-corrected), astrocyte=cluster5',
     'strict_threshold':float(strict),
     'astrocyte_cluster':{'cluster':5,'n':int(astro.sum()),'cnv_pos_pct':round(float(100*(cnv[astro]>strict).mean()),1)},
     'astrocyte_by_tissue':{t:{'n':int((astro&(TN==t)).sum()),'cnv_pos_pct':round(float(100*(cnv[astro&(TN==t)]>strict).mean()),1)} for t in ['Tumor','Normal'] if (astro&(TN==t)).sum()>0},
     'reference_FPR':{'myeloid':round(float(100*(cnv[mac]>strict).mean()),1),'oligodendrocyte':round(float(100*(cnv[oli]>strict).mean()),1)}}
json.dump(out, open('gse174554_leiden_cnv.json','w'), indent=2)
print("\nsaved json")
