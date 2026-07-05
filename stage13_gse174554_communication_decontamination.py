#!/usr/bin/env python3
"""
Stage 13: GSE174554 communication decontamination
Source artifact: gse174554_communication_decontam.json (version: 30c08be6-5b7b-4f10-b64e-cf37eacbbab6)

GBM AC-like Masquerade Analysis Pipeline
Malignant cells masquerade as astrocytes in glioblastoma and confound
immune-microenvironment analysis — a multi-cohort study.

This script was extracted from the interactive analysis session.
Adjust file paths as needed for your environment.
"""

import gzip, glob, os, gc, numpy as np, scipy.sparse as sp
from scipy.io import mmread
import pandas as pd, json

D='data/GSE174554'
SKIP={'GSM5319529_SF8963'}
ref_feat=sorted(glob.glob(f'{D}/*_features.tsv.gz'))[0]
genes=np.array([l.split('\t')[0].strip() for l in gzip.open(ref_feat,'rt')])
gidx={g:i for i,g in enumerate(genes)}

# communication genes (same 15 as in-house / GSE211376)
lig=['ANXA1','LGALS1','TGFB1','TGFB2','CSF1','CCL2','CD274']
rec=['FPR1','FPR3','PTPRC','TGFBR1','TGFBR2','CSF1R','CCR2','PDCD1']
comm=lig+rec
comm=[g for g in comm if g in gidx]
cidx=[gidx[g] for g in comm]
imm_g=[g for g in ['PTPRC','CD68','CD14','AIF1','ITGAM','CSF1R'] if g in gidx]
imm_idx=[gidx[g] for g in imm_g]

# cluster labels + cnv (same order as cnv174554_leiden.py build)
npz=np.load('data/intermediate/gse174554_cluster.npz', allow_pickle=True)
leiden=npz['leiden']
cnvz=np.load('data/intermediate/gse174554_cnv_leiden.npz', allow_pickle=True)
cnv=cnvz['cnv']; strict=float(cnvz['strict'])
assert len(leiden)==len(cnv)

# extract per-cell CP10K-lognorm expression for comm genes + immune score, SAME iteration order
mtx_files=sorted(glob.glob(f'{D}/*_matrix.mtx.gz'))
mtx_files=[m for m in mtx_files if not any(s in m for s in SKIP)]
expr_rows=[]; imm_rows=[]
for mf in mtx_files:
    base=mf.replace('_matrix.mtx.gz',''); bf=base+'_barcodes.tsv.gz'
    if not os.path.exists(bf): continue
    M=mmread(mf).tocsr().T.astype(np.float32)  # cells x genes
    if M.shape[1]!=len(genes): continue
    tot=np.asarray(M.sum(1)).ravel(); sf=tot/1e4; sf[sf==0]=1e-9
    sub=M[:,cidx].toarray()/sf[:,None]; sub=np.log1p(sub)
    expr_rows.append(sub.astype(np.float32))
    im=np.asarray(M[:,imm_idx].sum(1)).ravel()/sf/len(imm_idx); imm_rows.append(np.log1p(im).astype(np.float32))
    del M; gc.collect()
E=np.vstack(expr_rows); imm=np.concatenate(imm_rows)
assert E.shape[0]==len(leiden), f"align {E.shape[0]} vs {len(leiden)}"
Edf=pd.DataFrame(E, columns=comm)

astro=leiden==5                      # sender = astrocyte cluster
myeloid=imm>np.quantile(imm,0.90)    # receiver = myeloid (immune-high)
cnvpos=cnv>strict

# receptor means in myeloid (fixed receiver)
recmean={g:float(Edf.loc[myeloid,g].mean()) for g in rec if g in comm}

def score(pair_l, pair_r, astro_mask):
    lm=float(Edf.loc[astro_mask, pair_l].mean()) if pair_l in comm else 0.0
    rm=recmean.get(pair_r,0.0)
    return lm*rm, lm

pairs=[('ANXA1','FPR1'),('ANXA1','FPR3'),('LGALS1','PTPRC'),('TGFB1','TGFBR1'),
       ('TGFB2','TGFBR1'),('TGFB2','TGFBR2'),('CSF1','CSF1R'),('CCL2','CCR2'),('CD274','PDCD1')]
before_astro=astro
after_astro=astro & (~cnvpos)   # remove CNV+ (keep clean astrocytes)
n_before=int(before_astro.sum()); n_after=int(after_astro.sum())
print(f"astro cluster n={n_before}, CNV+ removed={n_before-n_after}, clean n={n_after}")

out={'dataset':'GSE174554','method':'Leiden astrocyte cluster5 sender; myeloid(immune-high) receiver; remove CNV+ from sender',
     'n_astro_before':n_before,'n_astro_after_clean':n_after,'strict_threshold':strict,'pairs':{}}
print(f"\n{'pair':18} {'before':>9} {'after':>9} {'%chg':>8}")
for l,r in pairs:
    sb,lb=score(l,r,before_astro); sa,la=score(l,r,after_astro)
    pct=(sa-sb)/sb*100 if sb>0 else 0.0
    out['pairs'][f'{l}->{r}']={'before':round(sb,4),'after':round(sa,4),'pct_change':round(pct,1),
                                'ligand_before':round(lb,4),'ligand_after':round(la,4)}
    print(f"{l+'->'+r:18} {sb:9.4f} {sa:9.4f} {pct:8.1f}")
json.dump(out, open('gse174554_communication_decontam.json','w'), indent=2)
print("\nsaved")
