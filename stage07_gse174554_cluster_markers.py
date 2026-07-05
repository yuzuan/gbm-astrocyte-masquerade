#!/usr/bin/env python3
"""
Stage 07: GSE174554 cluster markers
Source artifact: gse174554_cluster_markers.csv (version: 3039f06e-0d97-4866-a499-60ef9456eee2)

GBM AC-like Masquerade Analysis Pipeline
Malignant cells masquerade as astrocytes in glioblastoma and confound
immune-microenvironment analysis — a multi-cohort study.

This script was extracted from the interactive analysis session.
Adjust file paths as needed for your environment.
"""

import gzip, glob, os, gc, numpy as np, scipy.sparse as sp
from scipy.io import mmread
import pandas as pd, h5py, time

t0 = time.time()
def log(m): print(f"[{time.time()-t0:6.1f}s] {m}", flush=True)

D = 'data/GSE174554'
SKIP = {'GSM5319529_SF8963'}  # corrupt mtx

# gene list (identical across samples)
ref_feat = sorted(glob.glob(f'{D}/*_features.tsv.gz'))[0]
genes = np.array([l.split('\t')[0].strip() for l in gzip.open(ref_feat, 'rt')])
ngenes = len(genes)
log(f"genes={ngenes}")

# metadata Tumor/Normal
meta = {}
with gzip.open('data/intermediate/GSE174554_metadata.txt.gz', 'rt') as fh:
    next(fh)
    for line in fh:
        p = line.split()
        if len(p) >= 2:
            meta[p[0]] = p[1]
log(f"meta cells={len(meta)}")

# sample list
mtx_files = sorted(glob.glob(f'{D}/*_matrix.mtx.gz'))
mtx_files = [m for m in mtx_files if not any(s in m for s in SKIP)]
log(f"samples={len(mtx_files)}")

# Pass 1: load all, build combined CSR (cells x genes), collect TN + sample
mats = []; TN = []; SAMP = []
for mf in mtx_files:
    base = mf.replace('_matrix.mtx.gz', '')
    bf = base + '_barcodes.tsv.gz'
    if not os.path.exists(bf):
        continue
    bcs = [l.strip() for l in gzip.open(bf, 'rt')]
    M = mmread(mf).tocsr().T.astype(np.float32)  # cells x genes
    if M.shape[1] != ngenes:
        log(f"SKIP shape {os.path.basename(base)} {M.shape}")
        continue
    mats.append(M)
    pref = os.path.basename(base)
    sfid = next((p for p in pref.split('_') if p.startswith('SF')), pref)
    TN.extend(meta.get(f'{sfid}_{bc.split("-")[0]}', 'NA') for bc in bcs)
    SAMP.extend([sfid] * len(bcs))
    del M; gc.collect()

X = sp.vstack(mats).tocsr()
del mats; gc.collect()
TN = np.array(TN)
SAMP = np.array(SAMP)
log(f"combined X={X.shape}")

# normalize CP10K + log1p
tot = np.asarray(X.sum(1)).ravel()
tot[tot == 0] = 1
X = X.multiply(1e4 / tot[:, None]).tocsr()
X.data = np.log1p(X.data)
log("normalized")

# HVG: pick top 2000 by variance on log-norm
mean = np.asarray(X.mean(0)).ravel()
sqmean = np.asarray(X.multiply(X).mean(0)).ravel()
var = sqmean - mean ** 2
disp = var / (mean + 1e-8)
hvg = np.argsort(disp)[-2000:]
hvg = np.sort(hvg)
Xh = X[:, hvg].toarray()
log(f"HVG matrix {Xh.shape}")

# scale (clip)
Xh -= Xh.mean(0)
sd = Xh.std(0)
sd[sd == 0] = 1
Xh /= sd
np.clip(Xh, -10, 10, out=Xh)

# PCA
from sklearn.decomposition import TruncatedSVD
pca = TruncatedSVD(n_components=50, random_state=0).fit_transform(Xh)
del Xh; gc.collect()
log("PCA done")

# harmony batch correction on sample
import harmonypy
ho = harmonypy.run_harmony(pca, pd.DataFrame({'sample': SAMP}), ['sample'], max_iter_harmony=10)
Z = ho.Z_corr.T
if Z.shape[0] != pca.shape[0]:
    Z = ho.Z_corr
    if Z.shape[0] != pca.shape[0]:
        Z = Z.T
assert Z.shape[0] == pca.shape[0] and Z.shape[1] == 50, f'Z bad {Z.shape}'
log(f"harmony done Z={Z.shape}")

# kNN graph
from sklearn.neighbors import NearestNeighbors
nn = NearestNeighbors(n_neighbors=15, metric='euclidean', n_jobs=-1).fit(Z)
knn = nn.kneighbors_graph(Z, mode='connectivity')
log("kNN done")

# Leiden
import leidenalg, igraph as ig
src, dst = knn.nonzero()
g = ig.Graph(edges=list(zip(src.tolist(), dst.tolist())), directed=False)
g.simplify()
part = leidenalg.find_partition(g, leidenalg.RBConfigurationVertexPartition, resolution_parameter=0.5, seed=0)
leiden = np.array(part.membership)
log(f"leiden done, {leiden.max()+1} clusters")

# save cluster array
np.savez_compressed('gse174554_cluster.npz', leiden=leiden, TN=TN, SAMP=SAMP, hvg=hvg, Z=Z.astype(np.float32))

# compute marker means per cluster
markers = {
    'astro': ['GFAP', 'AQP4', 'SLC1A3', 'S100B', 'ALDOC', 'CLU'],
    'imm': ['PTPRC', 'CD68', 'CD14', 'AIF1', 'ITGAM', 'CSF1R'],
    'oligo': ['MOG', 'PLP1', 'MBP', 'MOBP'],
    'tumor': ['EGFR', 'SOX2', 'OLIG1', 'OLIG2'],
    'tcell': ['CD3D', 'CD3E', 'CD2'],
    'endo': ['PECAM1', 'VWF', 'CLDN5'],
    'neuron': ['RBFOX3', 'SYT1', 'SNAP25']
}
gidx = {g: i for i, g in enumerate(genes)}
rows = []
for cl in range(leiden.max() + 1):
    m = leiden == cl
    row = {'cluster': cl, 'n': int(m.sum())}
    for name, gl in markers.items():
        idx = [gidx[g] for g in gl if g in gidx]
        row[name] = float(np.asarray(X[m][:, idx].mean()))
    rows.append(row)

pd.DataFrame(rows).to_csv('gse174554_cluster_markers.csv', index=False)
log("saved cluster markers")
print("DONE")
