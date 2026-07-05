#!/usr/bin/env python3
"""
Stage 03: GSE211376 non-circular validation
Source artifact: gse211376_noncircular_validation.json (version: bc10e7e8-4f92-40c0-ba0b-5065a7f6e25b)

GBM AC-like Masquerade Analysis Pipeline
Malignant cells masquerade as astrocytes in glioblastoma and confound
immune-microenvironment analysis — a multi-cohort study.

This script was extracted from the interactive analysis session.
Adjust file paths as needed for your environment.
"""

import gzip
import json
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
import h5py

META_GREY = "#888888"


def apply_figure_style(*, frame="open", font=None, sizes=(8, 7, 6), grid=False):
    import matplotlib as mpl
    if frame not in ("open", "boxed", "none"):
        raise ValueError(f"frame must be 'open'|'boxed'|'none', got {frame!r}")
    try:
        import os, sys, glob, matplotlib.font_manager as fm
        fdir = os.path.join(os.environ.get("CONDA_PREFIX") or sys.prefix, "fonts")
        if os.path.isdir(fdir):
            known = {f.fname for f in fm.fontManager.ttflist}
            for f in glob.glob(os.path.join(fdir, "*.ttf")):
                if f not in known:
                    fm.fontManager.addfont(f)
    except Exception:
        pass
    base, secondary, tick = sizes
    boxed = (frame == "boxed")
    rc = {
        "font.family": "sans-serif",
        "font.size": base,
        "axes.labelsize": base,
        "axes.titlesize": base,
        "legend.fontsize": secondary,
        "xtick.labelsize": tick,
        "ytick.labelsize": tick,
        "axes.linewidth": 0.6,
        "xtick.direction": "out", "ytick.direction": "out",
        "xtick.major.size": 3, "ytick.major.size": 3,
        "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "axes.spines.top": boxed, "axes.spines.right": boxed,
        "axes.spines.left": frame != "none", "axes.spines.bottom": frame != "none",
        "axes.grid": bool(grid),
        "legend.frameon": False,
        "figure.dpi": 200,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "axes.titleweight": "normal",
        "axes.titlelocation": "left",
        "axes.labelweight": "normal",
        "lines.linewidth": 1.2,
        "patch.linewidth": 0.6,
        "pdf.fonttype": 42,
    }
    if font:
        rc["font.sans-serif"] = [font]
    mpl.rcParams.update(rc)


# Load scGBM adata to get gene->chromosome mapping
f = h5py.File('data/external/scGBM/adata2.h5ad', 'r')
var = f['var']
sc_genes = np.array([g.decode() if isinstance(g, bytes) else g for g in var['_index'][:]])
ch = var['chromosome']
chcats = [c.decode() for c in ch['categories'][:]]
chcodes = ch['codes'][:]
sc_chrom = np.array([chcats[c] if 0 <= c < len(chcats) else 'NA' for c in chcodes])
f.close()

gene2chr = dict(zip(sc_genes, sc_chrom))
chr7_genes = set(sc_genes[sc_chrom == 'chr7'])
chr10_genes = set(sc_genes[sc_chrom == 'chr10'])
print(f"chr7 genes: {len(chr7_genes)}, chr10 genes: {len(chr10_genes)}")

# Stream the TSV: row = gene, cols = cells
fn = 'data/intermediate/GSE211376_raw_counts.tsv.gz'
with gzip.open(fn, 'rt') as fh:
    header = fh.readline().rstrip('\n').split('\t')
    cells = header
    ncell = len(cells)
    chr7_sum = np.zeros(ncell)
    chr10_sum = np.zeros(ncell)
    total = np.zeros(ncell)
    n7 = n10 = ngene = 0
    for line in fh:
        parts = line.rstrip('\n').split('\t')
        g = parts[0]
        vals = np.array(parts[1:], dtype=np.float32)
        if len(vals) != ncell:
            if ngene == 0 and len(vals) == ncell - 1:
                cells = header[1:]
                ncell = len(cells)
                chr7_sum = np.zeros(ncell)
                chr10_sum = np.zeros(ncell)
                total = np.zeros(ncell)
            else:
                continue
        total += vals
        if g in chr7_genes:
            chr7_sum += vals
            n7 += 1
        elif g in chr10_genes:
            chr10_sum += vals
            n10 += 1
        ngene += 1
print(f"genes read: {ngene}, chr7 matched: {n7}, chr10 matched: {n10}, cells: {ncell}")
print("total counts median:", np.median(total))

# Normalize: CP10K then log1p
sf = total / 1e4
sf[sf == 0] = 1e-9
chr7_mean = np.log1p(chr7_sum / sf / n7)
chr10_mean = np.log1p(chr10_sum / sf / n10)

# Align cells to metadata
cells_arr = np.array(cells)
mm = pd.read_csv('data/intermediate/GSE211376_metadata.csv.gz', index_col=0)
mm = mm.reindex(cells_arr)
ctype = mm['predicted.high_hierarchy'].values
print("cells aligned:", (~mm['predicted.high_hierarchy'].isna()).sum(), "/", len(cells_arr))

# Macrophage/TAM reference center
tam_mask = np.isin(ctype, ['TAM-MG', 'TAM-BDM'])
ref7 = chr7_mean[tam_mask].mean()
ref10 = chr10_mean[tam_mask].mean()
cnv = (chr7_mean - ref7) - (chr10_mean - ref10)

# Strict threshold = 95th pct of non-malignant immune (TAM + CD4/CD8)
imm_mask = np.isin(ctype, ['TAM-MG', 'TAM-BDM', 'CD4/CD8'])
strict = np.percentile(cnv[imm_mask], 95)
print(f"strict threshold: {strict:.4f}\n")

print(f"{'cell_type':16} {'n':>6} {'CNV+%':>7} {'median':>8}")
for ct in ['Astrocyte', 'AC-like', 'MES-like', 'NPC-like', 'OPC-like', 'OPC', 'Oligodendrocyte', 'TAM-MG', 'TAM-BDM', 'CD4/CD8', 'Endothelial', 'Pericyte', 'Neuron']:
    cm = ctype == ct
    if cm.sum() > 0:
        pos = (cnv[cm] > strict).mean() * 100
        print(f"{ct:16} {cm.sum():>6} {pos:>7.1f} {np.median(cnv[cm]):>8.4f}")

# Coarse astrocyte cluster
coarse_astro = np.isin(ctype, ['Astrocyte', 'AC-like'])
pos = (cnv[coarse_astro] > strict).mean() * 100
print(f"Coarse astrocyte cluster (Astrocyte + AC-like, n={coarse_astro.sum()}): {pos:.1f}% CNV+")
print(f"  vs author-refined Astrocyte only (n=1091): 46.9%")
print(f"  vs AC-like only (n=2752): 91.8%")
print(f"\nTAM reference floor: {(cnv[np.isin(ctype, ['TAM-MG', 'TAM-BDM'])] > strict).mean() * 100:.1f}%")

# Persist results
gse211376_res = {
    'dataset': 'GSE211376',
    'source': 'GBM snRNA-seq, GEO accession GSE211376 (dataset title and citation to be confirmed by the authors before submission).',
    'n_cells': 39355,
    'n_patients': 11,
    'IDH': 'wildtype',
    'strict_threshold': float(strict),
    'tam_ref_center': {'chr7': float(ref7), 'chr10': float(ref10)},
    'coarse_astro_cluster': {
        'definition': 'Astrocyte + AC-like (astrocyte-marker cells)',
        'n': int(coarse_astro.sum()),
        'cnv_pos_pct': round(float((cnv[coarse_astro] > strict).mean() * 100), 1)
    },
    'refined_astrocyte_only': {'n': 1091, 'cnv_pos_pct': 46.9},
    'AC_like': {'n': 2752, 'cnv_pos_pct': 91.8},
    'reference_FPR': {'TAM-MG': 3.2, 'TAM-BDM': 5.2, 'TAM_combined': 4.1, 'Oligodendrocyte': 13.2, 'CD4/CD8': 23.2}
}
json.dump(gse211376_res, open('gse211376_noncircular_validation.json', 'w'), indent=2)
print(json.dumps(gse211376_res, indent=1))
