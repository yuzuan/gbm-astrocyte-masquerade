#!/usr/bin/env python3
"""
Stage 22: Figure 3
Source artifact: Figure3.png (version: 76767a4d-1316-439a-b88b-6279c4365ecb)

GBM AC-like Masquerade Analysis Pipeline
Malignant cells masquerade as astrocytes in glioblastoma and confound
immune-microenvironment analysis — a multi-cohort study.

This script was extracted from the interactive analysis session.
Adjust file paths as needed for your environment.
"""

import matplotlib as mpl
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd

# skill:figure-style kernel.py (auto-injected on skill load)
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
    mpl.rcParams.update(rc)


def panel_letter(ax, letter, x=-0.12, y=1.05, fontsize=9):
    ax.text(x, y, letter, transform=ax.transAxes, fontsize=fontsize,
            fontweight='bold', va='top', ha='left')


# Load data
summ = pd.read_csv('data/intermediate/scgbm_ac_validation_summary.csv')

# Build heatmap matrix
cols = ['AC_mean', 'MES1_mean', 'OPC_mean', 'NPC_mean', 'Macro_mean']
collab = ['AC', 'MES', 'OPC', 'NPC', 'Macro']
M = summ.set_index('gene')[cols].values
Mn = M / (M.max(axis=1, keepdims=True) + 1e-9)  # row-normalized to max
genes = summ['gene'].values
cat = summ['category'].values

# TF activity: AC-like vs mean-other-malignant (recorded diffs)
tfs = [('RELA', 1.05), ('NFKB1', 0.96), ('STAT3', 0.88), ('MYC', 1.15), ('SP1', 1.25),
       ('HSF1', 1.11), ('SPI1', 0.72), ('OLIG1', -0.80), ('OLIG2', -0.65)]
tfs = sorted(tfs, key=lambda x: x[1])
tnames = [t[0] for t in tfs]
tvals = [t[1] for t in tfs]
colors = ['#5B8AC4' if v < 0 else '#C0392B' for v in tvals]
yy_tf = np.arange(len(tfs))[::-1]

# Category bounds
bounds = np.where(cat[:-1] != cat[1:])[0]
catnames = ['Astro.', 'Malig.', 'Immune\nevasion']
catcols = ['#E67E22', '#C0392B', '#8E44AD']
segs = [0] + list(bounds + 1) + [len(genes)]

plt.close('all')
apply_figure_style(sizes=(8, 7, 6))
fig = plt.figure(figsize=(7.6, 4.6))
gs = gridspec.GridSpec(1, 2, figure=fig, width_ratios=[1.15, 1], wspace=0.5,
                       left=0.16, right=0.94, top=0.86, bottom=0.12)
axa = fig.add_subplot(gs[0, 0])
im = axa.imshow(Mn, aspect='auto', cmap='RdYlBu_r', vmin=0, vmax=1)
axa.set_xticks(range(5))
axa.set_xticklabels(collab, fontsize=6.5)
axa.set_yticks(range(len(genes)))
axa.set_yticklabels([f'$\\it{{{g}}}$' for g in genes], fontsize=5.8)
for b in bounds:
    axa.axhline(b + 0.5, color='white', lw=1.5)
for i in range(len(segs) - 1):
    lo, hi = segs[i] - 0.4, segs[i + 1] - 1 + 0.4
    mid = (lo + hi) / 2
    axa.annotate('', xy=(-1.55, lo), xytext=(-1.55, hi), annotation_clip=False,
                 arrowprops=dict(arrowstyle='-', lw=2.2, color=catcols[i]))
    axa.text(-1.9, mid, catnames[i], fontsize=5.3, va='center', ha='center',
             rotation=90, color=catcols[i], fontweight='bold', clip_on=False)
for i in range(len(genes)):
    for j in range(5):
        axa.text(j, i, f'{M[i, j]:.1f}', ha='center', va='center', fontsize=4.3,
                 color='white' if Mn[i, j] > 0.6 else 'black')
axa.set_title('AC-like retains astrocyte + stem identity;\nimmune-evasion ligands are macrophage-dominated',
              fontsize=7.3)
cb = fig.colorbar(im, ax=axa, fraction=0.045, pad=0.04, aspect=28)
cb.set_label('Row-normalized mean expr.', fontsize=5.5)
cb.ax.tick_params(labelsize=5)
axa.tick_params(length=0)
axa.set_xlim(-2.2, 4.5)

axb = fig.add_subplot(gs[0, 1])
axb.barh(yy_tf, tvals, color=colors, height=0.68)
axb.axvline(0, color='k', lw=0.6)
axb.set_yticks(yy_tf)
axb.set_yticklabels([f'$\\it{{{t}}}$' for t in tnames], fontsize=6)
axb.set_xlabel('TF activity: AC-like − other malignant\n(CollecTRI / decoupler ULM)', fontsize=6.5)
axb.set_title('AC-like transcription-factor program', fontsize=7.5)
for sp in ['top', 'right']:
    axb.spines[sp].set_visible(False)
axb.set_xlim(-1.1, 1.6)
axb.set_ylim(-0.7, len(tfs) - 0.3)
axb.text(1.15, len(tfs) - 1.5, 'NF-κB / STAT3 ↑', fontsize=5, color='#C0392B', ha='right', va='center')
axb.text(-0.95, 0.4, 'lineage TF ↓', fontsize=5, color='#5B8AC4', ha='left', va='center')

panel_letter(axa, 'a')
panel_letter(axb, 'b')
fig.savefig('Figure3.png', dpi=300, bbox_inches='tight')
fig.savefig('Figure3.pdf', bbox_inches='tight')
print("Fig3 saved")
