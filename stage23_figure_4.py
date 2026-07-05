#!/usr/bin/env python3
"""
Stage 23: Figure 4
Source artifact: Figure4.png (version: f11c809f-28e5-4edf-ac13-24f949003dfd)

GBM AC-like Masquerade Analysis Pipeline
Malignant cells masquerade as astrocytes in glioblastoma and confound
immune-microenvironment analysis — a multi-cohort study.

This script was extracted from the interactive analysis session.
Adjust file paths as needed for your environment.
"""

import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.path import Path
from matplotlib import gridspec
import numpy as np
import pandas as pd
import json

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


def set_frame(ax):
    for sp in ['top', 'right']:
        ax.spines[sp].set_visible(False)


apply_figure_style(sizes=(8, 7, 6))

comm = json.load(open('data/intermediate/communication_analysis_results.json'))

ts = comm['tam_scores']

subt = ['AC', 'MES1', 'OPC', 'NPC']
subtcol = {'AC': '#C0392B', 'MES1': '#E67E22', 'OPC': '#7FB77E', 'NPC': '#5B8AC4'}
lcolors = {'Ligand': '#C0392B', 'Receptor': '#E67E22', 'TF': '#8E44AD', 'Target': '#34495e'}

tot = {s: sum(ts[lr][s] for lr in ts) for s in subt}
mx = max(tot.values())

angs = np.linspace(120, 240, 4)
srcpos = {}
mac = (0.9, 0.0)

pw = ['TGF-β', 'CSF1', 'Chemokine']
pwvals = {s: [ts['TGFB2-TGFBR1'][s] + ts['TGFB2-TGFBR2'][s], ts['CSF1-CSF1R'][s], ts['CCL2-CCR2'][s] + ts['CXCL12-CXCR4'][s]] for s in subt}
xx = np.arange(3)
bw = 0.2

edges = [('Ligand', 'TGFB2', 'Receptor', 'TGFBR'), ('Ligand', 'SPP1', 'Receptor', 'CD44'), ('Ligand', 'CSF1', 'Receptor', 'CSF1R'),
         ('Receptor', 'TGFBR', 'TF', 'SMAD3'), ('Receptor', 'CD44', 'TF', 'NF-κB'), ('Receptor', 'CSF1R', 'TF', 'STAT3'),
         ('TF', 'SMAD3', 'Target', 'IL10'), ('TF', 'NF-κB', 'Target', 'ARG1'), ('TF', 'STAT3', 'Target', 'IL10'),
         ('TF', 'STAT3', 'Target', 'VEGFA'), ('TF', 'NF-κB', 'Target', 'VEGFA')]

plt.close('all')
fig = plt.figure(figsize=(7.6, 3.2))
gs = gridspec.GridSpec(1, 3, figure=fig, width_ratios=[1, 1, 1.25], wspace=0.42, left=0.04, right=0.98, top=0.86, bottom=0.14)

axa = fig.add_subplot(gs[0, 0])
axa.axis('off')
axa.set_xlim(-1.3, 1.7)
axa.set_ylim(-1.3, 1.3)
for s, a in zip(subt, angs):
    x, y = 0.9 * np.cos(np.radians(a)), 0.9 * np.sin(np.radians(a))
    srcpos[s] = (x, y)
    axa.scatter([x], [y], s=120, color=subtcol[s], zorder=3)
    axa.text(x * 1.35, y * 1.35, s, ha='center', va='center', fontsize=6, color=subtcol[s], fontweight='bold')
axa.scatter([mac[0]], [mac[1]], s=200, color='#34495e', zorder=3)
axa.text(mac[0], mac[1] + 0.32, 'Macrophage', ha='center', va='bottom', fontsize=6, fontweight='bold')
for s in subt:
    x0, y0 = srcpos[s]
    lw = 0.5 + 5 * tot[s] / mx
    p = mpatches.PathPatch(Path([(x0, y0), (0, y0 * 0.4), (mac[0], mac[1])], [Path.MOVETO, Path.CURVE3, Path.CURVE3]),
                           fill=False, edgecolor=subtcol[s], lw=lw, alpha=0.6, zorder=1)
    axa.add_patch(p)
axa.set_title('Malignant→macrophage\nsignaling strength', fontsize=7.5)

axb = fig.add_subplot(gs[0, 1])
for i, s in enumerate(subt):
    axb.bar(xx + (i - 1.5) * bw, pwvals[s], width=bw, color=subtcol[s], label=s)
axb.set_xticks(xx)
axb.set_xticklabels(pw, fontsize=6)
axb.set_ylabel('Communication score (→ macrophage)')
axb.legend(fontsize=5.5, frameon=False, ncol=2, columnspacing=0.8, handletextpad=0.3, loc='upper right')
axb.set_title('AC-like leads TGF-β and CSF1', fontsize=7.5)
for sp in ['top', 'right']:
    axb.spines[sp].set_visible(False)

axc = fig.add_subplot(gs[0, 2])
axc.axis('off')
axc.set_xlim(0, 4.5)
axc.set_ylim(0, 6)
layers = {'Ligand': (0.5, ['TGFB2', 'SPP1', 'CSF1']), 'Receptor': (1.8, ['TGFBR', 'CD44', 'CSF1R']),
          'TF': (3.0, ['SMAD3', 'NF-κB', 'STAT3']), 'Target': (4.2, ['IL10', 'ARG1', 'VEGFA'])}
pos = {}
for lname, (lx, nodes) in layers.items():
    ys = np.linspace(5.0, 1.0, len(nodes))
    for n, ny in zip(nodes, ys):
        pos[(lname, n)] = (lx, ny)
        axc.scatter([lx], [ny], s=95, color=lcolors[lname], zorder=3, alpha=0.9)
        axc.text(lx, ny - 0.35, n, ha='center', va='top', fontsize=4.8, color=lcolors[lname], fontweight='bold', zorder=4)
    axc.text(lx, 5.6, lname, ha='center', fontsize=5.5, fontweight='bold', color=lcolors[lname])
for l1, n1, l2, n2 in edges:
    x0, y0 = pos[(l1, n1)]
    x1, y1 = pos[(l2, n2)]
    axc.annotate('', xy=(x1 - 0.14, y1), xytext=(x0 + 0.14, y0),
                 arrowprops=dict(arrowstyle='-', lw=0.5, color='#bbb', alpha=0.8), zorder=1)
axc.set_title('AC-like → macrophage signaling model\n(literature-informed)', fontsize=7.5)

panel_letter(axa, 'a')
panel_letter(axb, 'b')
panel_letter(axc, 'c')

fig.savefig('Figure4.png', dpi=300, bbox_inches='tight')
fig.savefig('Figure4.pdf', bbox_inches='tight')
print("saved Figure4.png")
