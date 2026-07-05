#!/usr/bin/env python3
"""
Stage 20: Figure 1
Source artifact: Figure1.png (version: 456f1503-b6f6-416d-a021-727570d83dcf)

GBM AC-like Masquerade Analysis Pipeline
Malignant cells masquerade as astrocytes in glioblastoma and confound
immune-microenvironment analysis — a multi-cohort study.

This script was extracted from the interactive analysis session.
Adjust file paths as needed for your environment.
"""

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

mpl.rcParams['font.family'] = 'DejaVu Sans'
MAL = '#C0392B'
AST = '#27AE60'
GREY = '#95A5A6'
DK = '#2C3E50'
LT = '#EBF5FB'
GRN = '#27AE60'

fig = plt.figure(figsize=(7.4, 6.9))
gs = fig.add_gridspec(3, 2, height_ratios=[0.92, 0.95, 1.15], width_ratios=[1, 1.1],
                      hspace=0.42, wspace=0.24, left=0.055, right=0.985, top=0.955, bottom=0.06)

# a
axa = fig.add_subplot(gs[0, :])
axa.axis('off')
axa.set_xlim(0, 10)
axa.set_ylim(0, 2.5)
axa.text(0, 2.42, 'a', fontsize=13, fontweight='bold', va='top')
axa.text(0.55, 2.4, 'The problem: AC-like malignant cells masquerade as astrocytes', fontsize=8.3, fontweight='bold', va='top')

def cell(ax, x, y, w, h, fc, ec, txt, tc='white', fs=6.6):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.04,rounding_size=0.12", fc=fc, ec=ec, lw=1.3))
    ax.text(x + w/2, y + h/2, txt, ha='center', va='center', fontsize=fs, color=tc, fontweight='bold')

cell(axa, 0.6, 0.85, 2.5, 1.05, MAL, '#7B241C', 'AC-like\nmalignant cell')
axa.text(1.85, 0.68, 'S100B⁺ SLC1A3⁺ ALDOC⁺\n+ chr7 gain / chr10 loss', ha='center', va='top', fontsize=5.9, color=DK)
axa.add_patch(FancyArrowPatch((3.25, 1.375), (4.35, 1.375), arrowstyle='-|>', mutation_scale=13, lw=1.6, color=DK))
axa.text(3.8, 1.6, 'standard\nmarker clustering', ha='center', va='bottom', fontsize=5.6, color=DK)
cell(axa, 4.5, 0.85, 2.5, 1.05, AST, '#1E8449', '"Astrocyte"\ncluster')
axa.text(5.75, 0.68, 'assumed non-malignant\nsupport cells', ha='center', va='top', fontsize=5.9, color=DK)
axa.add_patch(FancyArrowPatch((7.15, 1.375), (8.15, 1.375), arrowstyle='-|>', mutation_scale=13, lw=1.6, color=MAL))
axa.text(8.6, 1.375, 'tumour cells\nmis-analysed\nas astrocytes', ha='center', va='center', fontsize=6.0, color=MAL, fontweight='bold')

# b
axb = fig.add_subplot(gs[1, :])
axb.axis('off')
axb.set_xlim(0, 10)
axb.set_ylim(0, 2.6)
axb.text(0, 2.52, 'b', fontsize=13, fontweight='bold', va='top')
axb.text(0.55, 2.5, 'Non-circular quantification (applied identically to all four cohorts)', fontsize=8.3, fontweight='bold', va='top')
steps = [('Public\nGBM cohort', '#D6EAF8', '#2E86C1'),
         ('Define "astrocyte"\ncluster from\nmarkers ALONE', '#D5F5E3', '#27AE60'),
         ('Score each cell\nfor +7/−10 vs\nimmune reference', '#FADBD8', '#C0392B'),
         ('Report % CNV+\n(no author\nlabels used)', '#E8DAEF', '#7D3C98')]
xs = [0.35, 2.75, 5.35, 7.85]
w = 2.1
y = 0.95
h = 1.2
for i, ((t, fc, ec), x) in enumerate(zip(steps, xs)):
    axb.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.04,rounding_size=0.1", fc=fc, ec=ec, lw=1.4))
    axb.text(x + w/2, y + h/2, t, ha='center', va='center', fontsize=6.3, color=DK, fontweight='bold')
    if i < 3:
        axb.add_patch(FancyArrowPatch((x + w + 0.02, y + h/2), (xs[i+1] - 0.02, y + h/2),
                                      arrowstyle='-|>', mutation_scale=11, lw=1.5, color=DK))
axb.text(5.0, 0.62,
         'Because the cluster is defined by markers and malignancy scored independently of the original\nlabels, the contamination estimate cannot be circular.',
         ha='center', va='top', fontsize=5.9, color=DK, style='italic')

# c
axc = fig.add_subplot(gs[2, 0])
coh = ['GSE211376', 'GSE162631\n+273274', 'GSE276177\n+286419', 'GSE174554']
rate = [72.5, 88.2, 71.0, 66.0]
axc.bar(range(4), rate, color=[GRN, MAL, MAL, MAL], zorder=3, width=0.68)
axc.axhline(5, ls='--', lw=1, color=GREY, zorder=2)
axc.text(3.7, 8.5, 'ref. floor ~5%', fontsize=6, color='#5D6D7E', ha='right', va='bottom',
         bbox=dict(fc='white', ec='#5D6D7E', lw=0.5, pad=2, boxstyle='round,pad=0.3'))
for i, v in enumerate(rate):
    axc.text(i, v + 2, f'{v:.0f}', ha='center', va='bottom', fontsize=7, fontweight='bold', color='#222')
axc.set_xticks(range(4))
axc.set_xticklabels(coh, fontsize=5.4)
axc.set_ylim(0, 102)
axc.set_ylabel('"Astrocyte" cluster CNV+ (%)', fontsize=7.0)
axc.set_title('66–88% of "astrocytes" are malignant', fontsize=7.4, loc='left')
for s in ['top', 'right']:
    axc.spines[s].set_visible(False)
axc.text(-0.20, 1.14, 'c', transform=axc.transAxes, fontsize=13, fontweight='bold', va='top')

# d
axd = fig.add_subplot(gs[2, 1])
axd.axis('off')
axd.set_xlim(-0.5, 10.5)
axd.set_ylim(0, 5.5)
axd.text(-0.5, 5.4, 'd', fontsize=13, fontweight='bold', va='top')
axd.text(5.0, 5.38, 'Four public GBM cohorts', fontsize=7.4, fontweight='bold', va='top', ha='center')

cols = ['GEO accession', 'Cells/nuclei', 'Pts', 'Platform']
cw = [3.4, 2.4, 1.0, 3.2]
cx = [sum(cw[:i]) for i in range(len(cw))]
tw = sum(cw)

rows = [['GSE211376 (disc.)', '39,355', '11', 'snRNA-seq'],
        ['GSE162631+273274', '135,535', '7', 'scRNA-seq'],
        ['GSE276177+286419', '68,920', '6', 'snRNA-seq'],
        ['GSE174554', '228,319', '80', 'snRNA-seq']]

top = 4.60
rh = 0.82
hdr_h = 0.52
x0 = 0.0

axd.add_patch(plt.Rectangle((x0, top), tw, hdr_h, fc=DK, ec='none'))
for c, x in enumerate(cx):
    axd.text(x0 + x + 0.15, top + hdr_h/2, cols[c],
             ha='left', va='center', fontsize=5.6, color='white', fontweight='bold')

for r, row in enumerate(rows):
    yy = top - (r + 1) * rh
    axd.add_patch(plt.Rectangle((x0, yy), tw, rh,
                                 fc=LT if r % 2 == 0 else 'white', ec='#D5DBDB', lw=0.6))
    for c, x in enumerate(cx):
        axd.text(x0 + x + 0.15, yy + rh/2, row[c],
                 ha='left', va='center', fontsize=5.6, color=DK,
                 fontweight='bold' if c == 0 else 'normal')

axd.text(x0 + tw/2, top - 4 * rh - 0.22,
         '>470,000 cells & nuclei · IDH-wildtype by design',
         ha='center', va='top', fontsize=5.8, color=DK, style='italic')

fig.savefig('Figure1.png', dpi=300, bbox_inches='tight', facecolor='white')
fig.savefig('Figure1.pdf', bbox_inches='tight', facecolor='white')
