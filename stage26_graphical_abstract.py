#!/usr/bin/env python3
"""
Stage 26: Graphical abstract
Source artifact: graphical_abstract.png (version: c4bd2723-5154-4c19-b00b-44ca82261072)

GBM AC-like Masquerade Analysis Pipeline
Malignant cells masquerade as astrocytes in glioblastoma and confound
immune-microenvironment analysis — a multi-cohort study.

This script was extracted from the interactive analysis session.
Adjust file paths as needed for your environment.
"""

import json
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

plt.close('all')
mpl.rcParams['font.family'] = 'DejaVu Sans'

# Load data
g211 = json.load(open('data/intermediate/gse211376_marker_noncircular.json'))
g174 = json.load(open('data/intermediate/gse174554_leiden_cnv.json'))
d211 = json.load(open('data/intermediate/gse211376_communication_decontam.json'))
d174 = json.load(open('data/intermediate/gse174554_communication_decontam.json'))

MAL = '#C0392B'; GRN = '#27AE60'; GREY = '#95A5A6'; DK = '#2C3E50'
BLU = '#2E86C1'; PURP = '#7D3C98'; PERI = '#5DADE2'

fig = plt.figure(figsize=(11, 4.2))
gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.0, 1.0],
                      wspace=0.32, left=0.04, right=0.97, top=0.88, bottom=0.14)

# === Panel a: Four-cohort CNV+ bar chart ===
axa = fig.add_subplot(gs[0, 0])
coh_labels = ['GSE211376\n(disc.)', 'GSE162631\n+273274', 'GSE276177\n+286419', 'GSE174554']
rates = [72.5, 88.2, 71.0, 66.0]
colors = [GRN, MAL, MAL, MAL]
bars = axa.bar(range(4), rates, color=colors, width=0.65, zorder=3)
axa.axhline(5, ls='--', lw=0.8, color=GREY, zorder=2)
axa.text(3.5, 9, 'ref. floor\n~5%', fontsize=5.5, color='#5D6D7E', ha='right', va='bottom',
         bbox=dict(fc='white', ec='#5D6D7E', lw=0.4, pad=1.5, boxstyle='round,pad=0.2'))
for i, v in enumerate(rates):
    axa.text(i, v + 2, f'{v:.0f}%', ha='center', va='bottom', fontsize=7.5, fontweight='bold', color='#222')
axa.set_xticks(range(4))
axa.set_xticklabels(coh_labels, fontsize=5.8)
axa.set_ylim(0, 105)
axa.set_ylabel('"Astrocyte" cluster\nCNV+ (%)', fontsize=7.5)
axa.set_title('66–88% of "astrocytes"\nare malignant', fontsize=8.5, fontweight='bold', pad=8)
for s in ['top', 'right']:
    axa.spines[s].set_visible(False)
axa.text(-0.15, 1.15, 'a', transform=axa.transAxes, fontsize=14, fontweight='bold', va='top')

# === Panel b: TGF-β2 decontamination (two cohorts) ===
axb = fig.add_subplot(gs[0, 1])
before = [d211['pairs']['TGFB2->TGFBR2']['before'],
          d174['pairs']['TGFB2->TGFBR2']['before']]
after = [d211['pairs']['TGFB2->TGFBR2']['after'],
         d174['pairs']['TGFB2->TGFBR2']['after']]
x = np.arange(2)
w = 0.32
b1 = axb.bar(x - w/2, before, width=w, color=GREY, label='Full cluster', zorder=3)
b2 = axb.bar(x + w/2, after, width=w, color=PURP, label='After removing\nmalignant cells', zorder=3)
for i in range(2):
    pct = (after[i] - before[i]) / before[i] * 100
    axb.text(x[i] + w/2, after[i] + 0.025, f'{pct:.0f}%', ha='center', va='bottom',
             fontsize=7.5, fontweight='bold', color=PURP)
axb.set_xticks(x)
axb.set_xticklabels(['GSE211376\n(discovery)', 'GSE174554\n(validation)'], fontsize=6.5)
axb.set_ylim(0, 1.15)
axb.set_ylabel('TGFB2→TGFBR2\ncommunication score', fontsize=7.5)
axb.set_title('TGF-β2 signal drops after\nremoving malignant cells', fontsize=8.5, fontweight='bold', pad=8)
axb.legend(fontsize=5.8, frameon=False, loc='upper right')
for s in ['top', 'right']:
    axb.spines[s].set_visible(False)
axb.text(-0.15, 1.15, 'b', transform=axb.transAxes, fontsize=14, fontweight='bold', va='top')

# === Panel c: Schematic summary ===
axc = fig.add_subplot(gs[0, 2])
axc.axis('off')
axc.set_xlim(0, 10)
axc.set_ylim(0, 10)
axc.text(-0.15, 1.15, 'c', transform=axc.transAxes, fontsize=14, fontweight='bold', va='top')
axc.text(5, 9.8, 'Conclusion', fontsize=9, fontweight='bold', ha='center', va='top')

axc.add_patch(FancyBboxPatch((0.3, 6.8), 9.2, 2.4, boxstyle='round,pad=0.15',
              fc='#FADBD8', ec=MAL, lw=1.3))
axc.text(5.0, 8.6, 'AC-like malignant cells masquerade\nas astrocytes in 4/4 cohorts',
         ha='center', va='top', fontsize=7.5, fontweight='bold', color=MAL)
axc.text(5.0, 7.3, '66–88% of marker-defined "astrocyte" clusters\ncarry +7/−10 tumour hallmark (>470k cells)',
         ha='center', va='top', fontsize=6.5, color=DK)

axc.annotate('', xy=(5, 6.3), xytext=(5, 6.7),
             arrowprops=dict(arrowstyle='->', lw=1.5, color=DK))

axc.add_patch(FancyBboxPatch((0.3, 3.5), 9.2, 2.5, boxstyle='round,pad=0.15',
              fc='#E8DAEF', ec=PURP, lw=1.3))
axc.text(5.0, 5.4, 'Malignant contamination inflates\nastrocyte→macrophage TGF-β2 signalling',
         ha='center', va='top', fontsize=7.5, fontweight='bold', color=PURP)
axc.text(5.0, 4.15, 'Removing CNV+ cells attenuates TGF-β2 score\nby −16% (discovery) and −28% (validation)',
         ha='center', va='top', fontsize=6.5, color=DK)

axc.annotate('', xy=(5, 2.9), xytext=(5, 3.4),
             arrowprops=dict(arrowstyle='->', lw=1.5, color=DK))

axc.add_patch(FancyBboxPatch((0.3, 0.8), 9.2, 1.8, boxstyle='round,pad=0.15',
              fc='#D5F5E3', ec=GRN, lw=1.3))
axc.text(5.0, 2.2, 'CNV scoring before downstream analysis\nis essential for GBM scRNA-seq studies',
         ha='center', va='center', fontsize=7.5, fontweight='bold', color='#1E8449')

fig.savefig('graphical_abstract.png', dpi=300, bbox_inches='tight', facecolor='white')
fig.savefig('graphical_abstract.pdf', bbox_inches='tight', facecolor='white')
