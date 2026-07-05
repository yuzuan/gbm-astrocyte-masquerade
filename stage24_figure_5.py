#!/usr/bin/env python3
"""
Stage 24: Figure 5
Source artifact: Figure5.png (version: 91ad6d9a-29d6-4e38-afc5-66bece3cec4f)

GBM AC-like Masquerade Analysis Pipeline
Malignant cells masquerade as astrocytes in glioblastoma and confound
immune-microenvironment analysis — a multi-cohort study.

This script was extracted from the interactive analysis session.
Adjust file paths as needed for your environment.
"""

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import json

mpl.rcParams['font.family'] = 'DejaVu Sans'
TGF = '#884EA0'; GREY = '#95A5A6'; DISC = '#27AE60'; VAL = '#C0392B'

g174 = json.load(open('data/intermediate/gse174554_communication_decontam.json'))['pairs']
g211 = json.load(open('data/intermediate/gse211376_communication_decontam.json'))['pairs']

fig = plt.figure(figsize=(7.4, 3.2))
gs = fig.add_gridspec(1, 2, wspace=0.42, left=0.10, right=0.985, top=0.83, bottom=0.30)

axa = fig.add_subplot(gs[0])
b211 = g211['TGFB2->TGFBR2']; b174 = g174['TGFB2->TGFBR2']
groups = ['GSE211376\n(discovery)', 'GSE174554\n(228k, valid.)']
before = [b211['before'], b174['before']]; after = [b211['after'], b174['after']]; pct = [b211['pct_change'], b174['pct_change']]
x = np.arange(2); w = 0.34
axa.bar(x - w/2, before, width=w, color=GREY, label='Full astrocyte cluster', zorder=3)
axa.bar(x + w/2, after, width=w, color=TGF, label='After removing malignant', zorder=3)
for i in x:
    axa.annotate(f'{pct[i]:.0f}%', xy=(i, before[i]), xytext=(i, before[i] + 0.05), ha='center', fontsize=8, fontweight='bold', color=TGF)
axa.set_xticks(x); axa.set_xticklabels(groups, fontsize=7)
axa.set_ylim(0, 1.15); axa.set_ylabel('TGFB2→TGFBR2\ncommunication score', fontsize=7.4)
axa.set_title('TGF-β2 signal drops after decontamination\nin both cohorts', fontsize=7.6, loc='left')
axa.legend(fontsize=6.4, frameon=False, loc='upper right')
for s in ['top', 'right']: axa.spines[s].set_visible(False)

axb = fig.add_subplot(gs[1])
pairs = ['TGFB2→\nTGFBR2', 'TGFB2→\nTGFBR1', 'CSF1→\nCSF1R', 'TGFB1→\nTGFBR1', 'ANXA1→\nFPR1']
keys = ['TGFB2->TGFBR2', 'TGFB2->TGFBR1', 'CSF1->CSF1R', 'TGFB1->TGFBR1', 'ANXA1->FPR1']
d211 = [g211[k]['pct_change'] for k in keys]; d174 = [g174[k]['pct_change'] for k in keys]
x = np.arange(len(pairs)); w = 0.38
axb.bar(x - w/2, d211, width=w, color=DISC, label='GSE211376 (disc.)', zorder=3)
axb.bar(x + w/2, d174, width=w, color=VAL, label='GSE174554 (val.)', zorder=3)
axb.axhline(0, color='#555', lw=0.8, zorder=2)
axb.axvspan(-0.5, 1.5, color=TGF, alpha=0.10, zorder=0)
axb.axvspan(2.5, 4.5, color='#2E86C1', alpha=0.05, zorder=0)
axb.text(0.5, 35, 'most consistent\nacross cohorts', ha='center', fontsize=5.6, color=TGF, style='italic')
axb.text(3.5, 35, 'also attenuate\nin both', ha='center', fontsize=5.6, color='#2E86C1', style='italic')
axb.text(2.0, 30, 'discordant', ha='center', fontsize=5.6, color='#888', style='italic')
axb.set_xticks(x); axb.set_xticklabels(pairs, fontsize=5.9)
axb.set_ylim(-40, 42); axb.set_ylabel('Change after decontamination (%)', fontsize=7.2)
axb.set_title('TGF-β2 most consistent attenuation;\nonly CSF1 is discordant', fontsize=7.4, loc='left')
axb.legend(fontsize=6.2, frameon=False, loc='lower right')
for s in ['top', 'right']: axb.spines[s].set_visible(False)

axa.text(-0.22, 1.11, 'a', transform=axa.transAxes, fontsize=12, fontweight='bold', va='top')
axb.text(-0.15, 1.11, 'b', transform=axb.transAxes, fontsize=12, fontweight='bold', va='top')

fig.savefig('Figure5.png', dpi=300, bbox_inches='tight', facecolor='white')
fig.savefig('Figure5.pdf', bbox_inches='tight', facecolor='white')
