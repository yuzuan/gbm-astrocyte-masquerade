#!/usr/bin/env python3
"""
Stage 21: Figure 2
Source artifact: Figure2.png (version: a3fc757d-6abe-43fa-a374-dfa677892877)

GBM AC-like Masquerade Analysis Pipeline
Malignant cells masquerade as astrocytes in glioblastoma and confound
immune-microenvironment analysis — a multi-cohort study.

This script was extracted from the interactive analysis session.
Adjust file paths as needed for your environment.
"""

import matplotlib.pyplot as plt, matplotlib as mpl, numpy as np, json

mpl.rcParams['font.family']='DejaVu Sans'

MAL='#C0392B'; BLU='#2E86C1'; PERI='#5DADE2'; GREY='#95A5A6'; GREEN='#27AE60'; PURP='#7D3C98'

g211=json.load(open('data/intermediate/gse211376_marker_noncircular.json'))
disc_rate=g211['marker_defined_astrocyte']['cnv_pos_pct']; disc_mye=g211['reference_FPR']['myeloid']; disc_oli=g211['reference_FPR']['oligodendrocyte']
disc_ac=g211['author_AClike']['cnv_pos_pct']; disc_ref=g211['author_astrocyte_only']['cnv_pos_pct']
g174=json.load(open('data/intermediate/gse174554_leiden_cnv.json'))
r174=g174['astrocyte_cluster']['cnv_pos_pct']; mye174=g174['reference_FPR']['myeloid']; oli174=g174['reference_FPR']['oligodendrocyte']
L4=['GSE211376','GSE162631\n+273274','GSE276177\n+286419','GSE174554']

fig=plt.figure(figsize=(7.4,6.6))
gs=fig.add_gridspec(2,2, hspace=0.56, wspace=0.34, left=0.09, right=0.975, top=0.99, bottom=0.10)

# a — FIXED: label on RIGHT side, no overlap with any bar
axa=fig.add_subplot(gs[0,0])
rate=[disc_rate,88.2,71.0,r174]
axa.bar(range(4), rate, color=[GREEN,MAL,MAL,MAL], zorder=3, width=0.66)
axa.axhline(5, ls='--', lw=1, color=GREY, zorder=2)
# Place text to the right of all bars, at y=5 level
axa.text(3.7, 8.5, 'ref. floor ~5%', fontsize=6, color='#5D6D7E', ha='right', va='bottom',
         bbox=dict(fc='white', ec='#5D6D7E', lw=0.5, pad=2, boxstyle='round,pad=0.3'))
for i,v in enumerate(rate): axa.text(i,v+1.5,f'{v:.1f}%',ha='center',va='bottom',fontsize=7.2,fontweight='bold',color='#222')
axa.set_xticks(range(4)); axa.set_xticklabels(L4, fontsize=5.6)
axa.set_ylim(0,102); axa.set_ylabel('Astrocyte cluster CNV+ (%)', fontsize=7.6)
axa.set_title('Non-circular replication across four cohorts', fontsize=7.6, loc='left')
for s in ['top','right']: axa.spines[s].set_visible(False)

# b
axb=fig.add_subplot(gs[0,1])
lab=['Marker-def.\nastrocyte','AC-like','Refined\n"Astrocyte"','Myeloid\n(ref)']
val=[disc_rate,disc_ac,disc_ref,disc_mye]
axb.bar(range(4), val, color=[GREEN,MAL,PURP,GREY], zorder=3, width=0.66)
for i,v in enumerate(val): axb.text(i,v+1.5,f'{v:.1f}%',ha='center',va='bottom',fontsize=7.2,fontweight='bold',color='#222')
axb.set_xticks(range(4)); axb.set_xticklabels(lab, fontsize=6.4)
axb.set_ylim(0,102); axb.set_ylabel('CNV+ (%)', fontsize=7.6)
axb.set_title('GSE211376 (discovery): masquerade\nsurvives expert refinement (47.9%)', fontsize=7.4, loc='left')
for s in ['top','right']: axb.spines[s].set_visible(False)

# c
axc=fig.add_subplot(gs[1,0])
myeloid=[disc_mye,4.7,4.9,mye174]; oligo=[disc_oli,None,1.3,oli174]
x=np.arange(4); w=0.38
axc.bar(x-w/2, myeloid, width=w, color=BLU, label='Myeloid ref.', zorder=3)
axc.bar(x+w/2, [o if o is not None else 0 for o in oligo], width=w, color='#AAB7B8', label='Oligo. ref.', zorder=3)
for i,v in enumerate(myeloid): axc.text(i-w/2,v+0.4,f'{v:.1f}',ha='center',va='bottom',fontsize=6.0,color='#1B4F72')
for i,o in enumerate(oligo):
    if o is not None: axc.text(i+w/2,o+0.4,f'{o:.1f}',ha='center',va='bottom',fontsize=6.0,color='#566573')
    else: axc.text(i+w/2,0.6,'n/a',ha='center',va='bottom',fontsize=5.6,color='#909497')
axc.set_xticks(x); axc.set_xticklabels(L4, fontsize=5.6)
axc.set_ylim(0,15); axc.set_ylabel('Reference false-positive rate (%)', fontsize=7.4)
axc.set_title('Reference specificity: 1–12% floor', fontsize=7.6, loc='left')
axc.legend(fontsize=6.0, frameon=False, loc='upper right')
for s in ['top','right']: axc.spines[s].set_visible(False)

# d — FIXED: 82.7% ABOVE its bar, offset right to avoid 93.8%
axd=fig.add_subplot(gs[1,1])
cohd=['GSE162631\n+273274','GSE276177\n+286419']; tum=[93.8,87.5]; per=[82.7,14.5]
x=np.arange(2); w=0.36
axd.bar(x-w/2, tum, width=w, color=MAL, label='Tumor', zorder=3)
axd.bar(x+w/2, per, width=w, color=PERI, label='Peritumoral', zorder=3)
axd.axhline(5, ls='--', lw=1, color=GREY, zorder=2)
# Tumor labels — centered above bars
axd.text(0-w/2, 95.5, '93.8%', ha='center', va='bottom', fontsize=6.4, fontweight='bold', color=MAL)
axd.text(1-w/2, 89.2, '87.5%', ha='center', va='bottom', fontsize=6.4, fontweight='bold', color=MAL)
# Peritumoral labels — ABOVE the peritumoral bars, shifted right of center
axd.text(0+w/2+0.08, 84.5, '82.7%', ha='center', va='bottom', fontsize=6.4, fontweight='bold', color='#2874A6')
axd.text(1+w/2, 16.2, '14.5%', ha='center', va='bottom', fontsize=6.4, fontweight='bold', color='#2874A6')
axd.set_xticks(x); axd.set_xticklabels(cohd, fontsize=5.8)
axd.set_ylim(0,112); axd.set_ylabel('Astrocyte cluster CNV+ (%)', fontsize=7.4)
axd.set_title('Tumor vs peritumoral (paired cohorts)', fontsize=7.4, loc='left')
axd.legend(fontsize=6.2, frameon=False, loc='upper right', ncol=2, columnspacing=1.0)
for s in ['top','right']: axd.spines[s].set_visible(False)

for ax,L in [(axa,'a'),(axb,'b'),(axc,'c'),(axd,'d')]:
    ax.text(-0.14,1.06,L,transform=ax.transAxes,fontsize=12,fontweight='bold',va='top')

fig.savefig('Figure2.png', dpi=300, bbox_inches='tight', facecolor='white')
fig.savefig('Figure2.pdf', bbox_inches='tight', facecolor='white')
