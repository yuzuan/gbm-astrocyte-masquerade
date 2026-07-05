#!/usr/bin/env python3
"""
Stage 25: Figure 6
Source artifact: Figure6.png (version: 7a6f5b35-b3ff-42d1-94d8-71b9d4b51586)

GBM AC-like Masquerade Analysis Pipeline
Malignant cells masquerade as astrocytes in glioblastoma and confound
immune-microenvironment analysis — a multi-cohort study.

This script was extracted from the interactive analysis session.
Adjust file paths as needed for your environment.
"""

import matplotlib.pyplot as plt, matplotlib as mpl, numpy as np, pandas as pd

mpl.rcParams['font.family']='DejaVu Sans'
MAL='#C0392B'; BLU='#2E86C1'; GREY='#95A5A6'; DK='#2C3E50'

v=pd.read_csv('data/intermediate/fig6_concordance_gse211376_vs_scgbm.csv')
n=len(v); top2=int(v['both_AC_top2'].sum())

fig=plt.figure(figsize=(7.6,3.0))
gs=fig.add_gridspec(1,3, wspace=0.50, left=0.075, right=0.985, top=0.80, bottom=0.24)

# a
axa=fig.add_subplot(gs[0])
cat=v.groupby('category').apply(lambda d: 100*(d['gse211376_AC_rank']==1).mean(), include_groups=False).sort_values()
short={'Checkpoint ligands':'Checkpoint','Complement regulators':'Complement','Dont-eat-me':"Don't-eat-me",
       'IFN response':'IFN resp.','Immunosuppressive':'Immunosupp.','Immunosuppressive chemokines':'Immun. chemok.','MHC-I':'MHC-I','MHC-II':'MHC-II'}
labels=[short.get(c,c) for c in cat.index]
axa.barh(range(len(cat)), cat.values, color=[MAL if x>=50 else GREY for x in cat.values], zorder=3, height=0.7)
axa.set_yticks(range(len(cat))); axa.set_yticklabels(labels, fontsize=6.2)
axa.set_xlim(0,105); axa.set_xlabel('AC-like ranked #1 (%)', fontsize=7.2)
axa.set_title('GSE211376: AC-like tops MHC-I,\nimmunosuppressive & IFN-response', fontsize=7.0, loc='left')
for s in ['top','right']: axa.spines[s].set_visible(False)

# b
axb=fig.add_subplot(gs[1])
rng=np.random.default_rng(1)
x=v['scGBM_AC_rank']+rng.uniform(-0.12,0.12,n); y=v['gse211376_AC_rank']+rng.uniform(-0.12,0.12,n)
conc=v['both_AC_top2'].values
axb.scatter(x[~conc], y[~conc], s=16, c=GREY, alpha=0.5, label='Discordant/other', zorder=2, edgecolors='none')
axb.scatter(x[conc], y[conc], s=18, c=BLU, alpha=0.8, label='Both AC top-2', zorder=3, edgecolors='none')
t=v[v['gene']=='TGFB2'].iloc[0]
axb.scatter([t['scGBM_AC_rank']],[t['gse211376_AC_rank']], s=60, facecolors='none', edgecolors=MAL, lw=1.6, zorder=5)
axb.annotate('TGFB2', (t['scGBM_AC_rank'],t['gse211376_AC_rank']), xytext=(1.75,1.7), fontsize=6.8, color=MAL, fontweight='bold', ha='left')
axb.plot([0.5,4.5],[0.5,4.5], ls=':', color='#B0B0B0', lw=1, zorder=1)
axb.set_xlim(4.6,0.4); axb.set_ylim(4.6,0.4)
axb.set_xticks([1,2,3,4]); axb.set_yticks([1,2,3,4])
axb.set_xlabel('AC rank — GSE162631+273274', fontsize=6.6)
axb.set_ylabel('AC rank — GSE211376', fontsize=6.8)
axb.set_title(f'Cross-cohort concordance\n(both AC top-2: {top2}/{n})', fontsize=7.2, loc='left')
axb.legend(fontsize=5.8, frameon=False, loc='lower left')
for s in ['top','right']: axb.spines[s].set_visible(False)

# c
axc=fig.add_subplot(gs[2])
lig=['TGFB2','NECTIN2','SPP1','TGFB1','MIF']; rk=[1,1,1,2,3]
cols=[MAL if r==1 else '#8FA5B0' for r in rk]
axc.hlines(range(len(lig)), 0, rk, color='#CCCCCC', lw=1.2, zorder=1)
axc.scatter(rk, range(len(lig)), s=60, c=cols, zorder=3)
for i,(r,g) in enumerate(zip(rk,lig)):
    axc.text(r+0.12, i, 'AC #1' if r==1 else f'#{r}', va='center', fontsize=6.2, color=MAL if r==1 else '#5D6D7E', fontweight='bold' if r==1 else 'normal')
axc.set_yticks(range(len(lig))); axc.set_yticklabels(lig, fontsize=6.8, fontstyle='italic')
axc.set_xlim(0.4,4.6); axc.set_xticks([1,2,3,4]); axc.invert_yaxis()
axc.set_xlabel('AC-like rank among subtypes\n(LIANA → macrophage)', fontsize=6.6)
axc.set_title('Cross-method (LIANA,\nGSE276177+286419): TGFB2 AC #1', fontsize=7.0, loc='left')
for s in ['top','right']: axc.spines[s].set_visible(False)

for ax,L in [(axa,'a'),(axb,'b'),(axc,'c')]:
    ax.text(-0.20,1.18,L,transform=ax.transAxes,fontsize=12,fontweight='bold',va='top')

fig.savefig('Figure6.png', dpi=300, bbox_inches='tight', facecolor='white')
