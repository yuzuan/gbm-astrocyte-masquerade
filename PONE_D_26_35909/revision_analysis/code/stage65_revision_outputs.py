"""返修：独立数值核验、森林图与数据附件。"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.stats import t
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from stage63_revision_preprocessing import OUT, PACKAGE

from plottie_figure_style import (configure_plottie_style, panel_label, open_axes, save_figure,
                                 TEAL, CORAL, INK, MUTED, GRID, GRAY_DARK, STATE_COLORS)

C1=TEAL; C2=CORAL; GRAY=GRAY_DARK
COHORT={'GSE274987':'GSE274986/7'}


def prepare_meta_inputs():
    rows=[]
    primary=pd.read_csv(PACKAGE/'tables/TableS3a_ligand_contrasts.csv')
    for r in primary[primary.gene=='TGFB2'].to_dict('records'):
        for d in json.loads(r['patient_values']):
            rows.append({'analysis':'primary_labels','cohort':r['cohort'],'patient':d['patient'],'difference':d['difference']})
    uniform=pd.read_csv(PACKAGE/'tables/TableS7a_uniform_annotation_patient_contrasts.csv')
    for r in uniform.to_dict('records'):
        rows.append({'analysis':'uniform_labels','cohort':r['cohort'],'patient':r['patient'],'difference':r['tgfb2_difference']})
    base=pd.DataFrame(rows);parts=[base]
    pre=pd.read_csv(OUT/'analysis/preprocessing_patient_contrasts.csv')
    for (label,scenario),df in pre[pre.scenario!='stored_X'].groupby(['labels','scenario']):
        df=df[['cohort','patient','difference']].copy();df['analysis']='preproc__'+label+'__'+scenario;parts.append(df)
    src=pd.read_csv(OUT/'analysis/GSE162631_source_sensitivity.csv')
    for scenario,df in src.groupby('scenario'):
        others=base[(base.analysis=='uniform_labels')&(base.cohort!='GSE162631')][['cohort','patient','difference']]
        mixed=pd.concat([others,df[['cohort','patient','difference']]],ignore_index=True)
        mixed['analysis']='raw__'+scenario;parts.append(mixed)
    pd.concat(parts,ignore_index=True).to_csv(OUT/'analysis/meta_patient_inputs.csv',index=False)


def validate():
    checks=[]
    def check(name,passed,detail=''):
        checks.append({'check':name,'passed':bool(passed),'detail':detail})
    pre=pd.read_csv(OUT/'analysis/preprocessing_patient_contrasts.csv')
    inp=pd.read_csv(OUT/'analysis/meta_patient_inputs.csv')
    for mode,analysis in [('primary','primary_labels'),('uniform','uniform_labels')]:
        a=pre[(pre.scenario=='stored_X')&(pre.labels==mode)]
        b=inp[inp.analysis==analysis]
        m=a.merge(b,on=['cohort','patient'],suffixes=('_new','_old'),validate='one_to_one')
        err=np.max(np.abs(m.difference_new-m.difference_old))
        check(mode+' frozen patient contrasts reproduced',len(m)==len(a)==len(b) and err<1e-6,f'max difference {err:.3g}')
    check('patient keys unique',not pre.duplicated(['scenario','labels','cohort','patient']).any())
    audit=pd.read_csv(OUT/'analysis/count_input_audit.csv')
    check('all four count inputs integer',len(audit)==4 and audit.nonnegative_integer_counts.all())
    check('three stored X matrices reconstructed',audit.max_abs_X_reconstruction_error.max()<1e-6)
    raw=pd.read_csv(OUT/'analysis/GSE162631_public_count_audit.csv')
    check('four public 10x tumor count files agree with retained panel',len(raw)==4 and (raw.max_panel_count_disagreement==0).all())
    ef=pd.read_csv(OUT/'analysis/meta_cohort_effects.csv'); summary=pd.read_csv(OUT/'analysis/meta_summary.csv')
    for _,r in summary.iterrows():
        s=ef[(ef.analysis==r.analysis)&ef.eligible];y=s.yi.to_numpy();v=s.vi.to_numpy();k=len(s)
        def objective(tau):
            w=1/(v+tau);mu=np.sum(w*y)/sum(w)
            return .5*(np.log(v+tau).sum()+np.log(w.sum())+np.sum(w*(y-mu)**2))
        result=minimize_scalar(objective,bounds=(0,max(1,np.var(y)*10)),method='bounded',options={'xatol':1e-13})
        tau=0. if objective(0)<=result.fun else result.x
        w=1/(v+tau);mu=np.sum(w*y)/sum(w)
        scale=max(1.,np.sum(w*(y-mu)**2)/(k-1));se=np.sqrt(scale/sum(w))
        low,high=mu+np.array([-1,1])*t.ppf(.975,k-1)*se
        prob=2*t.sf(abs(mu/se),k-1)
        check('independent REML and modified-KH: '+r.analysis,
              np.max(np.abs(np.array([mu,low,high,prob,tau])-np.array([r.estimate,r.ci_low,r.ci_high,r.p_value,r.tau2])))<5e-6)
    check('all meta CI ordered and finite',np.isfinite(summary[['estimate','ci_low','ci_high']]).all().all() and (summary.ci_low<summary.estimate).all() and (summary.ci_high>summary.estimate).all())
    check('cohort weights sum to 100',all(abs(x.weight_pct.sum()-100)<1e-8 for _,x in ef.groupby('analysis')))
    check('joint regional source study not split',not inp.cohort.isin(['GSE274986']).any())
    src=pd.read_csv(OUT/'analysis/GSE162631_source_sensitivity.csv')
    check('source count reprocessing retains discordance',all((x.difference>0).sum()==1 and len(x)==3 for _,x in src.groupby('scenario')))
    el=pd.read_csv(OUT/'analysis/preprocessing_eligibility.csv')
    check('20-cell eligibility consistent',np.array_equal(el.eligible,(el.n_ac>=20)&(el.n_other>=20)))
    check('all displayed eligible contrasts have >=20 cells',((pre.n_ac>=20)&(pre.n_other>=20)).all())
    d={'passed':sum(c['passed'] for c in checks),'failed':sum(not c['passed'] for c in checks),'checks':checks,
       'scope':'Scientific numeric/input checks, not journal acceptance or server-upload validation'}
    (OUT/'validation/analysis_checks.json').write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n')
    print('Scientific checks:',d['passed'],'passed,',d['failed'],'failed')
    assert d['failed']==0,[c for c in checks if not c['passed']]
    return d


def setup_style():
    configure_plottie_style()
    plt.rcParams['text.color'] = INK


def save(fig,name):
    save_figure(fig, OUT/'figures', name)


def forest():
    ef=pd.read_csv(OUT/'analysis/meta_cohort_effects.csv')
    meta=pd.read_csv(OUT/'analysis/meta_summary.csv').set_index('analysis')
    fig,axs=plt.subplots(2,1,figsize=(7.1,6.8))
    fig.subplots_adjust(hspace=.95,left=.225,right=.70,bottom=.17,top=.94)
    for ax,key,title,color,letter in zip(axs,['primary_labels','uniform_labels'],
            ['Original labels: 20 patients, 4 cohorts','Uniform labels: 26 patients, 5 cohorts'],[C1,C2],['a','b']):
        d=ef[(ef.analysis==key)&ef.eligible].reset_index(drop=True)
        s=meta.loc[key];yy=np.arange(len(d),0,-1)
        ax.axvline(0,color=GRAY,ls='--',lw=.65)
        for y,(_,r) in zip(yy,d.iterrows()):
            ax.plot([r.ci_low,r.ci_high],[y,y],color=color,lw=1)
            ax.scatter(r.yi,y,s=r.weight_pct*1.6,marker='s',color=color,zorder=3)
            ax.text(1.04,y,f'{r.yi:.3f} [{r.ci_low:.3f}, {r.ci_high:.3f}]',
                    transform=ax.get_yaxis_transform(),va='center',fontsize=6.8)
        ax.add_patch(Polygon([[s.ci_low,0],[s.estimate,.20],[s.ci_high,0],[s.estimate,-.20]],color=color))
        ax.text(1.04,0,f'{s.estimate:.3f} [{s.ci_low:.3f}, {s.ci_high:.3f}]',
                transform=ax.get_yaxis_transform(),va='center',fontsize=6.8,fontweight='semibold')
        ax.set_yticks(list(yy)+[0], [COHORT.get(r.cohort,r.cohort)+f'  (n={r.n})' for _,r in d.iterrows()]+['Random effects'])
        ax.set_ylim(-.7,len(d)+.8);ax.set_xlim(-.22,.53)
        ax.set_title(title,loc='left')
        ax.set_xlabel('Mean within-patient difference (log1p / 10,000)')
        ax.text(1.04,len(d)+.55,'Estimate [95% CI]',transform=ax.get_yaxis_transform(),fontweight='semibold',fontsize=7.0)
        hetero=f'REML; modified KH; P={s.p_value:.4f}; τ²={s.tau2:.4f}; Q P={s.Q_p:.3g}'
        if key=='uniform_labels':
            hetero+=f'\nI²={s.I2:.1f}%; prediction interval [{s.prediction_low:.3f}, {s.prediction_high:.3f}]'
        else:
            hetero+='\nτ² 95% CI [0.0000, '+f'{s.tau2_ci_high:.4f}'+']; heterogeneity is imprecisely estimated'
        ax.text(0,-.40,hetero,transform=ax.transAxes,fontsize=6.8,color=MUTED)
        open_axes(ax);panel_label(ax,letter)
    save(fig,'S15_Fig_random_effects')


def preprocessing_plot():
    meta=pd.read_csv(OUT/'analysis/meta_summary.csv').set_index('analysis')
    fig,axs=plt.subplots(3,1,figsize=(7.1,7.4),gridspec_kw={'height_ratios':[5,3,2]})
    fig.subplots_adjust(hspace=.95,left=.27,right=.95,bottom=.16,top=.95)
    configs=[('Log1p / 10,000',[('Original labels','primary_labels'),('Rebuilt counts','preproc__primary__counts_10k'),('Stricter QC','preproc__primary__counts_10k_strictQC'),('Uniform labels','uniform_labels'),('Uniform + stricter QC','preproc__uniform__counts_10k_strictQC')]),
      ('Log1p / 1,000',[('Original labels','preproc__primary__counts_1k'),('Uniform: labels frozen','preproc__uniform_frozen__counts_1k'),('Uniform: reassigned','preproc__uniform__counts_1k')]),
      ('Pseudobulk log2 CPM',[('Original labels','preproc__primary__pseudobulk'),('Uniform labels','preproc__uniform__pseudobulk')])]
    for ax,(title,items),letter in zip(axs,configs,['a','b','c']):
        for y,(label,key) in enumerate(items[::-1]):
            s=meta.loc[key];color=C2 if 'uniform' in key else C1
            ax.errorbar(s.estimate,y,xerr=[[s.estimate-s.ci_low],[s.ci_high-s.estimate]],fmt='o',color=color,capsize=2,ms=4.2,lw=1)
        ax.axvline(0,color=GRAY,lw=.65,ls='--')
        ax.set_yticks(range(len(items)),[label+f'\nn={int(meta.loc[key,"n"])}, k={int(meta.loc[key,"k"])}' for label,key in items[::-1]])
        ax.set_ylim(-.65,len(items)-.35);ax.set_title(title,loc='left')
        ax.set_xlabel('Pooled difference [95% CI]')
        open_axes(ax);panel_label(ax,letter)
    fig.text(.08,.045,'Each panel has its own expression scale. Uniform reassignment / stricter QC can exclude GSE162631.\nChanges in eligibility are reported, not treated as a correction of the discordant cohort.',fontsize=6.8,color=MUTED)
    save(fig,'S16_Fig_preprocessing')


def discordance_plot():
    p=pd.read_csv(OUT/'analysis/preprocessing_patient_contrasts.csv')
    baseline=p[(p.cohort=='GSE162631')&(p.scenario=='stored_X')&(p.labels=='uniform')].set_index('patient')
    e=pd.read_csv(OUT/'analysis/preprocessing_eligibility.csv')
    e=e[(e.cohort=='GSE162631')&(e.labels=='uniform')]
    raw=pd.read_csv(OUT/'analysis/GSE162631_source_sensitivity.csv')
    fig,axs=plt.subplots(1,3,figsize=(7.1,3.45))
    fig.subplots_adjust(wspace=.60,left=.075,right=.975,bottom=.35,top=.85)
    pats=[f'P{i}' for i in range(1,5)] if set(e.patient).issubset({'P1','P2','P3','P4'}) else [f'GSE162631_R{i}' for i in range(1,5)];x=np.arange(4)
    for scenario,label,col in [('stored_X','Submitted',C1),('counts_1k','1,000 + reassignment',C2),('counts_10k_strictQC','Stricter QC',GRAY)]:
        v=e[e.scenario==scenario].set_index('patient').loc[pats,'n_ac']
        axs[0].plot(x,v,'o-',label=label,color=col,lw=1,ms=3.2)
    axs[0].axhline(20,color=GRAY,ls='--',lw=.65)
    axs[0].set_xticks(x,['P1','P2','P3','P4']);axs[0].set_ylabel('AC-like cells');axs[0].set_ylim(0,55)
    axs[0].legend(loc='upper left',bbox_to_anchor=(0,-.15),borderaxespad=0,fontsize=6.5)
    for i,pat in enumerate(baseline.index):
        r=baseline.loc[pat]
        axs[1].plot([i-.1,i+.1],[r.ac_detected_fraction*100,r.other_detected_fraction*100],color=GRID,lw=.8)
        axs[1].scatter(i-.1,r.ac_detected_fraction*100,s=18,color=STATE_COLORS['AC-like'],label='AC-like' if i==0 else None)
        axs[1].scatter(i+.1,r.other_detected_fraction*100,s=18,color=STATE_COLORS['Other malignant'],label='Other malignant' if i==0 else None)
        axs[1].text(i-.1,r.ac_detected_fraction*100+.7,f'{round(r.ac_detected_fraction*r.n_ac)}/{int(r.n_ac)}',ha='center',fontsize=6.8)
    axs[1].set_xlim(-.4,2.4);axs[1].set_xticks(range(3),['P1','P2','P4']);axs[1].set_ylabel('TGFB2-detected cells (%)');axs[1].set_ylim(-.8,15)
    axs[1].legend(loc='upper left',bbox_to_anchor=(0,-.15),borderaxespad=0,fontsize=6.5)
    for j,(scenario,label,color) in enumerate([('stored_X','Submitted',C1),('source_full_gene_10k','Source counts',C2),('source_original_tumor_concordant','Concordant labels',GRAY)]):
        df=baseline if scenario=='stored_X' else raw[raw.scenario==scenario].set_index('patient')
        axs[2].plot(np.arange(3)+(j-1)*.10,df.loc[baseline.index,'difference'],'o',label=label,color=color,ms=3.8)
    axs[2].axhline(0,color=GRAY,ls='--',lw=.65)
    axs[2].set_xticks(range(3),['P1','P2','P4']);axs[2].set_ylabel('AC-like minus other\n(log1p / 10,000)');axs[2].set_ylim(-.065,.13)
    axs[2].legend(loc='upper left',bbox_to_anchor=(0,-.15),borderaxespad=0,fontsize=6.5)
    for ax,title,letter in zip(axs,['AC-like cell counts','TGFB2 detection','Source-count audit'],['a','b','c']):
        ax.set_title(title,loc='left');open_axes(ax);panel_label(ax,letter)
    fig.text(.075,.055,'P1–P4 are study-local patient codes; P3 has five AC-like cells at baseline.\nSource-count and concordant-label checks retain one positive and two negative contrasts.',fontsize=6.8,color=MUTED)
    save(fig,'S17_Fig_GSE162631')


def workbook():
    """公开新增结果使用研究内患者编码；原始对应关系留在本地分析缓存。"""
    frames={p.stem:pd.read_csv(p) for p in sorted((OUT/'analysis').glob('*.csv'))}
    # 优先采用已存在GSE162631图面P1-P4，其余患者按队列内有序编码。
    inp=pd.read_csv(OUT/'analysis/preprocessing_eligibility.csv')
    maps={}
    for co,g in inp.groupby('cohort'):
        values=sorted(g.patient.unique())
        if co=='GSE162631': values=[f'GSE162631_R{i}' for i in range(1,5)]
        maps.update({v:f'P{i+1}' for i,v in enumerate(values)})
    rows=[]
    with pd.ExcelWriter(OUT/'supporting_information/S14_Data.xlsx',engine='openpyxl') as w:
        used=set()
        for i,(name,df) in enumerate(frames.items(),1):
            for col in ['patient','omitted']:
                if col in df: df[col]=df[col].map(lambda v:maps.get(v,v))
            # 不导出细胞索引或条形码；队列字段保留GEO来源。
            sheet=f'T{i:02d}_'+name[:26];assert sheet not in used;used.add(sheet)
            df.to_excel(w,sheet_name=sheet,index=False);rows.append({'sheet':sheet,'source_table':name,'rows':len(df)})
        pd.DataFrame(rows).to_excel(w,sheet_name='README',index=False)
        for ws in w.book:
            ws.freeze_panes='A2';ws.auto_filter.ref=ws.dimensions
            for cell in ws[1]: cell.font=__import__('openpyxl').styles.Font(bold=True)
            for column in ws.columns:
                letter=column[0].column_letter
                ws.column_dimensions[letter].width=min(42,max(13,len(str(column[0].value))+2))
    pd.DataFrame(rows).to_csv(OUT/'validation/S14_sheet_index.csv',index=False)


if __name__=='__main__':
    validate();setup_style();forest();preprocessing_plot();discordance_plot();workbook()
