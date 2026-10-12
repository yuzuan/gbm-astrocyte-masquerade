"""返修：全保留基因计数重归一化、患者级敏感性及反向队列审计。"""
from pathlib import Path
import gzip
import json
import importlib.metadata
import os

import h5py
import numpy as np
import pandas as pd
from anndata.io import read_elem
from scipy import sparse
from scipy.stats import norm, wilcoxon

ROOT = Path(os.environ.get('GLIOMA_VALIDATION_ROOT', Path(__file__).resolve().parents[1]))
PACKAGE = ROOT / 'claude_outputs/public_only_submission'
OUT = Path(os.environ.get('PLOS_REVISION_DIR', PACKAGE / 'plos_one_revision_2026-10-12'))
INPUT = ROOT / 'public_inputs'
MARKERS = {
    'AC-like': ['GFAP','S100B','HOPX','SLC1A3','MLC1'],
    'MES-like': ['VIM','CD44','CHI3L1','HILPDA','DDIT3','ENO2'],
    'OPC-like': ['PLP1','OLIG1','OMG','TNR','ALCAM'],
    'NPC-like': ['SOX4','SOX11','DCX','CD24','STMN1','STMN2'],
}
STATES = list(MARKERS)
LINEAGES = {'Endothelial':['PECAM1','VWF','CLDN5','KDR'],
            'Myeloid':['PTPRC','LST1','TYROBP'],
            'Oligodendrocyte':['MBP','MOG','MOBP']}
GENES = list(dict.fromkeys(['TGFB2'] + sum(MARKERS.values(),[]) + sum(LINEAGES.values(),[])))


def csr_block(group, start, stop):
    """按连续细胞读稀疏矩阵，不将整个counts层载入内存。"""
    assert group.attrs['encoding-type'] == 'csr_matrix'
    ptr = group['indptr'][start:stop+1]
    left, right = int(ptr[0]), int(ptr[-1])
    return sparse.csr_matrix((group['data'][left:right], group['indices'][left:right], ptr-left),
                             shape=(stop-start, int(group.attrs['shape'][1])))


def extract_object(name):
    path = INPUT / 'processed_inputs' / name
    with h5py.File(path,'r') as f:
        obs = read_elem(f['obs']); var = read_elem(f['var'])
        assert obs.index.is_unique and var.index.is_unique
        idx = var.index.get_indexer(GENES)
        assert np.all(idx>=0), [GENES[i] for i in np.where(idx<0)[0]]
        mt = var.index.str.upper().str.startswith('MT-')
        selected = np.zeros((len(obs),len(GENES)),dtype=np.float32)
        stored = np.zeros_like(selected)
        library = np.zeros(len(obs)); detected = np.zeros(len(obs),dtype=int); mtpct=np.zeros(len(obs))
        all_integer=True
        for start in range(0,len(obs),2048):
            end=min(start+2048,len(obs))
            x=csr_block(f['layers/counts'],start,end)
            assert np.isfinite(x.data).all() and np.all(x.data>=0)
            all_integer &= bool(np.all(x.data==np.rint(x.data)))
            library[start:end]=np.asarray(x.sum(axis=1)).ravel()
            detected[start:end]=np.asarray((x>0).sum(axis=1)).ravel()
            mtpct[start:end]=100*np.asarray(x[:,mt].sum(axis=1)).ravel()/library[start:end]
            selected[start:end]=x[:,idx].toarray()
            stored[start:end]=csr_block(f['X'],start,end)[:,idx].toarray()
        assert all_integer and np.all(library>0)
        obs=obs.copy(); obs['cell_id']=obs.index.astype(str)
        obs['cohort']=obs['dataset'].astype(str)
        obs['patient']=obs['patient_id' if 'patient_id' in obs else 'patient'].astype(str)
        obs['region']=obs['tissue_type' if 'tissue_type' in obs else 'tissue'].astype(str)
        if name.startswith('scGBM'):
            obs['primary_state']=obs['Minor_cluster'].astype(str).map({'AC':'AC-like','MES1':'MES-like','OPC':'OPC-like','NPC1':'NPC-like','NPC2':'NPC-like'}).fillna('Excluded')
            obs['malignant_pool']=obs['Major_category'].astype(str).eq('Tumor')
        else:
            obs['primary_state']=obs['celltype'].astype(str).where(obs['celltype'].isin(STATES),'Excluded')
            obs['malignant_pool']=obs['celltype'].isin(STATES)
        obs['count_library']=library; obs['count_n_genes']=detected; obs['count_mt_pct']=mtpct
        keep=obs['cohort']!='GSE241037'
        obs=obs.loc[keep].reset_index(drop=True); selected=selected[keep]; stored=stored[keep]
        # 此处比较的是完整保留基因集的归一化，并非33基因子集归一化。
        rebuilt=np.log1p(selected/obs['count_library'].to_numpy()[:,None]*1e4)
        audit={'object':name,'n_cells':len(obs),'n_full_genes':len(var),'nonnegative_integer_counts':all_integer,
               'max_abs_X_reconstruction_error':float(np.max(np.abs(rebuilt-stored))),
               'mean_abs_X_reconstruction_error':float(np.mean(np.abs(rebuilt-stored))),
               'input_scope':'retained cells and genes in complete project AnnData'}
        print(json.dumps(audit),flush=True)
        return obs,selected,stored,audit


def extract_211():
    meta=pd.read_csv(INPUT/'raw_inputs/GSE211376_metadata.csv.gz',index_col=0)
    found=set()
    with gzip.open(INPUT/'raw_inputs/GSE211376_raw_counts.tsv.gz','rt') as f:
        header=f.readline().rstrip('\n').split('\t'); cells=header[1:] if header[0]=='' else header
        selected=np.zeros((len(cells),len(GENES)),dtype=np.float32)
        library=np.zeros(len(cells)); detected=np.zeros(len(cells),dtype=int); mitochondrial=np.zeros(len(cells))
        n_genes=0
        for line in f:
            gene, values=line.rstrip('\n').split('\t',1)
            v=np.fromstring(values,sep='\t',dtype=np.float64)
            assert len(v)==len(cells) and np.all(np.isfinite(v)) and np.all(v>=0) and np.all(v==np.rint(v))
            library+=v; detected+=(v>0); n_genes+=1
            if gene.upper().startswith('MT-'): mitochondrial+=v
            if gene in GENES:
                assert gene not in found
                selected[:,GENES.index(gene)]=v; found.add(gene)
    assert found==set(GENES) and np.all(library>0)
    obs=meta.reindex(cells).copy(); assert obs['patient'].notna().all()
    obs['cell_id']=cells; obs['cohort']='GSE211376'; obs['region']='Tumor'
    obs['primary_state']=obs['predicted.high_hierarchy'].where(obs['predicted.high_hierarchy'].isin(STATES),'Excluded')
    obs['malignant_pool']=obs['primary_state'].isin(STATES)
    obs['count_library']=library; obs['count_n_genes']=detected; obs['count_mt_pct']=100*mitochondrial/library
    stored=np.log1p(selected/library[:,None]*1e4).astype(np.float32)
    audit={'object':'GSE211376 GEO counts','n_cells':len(obs),'n_full_genes':n_genes,'nonnegative_integer_counts':True,
           'max_abs_X_reconstruction_error':0.,'mean_abs_X_reconstruction_error':0.,'input_scope':'public count matrix'}
    print(json.dumps(audit),flush=True)
    return obs.reset_index(drop=True),selected,stored,audit


def extract_inputs():
    (OUT/'analysis/cache').mkdir(parents=True,exist_ok=True)
    audits=[]
    sources=['scGBM_GSE162631_GSE273274.h5ad','snGBM_GSE276177_GSE286419.h5ad','regional_GSE274987_GSE241037.h5ad']
    for name in sources+['GSE211376']:
        obs,counts,stored,audit=extract_211() if name=='GSE211376' else extract_object(name)
        audits.append(audit)
        for cohort, sub in obs.groupby('cohort',sort=True):
            positions=sub.index.to_numpy()
            sub.to_csv(OUT/f'analysis/cache/{cohort}_metadata.csv.gz',index=False)
            np.savez_compressed(OUT/f'analysis/cache/{cohort}_expression.npz',counts=counts[positions],stored=stored[positions],genes=np.array(GENES))
    pd.DataFrame(audits).to_csv(OUT/'analysis/count_input_audit.csv',index=False)


def assign_states(x):
    scores=np.column_stack([x[:,[GENES.index(g) for g in m]].mean(axis=1) for m in MARKERS.values()])
    order=np.argsort(scores,axis=1); top=order[:,-1]
    margin=scores[np.arange(len(x)),top]-scores[np.arange(len(x)),order[:,-2]]
    labels=np.array(STATES,dtype=object)[top]; labels[margin<.05]='Ambiguous'
    return labels,margin,scores


def summarize_patients(obs,x,counts,labels,mode,scenario,pseudobulk=False):
    rows=[]; eligibility=[]
    target='Core' if obs['cohort'].iat[0]=='GSE274987' else 'Tumor'
    cohort=obs['cohort'].iat[0]
    df=obs.copy(); df['state']=labels
    for patient,frame in df[df['region']==target].groupby('patient'):
        n_by=frame['state'].value_counts()
        ac=frame.index[frame['state']=='AC-like'].to_numpy()
        other_states=STATES[1:]
        if mode=='primary': other_states=[s for s in other_states if n_by.get(s,0)>=20]
        other=frame.index[frame['state'].isin(other_states)].to_numpy()
        ok=len(ac)>=20 and len(other)>=20
        eligibility.append({'scenario':scenario,'labels':mode,'cohort':cohort,'patient':patient,'region':target,
                            'n_ac':len(ac),'n_other':len(other),'eligible':ok})
        if not ok: continue
        if pseudobulk:
            a=np.log2((counts[ac,0].sum()+.5)/(obs.loc[ac,'count_library'].sum()+1)*1e6)
            b=np.log2((counts[other,0].sum()+.5)/(obs.loc[other,'count_library'].sum()+1)*1e6)
        else:
            a=float(x[ac,0].mean()); b=float(x[other,0].mean())
        rows.append({'scenario':scenario,'labels':mode,'cohort':cohort,'patient':patient,'region':target,
          'n_ac':len(ac),'n_other':len(other),'ac_mean':a,'other_mean':b,'difference':a-b,
          'ac_detected_fraction':float(np.mean(counts[ac,0]>0)),
          'other_detected_fraction':float(np.mean(counts[other,0]>0)),
          'ac_median_library':float(np.median(obs.loc[ac,'count_library'])),
          'other_median_library':float(np.median(obs.loc[other,'count_library'])),
          'ac_median_genes':float(np.median(obs.loc[ac,'count_n_genes'])),
          'other_median_genes':float(np.median(obs.loc[other,'count_n_genes'])),
          'ac_median_mt_pct':float(np.median(obs.loc[ac,'count_mt_pct'])),
          'other_median_mt_pct':float(np.median(obs.loc[other,'count_mt_pct'])),
          'scale':'pseudobulk log2 CPM' if pseudobulk else ('mean log1p counts per 1000' if scenario=='counts_1k' else 'mean log1p counts per 10000')})
    return rows,eligibility


def combine(frame):
    cs=[]
    for co,g in frame.groupby('cohort'):
        v=g['difference'].to_numpy()
        if len(v)<3: continue
        p=1. if np.all(v==0) else float(wilcoxon(v,method='auto').pvalue)
        cs.append({'cohort':co,'n':len(v),'median':float(np.median(v)),'positive_patients':int((v>0).sum()),'p':p})
    if len(cs)<2: return None,cs
    def stouffer(cc):
        z=sum(np.sqrt(c['n'])*np.sign(c['median'])*norm.isf(max(c['p'],1e-15)/2) for c in cc)/np.sqrt(sum(c['n'] for c in cc))
        return float(z),float(2*norm.sf(abs(z)))
    z,p=stouffer(cs)
    return {'n':sum(c['n'] for c in cs),'k':len(cs),'positive_cohorts':sum(c['median']>0 for c in cs),'meta_z':z,'p_value':p,
            'loo_max_p':max(stouffer([c for c in cs if c['cohort']!=co['cohort']])[1] for co in cs)},cs


def analyze_inputs():
    all_rows=[]; all_elig=[]; gse_state=[]; transitions=[]; memberships=[]
    for path in sorted((OUT/'analysis/cache').glob('*_metadata.csv.gz')):
        obs=pd.read_csv(path); cohort=obs['cohort'].iat[0]
        z=np.load(OUT/f'analysis/cache/{cohort}_expression.npz'); counts=z['counts']; stored=z['stored']
        assert z['genes'].tolist()==GENES
        expr10=np.log1p(counts/obs['count_library'].to_numpy()[:,None]*1e4).astype(np.float32)
        expr1=np.log1p(counts/obs['count_library'].to_numpy()[:,None]*1e3).astype(np.float32)
        strict=(obs['count_n_genes']>=500)&(obs['count_mt_pct']<=5.)
        baseline_uniform,_,_=assign_states(stored)
        baseline_uniform[~obs['malignant_pool'].to_numpy()]='Excluded'
        settings=[('stored_X',stored,np.ones(len(obs),bool),False),('counts_10k',expr10,np.ones(len(obs),bool),False),
                  ('counts_1k',expr1,np.ones(len(obs),bool),False),('counts_10k_strictQC',expr10,strict.to_numpy(),False),
                  ('pseudobulk',expr10,np.ones(len(obs),bool),True)]
        for scenario,x,keep,pb in settings:
            unif,margin,scores=assign_states(x)
            unif[~obs['malignant_pool'].to_numpy()|~keep]='Excluded'
            primary=obs['primary_state'].to_numpy().copy(); primary[~keep]='Excluded'
            frozen=baseline_uniform.copy(); frozen[~keep]='Excluded'
            for mode,lab in [('primary',primary),('uniform',unif),('uniform_frozen',frozen)]:
                if mode=='primary' and cohort=='GSE162631': continue
                rows,el=summarize_patients(obs,x,counts,lab,mode,scenario,pb)
                all_rows.extend(rows); all_elig.extend(el)
            memberships.append({'cohort':cohort,'scenario':scenario,'retained_total':int(keep.sum()),
              'malignant_pool_retained':int((keep&obs['malignant_pool'].to_numpy()).sum()),
              'uniform_AC':int((unif=='AC-like').sum()),'uniform_ambiguous':int((unif=='Ambiguous').sum())})
            if cohort=='GSE162631':
                df=obs.copy(); df['uniform_state']=unif; df['margin']=margin
                for (pat,reg,st),g in df[df['malignant_pool']].groupby(['patient','region','uniform_state']):
                    ix=g.index.to_numpy()
                    row={'scenario':scenario,'patient':pat,'region':reg,'state':st,'n_cells':len(g),
                         'tgfb2_detected':float((counts[ix,0]>0).mean()),'tgfb2_mean_log10k':float(expr10[ix,0].mean()),
                         'median_library':float(g['count_library'].median()),'median_genes':float(g['count_n_genes'].median()),
                         'median_mt_pct':float(g['count_mt_pct'].median()),'median_assignment_margin':float(g['margin'].median())}
                    for lineage,genes in LINEAGES.items(): row[lineage+'_score']=float(expr10[ix][:,[GENES.index(gene) for gene in genes]].mean())
                    gse_state.append(row)
                if scenario=='stored_X':
                    for col in ['Major_category','Minor_cluster','cell_type_original','cell_type_cnv_corrected','cnv_score','cnv_leiden']:
                        if col in df:
                            tt=df.groupby(['patient','region',col,'uniform_state'],dropna=False).size().reset_index(name='n_cells')
                            tt=tt.rename(columns={col:'original_value'});tt['original_field']=col
                            transitions.append(tt)
        print(cohort,'sensitivity complete',flush=True)
    pt=pd.DataFrame(all_rows); pt.to_csv(OUT/'analysis/preprocessing_patient_contrasts.csv',index=False)
    pd.DataFrame(all_elig).to_csv(OUT/'analysis/preprocessing_eligibility.csv',index=False)
    pd.DataFrame(memberships).to_csv(OUT/'analysis/preprocessing_cell_membership.csv',index=False)
    pd.DataFrame(gse_state).to_csv(OUT/'analysis/GSE162631_state_audit.csv',index=False)
    pd.concat(transitions,ignore_index=True).to_csv(OUT/'analysis/GSE162631_annotation_crosswalk.csv',index=False)
    metarows=[]; corows=[]
    for mode in ['primary','uniform','uniform_frozen']:
        base=pt[(pt['scenario']=='stored_X')&(pt['labels']==mode)][['cohort','patient']]
        for scenario,df in pt[pt['labels']==mode].groupby('scenario'):
            for pop,g in [('available',df),('baseline_intersection',df.merge(base,on=['cohort','patient'],validate='one_to_one'))]:
                result,cs=combine(g)
                if result: metarows.append({'labels':mode,'scenario':scenario,'population':pop,**result})
                corows.extend({'labels':mode,'scenario':scenario,'population':pop,**c} for c in cs)
    pd.DataFrame(metarows).to_csv(OUT/'analysis/preprocessing_meta_stouffer.csv',index=False)
    pd.DataFrame(corows).to_csv(OUT/'analysis/preprocessing_cohort_summary.csv',index=False)
    versions={p:importlib.metadata.version(p) for p in ['numpy','pandas','scipy','anndata','h5py','matplotlib']}
    (OUT/'validation/python_versions.json').write_text(json.dumps(versions,indent=2)+'\n')
    return pt


if __name__=='__main__':
    extract_inputs()
    analyze_inputs()
