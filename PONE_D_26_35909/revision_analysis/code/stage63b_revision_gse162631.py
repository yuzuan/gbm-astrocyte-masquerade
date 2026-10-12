"""从GSE162631公开10x计数独立读回肿瘤样本，核对保留细胞及完整基因分母。"""
from pathlib import Path
import gzip
import os
import numpy as np
import pandas as pd
from scipy.io import mmread
from anndata.utils import make_index_unique
from stage63_revision_preprocessing import OUT, GENES, assign_states, summarize_patients


def run_raw_audit():
    source_value = os.environ.get('GSE162631_RAW_ROOT')
    if not source_value:
        raise ValueError('GSE162631_RAW_ROOT is required for the source-count audit')
    source = Path(source_value).expanduser()
    if not source.is_dir():
        raise FileNotFoundError('GSE162631_RAW_ROOT must point to an existing source directory')
    obs=pd.read_csv(OUT/'analysis/cache/GSE162631_metadata.csv.gz')
    data=np.load(OUT/'analysis/cache/GSE162631_expression.npz')
    assert data['genes'].tolist()==GENES
    rows=[]; patient=[]; eligibility=[]
    for folder in sorted(source.glob('R*_T')):
        features=folder/'features.tsv.gz'
        if not features.exists(): features=folder/'genes.tsv.gz'
        genes=make_index_unique(pd.Index(pd.read_csv(features,sep='\t',header=None)[1].astype(str)))
        barcodes=pd.Index(pd.read_csv(folder/'barcodes.tsv.gz',header=None)[0].astype(str))
        indices=genes.get_indexer(GENES); assert np.all(indices>=0) and barcodes.is_unique
        print('Reading',folder.name,flush=True)
        with gzip.open(folder/'matrix.mtx.gz','rb') as f:
            x=mmread(f).tocsc()
        assert x.shape==(len(genes),len(barcodes)) and np.all(x.data>=0) and np.all(x.data==np.rint(x.data))
        total=np.asarray(x.sum(axis=0)).ravel()
        detected=np.asarray((x>0).sum(axis=0)).ravel()
        mitochondrial=np.asarray(x[genes.str.upper().str.startswith('MT-')].sum(axis=0)).ravel()
        sub=obs[(obs['patient']=='GSE162631_'+folder.name.split('_')[0]) & (obs['region']=='Tumor')].copy()
        original_positions=sub.index.to_numpy()
        baridx=barcodes.get_indexer(sub['cell_id'].str.rsplit('_',n=1).str[-1])
        assert len(sub)>0 and np.all(baridx>=0)
        panel=x[indices,:][:,baridx].T.toarray().astype(np.float32)
        delta=np.max(np.abs(panel-data['counts'][original_positions]))
        assert delta==0, 'Public-source counts disagree with retained counts'
        sub['count_library']=total[baridx]; sub['count_n_genes']=detected[baridx]
        sub['count_mt_pct']=100*mitochondrial[baridx]/total[baridx]
        sub=sub.reset_index(drop=True)
        norm10=np.log1p(panel/total[baridx,None]*1e4).astype(np.float32)
        uniform,margin,scores=assign_states(norm10)
        uniform[~sub['malignant_pool'].to_numpy()]='Excluded'
        ps,el=summarize_patients(sub,norm10,panel,uniform,'uniform','source_full_gene_10k')
        patient.extend(ps); eligibility.extend(el)
        # 只检验已存在、方向未知的独立注释一致性，不用TGFB2结果决定细胞去留。
        original_tumor=sub['cell_type_original'].isin(['Tumor','Prol.tumor']).to_numpy()
        restricted=uniform.copy();restricted[~original_tumor]='Excluded'
        ps,el=summarize_patients(sub,norm10,panel,restricted,'uniform','source_original_tumor_concordant')
        patient.extend(ps);eligibility.extend(el)
        rows.append({'sample':folder.name,'source_cells':len(barcodes),'source_genes':len(genes),
                     'retained_cells':len(sub),'retained_fraction':len(sub)/len(barcodes),
                     'malignant_pool_cells':int(sub['malignant_pool'].sum()),
                     'max_panel_count_disagreement':float(delta),
                     'median_full_over_retained_library_ratio':float(np.median(total[baridx]/obs.loc[original_positions,'count_library'])),
                     'max_full_over_retained_library_ratio':float(np.max(total[baridx]/obs.loc[original_positions,'count_library'])),
                     'n_original_tumor_concordant':int((original_tumor&sub['malignant_pool'].to_numpy()).sum())})
        print(rows[-1],flush=True)
    assert len(rows)==4
    pd.DataFrame(rows).to_csv(OUT/'analysis/GSE162631_public_count_audit.csv',index=False)
    pd.DataFrame(patient).to_csv(OUT/'analysis/GSE162631_source_sensitivity.csv',index=False)
    pd.DataFrame(eligibility).to_csv(OUT/'analysis/GSE162631_source_eligibility.csv',index=False)
    base=pd.read_csv(OUT/'analysis/preprocessing_patient_contrasts.csv')
    g=base[(base['scenario']=='stored_X')&(base['labels']=='uniform')&(base['cohort']=='GSE162631')]
    influence=[]
    for p in g['patient']:
        values=g.loc[g['patient']!=p,'difference']
        influence.append({'omitted':p,'n':len(values),'median_difference':values.median(),'mean_difference':values.mean(),
                          'positive_patients':int((values>0).sum()),'inference':'descriptive n=2; no p value'})
    pd.DataFrame(influence).to_csv(OUT/'analysis/GSE162631_leave_patient_out.csv',index=False)


if __name__=='__main__':
    run_raw_audit()
