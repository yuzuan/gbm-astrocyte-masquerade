"""Reproduce primary summaries from the recoded minimal expression inputs."""
from pathlib import Path
import sys,json
import numpy as np
import pandas as pd
base=Path(__file__).resolve().parent
sys.path.insert(0,str(base/'code'))
import stage45_public_only_analysis as m
out=base/'reproduced';out.mkdir(exist_ok=True)
common={s:s for s in m.MALIGNANT_STATES};parts=[];comps=[]
configs=[
 ('GSE211376_minimal.h5ad','GSE211376','patient','region','predicted.high_hierarchy',common,lambda o:o['predicted.high_hierarchy'].astype(str).isin(['TAM-MG','TAM-BDM'])),
 ('scGBM_minimal.h5ad','GSE273274','patient_id','tissue_type','Minor_cluster',{'AC':'AC-like','MES1':'MES-like','OPC':'OPC-like','NPC1':'NPC-like','NPC2':'NPC-like'},lambda o:o.Major_category.astype(str).isin(['TAM/Macrophage','Microglia','Monocyte'])),
 ('snGBM_minimal.h5ad','GSE276177','patient','tissue','celltype',common,lambda o:o.celltype.astype(str).isin(['Macrophage','Microglia','Monocyte'])),
 ('snGBM_minimal.h5ad','GSE286419','patient','tissue','celltype',common,lambda o:o.celltype.astype(str).isin(['Macrophage','Microglia','Monocyte'])),
 ('regional_GSE274987_minimal.h5ad','GSE274987','patient_id','tissue_type','celltype',common,lambda o:o.celltype.astype(str).isin(['Macrophage','Microglia','Monocyte'])),
]
for file,co,pc,rc,sc,sm,mask in configs:
 exp,comp=m.read_h5ad_cohort(base/'minimal_cell_data'/file,co,co,pc,rc,'dataset',sm,sc,mask);parts.append(exp);comps.append(comp)
exp=pd.concat(parts,ignore_index=True);comp=pd.concat(comps,ignore_index=True)
contrasts=m.ac_vs_other_contrasts(exp);scores=m.communication_scores(exp);ac,regions=m.composition_tests(comp)
outputs={'patient_state_expression':exp,'patient_state_composition':comp,'ac_vs_other_ligand_contrasts':contrasts,'meta_ligand_results':m.meta_ligand_results(contrasts),'patient_communication_scores':scores,'regional_communication_tests':m.regional_pair_tests(scores),'patient_ac_fraction':ac,'regional_ac_fraction_tests':regions}
keys={'patient_state_expression':['cohort','patient','region','state'],'patient_state_composition':['cohort','patient','region','state'],'ac_vs_other_ligand_contrasts':['cohort','region','gene'],'meta_ligand_results':['gene'],'patient_communication_scores':['cohort','patient','region','pair'],'regional_communication_tests':['cohort','comparison','pair'],'patient_ac_fraction':['cohort','patient','region','state'],'regional_ac_fraction_tests':['cohort','comparison']}
checks={}
for name,d in outputs.items():
 d.to_csv(out/(name+'.csv'),index=False);ref=pd.read_csv(base/'data'/(name+'.csv'));k=keys[name];x=d.sort_values(k).reset_index(drop=True);y=ref.sort_values(k).reset_index(drop=True)
 assert len(x)==len(y) and x[k].astype(str).equals(y[k].astype(str)),name
 nums=[c for c in y.select_dtypes(include='number') if c in x]
 checks[name]=bool(np.allclose(x[nums].to_numpy(float),y[nums].to_numpy(float),rtol=1e-5,atol=1e-7,equal_nan=True))
assert all(checks.values()),checks
(out/'checks.json').write_text(json.dumps(checks,indent=2));print('Primary reproduction:',sum(checks.values()),'passed, 0 failed')
