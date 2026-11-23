#!/usr/bin/env python3
"""Independent checks of Phase 2 saved evidence, without retraining."""
from pathlib import Path
import json,xml.etree.ElementTree as ET
import numpy as np
import pandas as pd
import hashlib,statistics
from sklearn.metrics import roc_auc_score,average_precision_score,mean_squared_error,r2_score
from scipy.stats import spearmanr
R=Path(__file__).resolve().parent
read=lambda p:json.loads((R/p).read_text())
def check_bbbp():
 r=read('results/module6_results.json');s=read('results/module6_split.json')
 sets=[set(s[k]) for k in ['train','validation','test']]
 assert len(set(s['smiles']))==len(s['smiles'])==r['n_molecules']
 assert all(sets[i].isdisjoint(sets[j]) for i,j in [(0,1),(0,2),(1,2)])
 assert set.union(*sets)==set(range(r['n_molecules']))
 y=r['test_labels'];p=r['test_probabilities'];assert len(y)==len(sets[2])
 assert np.isclose(roc_auc_score(y,p),r['unimol_test']['ROC_AUC'])
 assert np.isclose(average_precision_score(y,p),r['unimol_test']['PR_AUC_AP'])
 assert max(r['epochs'],key=lambda e:e['validation']['ROC_AUC'])['epoch']==r['best_epoch']
 assert len(r['epochs'])==20
 for name,a in r['attribution'].items():
  assert len(a['atoms'])==len(a['occluded_logits'])==len(a['delta_logit'])
  assert np.allclose(a['full_logit']-np.asarray(a['occluded_logits']),a['delta_logit'])
  assert np.isclose(1/(1+np.exp(-a['full_logit'])),a['probability'])
 assert r.get('pairbias'), 'Missing cross-vendor Uni-Mol core evidence'
 pb=r['pairbias'];assert pb['pass']
 for c in pb['comparisons'].values():
  assert c['CUDA_vs_HIP_max_abs']<1e-3
  assert c['CUDA_vs_official_PyTorch_max_abs']<1e-3 and c['HIP_vs_official_PyTorch_max_abs']<1e-3
 return r

def check_forcefield():
 r=read('results/module7_results.json');v=r['validation']
 assert v['force_relative_l2']<1e-12 and v['energy_relative_error']<1e-12
 assert v['finite_difference_max_abs_error']<1e-7
 assert v['periodic_translation_force_relative_l2']<1e-12
 assert {a['n'] for a in r['systems']}=={32000,128000}
 for a in r['systems']:
  n=a['n'];arrays={}
  for backend in ['cpu','cuda','hip']:
   folder=R/'results'/f'module7_{backend}_{n}'
   e=np.fromfile(folder/'energy.bin',dtype='<f8');f=np.fromfile(folder/'forces.bin',dtype='<f8')
   assert e.size==n and f.size==n*3 and np.all(np.isfinite(e)) and np.all(np.isfinite(f))
   arrays[backend]=(e,f)
   t=a['backends'][backend]
   assert np.isclose(statistics.median(t['pipeline_ms']),t['pipeline_median_ms'])
  for backend in ['cuda','hip']:
   e,f=arrays[backend];ce,cf=arrays['cpu']
   er=abs(e.sum()-ce.sum())/max(abs(ce.sum()),1.)
   fr=np.linalg.norm(f-cf)/np.linalg.norm(cf)
   assert er<1e-12 and fr<1e-12
   assert np.isclose(fr,a['backends'][backend]['force_relative_l2_vs_cpu'],atol=1e-20)
  assert np.linalg.norm(arrays['hip'][1]-arrays['cuda'][1])/np.linalg.norm(arrays['cuda'][1])<1e-12
 return r

def check_protein():
 r=read('results/module8_results.json');d=pd.read_csv(R/'data/gb1_experiment.csv')
 assert len(d)==3000 and d.sequence.nunique()==3000
 assert all(set(seq)<=set('ACDEFGHIKLMNPQRSTVWY') and len(seq)==56 for seq in d.sequence)
 sets=[set(d.loc[d.split==k,'sequence']) for k in ['train','val','test']]
 assert all(sets[i].isdisjoint(sets[j]) for i,j in [(0,1),(0,2),(1,2)])
 p=pd.read_csv(R/'results/module8_test_predictions.csv');assert len(p)==300
 for key,m in r['test'].items():
  y=p.target.to_numpy();pred=p[key].to_numpy()
  assert np.isclose(np.sqrt(mean_squared_error(y,pred)),m['RMSE'])
  assert np.isclose(r2_score(y,pred),m['R2'])
  if m['Spearman'] is not None:assert np.isclose(spearmanr(y,pred).statistic,m['Spearman'])
 assert min(r['history'],key=lambda e:e['val']['RMSE'])['epoch']==r['best_epoch']
 assert len(r['history'])==r['epochs']>=10 and r['architecture']['position_embedding_type']=='rotary'
 scan=pd.read_csv(R/'results/module8_mutation_scan.csv');assert len(scan)==80
 assert np.allclose(scan.prediction-r['scan']['WT_prediction'],scan.delta_vs_WT)
 assert np.allclose(scan.loc[scan.WT==scan.mutant,'delta_vs_WT'],0,atol=1e-6)
 assert hashlib.sha256((R/'data/gb1_experiment.csv').read_bytes()).hexdigest()==r['dataset']['experiment_csv_sha256']
 return r

def check_figures():
 names=['fig5_bbbp_unimol_roc_and_atom_attribution','fig6_open_cupotential_md_forcefield_amd_vs_nvidia','fig7_protein_esm2_finetune_and_mutation_scan']
 for name in names:
  p=R/'visuals'/f'{name}.svg';text=p.read_text();root=ET.fromstring(text)
  assert 560<=int(root.attrib['width'])<=680
  assert 'WenQuanYi Micro Hei' in root.attrib['font-family']
  assert any(x.tag.endswith('rect') and x.attrib.get('fill')=='#ffffff' for x in root)
  assert '\u1d40' not in text
  assert p.with_suffix('.png').is_file()

def main():
 check_bbbp()
 check_forcefield()
 check_protein()
 check_figures()
 print('PASS Phase2: BBBP split/held-out metrics/checkpoint/attribution, raw force/energy arrays and physics checks, protein holdout metrics/splits/mutation deltas, three SVG/PNG formatting checks')
if __name__=='__main__':main()
