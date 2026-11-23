#!/usr/bin/env python3
"""Independent artifact consistency checks, without retraining or retiming."""
from pathlib import Path
import json,pickle,hashlib,xml.etree.ElementTree as ET
import numpy as np
from sklearn.metrics import mean_squared_error,r2_score
R=Path(__file__).resolve().parent
load=lambda p:json.loads((R/p).read_text())
a,b,c=[load(f'results/module{i}_results.json') for i in [1,2,3]]
split=load('results/module2_split.json')
sets=[set(split[n]) for n in ['train','validation','test']]
assert all(sets[i].isdisjoint(sets[j]) for i,j in [(0,1),(0,2),(1,2)])
assert set.union(*sets)==set(range(b['n_molecules']))
assert len(set(split['smiles']))==b['n_molecules']
y=np.asarray(b['test_targets']);p=np.asarray(b['test_predictions'])
assert len(y)==len(sets[2])
assert np.isclose(np.sqrt(mean_squared_error(y,p)),b['unimol_test']['RMSE'])
assert np.isclose(r2_score(y,p),b['unimol_test']['R2'])
epochs=[json.loads(x) for x in (R/'results/module2_epochs.jsonl').read_text().splitlines()]
assert min(epochs,key=lambda x:x['validation']['RMSE'])['epoch']==b['best_epoch']
f=a['folded_molecule'];xyz=np.array(f['coordinates']);i,j=f['atom_pair_zero_based']
assert f['mmff_converged']>0 and f['topological_bonds']>=5
assert np.isclose(np.linalg.norm(xyz[i]-xyz[j]),f['distance_angstrom']) and f['distance_angstrom']<3.5
z=np.load(R/'results/folded_molecule_distance.npz');d=z['distance']
assert np.allclose(d,d.T) and np.allclose(np.diag(d),0)
cache=pickle.load(open(R/'results/esol_conformers.pkl','rb'))
assert len(cache)==1128 and len([v for v in cache if 'coordinates' in v])==1128
assert c['assignment_mismatches']==0 and c['count_matrix_mismatches']==0
assert np.isclose(c['cpu_pipeline_median_s']/c['gpu_pipeline_median_s'],c['pipeline_speedup'])
assert all(v['match'] for v in load('results/drug_structure_audit.json'))
for key in ['esol','bbbp']:
    name='delaney-processed.csv' if key=='esol' else 'BBBP.csv'
    assert hashlib.sha256((R/'data'/name).read_bytes()).hexdigest()==a[key]['sha256']
figs=list((R/'visuals').glob('fig*.svg'));assert len(figs)==4
for path in figs:
    root=ET.parse(path).getroot();assert root.attrib['width']=='640'
    assert 'WenQuanYi Micro Hei' in root.attrib['font-family']
    assert any(x.tag.endswith('rect') and x.attrib.get('fill')=='#ffffff' for x in root)
    assert path.with_suffix('.png').is_file()
for n in [2,3]:
    g=load(f'logs/module{n}_memory.json');assert g['returncode']==0 and not g['killed']
print('PASS: split disjointness, metrics, validation checkpoint, complete 1128 cache, geometry, CPU/GPU agreement, drug identities, data hashes, 4 SVG/PNG, memory guards')
