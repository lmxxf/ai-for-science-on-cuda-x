#!/usr/bin/env python3
"""Full-array and timing checks for the nonbonded layout trials; no GPU execution."""
from pathlib import Path
import json,hashlib,statistics
import numpy as np
R=Path(__file__).resolve().parent;D=R.parent/'ai-for-science-on-amd-9070xt'
load=lambda p:json.loads((R/p).read_text())
def main():
 new=load('results/module7_optimized_results.json');old=load('results/module7_optimized_round1_results.json');root=load('results/module7_results.json');assert root['current_steady_state']['default_variant']=='v0'
 source=Path('amd_hip_port/cupotential_open/open_nonbonded.cpp');assert hashlib.sha256((R/source).read_bytes()).hexdigest()==hashlib.sha256((D/source).read_bytes()).hexdigest()==new['source_sha256']
 assert new['status']=='PASS'
 result=[]
 for s in new['systems']:
  n=s['n'];atoms=R/f'results/module7_data/{n}/atoms.bin';assert hashlib.sha256(atoms.read_bytes()).hexdigest()==s['input_sha256']
  e0=np.fromfile(R/f'results/module7_optimized/cpu/{n}/v0/energy.bin','<f8');f0=np.fromfile(R/f'results/module7_optimized/cpu/{n}/v0/forces.bin','<f8');assert e0.size==n and f0.size==3*n
  outputs={}
  for bk in ['cuda','hip']:
   for v,folder in [('v0',f'results/module7_optimized_soa/{bk}/{n}/v0'),('v2',f'results/module7_optimized/{bk}/{n}/v2'),('v3',f'results/module7_optimized_soa/{bk}/{n}/v3')]:
    p=R/folder;e=np.fromfile(p/'energy.bin','<f8');f=np.fromfile(p/'forces.bin','<f8');assert e.size==n and f.size==3*n and np.all(np.isfinite(e)) and np.all(np.isfinite(f));er=np.linalg.norm(e-e0)/np.linalg.norm(e0);fr=np.linalg.norm(f-f0)/np.linalg.norm(f0);assert er<1e-15 and fr<1e-15
    if v=='v0':assert np.array_equal(e,e0) and np.array_equal(f,f0)
    outputs[bk,v]=(e,f);result.append({'atoms':n,'backend':bk,'variant':v,'energy_L2_vs_cpu0':float(er),'force_L2_vs_cpu0':float(fr)})
   t=s['backends'][bk];assert t['warmups']==5 and t['repeats']==7
   for v,z in t['variants'].items():
    assert len(z['pipeline_ms'])==7 and len(z['warmup_pipeline_ms'])==5
    for phase in ['index','reorder','force','pipeline']:assert np.isclose(statistics.median(z[phase+'_ms']),z[phase+'_median_ms'])
  for v in ['v0','v2','v3']:
   for i in [0,1]:assert np.array_equal(outputs['cuda',v][i].view(np.uint64),outputs['hip',v][i].view(np.uint64))
 physics=new['physics_validation']['v3'];assert physics['force_relative_l2_vs_independent_bruteforce']<1e-15 and physics['energy_relative_error_vs_independent_bruteforce']<1e-15
 assert physics['net_force_l2']<1e-10 and physics['periodic_translation_force_relative_l2']<1e-14 and max(x['abs_error'] for x in physics['finite_difference_checks'])<1e-8
 print(json.dumps({'status':'PASS','default_retained':'v0','checks':result,'physics':physics},indent=2))
if __name__=='__main__':main()
