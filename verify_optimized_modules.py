#!/usr/bin/env python3
"""Read full optimized outputs and independently verify medians, precision and input/source identity."""
from pathlib import Path
import json,hashlib,statistics
import numpy as np
R=Path(__file__).resolve().parent;D=R.parent/'ai-for-science-on-amd-9070xt'
load=lambda p:json.loads((R/p).read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 from exp9_cuphoton_blackhole_fx_hip import image
 fx=load('results/module11_optimized_results.json');cf=load('results/module14_optimized_results.json');v=fx['selected_common_variant'];assert v=='v2'
 for port,file,r,key in [('cuphoton_xbeam_fx','fx.cpp',fx,'new_source_sha256'),('nvmolkit_batched_conformer','conformer.cpp',cf,'source_sha256')]:
  p=Path('amd_hip_port')/port/file;assert sha(R/p)==sha(D/p)==r[key]
 for script in ['exp9_cuphoton_blackhole_fx_hip.py','exp12_open_nvmolkit_batched_conformer_hip.py','benchmark_conformer_optimized.py']:assert sha(R/script)==sha(D/script)
 shape=(64,120,1024);cpu=np.memmap(R/'results/module11_optimized_cpu_vis.bin',dtype='<c16',mode='r',shape=shape);_,reference_clean,_,_,_=image(cpu);allvis={};allclean={};errors={}
 for bk in ['cuda','hip']:
  out=R/f'results/module11_optimized_stable_{bk}';a=np.memmap(out/f'{v}_vis.bin',dtype='<c16',mode='r',shape=shape);allvis[bk]=a
  assert np.all(np.isfinite(a));rel=np.linalg.norm(a-cpu)/np.linalg.norm(cpu);assert rel<1e-14
  _,clean,_,_,_=image(a);allclean[bk]=clean;crel=np.linalg.norm(clean-reference_clean)/np.linalg.norm(reference_clean);assert crel<1e-14
  t=fx['backends'][bk];assert t['warmup_passes']==5 and t['repeats']==7
  tv=t['variants'][v]
  for part in ['FFT','X','pipeline']:assert np.isclose(statistics.median(tv[part+'_wall_seconds']),tv[part+'_wall_median_seconds'])
  errors[bk]={'visibility_L2_vs_CPU':float(rel),'CLEAN_L2_vs_CPU':float(crel)}
 cross=np.linalg.norm(allvis['hip']-allvis['cuda'])/np.linalg.norm(allvis['cuda']);crossclean=np.linalg.norm(allclean['hip']-allclean['cuda'])/np.linalg.norm(allclean['cuda']);assert cross<1e-14 and crossclean<1e-14
 assert sha(R/'results/module11_data/voltage.bin')==fx['metadata']['voltage_sha256']
 dat=R/'results/module14_data';nc,nm,stride,*_=map(int,(dat/'params.txt').read_text().split());assert nc==10000 and nm==1000
 mols=np.fromfile(dat/'mols.bin','<i4').reshape(nm,4);mask=np.zeros((nc,stride),bool)
 for i,m in enumerate(mols):mask[i*10:i*10+10,:m[0]]=True
 oldcoords=np.fromfile(R/'results/module14_cpu/coordinates.bin','<f8').reshape(nc,stride,3);oldhist=np.fromfile(R/'results/module14_cpu/history.bin','<f8').reshape(nc,201)
 assert cf['gradient_validation']['directions']==18 and cf['gradient_validation']['max_abs_error']<1e-8
 for file,h in cf['inputs']['hashes'].items():assert sha(dat/file)==h
 for bk in ['cpu','cuda','hip']:
  folder=R/f'results/module14_optimized_{bk}';x=np.fromfile(folder/'coordinates.bin','<f8').reshape(nc,stride,3);h=np.fromfile(folder/'history.bin','<f8').reshape(nc,201);assert np.array_equal(x[mask],oldcoords[mask]) and np.array_equal(h,oldhist)
  t=cf['backends'][bk];assert t['steps']==200 and t['conformers']==10000 and t['collision_pairs_initial']==47 and t['collision_pairs_final']==0
  assert np.max(np.diff(h,axis=1))==0 and np.isclose(statistics.median(t['raw_seconds']),t['seconds'])
  if bk!='cpu':assert t['warmup_runs']==5 and t['measured_repetitions']==7 and t['coverage']['total_conformers']==10000
 # Recalculate contact and bond diagnostics directly, without trusting saved quality statistics.
 from verify_phase4 import TD
 terms=np.fromfile(dat/'terms.bin',TD);initial=np.fromfile(dat/'initial.bin','<f8').reshape(nc,stride,3);last=np.fromfile(R/'results/module14_optimized_hip/coordinates.bin','<f8').reshape(nc,stride,3);contacts=[0,0];bond=[[],[]]
 for mi,(n,start,count,off) in enumerate(mols):
  ts=terms[start:start+count];nb=ts[ts['ints'][:,0]==3];bt=ts[ts['ints'][:,0]==0]
  for z,xx in enumerate([initial[mi*10:mi*10+10],last[mi*10:mi*10+10]]):
   if len(nb):contacts[z]+=int((np.linalg.norm(xx[:,nb['ints'][:,1]]-xx[:,nb['ints'][:,2]],axis=-1)<.8*nb['params'][:,1]).sum())
   if len(bt):bond[z].extend(abs(np.linalg.norm(xx[:,bt['ints'][:,1]]-xx[:,bt['ints'][:,2]],axis=-1)-bt['params'][:,1]).ravel())
 assert contacts==[47,0] and np.mean(bond[1])<np.mean(bond[0])
 for n in [11,14]:assert load(f'results/module{n}_results.json')['active_performance_record']=='same_source_optimization_20261125_warm5_rep7'
 print(json.dumps({'status':'PASS','FX':errors,'FX_cross_visibility_L2':float(cross),'FX_cross_CLEAN_L2':float(crossclean),'conformer_all_coordinates_and_histories_bitwise_equal':True,'contacts':contacts,'bond_MAE_A':[float(np.mean(x)) for x in bond]},indent=2))
if __name__=='__main__':main()
