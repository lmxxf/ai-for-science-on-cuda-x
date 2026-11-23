#!/usr/bin/env python3
"""Verify corrected conformer physics, resident eigensolvers and read-only audit evidence."""
from pathlib import Path
import hashlib,json,xml.etree.ElementTree as ET
import numpy as np
R=Path(__file__).resolve().parent
load=lambda p:json.loads((R/p).read_text())
TD=np.dtype([('ints','<i4',(6,)),('params','<f8',(4,))])
def conformers():
 r=load('results/module14_results.json');p=r['preparation'];assert r['status']=='PASS'
 assert p['molecules']==1000 and p['conformers']==10000
 assert r['gradient_validation']['directions']>=6 and r['gradient_validation']['max_abs_error']<1e-8
 data=R/'results/module14_data';nc,nm,stride,_,_,_=map(int,(data/'params.txt').read_text().split());mols=np.fromfile(data/'mols.bin','<i4').reshape(nm,4)
 initial=np.fromfile(data/'initial.bin','<f8').reshape(nc,stride,3);terms=np.fromfile(data/'terms.bin',TD);mask=np.zeros((nc,stride),bool)
 for mi,(n,start,count,offset) in enumerate(mols):mask[mi*10:mi*10+10,:n]=True
 assert mask.sum()==p['heavy_atoms_total_all_conformers']
 for file,sha in p['hashes'].items():assert hashlib.sha256((data/file).read_bytes()).hexdigest()==sha
 histories={};xyz={}
 for bk,b in r['backends'].items():
  folder=R/f'results/module14_{bk}';x=np.fromfile(folder/'coordinates.bin','<f8').reshape(nc,stride,3);h=np.fromfile(folder/'history.bin','<f8').reshape(nc,201)
  assert b['steps']==200 and np.all(np.isfinite(x[mask])) and np.all(np.isfinite(h))
  assert np.diff(h,axis=1).max()<1e-9 and np.isclose(h[:,0].mean(),b['energy_initial_mean']) and np.isclose(h[:,-1].mean(),b['energy_final_mean'])
  assert int((h[:,-1]<h[:,0]-1e-10).sum())==b['strict_decrease_count']==9990
  assert np.all(h[:,-1]<=h[:,0]+1e-10)
  pre=post=0;bp=[];ba=[]
  for mi,(n,start,count,offset) in enumerate(mols):
   ts=terms[start:start+count];nb=ts[ts['ints'][:,0]==3];bonds=ts[ts['ints'][:,0]==0]
   a=initial[mi*10:mi*10+10];z=x[mi*10:mi*10+10]
   if len(nb):
    i,j=nb['ints'][:,1],nb['ints'][:,2];threshold=.8*nb['params'][:,1]
    pre+=int((np.linalg.norm(a[:,i]-a[:,j],axis=-1)<threshold).sum());post+=int((np.linalg.norm(z[:,i]-z[:,j],axis=-1)<threshold).sum())
   if len(bonds):
    i,j=bonds['ints'][:,1],bonds['ints'][:,2];r0=bonds['params'][:,1]
    bp.extend(abs(np.linalg.norm(a[:,i]-a[:,j],axis=-1)-r0).ravel());ba.extend(abs(np.linalg.norm(z[:,i]-z[:,j],axis=-1)-r0).ravel())
  assert pre==b['nonbonded_collision_pairs_initial']==47 and post==b['nonbonded_collision_pairs_final']==0
  assert np.mean(ba)<np.mean(bp) and np.isclose(np.mean(ba),b['bond_abs_deviation_final_mean_A'])
  histories[bk]=h;xyz[bk]=x
 for bk in ['cuda','hip']:
  assert np.array_equal(histories[bk],histories['cpu']) and np.array_equal(xyz[bk][mask],xyz['cpu'][mask])
 failure=r['failure_case'];assert not failure['collision_threshold_changed']
 assert failure['initial_coordinate_sha256_before']==failure['initial_coordinate_sha256_after']==p['hashes']['initial.bin']
 old=load(failure['raw_failure_summary']);assert old['backends']['cpu']['nonbonded_collision_pairs_final']>old['backends']['cpu']['nonbonded_collision_pairs_initial']
 assert hashlib.sha256((R/'results/module14_failed_weak_lj_data/terms.bin').read_bytes()).hexdigest()==failure['old_terms_sha256']
 oldterms=np.fromfile(R/'results/module14_failed_weak_lj_data/terms.bin',TD)
 assert np.array_equal(oldterms['ints'],terms['ints']) and np.array_equal(oldterms['params'][:,1:],terms['params'][:,1:])
 nb=terms['ints'][:,0]==3
 assert np.all(oldterms['params'][nb,0]==.05) and np.all(terms['params'][nb,0]==.5)
 assert np.array_equal(oldterms['params'][~nb,0],terms['params'][~nb,0])
 return r

def resident():
 r=load('results/module15_results.json');assert r['status']=='PASS' and r['iteration_host_transfers']==r['iteration_host_eigensolves']==0
 assert {x['B'] for x in r['systems']}=={64,256}
 for a in r['systems']:
  b=a['B'];folder=R/f'results/module15_data/{b}';H=np.fromfile(folder/'H.bin','<f8').reshape(b,384,384);ref=np.load(folder/'reference_eigenvalues.npy')
  assert hashlib.sha256((folder/'H.bin').read_bytes()).hexdigest()==a['metadata']['H_sha256']
  arrays={}
  for bk,t in a['backends'].items():
   prefix=R/f'results/module15_{bk}_{b}' if bk=='cpu' else R/f"results/module15_{bk.split('_')[0]}_{b}_{bk.split('_')[1]}"
   e=np.fromfile(str(prefix)+'_eigenvalues.bin','<f8').reshape(b,16);x=np.fromfile(str(prefix)+'_eigenvectors.bin','<f8').reshape(b,384,16)
   assert np.all(np.isfinite(e)) and np.all(np.isfinite(x));assert np.max(abs(e-ref))<1e-13
   residual=np.linalg.norm(H@x-x*e[:,None,:],axis=1).max();assert residual<1e-7
   assert np.max(abs(np.swapaxes(x,1,2)@x-np.eye(16)))<1e-10
   if bk!='cpu':
    assert t['iteration_host_transfers']==t['iteration_host_eigensolves']==0
    assert t['max_residual']<1e-10 and t['launch_count']==(b if bk.endswith('seq') else 1)
   arrays[bk]=e
  assert np.array_equal(arrays['cuda_seq'],arrays['cuda_batch']) and np.array_equal(arrays['cuda_batch'],arrays['hip_batch']) and np.array_equal(arrays['hip_seq'],arrays['hip_batch'])
 stored=load('results/module10_results.json')['resident_upgrade_module15'];assert stored==r
 return r

def audit():
 r=load('results/module16_results.json');article=Path(r['article_path']);review=Path(r['review_path'])
 assert not r['article_modified'] and r['article_sha256_before']==r['article_sha256_after']==hashlib.sha256(article.read_bytes()).hexdigest()
 assert hashlib.sha256(review.read_bytes()).hexdigest()==r['review_sha256']
 assert len(r['protected_assets'])==6
 for asset in r['protected_assets']:
  assert asset['hash_recorded_in_review'] and hashlib.sha256(Path(asset['path']).read_bytes()).hexdigest()==asset['sha256']
 assert r['status']=='AUDIT_COMPLETED_WITH_FINDINGS' and r['body_remains_with_findings']
 return r

def figures():
 for name in ['fig9_daviator_batched_davidson_vasp_band','fig13_open_nvmolkit_10000_conformers_amd_vs_nvidia']:
  p=R/'visuals'/f'{name}.svg';text=p.read_text();root=ET.fromstring(text)
  assert 560<=float(root.attrib['width'].replace('px',''))<=680 and 'WenQuanYi Micro Hei' in root.attrib.get('font-family','')
  assert '\u1d40' not in text and p.with_suffix('.png').is_file()
  assert any(e.attrib.get('fill','').lower()=='#ffffff' or '#ffffff' in e.attrib.get('style','').lower() for e in root.iter())
def main():
 conformers();resident();audit();figures();print('PASS Phase4: full10k energies/coordinates/monotonicity/contact/bond metrics and archived failure; all64/256 eigenpairs/residuals/orthogonality/resident metadata; unchanged381+audit hashes; figure9/13')
if __name__=='__main__':main()
