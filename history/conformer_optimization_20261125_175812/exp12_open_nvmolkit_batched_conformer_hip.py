#!/usr/bin/env python3
"""Fresh ETKDG -> explicit simplified united-heavy-atom FF; NOT MMFF94 parity."""
from pathlib import Path
import argparse,concurrent.futures,hashlib,itertools,json,subprocess,sys,time,os
os.environ.setdefault('OMP_NUM_THREADS','1')
import numpy as np,pandas as pd
from rdkit import Chem,RDLogger
from rdkit.Chem import AllChem
RDLogger.DisableLog('rdApp.*')
ROOT=Path(__file__).resolve().parent;SRC=ROOT/'amd_hip_port/nvmolkit_batched_conformer';RES=ROOT/'results';DAT=RES/'module14_data';BUILD=RES/'module14_build';TD=np.dtype([('ints','<i4',(6,)),('params','<f8',(4,))]);assert TD.itemsize==56

def run(c,**kw):print('RUN',*map(str,c),flush=True);return subprocess.run(list(map(str,c)),check=True,**kw)
def guard(c,n):return run([sys.executable,ROOT/'guard_process.py','--audit',ROOT/f'logs/module14_{n}_memory.json','--',*c])

def molecule(job):
 idx,smiles=job;t=time.perf_counter();m=Chem.AddHs(Chem.MolFromSmiles(smiles));nh=m.GetNumHeavyAtoms();p=AllChem.ETKDGv3();p.numThreads=1;p.randomSeed=20261124+idx;p.maxIterations=1000;p.pruneRmsThresh=-1
 ids=list(AllChem.EmbedMultipleConfs(m,10,p));retry=False
 if len(ids)!=10:p.useRandomCoords=True;ids=list(AllChem.EmbedMultipleConfs(m,10,p));retry=True
 if len(ids)!=10:raise RuntimeError(f'ETKDG {idx}: only{len(ids)}')
 xyz=np.array([m.GetConformer(i).GetPositions()[:nh] for i in ids]);rng=np.random.default_rng(720260000+idx);xyz+=rng.normal(0,.06,xyz.shape);embed_s=time.perf_counter()-t
 # MMFF API is used ONLY for equilibrium geometry parameters, never MMFF optimization.
 props=AllChem.MMFFGetMoleculeProperties(m,mmffVariant='MMFF94');ptable=Chem.GetPeriodicTable();adj=[[a.GetIdx() for a in m.GetAtomWithIdx(i).GetNeighbors() if a.GetIdx()<nh] for i in range(nh)];terms=[];mmff_bond=mmff_angle=0
 for bond in m.GetBonds():
  i,j=bond.GetBeginAtomIdx(),bond.GetEndAtomIdx()
  if i>=nh or j>=nh:continue
  params=props.GetMMFFBondStretchParams(m,i,j) if props else None
  if params:r0=params[2];mmff_bond+=1
  else:r0=ptable.GetRcovalent(m.GetAtomWithIdx(i).GetAtomicNum())+ptable.GetRcovalent(m.GetAtomWithIdx(j).GetAtomicNum())-.12*max(0,bond.GetBondTypeAsDouble()-1)
  terms.append(([0,i,j,-1,-1,0],[400.,r0,0.,0.]))
 for j,neighbors in enumerate(adj):
  for i,k in itertools.combinations(neighbors,2):
   params=props.GetMMFFAngleBendParams(m,i,j,k) if props else None
   if params:theta=params[2]*np.pi/180;mmff_angle+=1
   else:theta=np.pi if m.GetAtomWithIdx(j).GetHybridization()==Chem.HybridizationType.SP else 2*np.pi/3 if m.GetAtomWithIdx(j).GetHybridization()==Chem.HybridizationType.SP2 else np.arccos(-1/3)
   terms.append(([1,i,j,k,-1,0],[40.,float(np.cos(theta)),0.,0.]))
 for b in range(nh):
  for c in adj[b]:
   if b>=c:continue
   for a in adj[b]:
    if a==c:continue
    for d in adj[c]:
     if d==b or d==a:continue
     period=3 if m.GetAtomWithIdx(b).GetHybridization()==Chem.HybridizationType.SP3 and m.GetAtomWithIdx(c).GetHybridization()==Chem.HybridizationType.SP3 else 2
     terms.append(([2,a,b,c,d,period],[.2,0.,0.,0.]))
 AllChem.ComputeGasteigerCharges(m);charges=[];badq=0
 for i in range(nh):
  a=m.GetAtomWithIdx(i);qq=float(a.GetProp('_GasteigerCharge'))
  for h in a.GetNeighbors():
   if h.GetAtomicNum()==1:qq+=float(h.GetProp('_GasteigerCharge'))
  if not np.isfinite(qq):qq=0.;badq+=1
  charges.append(qq)
 graph=Chem.GetDistanceMatrix(Chem.RemoveHs(m))
 for i in range(nh):
  for j in range(i+1,nh):
   if graph[i,j]<3:continue
   sigma=.85*(ptable.GetRvdw(m.GetAtomWithIdx(i).GetAtomicNum())+ptable.GetRvdw(m.GetAtomWithIdx(j).GetAtomicNum()));terms.append(([3,i,j,-1,-1,0],[.5,sigma,1389.354576*charges[i]*charges[j],0.]))
 return dict(index=idx,smiles=smiles,atoms=[m.GetAtomWithIdx(i).GetSymbol() for i in range(nh)],xyz=xyz,terms=terms,etkdg_seconds=embed_s,retry=retry,mmff_bond_params=mmff_bond,mmff_angle_params=mmff_angle,bad_charge_atoms=badq)

def prepare(workers=8):
 DAT.mkdir(exist_ok=True);BUILD.mkdir(exist_ok=True);df=pd.read_csv(ROOT/'data/esol_clean.csv').iloc[:1000];st=time.perf_counter()
 with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as ex:items=list(ex.map(molecule,[(i,s) for i,s in enumerate(df.smiles)],chunksize=4))
 elapsed=time.perf_counter()-st;stride=max(len(x['atoms']) for x in items);coords=np.zeros((10000,stride,3));mols=[];terms=[];ptr=[0];incident=[];meta=[];astart=0
 for j,m in enumerate(items):
  n=len(m['atoms']);coords[j*10:j*10+10,:n]=m['xyz'];start=len(terms);localadj=[[] for _ in range(n)]
  for ii,pp in m['terms']:
   ti=len(terms);terms.append((ii,pp));indices=ii[1:3] if ii[0] in [0,3] else ii[1:4] if ii[0]==1 else ii[1:5]
   for a in indices:localadj[a].append(ti)
  mols.append([n,start,len(m['terms']),astart]);astart+=n
  for a in localadj:incident.extend(a);ptr.append(len(incident))
  meta.append({k:v for k,v in m.items() if k not in ['xyz','terms']})
 ta=np.empty(len(terms),TD)
 for i,(ii,pp) in enumerate(terms):ta['ints'][i]=ii;ta['params'][i]=pp
 coords.astype('<f8').tofile(DAT/'initial.bin');np.array(mols,dtype='<i4').tofile(DAT/'mols.bin');ta.tofile(DAT/'terms.bin');np.array(ptr,dtype='<i4').tofile(DAT/'ptr.bin');np.array(incident,dtype='<i4').tofile(DAT/'adj.bin');(DAT/'params.txt').write_text(f'10000 1000 {stride} {len(terms)} {astart} {len(incident)}\n');(DAT/'molecules.json').write_text(json.dumps(meta,indent=2))
 r=dict(molecules=1000,conformers=10000,conformers_per_molecule=10,max_heavy_atoms=stride,heavy_atoms_total_all_conformers=astart*10,etkdg_workers=workers,preparation_wall_seconds=elapsed,preparation_timing_scope='8-worker pool elapsed including fresh ETKDG+jitter and per-molecule parameter extraction; excludes subsequent flattening/binary serialization',etkdg_plus_jitter_sum_worker_seconds=sum(m['etkdg_seconds'] for m in items),retry_molecules=sum(m['retry'] for m in items),bad_charge_atoms=sum(m['bad_charge_atoms'] for m in items),input='fresh ETKDGv3(seed20261124+row), ten conformers per molecule, never MMFF optimized; deterministic Gaussian sigma0.06A positional perturbation added',force_field='original simplified united-heavy-atom model: bond k400/r0 MMFF where available; cosangle k40/theta0 MMFF where available; torsion k0.2 with period2/3; softcore full LJ eps0.5 sigma0.85*(rvdw_i+rvdw_j) and Coulomb >=3 graph bonds, Gasteiger charges folded from H; energy kJ/mol, length A',not_comparable_to_mmff94='original66.607885s singlecore/9.986772s8core includes new ETKDG plus actual RDKit MMFF94, not this model',topology_terms=dict(zip(['bond','angle','torsion','nonbonded'],map(int,np.bincount(ta['ints'][:,0],minlength=4)))) ,hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in DAT.glob('*.bin')})
 (RES/'module14_preparation.json').write_text(json.dumps(r,indent=2));print(json.dumps(r,indent=2),flush=True)

def vector_reference(x,terms):
 # Independent vectorized energy only, for finite differences of full FF.
 E=0.
 for ii,p in zip(terms['ints'],terms['params']):
  typ,i,j,k,l,period=ii;a=x[i]-x[j]
  if typ==0:E+=.5*p[0]*(np.sqrt(a@a+1e-20)-p[1])**2
  elif typ==3:
   rr=np.sqrt(a@a+.01);sr6=(p[1]/rr)**6;E+=4*p[0]*(sr6*sr6-sr6)+p[2]/rr
  elif typ==1:
   b=x[k]-x[j];co=(a@b)/np.sqrt((a@a+1e-20)*(b@b+1e-20));E+=.5*p[0]*(co-p[1])**2
  else:
   b=x[k]-x[j];c=x[l]-x[k];n1=np.cross(a,b);n2=np.cross(b,c);co=(n1@n2)/np.sqrt((n1@n1+1e-16)*(n2@n2+1e-16));E+=p[0]*(1+4*co**3-3*co) if period==3 else 2*p[0]*(1-co*co)
 return E

def validate_gradients():
 nc,nm,stride,*_=map(int,(DAT/'params.txt').read_text().split());mols=np.fromfile(DAT/'mols.bin','<i4').reshape(-1,4);terms=np.fromfile(DAT/'terms.bin',TD);initial=np.fromfile(DAT/'initial.bin').reshape(nc,stride,3);checks=[]
 for mi in [0,1,48]:
  ci=mi*10;folder=RES/f'module14_gradient_{mi}';guard([BUILD/'conformer_cpu',DAT,folder,'gradient',ci],f'gradient_{mi}');n,start,count,ast=mols[mi];xx=initial[ci,:n].copy();ts=terms[start:start+count];g=np.fromfile(folder/'gradient.bin').reshape(n,3);analytic_E=float((folder/'energy.txt').read_text());reference_E=vector_reference(xx,ts);assert abs(analytic_E-reference_E)<1e-9
  for atom,axis in [(0,0),(0,1),(0,2),(min(n-1,2),0),(min(n-1,2),1),(min(n-1,2),2)]:
   h=2e-4;ener=[]
   for step in [-2,-1,1,2]:
    xp=xx.copy();xp[atom,axis]+=step*h;ener.append(vector_reference(xp,ts))
   fd=(ener[0]-8*ener[1]+8*ener[2]-ener[3])/(12*h);checks.append(dict(molecule=mi,atom=atom,axis=axis,finite_difference=float(fd),analytic=float(g[atom,axis]),abs_error=float(abs(fd-g[atom,axis]))))
 r=dict(method='independent NumPy energy, five-point finite difference h=2e-4 A vs common C++ analytic gradients',directions=len(checks),max_abs_error=max(z['abs_error'] for z in checks),checks=checks);assert r['max_abs_error']<1e-8,r;(RES/'module14_gradient_validation.json').write_text(json.dumps(r,indent=2));return r

def compile_all(cpu_only=False):
 run(['g++','-std=c++17','-O3','-ffp-contract=off',SRC/'conformer.cpp','-o',BUILD/'conformer_cpu'])
 if not cpu_only:run(['nvcc','-x','cu','-std=c++17','-O3','--fmad=false','-arch=sm_121',SRC/'conformer.cpp','-o',BUILD/'conformer_cuda'])

def execute(cpu_only=False,gpu_only=False):
 compile_all(cpu_only);validate_gradients()
 if not gpu_only:guard([BUILD/'conformer_cpu',DAT,RES/'module14_cpu'],'cpu')
 if cpu_only:return
 guard([BUILD/'conformer_cuda',DAT,RES/'module14_cuda'],'cuda');run(['scp','-r',SRC,'amd9070:C:/AI4S-HIP-Lab/']);run(['ssh','amd9070','cmd /c "mkdir C:\\AI4S-HIP-Lab\\nvmolkit_batched_conformer\\data 2>nul & exit /b 0"']);run(['scp',*sorted(DAT.glob('*.bin')),DAT/'params.txt','amd9070:C:/AI4S-HIP-Lab/nvmolkit_batched_conformer/data/']);run(['ssh','amd9070','C:\\AI4S-HIP-Lab\\nvmolkit_batched_conformer\\build_windows.cmd']);run(['ssh','amd9070','C:\\AI4S-HIP-Lab\\nvmolkit_batched_conformer\\run_windows.cmd']);dst=RES/'module14_hip';dst.mkdir(exist_ok=True);run(['scp','-r','amd9070:C:/AI4S-HIP-Lab/nvmolkit_batched_conformer/out_hip/.',dst]);run(['scp','amd9070:C:/AI4S-HIP-Lab/nvmolkit_batched_conformer/build.log',SRC/'build_windows.log'])

def summarize():
 nc,nm,stride,nt,nap,nadj=map(int,(DAT/'params.txt').read_text().split());mols=np.fromfile(DAT/'mols.bin','<i4').reshape(-1,4);terms=np.fromfile(DAT/'terms.bin',TD);initial=np.fromfile(DAT/'initial.bin').reshape(nc,stride,3);r=dict(preparation=json.loads((RES/'module14_preparation.json').read_text()),gradient_validation=json.loads((RES/'module14_gradient_validation.json').read_text()),backends={},comparison={});mask=np.zeros((nc,stride,3),bool)
 for j,m in enumerate(mols):mask[j*10:j*10+10,:m[0]]=True
 coords={};hists={}
 for b in ['cpu','cuda','hip']:
  path=RES/f'module14_{b}';tim=json.loads((path/'timings.json').read_text());xx=np.fromfile(path/'coordinates.bin').reshape(nc,stride,3);hh=np.fromfile(path/'history.bin').reshape(nc,201);coords[b]=xx;hists[b]=hh;bond_pre=[];bond_post=[];coll_pre=coll_post=0;max_inc=float(np.max(np.diff(hh,axis=1)));finite=bool(np.isfinite(xx[mask]).all() and np.isfinite(hh).all());assert finite;assert max_inc<1e-9,(b,max_inc)
  for j,m in enumerate(mols):
   n,start,count,ast=m;bef=initial[j*10:j*10+10,:n];aft=xx[j*10:j*10+10,:n]
   for t in terms[start:start+count]:
    typ,i,k,_,_,_=t['ints'];p=t['params']
    if typ==0:bond_pre.extend(abs(np.linalg.norm(bef[:,i]-bef[:,k],axis=-1)-p[1]));bond_post.extend(abs(np.linalg.norm(aft[:,i]-aft[:,k],axis=-1)-p[1]))
    if typ==3:
     coll_pre+=int((np.linalg.norm(bef[:,i]-bef[:,k],axis=-1)<.8*p[1]).sum());coll_post+=int((np.linalg.norm(aft[:,i]-aft[:,k],axis=-1)<.8*p[1]).sum())
  assert coll_post<=coll_pre,(b,'declared close-contact quality failed',coll_pre,coll_post)
  tim.update(energy_initial_mean=float(hh[:,0].mean()),energy_final_mean=float(hh[:,-1].mean()),energy_sum_drop=float((hh[:,0]-hh[:,-1]).sum()),strict_decrease_count=int((hh[:,-1]<hh[:,0]-1e-10).sum()),nonincrease_count=int((hh[:,-1]<=hh[:,0]+1e-10).sum()),max_step_energy_increase=max_inc,bond_abs_deviation_initial_mean_A=float(np.mean(bond_pre)),bond_abs_deviation_final_mean_A=float(np.mean(bond_post)),bond_abs_deviation_initial_max_A=float(np.max(bond_pre)),bond_abs_deviation_final_max_A=float(np.max(bond_post)),bond_abs_deviation_final_p95_A=float(np.percentile(bond_post,95)),nonbonded_collision_pairs_initial=coll_pre,nonbonded_collision_pairs_final=coll_post,collision_definition='topological graph distance>=3 bonds and Euclidean r<0.8*sigma_ij',all_finite=finite,max_gradient_final_mean=float(np.fromfile(path/'max_gradient.bin').mean()));r['backends'][b]=tim
 for b in ['cuda','hip']:
  de=hists[b][:,-1]-hists['cpu'][:,-1];dx=coords[b][mask]-coords['cpu'][mask];r['comparison'][b+'_vs_cpu']=dict(energy_relative_l2=float(np.linalg.norm(de)/np.linalg.norm(hists['cpu'][:,-1])),coordinate_relative_l2=float(np.linalg.norm(dx)/np.linalg.norm(coords['cpu'][mask])),coordinate_max_abs=float(np.max(abs(dx))),energy_max_abs=float(np.max(abs(de))))
 de=hists['hip'][:,-1]-hists['cuda'][:,-1];dx=coords['hip'][mask]-coords['cuda'][mask];r['comparison']['hip_vs_cuda']=dict(energy_relative_l2=float(np.linalg.norm(de)/np.linalg.norm(hists['cuda'][:,-1])),coordinate_relative_l2=float(np.linalg.norm(dx)/np.linalg.norm(coords['cuda'][mask])),coordinate_max_abs=float(np.max(abs(dx))),energy_max_abs=float(np.max(abs(de))),exact_energy_count=int((de==0).sum()),exact_coordinate_count=int((dx==0).sum()))
 failure=RES/'module14_failed_weak_lj_results.json'
 if failure.exists():
  # Parameter inputs are exactly regenerable; do not fabricate archived solver outputs.
  archived_terms=RES/'module14_failed_weak_lj_data/terms.bin'
  if not archived_terms.exists():
   archived_terms.parent.mkdir(parents=True,exist_ok=True);weak=terms.copy();weak['params'][weak['ints'][:,0]==3,0]=.05;weak.tofile(archived_terms)
  old=json.loads(failure.read_text());current_hash=hashlib.sha256((DAT/'initial.bin').read_bytes()).hexdigest();oldhash=old['preparation']['hashes']['initial.bin'];assert current_hash==oldhash;r['failure_case']=dict(original_epsilon_kj_mol=.05,revised_epsilon_kj_mol=.5,original_contact_initial=old['backends']['cpu']['nonbonded_collision_pairs_initial'],original_contact_final=old['backends']['cpu']['nonbonded_collision_pairs_final'],collision_threshold_changed=False,initial_coordinate_sha256_before=oldhash,initial_coordinate_sha256_after=current_hash,old_terms_sha256=hashlib.sha256((RES/'module14_failed_weak_lj_data/terms.bin').read_bytes()).hexdigest(),revised_terms_sha256=hashlib.sha256((DAT/'terms.bin').read_bytes()).hexdigest(),raw_failure_summary=str(failure.relative_to(ROOT)),explanation='energy descent alone failed contact-quality check; stronger explicitly defined LJ restored it without changing inputs or diagnostic threshold')
 r['status']='PASS'
 (RES/'module14_results.json').write_text(json.dumps(r,indent=2));from render_module14 import render;render();print(json.dumps(r,indent=2),flush=True)

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--stage',choices=['prepare','run','gpu','summarize','all'],default='all');a=ap.parse_args()
 if a.stage in ['prepare','all']:prepare()
 if a.stage in ['run','all']:execute()
 if a.stage=='gpu':execute(gpu_only=True)
 if a.stage in ['summarize','all']:summarize()
if __name__=='__main__':main()
