#!/usr/bin/env python3
"""Original MIT CPU/CUDA/HIP nonbonded benchmark; not libcupo compatible.
Use --prepare-only while module six owns GPU. Default executes the full benchmark.
"""
from pathlib import Path
import argparse, json, hashlib, subprocess, sys, math, shutil, time
import numpy as np
ROOT=Path(__file__).resolve().parent
SRC=ROOT/'amd_hip_port/cupotential_open'
RES=ROOT/'results';DAT=RES/'module7_data';BUILD=RES/'module7_build'
PY=sys.executable

def run(cmd,**kw):
 print('RUN',*[str(x) for x in cmd],flush=True)
 return subprocess.run([str(x) for x in cmd],check=True,**kw)

def guard(cmd,name):return run([PY,ROOT/'guard_process.py','--audit',ROOT/f'logs/module7_{name}_memory.json','--',*cmd])

def write_system(folder,atoms,box,rc=10.,erf=78.3):
 folder.mkdir(parents=True,exist_ok=True);np.asarray(atoms,dtype='<f8').tofile(folder/'atoms.bin');(folder/'params.txt').write_text(f'{len(atoms)} {box:.17g} {rc:.17g} {erf:.17g}\n')

def generate(n,spacing=3.5,seed=20261123):
 rng=np.random.default_rng(seed);side=int(np.ceil(n**(1/3)));side+=int(side**3<n)
 # Jittered lattice, >= spacing-0.3A separation, no divergent close contacts.
 grid=np.indices((side,side,side)).reshape(3,-1).T[:n]
 pos=(grid+.5)*spacing+rng.uniform(-.15,.15,(n,3));q=rng.uniform(-.15,.15,n);q-=q.mean()
 atoms=np.column_stack([pos,q,rng.uniform(2.8,3.2,n),rng.uniform(.12,.28,n)]).astype('<f8');return atoms,side*spacing

def brute(atoms,L,rc=10.,erf=78.3):
 # Independent NumPy all-pairs implementation, not the C++ cell-list helper.
 d=atoms[:,None,:3]-atoms[None,:,:3];d-=L*np.floor(d/L+.5);r2=(d*d).sum(-1);mask=(r2>0)&(r2<rc*rc)
 safe=np.where(mask,r2,1.);r=np.sqrt(safe);sigma=(atoms[:,None,4]+atoms[None,:,4])/2;eps=np.sqrt(atoms[:,None,5]*atoms[None,:,5]);sr=sigma/r;s6=sr**6;s12=sr**12;qq=1389.354576*atoms[:,None,3]*atoms[None,:,3]
 kr=(erf-1)/(2*erf+1)/rc**3;cr=3*erf/(2*erf+1)/rc
 E=np.where(mask,4*eps*(s12-s6)+qq*(1/r+kr*safe-cr),0)
 coeff=np.where(mask,24*eps*(2*s12-s6)/safe+qq*(1/r**3-2*kr),0)
 return E.sum()/2,(coeff[:,:,None]*d).sum(1)

def execute_cpu(folder,out):
 guard([BUILD/'open_nonbonded_cpu',folder,out,'3'],out.name)
 return np.fromfile(out/'energy.bin','<f8'),np.fromfile(out/'forces.bin','<f8').reshape(-1,3)

def validate():
 atoms,L=generate(216,spacing=5.5);folder=DAT/'validation';write_system(folder,atoms,L);energy,forces=execute_cpu(folder,RES/'module7_validation_cpu');be,bf=brute(atoms,L)
 erel=abs(energy.sum()-be)/max(abs(be),1.);frel=np.linalg.norm(forces-bf)/np.linalg.norm(bf)
 # Independent finite difference of total all-pairs energy, away from cutoff.
 d=atoms[:,None,:3]-atoms[None,:,:3];d-=L*np.floor(d/L+.5);dist=np.sqrt((d*d).sum(-1));margin=np.min(np.abs(dist[np.triu_indices(len(atoms),1)]-10))
 h=min(1e-4,float(margin)/100);checks=[]
 for atom,axis in [(0,0),(1,1),(17,2),(53,0),(100,1),(200,2)]:
  plus=atoms.copy();minus=atoms.copy();plus[atom,axis]+=h;minus[atom,axis]-=h;ep,_=brute(plus,L);em,_=brute(minus,L);fd=-(ep-em)/(2*h);actual=forces[atom,axis];checks.append(dict(atom=atom,axis=axis,finite_difference=float(fd),analytic=float(actual),abs_error=float(abs(fd-actual))))
 ferr=max(x['abs_error'] for x in checks);net=np.linalg.norm(forces.sum(0));assert erel<1e-12,(erel,'brute energy');assert frel<1e-12,(frel,'brute force');assert ferr<1e-6,(ferr,'finite difference');assert net<1e-10,(net,'momentum')
 result=dict(atoms=216,box_angstrom=L,cutoff_angstrom=10,cell_side=3,bruteforce_energy_kj_mol=float(be),cell_list_energy_kj_mol=float(energy.sum()),energy_relative_error=erel,force_relative_l2=frel,net_force_l2=float(net),finite_difference_step=h,cutoff_margin=float(margin),finite_difference_max_abs_error=ferr,finite_difference_checks=checks)
 # Periodic translation invariance independently checks wrap and MIC.
 translated=atoms.copy();translated[:,:3]=(translated[:,:3]+[11.3,19.7,5.1])%L;write_system(DAT/'validation_translate',translated,L);te,tf=execute_cpu(DAT/'validation_translate',RES/'module7_validation_translate_cpu');result['periodic_translation_force_relative_l2']=float(np.linalg.norm(tf-forces)/np.linalg.norm(forces));result['periodic_translation_energy_abs_diff']=float(abs(te.sum()-energy.sum()));assert result['periodic_translation_force_relative_l2']<1e-12
 (RES/'module7_cpu_validation.json').write_text(json.dumps(result,indent=2));return result

def compile_cpu():
 BUILD.mkdir(parents=True,exist_ok=True);run(['g++','-std=c++17','-O3','-ffp-contract=off',SRC/'open_nonbonded.cpp','-o',BUILD/'open_nonbonded_cpu'])

def prepare():
 compile_cpu();validate()
 for n in [32000,128000]:
  atoms,L=generate(n);folder=DAT/str(n);write_system(folder,atoms,L);(folder/'metadata.json').write_text(json.dumps(dict(atoms=n,box_angstrom=L,seed=20261123,minimum_spacing_lower_bound_angstrom=3.2,units=dict(position='angstrom',charge='elementary e',sigma='angstrom',epsilon='kJ/mol',force='kJ/(mol angstrom)'),rf_dielectric=78.3,net_charge=float(atoms[:,3].sum()),input_sha256=hashlib.sha256((folder/'atoms.bin').read_bytes()).hexdigest(),workload='synthetic jittered lattice of neutral total charge, NOT a validated molecular topology or solvent trajectory'),indent=2))
  execute_cpu(folder,RES/f'module7_cpu_{n}')

def compile_and_run_gpus():
 run(['nvcc','-x','cu','-std=c++17','-O3','--fmad=false','-arch=sm_121',SRC/'open_nonbonded.cpp','-o',BUILD/'open_nonbonded_cuda'])
 for n in [32000,128000]:guard([BUILD/'open_nonbonded_cuda',DAT/str(n),RES/f'module7_cuda_{n}','7'],f'cuda_{n}')
 # Windows isolated workspace; no mutation of the existing ROCm/Strata installation.
 run(['scp','-r',SRC,'amd9070:C:/AI4S-HIP-Lab/'])
 for n in [32000,128000]:run(['scp','-r',DAT/str(n),'amd9070:C:/AI4S-HIP-Lab/cupotential_open/'])
 run(['ssh','amd9070','C:\\AI4S-HIP-Lab\\cupotential_open\\build_windows.cmd'])
 run(['ssh','amd9070','C:\\AI4S-HIP-Lab\\cupotential_open\\run_windows.cmd'])
 for n in [32000,128000]:
  dst=RES/f'module7_hip_{n}';dst.mkdir(exist_ok=True);run(['scp','-r',f'amd9070:C:/AI4S-HIP-Lab/cupotential_open/out_{n}/.',str(dst)])
 run(['scp','amd9070:C:/AI4S-HIP-Lab/cupotential_open/build.log',str(SRC/'build_windows.log')])
 run(['scp','amd9070:C:/AI4S-HIP-Lab/cupotential_open/run.log',str(SRC/'run_windows.log')])

def summarize():
 r=dict(physics=dict(pair_potential='Lennard-Jones12-6 plus reaction-field Coulomb, Lorentz-Berthelot mixing',cutoff_angstrom=10.,rf_dielectric=78.3,ke_kj_angstrom_mol_e2=1389.354576,lj_shift='none: hard cutoff, LJ energy/force discontinuous at cutoff; finite differences avoid boundary',electrostatics='reaction field, not PME/Ewald; RF energy shifted to zero at cutoff',exclusions='none; synthetic atoms, no bonds/1-4 scaling',pbc='cubic minimum image; cell width >= cutoff and >=3 cells/axis',precision='float64, contraction disabled'),validation=json.loads((RES/'module7_cpu_validation.json').read_text()),systems=[])
 for n in [32000,128000]:
  a=dict(n=n,metadata=json.loads((DAT/str(n)/'metadata.json').read_text()),backends={})
  ce=np.fromfile(RES/f'module7_cpu_{n}'/'energy.bin','<f8');cf=np.fromfile(RES/f'module7_cpu_{n}'/'forces.bin','<f8');
  for backend in ['cpu','cuda','hip']:
   folder=RES/f'module7_{backend}_{n}';tim=json.loads((folder/'timings.json').read_text());e=np.fromfile(folder/'energy.bin','<f8');f=np.fromfile(folder/'forces.bin','<f8');tim['energy_relative_error_vs_cpu']=float(abs(e.sum()-ce.sum())/max(abs(ce.sum()),1.));tim['force_relative_l2_vs_cpu']=float(np.linalg.norm(f-cf)/np.linalg.norm(cf));tim['force_max_abs_diff_vs_cpu']=float(np.max(np.abs(f-cf)));tim['net_force_l2']=float(np.linalg.norm(f.reshape(-1,3).sum(0)));tim['canonical_numpy_total_energy_kj_mol']=float(e.sum());tim['reported_host_total_relative_error_vs_cpu']=float(abs(tim['energy_kj_per_mol']-a['backends'].get('cpu',tim)['energy_kj_per_mol'])/max(abs(tim['energy_kj_per_mol']),1.));tim['energy_array_bitwise_equal_cpu']=bool(np.array_equal(e.view(np.uint64),ce.view(np.uint64)));tim['force_array_bitwise_equal_cpu']=bool(np.array_equal(f.view(np.uint64),cf.view(np.uint64)));assert tim['energy_relative_error_vs_cpu']<1e-12;assert tim['force_relative_l2_vs_cpu']<1e-12;a['backends'][backend]=tim
  ge=np.fromfile(RES/f'module7_cuda_{n}'/'energy.bin','<f8');gf=np.fromfile(RES/f'module7_cuda_{n}'/'forces.bin','<f8');he=np.fromfile(RES/f'module7_hip_{n}'/'energy.bin','<f8');hf=np.fromfile(RES/f'module7_hip_{n}'/'forces.bin','<f8');a['hip_vs_cuda']=dict(energy_relative_error=float(abs(he.sum()-ge.sum())/max(abs(ge.sum()),1.)),force_relative_l2=float(np.linalg.norm(hf-gf)/np.linalg.norm(gf)),force_max_abs_diff=float(np.max(np.abs(hf-gf))));assert a['hip_vs_cuda']['force_relative_l2']<1e-12;r['systems'].append(a)
 (RES/'module7_results.json').write_text(json.dumps(r,indent=2,ensure_ascii=False));from render_module7 import render;render();print(json.dumps(r,indent=2,ensure_ascii=False))

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--prepare-only',action='store_true');ap.add_argument('--gpu-only',action='store_true');ap.add_argument('--summarize-only',action='store_true');a=ap.parse_args()
 if not a.gpu_only and not a.summarize_only:prepare()
 if a.prepare_only:return
 if not a.summarize_only:compile_and_run_gpus()
 summarize()
if __name__=='__main__':main()
