#!/usr/bin/env python3
"""Steady-state same-input same-FF conformer implementation audit; no root result rewrite."""
import os,sys,json,subprocess,hashlib,shutil,argparse
from pathlib import Path
os.environ.setdefault('OPENBLAS_NUM_THREADS','4');os.environ.setdefault('OMP_NUM_THREADS','1')
import numpy as np
ROOT=Path(__file__).resolve().parent;RES=ROOT/'results';SRC=ROOT/'amd_hip_port/nvmolkit_batched_conformer';DAT=RES/'module14_data';BUILD=RES/'module14_optimized_build';PY=sys.executable
GPU_WARMUPS=5;GPU_REPETITIONS=7

def run(c,**kw):print('RUN',*map(str,c),flush=True);return subprocess.run(list(map(str,c)),check=True,**kw)
def guard(c,n):return run([PY,ROOT/'guard_process.py','--audit',ROOT/f'logs/module14_optimized_{n}_guard.json','--',*c])
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def cpu():
 BUILD.mkdir(exist_ok=True);run(['g++','-std=c++17','-O3','-ffp-contract=off',SRC/'conformer.cpp','-o',BUILD/'conformer_cpu']);guard([BUILD/'conformer_cpu',DAT,RES/'module14_optimized_cpu','benchmark','3'],'cpu');gradients()
def gradients():
 from exp12_open_nvmolkit_batched_conformer_hip import vector_reference,TD
 nc,nm,stride,*_=map(int,(DAT/'params.txt').read_text().split());mols=np.fromfile(DAT/'mols.bin','<i4').reshape(-1,4);terms=np.fromfile(DAT/'terms.bin',TD);initial=np.fromfile(DAT/'initial.bin').reshape(nc,stride,3);checks=[]
 for mi in [0,1,48]:
  ci=mi*10;folder=RES/f'module14_optimized_gradient_{mi}';guard([BUILD/'conformer_cpu',DAT,folder,'gradient',ci],f'gradient_{mi}');n,start,count,ast=mols[mi];xx=initial[ci,:n].copy();ts=terms[start:start+count];g=np.fromfile(folder/'gradient.bin').reshape(n,3);analytic_E=float((folder/'energy.txt').read_text());assert abs(analytic_E-vector_reference(xx,ts))<1e-9
  for atom,axis in [(0,0),(0,1),(0,2),(min(n-1,2),0),(min(n-1,2),1),(min(n-1,2),2)]:
   h=2e-4;ener=[]
   for step in [-2,-1,1,2]:xp=xx.copy();xp[atom,axis]+=step*h;ener.append(vector_reference(xp,ts))
   fd=(ener[0]-8*ener[1]+8*ener[2]-ener[3])/(12*h);checks.append({'molecule':mi,'atom':atom,'axis':axis,'finite_difference':float(fd),'analytic':float(g[atom,axis]),'abs_error':float(abs(fd-g[atom,axis]))})
 r={'directions':18,'max_abs_error':max(x['abs_error'] for x in checks),'checks':checks,'method':'unchanged analytic CPU force vs independent NumPy energy five-point h2e-4'};assert r['max_abs_error']<1e-8;rpath=RES/'module14_optimized_gradient_validation.json';rpath.write_text(json.dumps(r,indent=2));return r

def check_remote_idle():
 cmd='powershell -NoProfile -Command "Get-Process | Select-Object ProcessName,Id,CPU | ConvertTo-Json -Compress"'
 cp=subprocess.run(['ssh','amd9070',cmd],check=True,text=True,stdout=subprocess.PIPE);(RES/'module14_optimized_remote_processes_before.json').write_text(cp.stdout);print('REMOTE PROCESS SNAPSHOT SAVED: inspect before GPU; no process killed',flush=True)
def gpus():
 check_remote_idle()
 run(['nvcc','-x','cu','-std=c++17','-O3','--fmad=false','-arch=sm_121',SRC/'conformer.cpp','-o',BUILD/'conformer_cuda']);guard([BUILD/'conformer_cuda',DAT,RES/'module14_optimized_cuda','benchmark',GPU_REPETITIONS,GPU_WARMUPS],'cuda')
 build=(SRC/'build_windows.cmd').read_text().replace('conformer_hip.exe','conformer_optimized_hip.exe').replace('> build.log','> optimized_build.log');(SRC/'optimized_build_windows.cmd').write_text(build)
 rs=(SRC/'run_windows.cmd').read_text().split('conformer_hip.exe')[0]+f'conformer_optimized_hip.exe data out_optimized_hip benchmark {GPU_REPETITIONS} {GPU_WARMUPS} > optimized_run.log 2>&1\nexit /b %ERRORLEVEL%\n';(SRC/'optimized_run_windows.cmd').write_text(rs)
 run(['scp','-r',SRC,'amd9070:C:/AI4S-HIP-Lab/']);run(['scp',*sorted(DAT.glob('*.bin')),DAT/'params.txt','amd9070:C:/AI4S-HIP-Lab/nvmolkit_batched_conformer/data/']);run(['ssh','amd9070','C:\\AI4S-HIP-Lab\\nvmolkit_batched_conformer\\optimized_build_windows.cmd']);run(['ssh','amd9070','C:\\AI4S-HIP-Lab\\nvmolkit_batched_conformer\\optimized_run_windows.cmd']);dst=RES/'module14_optimized_hip';dst.mkdir(exist_ok=True);run(['scp','-r','amd9070:C:/AI4S-HIP-Lab/nvmolkit_batched_conformer/out_optimized_hip/.',dst]);run(['scp','amd9070:C:/AI4S-HIP-Lab/nvmolkit_batched_conformer/optimized_build.log',SRC/'optimized_build.log']);run(['scp','amd9070:C:/AI4S-HIP-Lab/nvmolkit_batched_conformer/optimized_run.log',SRC/'optimized_run.log'])
 # Fetch and hash actual compiled remote source, not just trusting local scp success.
 remote=RES/'module14_optimized_remote_source.cpp';run(['scp','amd9070:C:/AI4S-HIP-Lab/nvmolkit_batched_conformer/conformer.cpp',remote]);assert sha(remote)==sha(SRC/'conformer.cpp')
def summary():
 from exp12_open_nvmolkit_batched_conformer_hip import TD
 nc,nm,stride,nt,nap,nadj=map(int,(DAT/'params.txt').read_text().split());mols=np.fromfile(DAT/'mols.bin','<i4').reshape(-1,4);terms=np.fromfile(DAT/'terms.bin',TD);initial=np.fromfile(DAT/'initial.bin').reshape(nc,stride,3);mask=np.zeros(initial.shape,bool)
 for j,m in enumerate(mols):mask[j*10:j*10+10,:m[0]]=True
 archive=json.loads((RES/'module14_optimized_archive.json').read_text());old=json.loads((ROOT/archive['history']/'results/module14_results.json').read_text());r={'status':'PASS','inputs':json.loads((RES/'module14_preparation.json').read_text()),'archived_baseline':archive,'source_sha256':sha(SRC/'conformer.cpp'),'gradient_validation':json.loads((RES/'module14_optimized_gradient_validation.json').read_text()),'backends':{},'comparisons':{},'scope':'same epsilon0.5 topology, same inputs, FP64 FMA disabled, same200steps/16backtracks/step clipping; once-per-term cached gradients and energy-only trial evaluation; atom ptr/adj and scalar energy accumulation order preserved. CPU reference unchanged arithmetic.'};coords={};hist={}
 assert r['inputs']['hashes']==old['preparation']['hashes'],'inputs changed'
 trial=ROOT/archive['history']/'optimized_warm1_rep3'
 prefix=lambda p:Path(p).read_text().split('int main(')[0]
 assert prefix(trial/'conformer.cpp')==prefix(SRC/'conformer.cpp'),'force or GPU kernel changed during thermal protocol retest'
 r['thermal_retest']={'first_trial_archive':str(trial.relative_to(ROOT)),'CPU_protocol':'one warmup + 3 repeats unchanged; only GPU retested hot','GPU_protocol':'five complete warmups + seven timed complete optimization runs, same context, same original inputs each run','force_and_GPU_kernel_prefix_sha256':hashlib.sha256(prefix(SRC/'conformer.cpp').encode()).hexdigest(),'only_source_change_after_trial':'main() warmup CLI argument parsing; force/kernel prefix byte-identical'}
 r['root_module14_result_unchanged']=sha(RES/'module14_results.json')==archive['old_result_sha256'];assert r['root_module14_result_unchanged']
 for backend in ['cpu','cuda','hip']:
  folder=RES/f'module14_optimized_{backend}';tim=json.loads((folder/'timings.json').read_text());assert tim['warmup_runs']>=1 and len(tim['raw_seconds'])>=3;assert tim['seconds']==float(np.median(tim['raw_seconds']));xx=np.fromfile(folder/'coordinates.bin').reshape(nc,stride,3);hh=np.fromfile(folder/'history.bin').reshape(nc,201);assert np.isfinite(xx).all() and np.isfinite(hh).all();coords[backend]=xx;hist[backend]=hh;max_increase=float(np.diff(hh,axis=1).max());assert max_increase<1e-9;pre=post=0;bpre=[];bpost=[]
  for j,m in enumerate(mols):
   n,start,count,ast=m;bef=initial[j*10:j*10+10,:n];aft=xx[j*10:j*10+10,:n]
   for term in terms[start:start+count]:
    typ,i,k,_,_,_=term['ints'];p=term['params']
    if typ==0:bpre.extend(abs(np.linalg.norm(bef[:,i]-bef[:,k],axis=-1)-p[1]));bpost.extend(abs(np.linalg.norm(aft[:,i]-aft[:,k],axis=-1)-p[1]))
    if typ==3:pre+=int((np.linalg.norm(bef[:,i]-bef[:,k],axis=-1)<.8*p[1]).sum());post+=int((np.linalg.norm(aft[:,i]-aft[:,k],axis=-1)<.8*p[1]).sum())
  assert (pre,post)==(47,0),(backend,pre,post)
  tim.update({'max_step_energy_increase':max_increase,'collision_pairs_initial':pre,'collision_pairs_final':post,'bond_abs_deviation_initial_mean_A':float(np.mean(bpre)),'bond_abs_deviation_final_mean_A':float(np.mean(bpost)),'mean_initial_energy':float(hh[:,0].mean()),'mean_final_energy':float(hh[:,-1].mean()),'last_output_raw_hashes':{p.name:sha(p) for p in folder.glob('*.bin')},'old_single_run_seconds':old['backends'][backend]['seconds'],'speedup_vs_old_single_run':old['backends'][backend]['seconds']/tim['seconds']})
  if (folder/'coverage.json').exists():tim['coverage']=json.loads((folder/'coverage.json').read_text());assert tim['coverage']['total_conformers']==10000
  if backend=='cpu':
   tim['raw_executable_GPU_launch_width_field']=tim['threads_per_conformer'];tim['threads_per_conformer']=1
  tim['raw_timing_peak_to_peak_percent_of_median']=float((max(tim['raw_seconds'])-min(tim['raw_seconds']))/tim['seconds']*100)
  oldfolder=ROOT/archive['history']/f'results/module14_{backend}';ox=np.fromfile(oldfolder/'coordinates.bin').reshape(nc,stride,3);oh=np.fromfile(oldfolder/'history.bin').reshape(nc,201)
  tim['old_raw_regression']={'coordinates_bitwise_equal':bool(np.array_equal(xx.view(np.uint64),ox.view(np.uint64))),'full_201step_history_bitwise_equal':bool(np.array_equal(hh.view(np.uint64),oh.view(np.uint64)))}
  assert tim['old_raw_regression']['coordinates_bitwise_equal'] and tim['old_raw_regression']['full_201step_history_bitwise_equal'],(backend,'changed old trajectory')
  r['backends'][backend]=tim
 for a,b in [('cuda','cpu'),('hip','cpu'),('hip','cuda')]:
  dx=coords[a][mask]-coords[b][mask];dh=hist[a]-hist[b];d={'coordinate_relative_l2':float(np.linalg.norm(dx)/np.linalg.norm(coords[b][mask])),'all_201step_history_relative_l2':float(np.linalg.norm(dh)/np.linalg.norm(hist[b])),'coordinate_max_abs':float(abs(dx).max()),'history_max_abs':float(abs(dh).max()),'coordinates_bitwise_equal':bool(np.array_equal(coords[a][mask].view(np.uint64),coords[b][mask].view(np.uint64))),'full_histories_bitwise_equal':bool(np.array_equal(hist[a].view(np.uint64),hist[b].view(np.uint64)))};assert d['coordinate_relative_l2']<1e-13 and d['all_201step_history_relative_l2']<1e-13,(a,b,d);r['comparisons'][f'{a}_vs_{b}']=d
 r['source_remote_sha256']=sha(RES/'module14_optimized_remote_source.cpp');assert r['source_remote_sha256']==r['source_sha256'];(RES/'module14_optimized_results.json').write_text(json.dumps(r,indent=2));print(json.dumps({k:r[k] for k in ['status','source_sha256','comparisons']},indent=2))
 # Explicitly synchronize implementation+driver to the separate AMD repo, no git/index changes.
 AMD=Path('/home/lmxxf/work/ai-for-science-on-amd-9070xt');(AMD/'amd_hip_port/nvmolkit_batched_conformer').mkdir(exist_ok=True)
 for f in ['conformer.cpp','optimized_build_windows.cmd','optimized_run_windows.cmd']:shutil.copy2(SRC/f,AMD/'amd_hip_port/nvmolkit_batched_conformer'/f)
 for f in ['benchmark_conformer_optimized.py','exp12_open_nvmolkit_batched_conformer_hip.py']:shutil.copy2(ROOT/f,AMD/f)
 (AMD/'results').mkdir(exist_ok=True);shutil.copy2(RES/'module14_optimized_results.json',AMD/'results/module14_optimized_results.json');r['AMD_repo_source_sha256']=sha(AMD/'amd_hip_port/nvmolkit_batched_conformer/conformer.cpp');assert r['AMD_repo_source_sha256']==r['source_sha256'];(RES/'module14_optimized_results.json').write_text(json.dumps(r,indent=2));shutil.copy2(RES/'module14_optimized_results.json',AMD/'results/module14_optimized_results.json')
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--stage',choices=['cpu','gpu','summary','all'],default='all');p.add_argument('--gpu-warmups',type=int,default=5);p.add_argument('--gpu-repetitions',type=int,default=7);a=p.parse_args();GPU_WARMUPS=a.gpu_warmups;GPU_REPETITIONS=a.gpu_repetitions;BUILD.mkdir(exist_ok=True)
 if a.stage in ['cpu','all']:cpu()
 if a.stage in ['gpu','all']:gpus()
 if a.stage in ['summary','all']:summary()
