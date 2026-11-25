#!/usr/bin/env python3
"""Same-input CPU/CUDA/HIP spatial reorder / epsilon sqrt benchmark, FP64."""
from pathlib import Path
import argparse,base64,hashlib,json,shutil,subprocess,sys
import numpy as np
import exp5_open_cupotential_hip as base
ROOT=Path(__file__).resolve().parent;RES=ROOT/'results';DAT=RES/'module7_data';OUT=RES/'module7_optimized';SRC=ROOT/'amd_hip_port/cupotential_open';BUILD=RES/'module7_build'

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def run(c,**kw):print('RUN',*map(str,c),flush=True);return subprocess.run(list(map(str,c)),check=True,**kw)
def guard(c,name):return run([sys.executable,ROOT/'guard_process.py','--audit',ROOT/f'logs/module7_opt_{name}_guard.json','--',*c])
def native(backend,data,out):
 out.mkdir(parents=True,exist_ok=True);guard([BUILD/f'open_nonbonded_optimized_{backend}',data,out,'opt','7','5'],out.name+'_'+backend)

def physics():
 # Fresh tiny reference data only; the benchmark input files remain untouched.
 atoms,L=base.generate(216,spacing=5.5);folder=OUT/'validation_input';base.write_system(folder,atoms,L);native('cpu',folder,OUT/'validation_cpu');E,F=base.brute(atoms,L);delta=atoms[:,None,:3]-atoms[None,:,:3];delta-=L*np.floor(delta/L+.5);dist=np.linalg.norm(delta,axis=-1);margin=float(np.min(abs(dist[np.triu_indices(216,1)]-10)));h=min(1e-4,margin/100);fd=[]
 for atom,axis in [(0,0),(1,1),(17,2),(53,0),(100,1),(200,2)]:
  p=atoms.copy();m=atoms.copy();p[atom,axis]+=h;m[atom,axis]-=h;ep,_=base.brute(p,L);em,_=base.brute(m,L);fd.append((atom,axis,-(ep-em)/(2*h)))
 translated=atoms.copy();translated[:,:3]=(translated[:,:3]+[11.3,19.7,5.1])%L;tpath=OUT/'translation_input';base.write_system(tpath,translated,L);native('cpu',tpath,OUT/'translation_cpu');r={}
 for v in ['v0','v1','v2']:
  p=OUT/'validation_cpu'/v;e=np.fromfile(p/'energy.bin');f=np.fromfile(p/'forces.bin').reshape(-1,3);tf=np.fromfile(OUT/'translation_cpu'/v/'forces.bin').reshape(-1,3);te=np.fromfile(OUT/'translation_cpu'/v/'energy.bin');checks=[dict(atom=i,axis=a,finite_difference=float(x),analytic=float(f[i,a]),abs_error=float(abs(x-f[i,a])))for i,a,x in fd];er=abs(e.sum()-E)/max(abs(E),1);fr=np.linalg.norm(f-F)/np.linalg.norm(F);tr=np.linalg.norm(tf-f)/np.linalg.norm(f);net=np.linalg.norm(f.sum(0));assert er<1e-15 and fr<1e-15,(v,er,fr);assert tr<1e-14 and net<1e-10;assert max(x['abs_error']for x in checks)<1e-8
  r[v]=dict(energy_relative_error_vs_independent_bruteforce=float(er),force_relative_l2_vs_independent_bruteforce=float(fr),net_force_l2=float(net),periodic_translation_force_relative_l2=float(tr),periodic_translation_energy_abs_diff=float(abs(te.sum()-e.sum())),finite_difference_step_A=h,cutoff_margin_A=margin,finite_difference_checks=checks)
 (RES/'module7_optimized_physics.json').write_text(json.dumps(r,indent=2));return r

def cpu_and_cuda():
 for n in [32000,128000]:
  actual=sha(DAT/str(n)/'atoms.bin');expected=json.loads((DAT/str(n)/'metadata.json').read_text())['input_sha256'];assert actual==expected
 physics()
 for n in [32000,128000]:native('cpu',DAT/str(n),OUT/'cpu'/str(n))
 for n in [32000,128000]:native('cuda',DAT/str(n),OUT/'cuda'/str(n))

def remote(output_prefix='optimized_soa',run_script='run_optimized_windows.cmd'):
 def powershell(script):
  encoded=base64.b64encode(script.encode('utf-16-le')).decode();return run(['ssh','amd9070','powershell -NoProfile -EncodedCommand '+encoded],capture_output=True,text=True,errors='replace')
 def audit_processes():
  p=powershell('Get-CimInstance Win32_Process | Select-Object Name,ExecutablePath | ConvertTo-Json -Compress');processes=json.loads(p.stdout);(RES/'module7_optimized_remote_processes.json').write_text(p.stdout);games=[x for x in processes if x.get('ExecutablePath') and ('steamapps' in x['ExecutablePath'].lower() or any(n in x['Name'].lower()for n in ['eldenring','cyberpunk','rdr2','witcher','wukong','palworld','kingdomcome']))];assert not games,('no process killed; stop GPU for game',games)
 audit_processes()
 # Fresh remote workspace gets the same fixed input bytes, not a generated replacement workload.
 for n in [32000,128000]:
  folder=f'C:/AI4S-HIP-Lab/cupotential_open/{n}';powershell(f"New-Item -ItemType Directory -Force '{folder}' | Out-Null");run(['scp',DAT/str(n)/'atoms.bin',DAT/str(n)/'params.txt',f'amd9070:{folder}/'])
 commands=["$ErrorActionPreference='Stop';"]
 for n in [32000,128000]:
  h=sha(DAT/str(n)/'atoms.bin');commands.append(f"if ((Get-FileHash 'C:/AI4S-HIP-Lab/cupotential_open/{n}/atoms.bin' -Algorithm SHA256).Hash.ToLower() -ne '{h}') {{throw 'input{n}SHA mismatch'}};")
 powershell(' '.join(commands));dest='amd9070:C:/AI4S-HIP-Lab/cupotential_open/'
 run(['scp',SRC/'open_nonbonded.cpp',SRC/'build_optimized_windows.cmd',SRC/run_script,dest]);run(['ssh','amd9070',r'C:\AI4S-HIP-Lab\cupotential_open\build_optimized_windows.cmd']);audit_processes();run(['ssh','amd9070','C:\\AI4S-HIP-Lab\\cupotential_open\\'+run_script])
 for n in [32000,128000]:
  out=OUT/'hip'/str(n);out.mkdir(parents=True,exist_ok=True);run(['scp','-r',f'amd9070:C:/AI4S-HIP-Lab/cupotential_open/{output_prefix}_{n}/.',out])
 for f in ['build_optimized.log','run_baseline.log' if run_script=='run_baseline_windows.cmd' else 'run_optimized.log']:run(['scp',dest+f,SRC/f])

def compare_arrays(e,f,refe,reff):
 return dict(energy_array_relative_l2=float(np.linalg.norm(e-refe)/np.linalg.norm(refe)),force_array_relative_l2=float(np.linalg.norm(f-reff)/np.linalg.norm(reff)),energy_array_max_abs=float(np.max(abs(e-refe))),force_array_max_abs=float(np.max(abs(f-reff))),energy_array_bitwise_equal=bool(np.array_equal(e.view(np.uint64),refe.view(np.uint64))),force_array_bitwise_equal=bool(np.array_equal(f.view(np.uint64),reff.view(np.uint64))),total_energy_relative_error=float(abs(e.sum()-refe.sum())/max(abs(refe.sum()),1)))

def summarize():
 r=dict(status='PASS',precision='same CPU/CUDA/HIP source, FP64 noFMA/no fastmath',protocol='5 complete warmups +7 measured complete repeats; independently measured index/reorder/force and full pipeline synchronized host wall; allocations I/O/H2D/D2H separate',source_sha256=sha(SRC/'open_nonbonded.cpp'),benchmark_sha256=sha(ROOT/'benchmark_nonbonded_optimized.py'),physics_validation=json.loads((RES/'module7_optimized_physics.json').read_text()),systems=[],common_variant=None,selection_scope='same best requested reorder candidate v1/v2 across both sizes/cards, not a claim to beat baseline v0 on every platform; legacy default remains v0',previous_artifacts='results/module7_optimized_previous; old source/driver/JSON/raw preserved')
 for n in [32000,128000]:
  s=dict(n=n,input_sha256=sha(DAT/str(n)/'atoms.bin'),backends={},comparisons={},previous_single_run_timings={});assert s['input_sha256']==json.loads((DAT/str(n)/'metadata.json').read_text())['input_sha256'];refe=np.fromfile(OUT/'cpu'/str(n)/'v0/energy.bin');reff=np.fromfile(OUT/'cpu'/str(n)/'v0/forces.bin');data={}
  for b in ['cpu','cuda','hip']:
   path=OUT/b/str(n);t=json.loads((path/'timings.json').read_text());s['backends'][b]=t;s['previous_single_run_timings'][b]=json.loads((RES/f'module7_{b}_{n}'/'timings.json').read_text()) if (RES/f'module7_{b}_{n}'/'timings.json').exists() else None
   for v in ['v0','v1','v2']:
    e=np.fromfile(path/v/'energy.bin');f=np.fromfile(path/v/'forces.bin');data[b,v]=(e,f);a=compare_arrays(e,f,refe,reff);assert a['energy_array_relative_l2']<1e-15,(n,b,v,a);assert a['force_array_relative_l2']<1e-15,(n,b,v,a);s['comparisons'][b+'_'+v+'_vs_cpu_v0']=a
   historical=RES/f'module7_{b}_{n}'
   if (historical/'energy.bin').exists() and (historical/'forces.bin').exists():s['comparisons'][b+'_v0_vs_historical']=compare_arrays(*data[b,'v0'],np.fromfile(historical/'energy.bin'),np.fromfile(historical/'forces.bin'))
   else:s['comparisons'][b+'_v0_vs_historical']={'available':False,'reason':'legacy raw not supplied; independent brute and full same-harness CPU reference remain mandatory'}
  for v in ['v0','v1','v2']:s['comparisons'][v+'_HIP_vs_CUDA']=compare_arrays(*data['hip',v],*data['cuda',v])
  r['systems'].append(s)
 # One candidate/config for both sizes and both cards.
 chosen=min(['v1','v2'],key=lambda v:float(np.exp(np.mean([np.log(s['backends'][b]['variants'][v]['pipeline_median_ms'])for s in r['systems']for b in ['cuda','hip']]))));r['common_variant']=chosen
 for s in r['systems']:
  for b,t in s['backends'].items():
   t['selected_variant']=chosen;t['selected_timing']=t['variants'][chosen];t['same_harness_speedup_vs_v0']=t['variants']['v0']['pipeline_median_ms']/t['variants'][chosen]['pipeline_median_ms'];z=np.array(t['variants'][chosen]['pipeline_ms']);t['selected_pipeline_CV_percent']=float(z.std()/z.mean()*100)
 (RES/'module7_optimized_results.json').write_text(json.dumps(r,indent=2));print(json.dumps(r,indent=2))

def compile_local():
 BUILD.mkdir(exist_ok=True)
 guard(['g++','-std=c++17','-O3','-ffp-contract=off',SRC/'open_nonbonded.cpp','-o',BUILD/'open_nonbonded_optimized_cpu'],'compile_cpu')
 guard(['nvcc','-x','cu','-std=c++17','-O3','--fmad=false','-arch=sm_121',SRC/'open_nonbonded.cpp','-o',BUILD/'open_nonbonded_optimized_cuda'],'compile_cuda')

SOA=RES/'module7_optimized_soa'
def soa_local():
 compile_local()
 # Fresh tiny physics outputs, original benchmark binaries unchanged.
 atoms,L=base.generate(216,spacing=5.5);folder=SOA/'validation_input';base.write_system(folder,atoms,L);out=SOA/'validation_cpu';out.mkdir(parents=True,exist_ok=True);guard([BUILD/'open_nonbonded_optimized_cpu',folder,out,'opt','7','5','03'],'soa_physics')
 translated=atoms.copy();translated[:,:3]=(translated[:,:3]+[11.3,19.7,5.1])%L;tfolder=SOA/'translation_input';base.write_system(tfolder,translated,L);tout=SOA/'translation_cpu';tout.mkdir(parents=True,exist_ok=True);guard([BUILD/'open_nonbonded_optimized_cpu',tfolder,tout,'opt','7','5','03'],'soa_translation')
 E,F=base.brute(atoms,L);checks=[];h=1e-4;e=np.fromfile(out/'v3/energy.bin');f=np.fromfile(out/'v3/forces.bin').reshape(-1,3);tforce=np.fromfile(tout/'v3/forces.bin').reshape(-1,3)
 for i,axis in [(0,0),(1,1),(17,2),(53,0),(100,1),(200,2)]:
  ap=atoms.copy();am=atoms.copy();ap[i,axis]+=h;am[i,axis]-=h;ep,_=base.brute(ap,L);em,_=base.brute(am,L);fd=-(ep-em)/(2*h);checks.append(dict(atom=i,axis=axis,finite_difference=float(fd),analytic=float(f[i,axis]),abs_error=float(abs(fd-f[i,axis]))))
 phys=dict(force_relative_l2_vs_independent_bruteforce=float(np.linalg.norm(f-F)/np.linalg.norm(F)),energy_relative_error_vs_independent_bruteforce=float(abs(e.sum()-E)/abs(E)),net_force_l2=float(np.linalg.norm(f.sum(0))),periodic_translation_force_relative_l2=float(np.linalg.norm(f-tforce)/np.linalg.norm(f)),finite_difference_checks=checks);assert phys['force_relative_l2_vs_independent_bruteforce']<1e-15;assert phys['energy_relative_error_vs_independent_bruteforce']<1e-15;assert phys['periodic_translation_force_relative_l2']<1e-14;assert phys['net_force_l2']<1e-10;assert max(q['abs_error']for q in checks)<1e-8;(RES/'module7_optimized_soa_physics.json').write_text(json.dumps(phys,indent=2))
 for n in [32000,128000]:
  assert sha(DAT/str(n)/'atoms.bin')==json.loads((DAT/str(n)/'metadata.json').read_text())['input_sha256']
  if not (OUT/'cpu'/str(n)/'v0/forces.bin').exists():
   reference=OUT/'cpu'/str(n);reference.mkdir(parents=True,exist_ok=True);guard([BUILD/'open_nonbonded_optimized_cpu',DAT/str(n),reference,'opt','7','5','03'],'fresh_cpu_reference_'+str(n))
  output=SOA/'cuda'/str(n);output.mkdir(parents=True,exist_ok=True);guard([BUILD/'open_nonbonded_optimized_cuda',DAT/str(n),output,'opt','7','5','03'],'soa_cuda_'+str(n))

def soa_remote():
 global OUT
 previous=OUT;OUT=SOA
 try:remote()
 finally:OUT=previous

def summarize_soa():
 round1=RES/'module7_optimized_round1_results.json';old=json.loads(round1.read_text()) if round1.exists() else {'systems':[]};r=dict(status='PASS',precision='same source FP64 noFMA, original27cell/same within-cell order; SoA pre-sqrt epsilon matches v2 path',protocol='5complete warmups +7raw repetitions; index+reorder+force complete and stages independently synchronized',source_sha256=sha(SRC/'open_nonbonded.cpp'),benchmark_sha256=sha(ROOT/'benchmark_nonbonded_optimized.py'),physics_validation={'v3':json.loads((RES/'module7_optimized_soa_physics.json').read_text())},previous_round_summary='results/module7_optimized_round1_results.json',unchanged_round1_kernel_prefix=json.loads((RES/'module7_round1_kernel_prefix.json').read_text()) if (RES/'module7_round1_kernel_prefix.json').exists() else {'available':False},systems=[],common_variant='v0',default_variant='v0',candidate_variant='v3')
 for n in [32000,128000]:
  prior=next((x for x in old['systems']if x['n']==n),None);s=dict(n=n,input_sha256=sha(DAT/str(n)/'atoms.bin'),backends={},comparisons={});assert s['input_sha256']==json.loads((DAT/str(n)/'metadata.json').read_text())['input_sha256'];refe=np.fromfile(OUT/'cpu'/str(n)/'v0/energy.bin');reff=np.fromfile(OUT/'cpu'/str(n)/'v0/forces.bin');cpu=json.loads((OUT/'cpu'/str(n)/'timings.json').read_text());cpu['selected_variant']='v0';cpu['selected_timing']=cpu['variants']['v0'];cpu['default_timing']=cpu['variants']['v0'];cpu['reference_scope']='measured5warm7 original-formula CPUv0 reference; no CPUv3 performance substituted';cpu['full_CPU_v3_timing_available']='v3' in cpu['variants'];s['backends']['cpu']=cpu;outputs={}
  for b in ['cuda','hip']:
   folder=SOA/b/str(n);t=json.loads((folder/'timings.json').read_text());s['backends'][b]=t
   for v in ['v0','v3']:
    e=np.fromfile(folder/v/'energy.bin');f=np.fromfile(folder/v/'forces.bin');outputs[b,v]=(e,f);c=compare_arrays(e,f,refe,reff);assert c['energy_array_relative_l2']<1e-15 and c['force_array_relative_l2']<1e-15,(n,b,v,c);s['comparisons'][b+'_'+v+'_vs_cpu_v0']=c
   legacy=OUT/b/str(n)/'v2'
   if (legacy/'energy.bin').exists() and (legacy/'forces.bin').exists():s['comparisons'][b+'_v3_vs_round1_v2']=compare_arrays(*outputs[b,'v3'],np.fromfile(legacy/'energy.bin'),np.fromfile(legacy/'forces.bin'))
   else:s['comparisons'][b+'_v3_vs_round1_v2']={'available':False,'reason':'optional historical v2 raw not supplied; full CPUv0/crossGPU/independent physics checks remain mandatory'}
   t['selected_variant']='v0';t['default_variant']='v0';t['selected_timing']=t['variants']['v0'];t['default_timing']=t['variants']['v0'];t['candidate_variant']='v3';t['candidate_timing']=t['variants']['v3'];t['candidate_speedup_vs_v0']=t['variants']['v0']['pipeline_median_ms']/t['variants']['v3']['pipeline_median_ms'];t['round1_AoS_variants']=prior['backends'][b]['variants'] if prior else {'available':False}
  s['comparisons']['v3_HIP_vs_CUDA']=compare_arrays(*outputs['hip','v3'],*outputs['cuda','v3']);r['systems'].append(s)
 r['recommendation']='retain_v0_default: v3 is a common tested candidate, not an accepted universal optimization. GB10 regresses; AMD modest gains have timing variation. Round1 v2 snapshots remain separate, never mix variants across cards.'
 (RES/'module7_optimized_results.json').write_text(json.dumps(r,indent=2));print(json.dumps(r,indent=2))

def ensure_fixed_inputs():
 expected={32000:'003cec4c76d285d8fbc377f6ebb0de43be5e6c41269d59dcaa54174686ad2a32',128000:'74e421bf4e3929aa8e6f4e2e54ef23a65491f9cc06e8bb7050c08bde8cccfa80'}
 for n,h in expected.items():
  folder=DAT/str(n)
  if not (folder/'atoms.bin').exists():
   atoms,L=base.generate(n);base.write_system(folder,atoms,L)
  assert sha(folder/'atoms.bin')==h,('fixed original input SHA mismatch',n)
  if not (folder/'metadata.json').exists():(folder/'metadata.json').write_text(json.dumps(dict(atoms=n,input_sha256=h,seed=20261123,cutoff_angstrom=10.,rf_dielectric=78.3),indent=2))
  assert json.loads((folder/'metadata.json').read_text())['input_sha256']==h

def round1_ready():
 required=[RES/'module7_optimized_round1_results.json',RES/'module7_optimized_physics.json']
 for b in ['cpu','cuda','hip']:
  for n in [32000,128000]:
   required.append(OUT/b/str(n)/'timings.json')
   for v in ['v0','v1','v2']:
    for f in ['energy.bin','forces.bin']:required.append(OUT/b/str(n)/v/f)
 return all(p.exists() for p in required)

def main():
 p=argparse.ArgumentParser();p.add_argument('--stage',choices=['all','local','remote','summarize','soa','soa-summary'],default='all');a=p.parse_args()
 if a.stage in ['all','local','remote','soa']:
  ensure_fixed_inputs()
 if a.stage in ['all','local','soa']:soa_local()
 if a.stage in ['all','remote','soa']:soa_remote()
 if a.stage in ['all','summarize','soa','soa-summary']:summarize_soa()
if __name__=='__main__':main()
