#!/usr/bin/env python3
"""Original educational plane-wave block Davidson; not VASP/Daviator binary."""
from pathlib import Path
import argparse,json,subprocess,sys,time,hashlib,os
os.environ.setdefault('OPENBLAS_NUM_THREADS','4');os.environ.setdefault('OMP_NUM_THREADS','4')
import numpy as np
ROOT=Path(__file__).resolve().parent;SRC=ROOT/'amd_hip_port/daviator_batched_dft';RES=ROOT/'results';DATA=RES/'module10_data';BUILD=RES/'module10_build';PY=sys.executable

def run(c,**kw):print('RUN',*[str(x) for x in c],flush=True);return subprocess.run([str(x) for x in c],check=True,**kw)
def guard(c,name):return run([PY,ROOT/'guard_process.py','--audit',ROOT/f'logs/module10_{name}_guard.json','--',*c])
def prepare():
 DATA.mkdir(parents=True,exist_ok=True);BUILD.mkdir(exist_ok=True)
 # Hermitian truncated plane-wave Hamiltonian: kinetic plus Fourier components of a periodic cosine lattice.
 g=np.array([(x,y,z) for x in range(-4,5) for y in range(-4,5) for z in range(-4,5)]);g=g[np.lexsort((g[:,2],g[:,1],g[:,0],(g*g).sum(1)))][:384];Hs=[];ks=[]
 delta=g[:,None,:]-g[None,:,:];v=np.zeros((384,384))
 for axis in range(3):
  other=[i for i in range(3) if i!=axis];v+=-.035*((abs(delta[:,:,axis])==1)&(delta[:,:,other]==0).all(-1))
 for b in range(64):
  k=np.array([-.23+.46*b/63,.071*np.sin(b*.17),.053*np.cos(b*.23)]);T=.5*((g+k)**2).sum(1);Hs.append(np.diag(T)+v);ks.append(k)
 H=np.asarray(Hs,dtype='<f8');H.tofile(DATA/'H.bin');np.save(DATA/'G_vectors.npy',g);np.save(DATA/'k_points.npy',ks);(DATA/'params.txt').write_text('64 384 16\n');t=time.perf_counter();e=np.linalg.eigvalsh(H)[:,:16];np.save(DATA/'reference_eigenvalues.npy',e);info={'systems':64,'plane_waves':384,'bands':16,'Hamiltonian':'T=0.5|G+k|²; V(r)=-0.07[cos(x)+cos(y)+cos(z)], reciprocal lattice units, not silicon DFT or self-consistent VASP','eigensolver_reference':'independent numpy.linalg.eigvalsh full dense H','reference_seconds':time.perf_counter()-t,'H_sha256':hashlib.sha256((DATA/'H.bin').read_bytes()).hexdigest()};(DATA/'metadata.json').write_text(json.dumps(info,indent=2));run(['g++','-O3','-std=c++17','-ffp-contract=off',SRC/'davidson.cpp','-o',BUILD/'davidson_cpu']);guard([BUILD/'davidson_cpu',DATA,RES/'module10_cpu','cpu','100'],'cpu')
def gpus():
 run(['nvcc','-x','cu','-std=c++17','-O3','--fmad=false','-arch=sm_121',SRC/'davidson.cpp','-o',BUILD/'davidson_cuda'])
 for mode in ['seq','batch']:guard([BUILD/'davidson_cuda',DATA,RES/f'module10_cuda_{mode}',mode,'100'],f'cuda_{mode}')
 (SRC/'build_windows.cmd').write_text('@echo off\nsetlocal\ncd /d %~dp0\nset "ROCM=C:\\Strata-Lab\\Strata\\.rocm-win\\Lib\\site-packages\\_rocm_sdk_devel"\ncall "D:\\DLSSNR-Lab\\build-tools\\VC\\Auxiliary\\Build\\vcvars64.bat" >nul || exit /b 1\nset "HIP_PLATFORM=amd"\nset "HIP_PATH=%ROCM%"\nset "ROCM_PATH=%ROCM%"\nset "PATH=%ROCM%\\bin;%ROCM%\\lib\\llvm\\bin;%PATH%"\n"%ROCM%\\lib\\llvm\\bin\\clang++.exe" -x hip --offload-arch=gfx1201 --rocm-path="%ROCM%" --rocm-device-lib-path="%ROCM%\\lib\\llvm\\amdgcn\\bitcode" -std=c++17 -O3 -ffp-contract=off -I"%ROCM%\\include" davidson.cpp -o davidson.exe -L"%ROCM%\\lib" -lamdhip64 > build.log 2>&1\nexit /b %ERRORLEVEL%\n')
 (SRC/'run_windows.cmd').write_text('@echo off\nsetlocal\ncd /d %~dp0\nset "ROCM=C:\\Strata-Lab\\Strata\\.rocm-win\\Lib\\site-packages\\_rocm_sdk_devel"\nset "PATH=%ROCM%\\bin;%ROCM%\\lib\\llvm\\bin;%PATH%"\ndavidson.exe data out_seq seq 100 > seq.log 2>&1\nif errorlevel 1 exit /b 1\ndavidson.exe data out_batch batch 100 > batch.log 2>&1\nexit /b %ERRORLEVEL%\n')
 run(['scp','-r',SRC,'amd9070:C:/AI4S-HIP-Lab/']);run(['ssh','amd9070','cmd /c "mkdir C:\\AI4S-HIP-Lab\\daviator_batched_dft\\data 2>nul & exit /b 0"']);run(['scp',DATA/'H.bin',DATA/'params.txt','amd9070:C:/AI4S-HIP-Lab/daviator_batched_dft/data/']);run(['ssh','amd9070','C:\\AI4S-HIP-Lab\\daviator_batched_dft\\build_windows.cmd']);run(['ssh','amd9070','C:\\AI4S-HIP-Lab\\daviator_batched_dft\\run_windows.cmd'])
 for mode in ['seq','batch']:
  for suffix in ['eigenvalues.bin','eigenvectors.bin','timings.json','history.csv']:run(['scp',f'amd9070:C:/AI4S-HIP-Lab/daviator_batched_dft/out_{mode}_{suffix}',RES/f'module10_hip_{mode}_{suffix}'])
 for name in ['build.log','seq.log','batch.log']:run(['scp',f'amd9070:C:/AI4S-HIP-Lab/daviator_batched_dft/{name}',SRC/name])
def summary():
 ref=np.load(DATA/'reference_eigenvalues.npy');H=np.fromfile(DATA/'H.bin','<f8').reshape(64,384,384);r={'status':'PASS','metadata':json.loads((DATA/'metadata.json').read_text()),'backends':{},'scope':'GPU computes H·V, V^T HV projection, V*C/HV*C Ritz reconstruction, residual and diagonal preconditioner in batched kernels. Small-subspace Jacobi diagonalization and two-pass MGS remain CPU. Includes per-iteration transfers. Sequential makes 64 launches per stage; batch one per stage. Not NVIDIA Daviator or full DFT.'}
 for name in ['cpu','cuda_seq','cuda_batch','hip_seq','hip_batch']:
  pre=RES/f'module10_{name}';e=np.fromfile(str(pre)+'_eigenvalues.bin','<f8').reshape(64,16);x=np.fromfile(str(pre)+'_eigenvectors.bin','<f8').reshape(64,384,16);rr=H@x-x*e[:,None,:];err=float(abs(e-ref).max());res=float(np.linalg.norm(rr,axis=1).max());orth=float(abs(np.swapaxes(x,1,2)@x-np.eye(16)).max());tim=json.loads(Path(str(pre)+'_timings.json').read_text());tim.update({'eigenvalue_max_abs_error':err,'independent_full_H_max_residual':res,'orthogonality_max_abs_error':orth});assert err<1e-10,(name,err);assert res<1e-7,(name,res);r['backends'][name]=tim
 r['hip_vs_cuda_eigenvalue_max_abs_diff']={mode:float(np.max(np.abs(np.fromfile(RES/f'module10_hip_{mode}_eigenvalues.bin','<f8')-np.fromfile(RES/f'module10_cuda_{mode}_eigenvalues.bin','<f8')))) for mode in ['seq','batch']}
 r['batch_speedups']={b:{'end_to_end':r['backends'][b+'_seq']['total_seconds']/r['backends'][b+'_batch']['total_seconds'],'GPU_core_only':r['backends'][b+'_seq']['H_apply_seconds']/r['backends'][b+'_batch']['H_apply_seconds']} for b in ['cuda','hip']}
 previous=json.loads((RES/'module10_results.json').read_text()) if (RES/'module10_results.json').exists() else {}
 if 'resident_upgrade_module15' in previous:r['resident_upgrade_module15']=previous['resident_upgrade_module15']
 (RES/'module10_results.json').write_text(json.dumps(r,indent=2));render(r)
def render(r):
 import matplotlib;matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 plt.rcParams.update({'font.family':'WenQuanYi Micro Hei','svg.fonttype':'none','font.size':12,'figure.facecolor':'white','axes.facecolor':'white'})
 fig,ax=plt.subplots(3,1,figsize=(6.4,11));fig.suptitle('周期晶格本征值：批处理 Block-Davidson',fontsize=17)
 for name in ['cpu','cuda_batch','hip_batch']:
  p=np.genfromtxt(RES/f'module10_{name}_history.csv',delimiter=',',names=True);ax[0].semilogy(p['iteration'],p['max_residual'],marker='o',label=name)
 ax[0].axhline(1e-7,color='gray',ls='--');ax[0].set(xlabel='实际迭代轮数',ylabel='最大本征对残差');ax[0].legend()
 names=list(r['backends'])+['NumPy全eigh参考'];ax[1].barh(names,[r['backends'][n]['total_seconds'] for n in list(r['backends'])]+[r['metadata']['reference_seconds']],color=['#94a3b8','#60a5fa','#2563eb','#fca5a5','#dc2626','#64748b']);ax[1].set(xlabel='端到端秒（含CPU子空间求解与传输）')
 e=np.load(DATA/'reference_eigenvalues.npy');ax[2].plot(np.arange(64),e[:,:8]);ax[2].set(xlabel='64个合成k点（非硅材料能带）',ylabel='最低8条能带 / 无量纲')
 fig.tight_layout(rect=(0,.08,1,.96));fig.text(.04,.035,'64系统 × 384平面波 × 16本征值；独立NumPy全矩阵参考\nGPU批处理投影/Ritz/残差/预条件，CPU子空间eigh与正交化。',fontsize=11)
 out=ROOT/'visuals/fig9_daviator_batched_davidson_vasp_band.svg';fig.savefig(out,facecolor='#ffffff');plt.close(fig);s=out.read_text();s=s.replace('<svg ', '<svg font-family="\'WenQuanYi Micro Hei\',\'Noto Sans CJK SC\',sans-serif" ',1).replace('width="460.8pt"','width="640"').replace('height="792pt"','height="1100"');out.write_text(s)
 import cairosvg;cairosvg.svg2png(url=str(out),write_to=str(out.with_suffix('.png')))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--prepare-only',action='store_true');p.add_argument('--gpu-only',action='store_true');p.add_argument('--summarize-only',action='store_true');a=p.parse_args()
 if not a.gpu_only and not a.summarize_only:prepare()
 if a.prepare_only:sys.exit(0)
 if not a.summarize_only:gpus()
 summary()
