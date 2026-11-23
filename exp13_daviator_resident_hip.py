#!/usr/bin/env python3
"""All-on-GPU Block-Davidson including shared Jacobi and two-pass MGS."""
from pathlib import Path
import os,sys,argparse,subprocess,json,time,hashlib
os.environ.setdefault('OMP_NUM_THREADS','4');os.environ.setdefault('OPENBLAS_NUM_THREADS','4')
import numpy as np
ROOT=Path(__file__).resolve().parent;SRC=ROOT/'amd_hip_port/daviator_batched_dft';RES=ROOT/'results';DATA=RES/'module15_data';BUILD=RES/'module15_build';PY=sys.executable

def run(c,**kw):print('RUN',*[str(x) for x in c],flush=True);return subprocess.run([str(x) for x in c],check=True,**kw)
def guard(c,n):return run([PY,ROOT/'guard_process.py','--audit',ROOT/f'logs/module15_{n}_guard.json','--',*c])
def prepare():
 DATA.mkdir(exist_ok=True);BUILD.mkdir(exist_ok=True)
 g=np.array([(x,y,z) for x in range(-4,5) for y in range(-4,5) for z in range(-4,5)]);g=g[np.lexsort((g[:,2],g[:,1],g[:,0],(g*g).sum(1)))][:384];delta=g[:,None]-g[None,:];pot=np.zeros((384,384))
 for ax in range(3):pot+=-.035*((abs(delta[:,:,ax])==1)&(delta[:,:,[i for i in range(3) if i!=ax]]==0).all(-1))
 for B in [64,256]:
  folder=DATA/str(B);folder.mkdir(exist_ok=True);H=[]
  for b in range(B):
   k=np.array([-.23+.46*b/(B-1),.071*np.sin(b*.17),.053*np.cos(b*.23)]);H.append(np.diag(.5*((g+k)**2).sum(1))+pot)
  H=np.asarray(H,dtype='<f8');H.tofile(folder/'H.bin');(folder/'params.txt').write_text(f'{B} 384 16\n');st=time.perf_counter();ref=np.linalg.eigvalsh(H)[:,:16];np.save(folder/'reference_eigenvalues.npy',ref);(folder/'metadata.json').write_text(json.dumps({'systems':B,'plane_waves':384,'bands':16,'CPU_numpy_full_eigvalsh_seconds':time.perf_counter()-st,'H_sha256':hashlib.sha256((folder/'H.bin').read_bytes()).hexdigest(),'Hamiltonian':'0.5|G+k|² + periodic cosine potential V=-0.07(cos x+cos y+cos z), same model as module10; not self-consistent DFT or silicon'},indent=2))
 # CPU educational iterative baseline retained for context; max subspace96 differs from resident fixed32.
 run(['g++','-std=c++17','-O3','-ffp-contract=off',SRC/'davidson.cpp','-o',BUILD/'davidson_cpu'])
 for B in [64,256]:guard([BUILD/'davidson_cpu',DATA/str(B),RES/f'module15_cpu_{B}','cpu','100'],f'cpu_{B}')
def gpu():
 run(['nvcc','-x','cu','-std=c++17','-O3','--fmad=false','-arch=sm_121',SRC/'resident.cpp','-o',BUILD/'resident_cuda'])
 for B in [64,256]:
  for mode in ['seq','batch']:guard([BUILD/'resident_cuda',DATA/str(B),RES/f'module15_cuda_{B}_{mode}',mode,'100'],f'cuda_{B}_{mode}')
 build=(SRC/'build_windows.cmd').read_text().replace('davidson.cpp','resident.cpp').replace('davidson.exe','resident.exe').replace('> build.log','> resident_build.log')
 (SRC/'resident_build_windows.cmd').write_text(build)
 runscript=(SRC/'run_windows.cmd').read_text().split('davidson.exe')[0]
 for B in [64,256]:
  for mode in ['seq','batch']:runscript+=f'resident.exe data{B} resident_{B}_{mode} {mode} 100 >> resident_run.log 2>&1\nif errorlevel 1 exit /b 1\n'
 runscript+='exit /b %ERRORLEVEL%\n';(SRC/'resident_run_windows.cmd').write_text(runscript)
 run(['scp','-r',SRC,'amd9070:C:/AI4S-HIP-Lab/'])
 for B in [64,256]:
  run(['ssh','amd9070',f'cmd /c "mkdir C:\\AI4S-HIP-Lab\\daviator_batched_dft\\data{B} 2>nul & exit /b 0"']);run(['scp',DATA/str(B)/'H.bin',DATA/str(B)/'params.txt',f'amd9070:C:/AI4S-HIP-Lab/daviator_batched_dft/data{B}/'])
 run(['ssh','amd9070','C:\\AI4S-HIP-Lab\\daviator_batched_dft\\resident_build_windows.cmd']);run(['ssh','amd9070','C:\\AI4S-HIP-Lab\\daviator_batched_dft\\resident_run_windows.cmd'])
 for B in [64,256]:
  for mode in ['seq','batch']:
   for suffix in ['eigenvalues.bin','eigenvectors.bin','timings.json','history.csv']:run(['scp',f'amd9070:C:/AI4S-HIP-Lab/daviator_batched_dft/resident_{B}_{mode}_{suffix}',RES/f'module15_hip_{B}_{mode}_{suffix}'])
 for f in ['resident_build.log','resident_run.log']:run(['scp',f'amd9070:C:/AI4S-HIP-Lab/daviator_batched_dft/{f}',SRC/f])
def summarize():
 r={'status':'PASS','method':'GPU persistent kernel: each block owns one k-point, all iterations and convergence control on device, shared-memory 32x32 cyclic Jacobi, two-pass modified Gram-Schmidt, thick-restarted 32-vector block Davidson retaining 24 Ritz vectors (8 guard roots) plus 8 largest target residual directions; 16 requested roots; no host interaction during iteration. Sequential launches one block per k-point in same stream, batch one grid covering all k-points.','iteration_host_transfers':0,'iteration_host_eigensolves':0,'systems':[],'caveats':['Original educational implementation, not NVIDIA Daviator or VASP. Simple periodic potential, no self-consistency.','CPU educational Davidson uses max96 subspace, resident uses thick-restarted32 (24 retained Ritz + 8 corrections); CPU full eigvalsh also provided as optimized reference.','GPU resident kernel time excludes upload/allocation and final result download; total_seconds includes them. Neither includes file I/O or compilation.']}
 for B in [64,256]:
  folder=DATA/str(B);H=np.fromfile(folder/'H.bin','<f8').reshape(B,384,384);ref=np.load(folder/'reference_eigenvalues.npy');entry={'B':B,'metadata':json.loads((folder/'metadata.json').read_text()),'backends':{}}
  for name in ['cpu','cuda_seq','cuda_batch','hip_seq','hip_batch']:
   prefix=RES/f'module15_cpu_{B}' if name=='cpu' else RES/f'module15_{name.split("_")[0]}_{B}_{name.split("_")[1]}'
   e=np.fromfile(str(prefix)+'_eigenvalues.bin','<f8').reshape(B,16);x=np.fromfile(str(prefix)+'_eigenvectors.bin','<f8').reshape(B,384,16);assert np.isfinite(e).all() and np.isfinite(x).all();err=float(abs(e-ref).max());res=float(np.linalg.norm(H@x-x*e[:,None,:],axis=1).max());orth=float(abs(np.swapaxes(x,1,2)@x-np.eye(16)).max());tim=json.loads(Path(str(prefix)+'_timings.json').read_text());tim.update({'eigenvalue_max_abs_error':err,'independent_full_H_max_residual':res,'orthogonality_max_abs_error':orth});assert err<1e-13,(B,name,'eigen',err);assert res<1e-7,(B,name,'residual',res);assert orth<1e-10,(B,name,'orth',orth);entry['backends'][name]=tim
  entry['batch_speedups']={b:{'resident_kernel':entry['backends'][b+'_seq']['resident_kernel_seconds']/entry['backends'][b+'_batch']['resident_kernel_seconds'],'end_to_end':entry['backends'][b+'_seq']['total_seconds']/entry['backends'][b+'_batch']['total_seconds']} for b in ['cuda','hip']};entry['hip_vs_cuda_eigenvalue_max_abs_diff']={m:float(np.max(abs(np.fromfile(RES/f'module15_hip_{B}_{m}_eigenvalues.bin','<f8')-np.fromfile(RES/f'module15_cuda_{B}_{m}_eigenvalues.bin','<f8')))) for m in ['seq','batch']};r['systems'].append(entry)
 (RES/'module15_results.json').write_text(json.dumps(r,ensure_ascii=False,indent=2));old=json.loads((RES/'module10_results.json').read_text());old['resident_upgrade_module15']=r;(RES/'module10_results.json').write_text(json.dumps(old,ensure_ascii=False,indent=2));render(r)
def render(r):
 import matplotlib;matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 plt.rcParams.update({'font.family':'WenQuanYi Micro Hei','svg.fonttype':'none','font.size':11,'figure.facecolor':'white','axes.facecolor':'white'})
 fig,axs=plt.subplots(4,1,figsize=(6.4,14));fig.suptitle('Block-Davidson：从混合版到全GPU驻留',fontsize=18)
 for backend in ['cuda','hip']:
  p=np.genfromtxt(RES/f'module15_{backend}_64_batch_history.csv',delimiter=',',names=True);axs[0].semilogy(p['iteration'],p['max_residual'],label=backend,marker='o',ms=3)
 axs[0].axhline(1e-7,color='gray',ls='--');axs[0].axhline(1e-10,color='#b91c1c',ls=':',label='设备阈值1e-10');axs[0].set(xlabel='真实迭代轮数（32维厚重启：24Ritz+8修正）',ylabel='全部系统最大残差');axs[0].legend()
 for ax,entry in zip(axs[1:3],r['systems']):
  names=['cpu','cuda_seq','cuda_batch','hip_seq','hip_batch'];times=[entry['backends'][n]['total_seconds'] for n in names];ax.barh(names+['NumPy全eigh参考'],times+[entry['metadata']['CPU_numpy_full_eigvalsh_seconds']],color=['#94a3b8','#93c5fd','#2563eb','#fca5a5','#dc2626','#64748b']);ax.set_xscale('log');ax.set_xlim(.1,max(times)*2);ax.set(xlabel=f"{entry['B']}系统 × 384平面波 × 16根：总秒（对数刻度）");
  for j,v in enumerate(times+[entry['metadata']['CPU_numpy_full_eigvalsh_seconds']]):ax.text(v*1.05,j,f'{v:.3f}s',va='center',fontsize=9)
 axs[3].axis('off');text=['共享内存Jacobi + 两遍MGS + 投影/Ritz/残差全部在GPU。','每k点一个持久Block，内部设备判断收敛；迭代无主机传输。']
 for e in r['systems']:
  text.append(f"B={e['B']}：GB10驻留核批处理 {e['batch_speedups']['cuda']['resident_kernel']:.2f}×；AMD {e['batch_speedups']['hip']['resident_kernel']:.2f}×")
 text+=['旧混合版B64端到端批处理仅1.055× / 1.149×。','CPU保留教学Davidson与优化NumPy参考，不宣称通用硬件排名。','旧混合版数据保留在module10_results；新驻留结果见module15。','原始周期余弦势教学原型，不是官方Daviator/VASP或硅材料。'];axs[3].text(0,1,'\n'.join(text),va='top',linespacing=1.7,fontsize=10.5)
 fig.tight_layout(rect=(0,0,1,.96));out=ROOT/'visuals/fig9_daviator_batched_davidson_vasp_band.svg'
 # Preserve hybrid plot for phase3 review.
 backup=out.with_name('fig9_daviator_hybrid_original.svg')
 if not backup.exists() and out.exists():backup.write_bytes(out.read_bytes())
 fig.savefig(out,facecolor='#ffffff');plt.close(fig);s=out.read_text().replace('<svg ','<svg font-family="\'WenQuanYi Micro Hei\',\'Noto Sans CJK SC\',sans-serif" ',1).replace('width="460.8pt"','width="640"').replace('height="1008pt"','height="1400"');out.write_text(s)
 import cairosvg;cairosvg.svg2png(url=str(out),write_to=str(out.with_suffix('.png')))
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--prepare-only',action='store_true');ap.add_argument('--gpu-only',action='store_true');ap.add_argument('--summarize-only',action='store_true');a=ap.parse_args()
 if not a.gpu_only and not a.summarize_only:prepare()
 if a.prepare_only:sys.exit(0)
 if not a.summarize_only:gpu()
 summarize()
