#!/usr/bin/env python3
"""Physically consistent synthetic sky covariance -> voltage -> FFT/FX -> dirty/CLEAN."""
from pathlib import Path
import argparse,hashlib,json,subprocess,sys,time,os
os.environ.setdefault('OPENBLAS_NUM_THREADS','4');os.environ.setdefault('OMP_NUM_THREADS','4')
import numpy as np
ROOT=Path(__file__).resolve().parent;SRC=ROOT/'amd_hip_port/cuphoton_xbeam_fx';RES=ROOT/'results';DAT=RES/'module11_data';BUILD=RES/'module11_build';PY=sys.executable

def run(c,**kw):print('RUN',*[str(x) for x in c],flush=True);return subprocess.run([str(x) for x in c],check=True,**kw)
def guard(c,n):return run([PY,ROOT/'guard_process.py','--audit',ROOT/f'logs/module11_{n}_guard.json','--',*c])
def prepare():
 DAT.mkdir(exist_ok=True);BUILD.mkdir(exist_ok=True);rng=np.random.default_rng(20261124);N=64;pix=(np.arange(N)-N//2)/N;xx,yy=np.meshgrid(pix,pix);radius=np.sqrt(xx**2+yy**2);angle=np.arctan2(yy,xx);sky=np.exp(-.5*((radius-.155)/.025)**2)*(1+.65*np.cos(angle-.6));sky/=sky.sum();np.save(DAT/'sky.npy',sky)
 # Non-collinear stations; rotated array at 64 sparse snapshots over a 12h toy track.
 az=rng.uniform(0,2*np.pi,16);rad=rng.uniform(4,14,16);stations=np.column_stack([np.cos(az)*rad,np.sin(az)*rad]);np.save(DAT/'stations.npy',stations);ab=np.array([(a,b) for a in range(16) for b in range(a+1,16)]);np.save(DAT/'baselines.npy',ab)
 freqs=1+np.linspace(-.02,.02,16);angles=np.linspace(0,np.pi,64,endpoint=False);uvs=[];expected=[];truth=sky.ravel();coords=np.stack([xx.ravel(),yy.ravel()]);start=time.perf_counter();cov_error=0;noise=.02
 with (DAT/'voltage.bin').open('wb') as f:
  for bin,ang in enumerate(angles):
   rot=np.array([[np.cos(ang),-np.sin(ang)],[np.sin(ang),np.cos(ang)]]);ant=stations@rot.T;spec=np.empty((64,16,1024),np.complex128);uvbin=[];exbin=[]
   for g,fr in enumerate(freqs):
    phase=np.exp(-2j*np.pi*((ant*fr)@coords));cov=(phase*truth)@phase.conj().T;cov=(cov+cov.conj().T)/2;L=np.linalg.cholesky(cov+noise*np.eye(16));z=(rng.normal(size=(64*64,16))+1j*rng.normal(size=(64*64,16)))/np.sqrt(2);z=z@L.T;spec[:,:,g*64:(g+1)*64]=z.reshape(64,64,16).transpose(0,2,1);uvbin.append((ant[ab[:,0]]-ant[ab[:,1]])*fr);exbin.append(cov[ab[:,0],ab[:,1]])
   voltage=np.fft.ifft(spec,axis=-1);voltage.astype('<c16').tofile(f);uvs.append(uvbin);expected.append(exbin)
   if bin%8==0:print('synthetic covariance voltage bin',bin,flush=True)
 np.save(DAT/'uv.npy',np.asarray(uvs));np.save(DAT/'expected_visibility_groups.npy',expected);metadata={'workload':'synthetic asymmetric photon-ring sky, NOT real EHT/M87/SgrA data','seed':20261124,'antennas':16,'frames':4096,'FFT_length':1024,'time_bins':64,'frames_per_time_bin':64,'baselines':120,'frequency_groups':16,'channels_per_group':64,'center_frequency_GHz':230.,'fractional_bandwidth':.04,'bandwidth_GHz':9.2,'geometry':'16 noncollinear stations in wavelength/image-FOV units, rigid rotation over pi radians at 64 sparse snapshots across nominal 12h; not geographical EHT stations','cadence':'64 independent FFT frames per snapshot; sparse synthetic observing windows, not contiguous 12h ADC samples','sky_to_voltage':'C_ab(f,t)=sum_p I_p exp(-2pi i (u_a-u_b)·l_p); S=L z where LL*=C+0.02I, z iid circular complex Gaussian; time voltage=IFFT(S)','frequency_approximation':'narrowband piecewise constant geometry/covariance in each 64-channel group; 16 group center frequencies spanning +/-2%','receiver_noise_covariance_diagonal':noise,'voltage_samples_complex':4096*16*1024,'generation_seconds':time.perf_counter()-start,'voltage_sha256':hashlib.sha256((DAT/'voltage.bin').read_bytes()).hexdigest()};(DAT/'metadata.json').write_text(json.dumps(metadata,indent=2));cpu()
def cpu():
 ft=xt=io=0.;all=time.perf_counter();ab=np.load(DAT/'baselines.npy');dst=np.memmap(RES/'module11_cpu_vis.bin',mode='w+',dtype='<c16',shape=(64,120,1024))
 with (DAT/'voltage.bin').open('rb') as f:
  for bin in range(64):
   t=time.perf_counter();v=np.fromfile(f,dtype='<c16',count=64*16*1024).reshape(64,16,1024);io+=time.perf_counter()-t;t=time.perf_counter();s=np.fft.fft(v,axis=-1);ft+=time.perf_counter()-t;t=time.perf_counter();dst[bin]=np.stack([(s[:,a,:]*s[:,b,:].conj()).mean(0) for a,b in ab]);xt+=time.perf_counter()-t
 dst.flush();(RES/'module11_cpu_timings.json').write_text(json.dumps({'FFT_seconds':ft,'X_seconds':xt,'IO_seconds':io,'transfer_seconds':0,'total_seconds':time.perf_counter()-all,'precision':'complex128'},indent=2))
def gpus():
 run(['nvcc','-x','cu','-std=c++17','-O3','--fmad=false','-arch=sm_121',SRC/'fx.cpp','-o',BUILD/'fx_cuda']);guard([BUILD/'fx_cuda',DAT,RES/'module11_cuda'],'cuda')
 (SRC/'build_windows.cmd').write_text('@echo off\nsetlocal\ncd /d %~dp0\nset "ROCM=C:\\Strata-Lab\\Strata\\.rocm-win\\Lib\\site-packages\\_rocm_sdk_devel"\ncall "D:\\DLSSNR-Lab\\build-tools\\VC\\Auxiliary\\Build\\vcvars64.bat" >nul || exit /b 1\nset "HIP_PLATFORM=amd"\nset "HIP_PATH=%ROCM%"\nset "ROCM_PATH=%ROCM%"\nset "PATH=%ROCM%\\bin;%ROCM%\\lib\\llvm\\bin;%PATH%"\n"%ROCM%\\lib\\llvm\\bin\\clang++.exe" -x hip --offload-arch=gfx1201 --rocm-path="%ROCM%" --rocm-device-lib-path="%ROCM%\\lib\\llvm\\amdgcn\\bitcode" -std=c++17 -O3 -ffp-contract=off -I"%ROCM%\\include" fx.cpp -o fx.exe -L"%ROCM%\\lib" -lamdhip64 > build.log 2>&1\nexit /b %ERRORLEVEL%\n')
 (SRC/'run_windows.cmd').write_text('@echo off\nsetlocal\ncd /d %~dp0\nset "ROCM=C:\\Strata-Lab\\Strata\\.rocm-win\\Lib\\site-packages\\_rocm_sdk_devel"\nset "PATH=%ROCM%\\bin;%ROCM%\\lib\\llvm\\bin;%PATH%"\nfx.exe data out_hip > run.log 2>&1\nexit /b %ERRORLEVEL%\n')
 run(['scp','-r',SRC,'amd9070:C:/AI4S-HIP-Lab/']);run(['ssh','amd9070','cmd /c "mkdir C:\\AI4S-HIP-Lab\\cuphoton_xbeam_fx\\data 2>nul & exit /b 0"']);run(['scp',DAT/'voltage.bin','amd9070:C:/AI4S-HIP-Lab/cuphoton_xbeam_fx/data/']);run(['ssh','amd9070','C:\\AI4S-HIP-Lab\\cuphoton_xbeam_fx\\build_windows.cmd']);run(['ssh','amd9070','C:\\AI4S-HIP-Lab\\cuphoton_xbeam_fx\\run_windows.cmd'])
 for f in ['vis.bin','timings.json']:run(['scp',f'amd9070:C:/AI4S-HIP-Lab/cuphoton_xbeam_fx/out_hip_{f}',RES/f'module11_hip_{f}'])
 for f in ['run.log','build.log']:run(['scp',f'amd9070:C:/AI4S-HIP-Lab/cuphoton_xbeam_fx/{f}',SRC/f])
def image(vis):
 # Only measured FX visibility enters image reconstruction; input sky never enters this function.
 uv=np.load(DAT/'uv.npy');vg=vis.reshape(64,120,16,64).mean(-1).transpose(0,2,1);grid=np.zeros((64,64),complex);weight=np.zeros((64,64));
 for b in range(64):
  for g in range(16):
   for a in range(120):
    u,v=np.rint(uv[b,g,a]).astype(int)
    for su,sv,z in [(u,v,vg[b,g,a]),(-u,-v,vg[b,g,a].conj())]:
     ix=su+32;iy=sv+32
     if 0<=ix<64 and 0<=iy<64:grid[iy,ix]+=z;weight[iy,ix]+=1
 mask=weight>0;grid[mask]/=weight[mask];dirty=np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(grid))).real;psf=np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(mask.astype(float)))).real;dirty/=psf[32,32];psf/=psf[32,32]
 residual=dirty.copy();components=np.zeros_like(dirty);gain=.1;initial=np.max(np.abs(residual));used=0
 for it in range(2000):
  iy,ix=np.unravel_index(np.abs(residual).argmax(),residual.shape);amp=residual[iy,ix]
  if abs(amp)<initial*.005:break
  components[iy,ix]+=gain*amp;residual-=gain*amp*np.roll(np.roll(psf,iy-32,axis=0),ix-32,axis=1);used=it+1
 # Fixed illustrative Gaussian restoring beam (sigma 0.7 pixel), not fitted from uv resolution and not a ring-shaped target prior.
 from scipy.ndimage import gaussian_filter
 clean=gaussian_filter(components,sigma=.7,mode='wrap')+residual
 return dirty,clean,psf,mask,int(used)
def summary():
 sky=np.load(DAT/'sky.npy');ref=np.memmap(RES/'module11_cpu_vis.bin',mode='r',dtype='<c16',shape=(64,120,1024));r={'status':'PASS','metadata':json.loads((DAT/'metadata.json').read_text()),'backends':{},'image_pipeline':'observed FX spectra -> average channels within each of 16 narrowband groups -> nearest-grid uv per-cell mean then uniform occupied-cell weighting with conjugate symmetry -> dirty IFFT -> Hogbom CLEAN gain0.1 up to2000 iterations threshold0.005*initialpeak -> Gaussian restore sigma0.7pix + residual; no target ring injected into reconstruction'};expected=np.load(DAT/'expected_visibility_groups.npy');observed=np.asarray(ref).reshape(64,120,16,64).mean(-1).transpose(0,2,1);r['sample_covariance_vs_sky_fourier_relative_l2']=float(np.linalg.norm(observed-expected)/np.linalg.norm(expected));refim=None
 # Independent direct pixel sum for selected baseline/frequency/time entries.
 uv=np.load(DAT/'uv.npy');coord=(np.arange(64)-32)/64;xx,yy=np.meshgrid(coord,coord);checks=[]
 for b,g,a in [(0,0,0),(7,3,11),(31,8,63),(63,15,119)]:
  u,v=uv[b,g,a];direct=np.sum(sky*np.exp(-2j*np.pi*(u*xx+v*yy)));checks.append({'timebin':b,'frequency_group':g,'baseline':a,'abs_diff':float(abs(direct-expected[b,g,a]))})
 r['independent_direct_fourier_pixel_sum_checks']=checks
 assert max(v['abs_diff'] for v in checks)<1e-12
 r['reconstruction_caveats']=['No zero-spacing/autocorrelation measurement: dirty map DC is missing and sum near zero. Dirty flux-normalized PSNR undefined.', 'Restored CLEAN image normalized by its recovered flux for PSNR; 2000 iteration cap reached, not a claim of converged optimal reconstruction.', 'Gaussian restoration sigma0.7pix is an illustrative beam, not an EHT-derived beam fit.']
 images={}
 for name in ['cpu','cuda','hip']:
  vis=np.memmap(RES/f'module11_{name}_vis.bin',mode='r',dtype='<c16',shape=(64,120,1024));rel=float(np.linalg.norm(vis-ref)/np.linalg.norm(ref));assert rel<1e-12,(name,rel);dirty,clean,psf,mask,it=image(vis);images[name]=clean
  # Flux-normalized evaluation, not an arbitrary fitted ring overlay.
  normalized=clean/clean.sum();mse=float(np.mean((normalized-sky)**2));psnr=float(10*np.log10(sky.max()**2/mse));tim=json.loads((RES/f'module11_{name}_timings.json').read_text());tim.update({'visibility_relative_l2_vs_cpu':rel,'clean_PSNR_vs_input_sky_dB':psnr,'clean_iterations':it,'clean_hit_iteration_cap':it==2000,'clean_relative_l2_vs_cpu':0. if refim is None else float(np.linalg.norm(clean-refim)/np.linalg.norm(refim)),'uv_grid_occupied':int(mask.sum()),'dirty_PSNR_vs_sky_dB':float(10*np.log10(sky.max()**2/np.mean((dirty/dirty.sum()-sky)**2))) if abs(dirty.sum())>1e-12 else None});r['backends'][name]=tim
  if refim is None:refim=clean
  np.savez(RES/f'module11_{name}_images.npz',dirty=dirty,clean=clean,psf=psf,uv_mask=mask)
 r['hip_vs_cuda_visibility_relative_l2']=float(np.linalg.norm(np.memmap(RES/'module11_hip_vis.bin',mode='r',dtype='<c16',shape=ref.shape)-np.memmap(RES/'module11_cuda_vis.bin',mode='r',dtype='<c16',shape=ref.shape))/np.linalg.norm(ref));r['hip_vs_cuda_clean_relative_l2']=float(np.linalg.norm(images['hip']-images['cuda'])/np.linalg.norm(images['cuda']));assert r['hip_vs_cuda_clean_relative_l2']<1e-12
 (RES/'module11_results.json').write_text(json.dumps(r,indent=2));render(r)
def render(r):
 import matplotlib;matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 plt.rcParams.update({'font.family':'WenQuanYi Micro Hei','svg.fonttype':'none','font.size':11,'figure.facecolor':'white','axes.facecolor':'white'})
 sky=np.load(DAT/'sky.npy');im=np.load(RES/'module11_cuda_images.npz');fig=plt.figure(figsize=(6.4,13));gs=fig.add_gridspec(4,2,height_ratios=[1,1,1,.9]);fig.suptitle('从相关电压流重建合成光子环',fontsize=18)
 for (row,col),data,title in [((0,0),sky,'输入天空（合成，非EHT数据）'),((0,1),im['uv_mask'],'64时段 × 120基线 uv覆盖'),((1,0),im['dirty'],'真实FX可见度 → 脏图'),((1,1),im['clean'],'Högbom CLEAN恢复图')]:
  ax=fig.add_subplot(gs[row,col]);ax.imshow(data,cmap='inferno' if col==0 or row==1 else 'gray',origin='lower');ax.set_title(title,fontsize=11);ax.set_xticks([]);ax.set_yticks([])
 ax=fig.add_subplot(gs[2,:]);names=['cpu','cuda','hip'];ax.barh(names,[r['backends'][n]['FFT_seconds']+r['backends'][n]['X_seconds'] for n in names],color=['#94a3b8','#2563eb','#dc2626']);ax.set(xlabel='F阶段 FFT + X阶段复数相关 / 秒（不含I/O）')
 ax=fig.add_subplot(gs[3,:]);ax.axis('off');t=[f"64,000,000级复采样：16天线 × 4096帧 × 1024FFT",f"GPU与CPU可见度相对L2：CUDA {r['backends']['cuda']['visibility_relative_l2_vs_cpu']:.2e}",f"HIP {r['backends']['hip']['visibility_relative_l2_vs_cpu']:.2e}；跨GPU图像 {r['hip_vs_cuda_clean_relative_l2']:.2e}",f"恢复PSNR {r['backends']['cuda']['clean_PSNR_vs_input_sky_dB']:.2f} dB；有限uv覆盖与采样噪声会限制清晰度。",'电压协方差来自同一输入天空的傅里叶变换。','恢复图只使用FX输出，不直接拼接输入环图。','频域/时间块独立高斯场；教学原型，非官方cuPhoton性能。'];ax.text(0,1,'\n'.join(t),va='top',linespacing=1.7,fontsize=11)
 fig.tight_layout(rect=(0,0,1,.96));out=ROOT/'visuals/fig10_cuphoton_eht_blackhole_correlator.svg';fig.savefig(out,facecolor='#ffffff');plt.close(fig);s=out.read_text().replace('<svg ','<svg font-family="\'WenQuanYi Micro Hei\',\'Noto Sans CJK SC\',sans-serif" ',1).replace('width="460.8pt"','width="640"').replace('height="936pt"','height="1300"');out.write_text(s)
 import cairosvg;cairosvg.svg2png(url=str(out),write_to=str(out.with_suffix('.png')))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--prepare-only',action='store_true');p.add_argument('--gpu-only',action='store_true');p.add_argument('--summarize-only',action='store_true');a=p.parse_args()
 if not a.gpu_only and not a.summarize_only:prepare()
 if a.prepare_only:sys.exit(0)
 if not a.summarize_only:gpus()
 summary()
