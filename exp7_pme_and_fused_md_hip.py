#!/usr/bin/env python3
"""Original PME+bonded MD prototype; independent NumPy direct-Ewald reference."""
from pathlib import Path
import argparse,subprocess,json,sys,time,itertools,hashlib
import numpy as np
from scipy.fft import fftn,ifftn
ROOT=Path(__file__).resolve().parent;SRC=ROOT/'amd_hip_port/cupotential_pme_md';RES=ROOT/'results';DAT=RES/'module9_data';BUILD=RES/'module9_build'
KE=1389.354576;KB=.00831446261815324;PI=np.pi

def run(c,**kw):print('RUN',*map(str,c),flush=True);return subprocess.run(list(map(str,c)),check=True,**kw)
def guard(c,name):return run([sys.executable,ROOT/'guard_process.py','--audit',ROOT/f'logs/module9_{name}_memory.json','--',*c])
def mic(d,L):return d-L*np.floor(d/L+.5)

def generate():
 rng=np.random.default_rng(20261124);x=[];q=[];mass=[];theta=1.9
 for ix,iy,iz in itertools.product(range(3),repeat=3):
  center=(np.array([ix,iy,iz])+.5)*8
  rot,_=np.linalg.qr(rng.normal(size=(3,3)));a=np.array([2.,0,0])@rot;b=np.array([2*np.cos(theta),2*np.sin(theta),0])@rot
  x.extend([center+a,center,center+b]);q.extend([.08,-.16,.08]);mass.extend([12.,16.,12.])
 x=np.array(x);q=np.array(q);mass=np.array(mass);v=rng.normal(size=x.shape)*np.sqrt(KB*300/(.01*mass[:,None]));v-=(v*mass[:,None]).sum(0)/mass.sum();v*=np.sqrt((3*len(x)*KB*300)/(.01*(mass[:,None]*v*v).sum()))
 DAT.mkdir(parents=True,exist_ok=True);np.column_stack([x,v,q,mass]).astype('<f8').tofile(DAT/'atoms.bin');(DAT/'params.txt').write_text(f'{len(x)} 64 24 0.4 10 200 30 2 1.9\n');return x,v,q,mass

def real_bond(x,q,L=24.,alpha=.4,rc=10.,kbond=200.,kangle=30.,l0=2.,theta0=1.9):
 n=len(x);d=mic(x[:,None,:]-x[None,:,:],L);r2=(d*d).sum(-1);valid=(r2>0)&(r2<rc*rc);safe=np.where(valid,r2,1.);r=np.sqrt(safe);from scipy.special import erfc
 er=erfc(alpha*r);qq=KE*q[:,None]*q[None,:];sr6=(1.44/safe)**3;ee=np.where(valid,.2*(sr6*sr6-sr6)+qq*er/r,0);coef=np.where(valid,1.2*(2*sr6*sr6-sr6)/safe+qq*(er/(r*safe)+2*alpha/np.sqrt(PI)*np.exp(-alpha*alpha*safe)/safe),0);f=(coef[:,:,None]*d).sum(1);E=ee.sum()/2-KE*alpha/np.sqrt(PI)*(q*q).sum()
 p=x.reshape(-1,3,3);a=mic(p[:,0]-p[:,1],L);b=mic(p[:,2]-p[:,1],L);ra=np.linalg.norm(a,axis=1);rb=np.linalg.norm(b,axis=1);co=(a*b).sum(1)/(ra*rb);co=np.clip(co,-.999999999999,.999999999999);theta=np.arccos(co);sn=np.sqrt(1-co*co)
 E+=.5*kbond*((ra-l0)**2+(rb-l0)**2).sum()+.5*kangle*((theta-theta0)**2).sum()
 f0=-kbond*(ra-l0)[:,None]*a/ra[:,None]-kangle*((theta-theta0)/sn)[:,None]*(co[:,None]*a/ra[:,None]**2-b/(ra*rb)[:,None]);f2=-kbond*(rb-l0)[:,None]*b/rb[:,None]-kangle*((theta-theta0)/sn)[:,None]*(co[:,None]*b/rb[:,None]**2-a/(ra*rb)[:,None]);f.reshape(-1,3,3)[:,0]+=f0;f.reshape(-1,3,3)[:,2]+=f2;f.reshape(-1,3,3)[:,1]-=f0+f2
 return E,f

class PME:
 def __init__(self,m=64,L=24.,alpha=.4):
  self.m=m;self.L=L;self.alpha=alpha;ks=np.fft.fftfreq(m)*m;k=2*PI/L*ks;k2=k[:,None,None]**2+k[None,:,None]**2+k[None,None,:]**2;b=np.sinc(ks/m)**8;deconv=b[:,None,None]*b[None,:,None]*b[None,None,:]
  self.G=KE/L**3*4*PI*np.exp(-k2/(4*alpha**2))/np.where(k2==0,1,k2)/deconv;self.G[0,0,0]=0;self.offsets=np.array(list(itertools.product(range(4),repeat=3)));self.shift=np.arange(4)
 def recip(self,x,q):
  u=x*self.m/self.L;fl=np.floor(u);t=u-fl;base=fl.astype(int)-1;w=np.stack([(1-t)**3/6,(3*t**3-6*t*t+4)/6,(-3*t**3+3*t*t+3*t+1)/6,t**3/6],axis=-1);dw=np.stack([-.5*(1-t)**2,1.5*t*t-2*t,-1.5*t*t+t+.5,.5*t*t],axis=-1)
  ix=(base[:,None,:]+self.offsets[None,:,:])%self.m;weights=w[:,0,self.offsets[:,0]]*w[:,1,self.offsets[:,1]]*w[:,2,self.offsets[:,2]];rho=np.zeros((self.m,)*3);np.add.at(rho,(ix[:,:,0].ravel(),ix[:,:,1].ravel(),ix[:,:,2].ravel()),(q[:,None]*weights).ravel());phi=ifftn(fftn(rho,workers=1)*self.G,workers=1).real*self.m**3;sample=phi[ix[:,:,0],ix[:,:,1],ix[:,:,2]];E=.5*np.sum(q[:,None]*weights*sample);f=np.zeros_like(x)
  for a in range(3):
   der=np.ones_like(weights)
   for b in range(3):der*=dw[:,b,self.offsets[:,b]] if a==b else w[:,b,self.offsets[:,b]]
   f[:,a]=-q*self.m/self.L*np.sum(der*sample,axis=1)
  return E,f
 def force(self,x,q):
  a,b=real_bond(x,q,self.L,self.alpha);c,d=self.recip(x,q);return a+c,b+d

def direct_ewald(x,q,L=24.,alpha=.4,rc=10.,kmax=16):
 E,F=real_bond(x,q,L,alpha,rc)
 # Complete real-space periodic Ewald sum over adjacent image boxes, beyond
 # the simulation's tiny erfc tail cutoff. LJ/bonds retain their stated model.
 from scipy.special import erfc
 d0=x[:,None,:]-x[None,:,:];dm=mic(d0,L);r2m=(dm*dm).sum(-1);valid=(r2m>0)&(r2m<rc*rc);sm=np.where(valid,r2m,1.);rm=np.sqrt(sm);qq=KE*q[:,None]*q[None,:]
 ereal=np.where(valid,qq*erfc(alpha*rm)/rm,0).sum()/2
 cf=np.where(valid,qq*(erfc(alpha*rm)/(rm*sm)+2*alpha/np.sqrt(PI)*np.exp(-alpha*alpha*sm)/sm),0)
 freal=(cf[:,:,None]*dm).sum(1);E-=ereal;F-=freal
 for shift in itertools.product([-1,0,1],repeat=3):
  d=d0+L*np.array(shift);r2=(d*d).sum(-1);valid=r2>0;safe=np.where(valid,r2,1.);rr=np.sqrt(safe);er=erfc(alpha*rr);E+=np.where(valid,qq*er/rr,0).sum()/2;coeff=np.where(valid,qq*(er/(rr*safe)+2*alpha/np.sqrt(PI)*np.exp(-alpha*alpha*safe)/safe),0);F+=(coeff[:,:,None]*d).sum(1)
 ki=np.array(list(itertools.product(range(-kmax,kmax+1),repeat=3)));ki=ki[(ki*ki).sum(1)>0];k=2*PI/L*ki;K2=(k*k).sum(1);G=4*PI*np.exp(-K2/(4*alpha**2))/K2;phase=np.exp(1j*x@k.T);S=(q[:,None]*phase.conj()).sum(0);E+=KE/(2*L**3)*np.sum(G*np.abs(S)**2);F+=KE/L**3*(q[:,None]*(phase*S).imag*G)@k;return E,F

def validate(x,q):
 E,F=direct_ewald(x,q);e2,f2=direct_ewald(x,q,kmax=18);checks={};
 for m in [32,64,128]:
  pe,pf=PME(m).force(x,q);checks[str(m)]=dict(energy_kj_mol=float(pe),energy_abs_error=float(abs(pe-E)),energy_relative_error=float(abs(pe-E)/abs(E)),force_relative_l2=float(np.linalg.norm(pf-F)/np.linalg.norm(F)),force_max_abs_error=float(np.max(abs(pf-F))),net_force_l2=float(np.linalg.norm(pf.sum(0))))
 p=PME(64);pe,pf=p.force(x,q);gradient=[];h=1e-5
 for i,a in [(0,0),(1,1),(7,2),(45,0),(70,2)]:
  xp=x.copy();xm=x.copy();xp[i,a]+=h;xm[i,a]-=h;ep,_=p.force(xp,q);em,_=p.force(xm,q);gradient.append(dict(atom=i,axis=a,finite_difference=float(-(ep-em)/(2*h)),analytic=float(pf[i,a]),abs_error=float(abs(-(ep-em)/(2*h)-pf[i,a]))))
 diff=mic(x[:,None,:]-x[None,:,:],24.);dist=np.sqrt((diff*diff).sum(-1));cutmargin=float(np.min(abs(dist[np.triu_indices(len(x),1)]-10.)));assert h<cutmargin/10;refnet=float(np.linalg.norm(F.sum(0)));assert refnet<1e-10
 maxerr=max(z['abs_error'] for z in gradient);assert maxerr<1e-6,(maxerr,'mesh energy force derivative');assert checks['64']['force_relative_l2']<checks['32']['force_relative_l2'];r=dict(ewald_reference_net_force_l2=refnet,finite_difference_cutoff_margin_angstrom=cutmargin,ewald_reference_energy_kj_mol=float(E),ewald_kmax=16,ewald_convergence_energy_abs_diff=float(abs(E-e2)),ewald_convergence_force_l2=float(np.linalg.norm(F-f2)/np.linalg.norm(f2)),mesh_sizes=checks,mesh_gradient_checks=gradient,mesh_gradient_max_abs_error=maxerr,notes='All comparisons include real erfc, reciprocal, self, LJ, bonds and angles. PME force is derivative of B-spline charge interpolation, hence gradient of same discrete energy; no ik-field inconsistency.')
 (RES/'module9_validation.json').write_text(json.dumps(r,indent=2));return r

def random_normals(step,n):
 def mix(x):
  with np.errstate(over='ignore'):
   x=x+np.uint64(0x9e3779b97f4a7c15);x=(x^(x>>np.uint64(30)))*np.uint64(0xbf58476d1ce4e5b9);x=(x^(x>>np.uint64(27)))*np.uint64(0x94d049bb133111eb);return x^(x>>np.uint64(31))
 a=mix(np.uint64(20261124+step*123457)+2*np.arange(n,dtype=np.uint64));b=mix(a);u=((a>>np.uint64(11)).astype(float)+.5)/9007199254740992.;v=((b>>np.uint64(11)).astype(float)+.5)/9007199254740992.;return np.sqrt(-2*np.log(u))*np.cos(2*PI*v)

def cpu_trajectories(x0,v0,q,mass):
 out=RES/'module9_cpu';out.mkdir(exist_ok=True);p=PME();ei,fi=p.force(x0,q);fi.astype('<f8').tofile(out/'initial_forces.bin');tim=[]
 def rec(x,v,e,step,dt):
  kin=.005*np.sum(mass[:,None]*v*v);return [step*dt,e,kin,e+kin,2*kin/(3*len(x)*KB),np.linalg.norm(mic(x[0]-x[1],24)),x[0,0]]
 for name,steps,dt in [('nve',1000,.0005),('nve_halfdt',2000,.00025),('nvt',1000,.001)]:
  x=x0.copy();v=v0.copy();e,f=p.force(x,q);hist=[rec(x,v,e,0,dt)];t=time.perf_counter()
  for step in range(1,steps+1):
   v+=.5*dt*100*f/mass[:,None]
   if name=='nvt':
    x+=.5*dt*v;c=np.exp(-10*dt);v=c*v+np.sqrt((1-c*c)*KB*300/(.01*mass[:,None]))*random_normals(step,x.size).reshape(x.shape);x+=.5*dt*v
   else:x+=dt*v
   x%=24;e,f=p.force(x,q);v+=.5*dt*100*f/mass[:,None];hist.append(rec(x,v,e,step,dt))
  sec=time.perf_counter()-t;np.array(hist,dtype='<f8').tofile(out/f'{name}_history.bin');x.astype('<f8').tofile(out/f'{name}_positions.bin');v.astype('<f8').tofile(out/f'{name}_velocities.bin');tim.append(dict(name=name,steps=steps,dt_ps=dt,seconds=sec));print('CPU',name,sec,flush=True)
 (out/'timings.json').write_text(json.dumps(dict(device='CPU NumPy/SciPy FFT workers=1, Python dynamics',grid=64,atoms=len(x0),trajectories=tim),indent=2))

def gpu():
 BUILD.mkdir(parents=True,exist_ok=True)
 guard(['nvcc','-x','cu','-std=c++17','-O3','--fmad=false','-arch=sm_121',SRC/'pme_md.cpp','-lcufft','-o',BUILD/'pme_md_cuda'],'compile_cuda')
 guard([BUILD/'pme_md_cuda',DAT,RES/'module9_cuda'],'cuda')
 run(['scp','-r',SRC,'amd9070:C:/AI4S-HIP-Lab/']);run(['ssh','amd9070','if not exist C:\\AI4S-HIP-Lab\\cupotential_pme_md\\data mkdir C:\\AI4S-HIP-Lab\\cupotential_pme_md\\data']);run(['scp',DAT/'atoms.bin',DAT/'params.txt','amd9070:C:/AI4S-HIP-Lab/cupotential_pme_md/data/'])
 run(['ssh','amd9070','C:\\AI4S-HIP-Lab\\cupotential_pme_md\\build_windows.cmd']);run(['ssh','amd9070','C:\\AI4S-HIP-Lab\\cupotential_pme_md\\run_windows.cmd']);dst=RES/'module9_hip';dst.mkdir(exist_ok=True);run(['scp','-r','amd9070:C:/AI4S-HIP-Lab/cupotential_pme_md/out_hip/.',dst]);run(['scp','amd9070:C:/AI4S-HIP-Lab/cupotential_pme_md/build.log',SRC/'build_windows.log']);run(['scp','amd9070:C:/AI4S-HIP-Lab/cupotential_pme_md/run.log',SRC/'run_windows.log'])

def summarize():
 r=dict(validation=json.loads((RES/'module9_validation.json').read_text()),backends={},scope='Original MIT prototype, 27 synthetic neutral flexible trimers/81 atoms; full real+reciprocal+self/LJ/bonds/angles. Not SETTLE or production libcupo replica.')
 init=None
 for b in ['cpu','cuda','hip']:
  folder=RES/f'module9_{b}';t=json.loads((folder/'timings.json').read_text());tr={};f=np.fromfile(folder/'initial_forces.bin').reshape(-1,3)
  if init is None:init=f
  t['initial_force_relative_l2_vs_cpu']=float(np.linalg.norm(f-init)/np.linalg.norm(init))
  for runinfo in t['trajectories']:
   name=runinfo['name'];h=np.fromfile(folder/f'{name}_history.bin').reshape(-1,7);np.savetxt(folder/f'{name}_history.csv',h,delimiter=',',header='time_ps,potential,kinetic,total,temperature,bond01_angstrom,x0_angstrom',comments='');runinfo.update(energy_initial=float(h[0,3]),energy_final=float(h[-1,3]),energy_relative_max_drift=float(np.max(abs(h[:,3]-h[0,3]))/max(abs(h[0,3]),1.)),temperature_mean_second_half=float(h[len(h)//2:,4].mean()),temperature_std_second_half=float(h[len(h)//2:,4].std()),final_bond01=float(h[-1,5]));tr[name]=runinfo
   if b!='cpu':
    cpuout=RES/'module9_cpu';fp=np.fromfile(folder/f'{name}_positions.bin');rp=np.fromfile(cpuout/f'{name}_positions.bin');delta=mic(fp-rp,24);runinfo['final_position_max_abs_error_vs_cpu_angstrom']=float(np.max(abs(delta)));ch=np.fromfile(cpuout/f'{name}_history.bin').reshape(-1,7);runinfo['history_total_energy_max_abs_error_vs_cpu']=float(np.max(abs(h[:,3]-ch[:,3])))
  t['trajectories']=tr;t['nve_dt_halving_drift_ratio']=tr['nve']['energy_relative_max_drift']/tr['nve_halfdt']['energy_relative_max_drift'];assert t['nve_dt_halving_drift_ratio']>2,t;assert t['initial_force_relative_l2_vs_cpu']<1e-10;t['nvt_temperature_target_K']=300;r['backends'][b]=t
 fc=np.fromfile(RES/'module9_cuda'/'initial_forces.bin');fh=np.fromfile(RES/'module9_hip'/'initial_forces.bin');r['hip_vs_cuda_initial_force_relative_l2']=float(np.linalg.norm(fc-fh)/np.linalg.norm(fc))
 (RES/'module9_results.json').write_text(json.dumps(r,indent=2));from render_module9 import render;render();print(json.dumps(r,indent=2))

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--cpu-only',action='store_true');ap.add_argument('--gpu-only',action='store_true');ap.add_argument('--summarize-only',action='store_true');a=ap.parse_args()
 if not a.gpu_only and not a.summarize_only:
  x,v,q,m=generate();validate(x,q);cpu_trajectories(x,v,q,m)
 if a.cpu_only:return
 if not a.summarize_only:gpu()
 summarize()
if __name__=='__main__':main()
