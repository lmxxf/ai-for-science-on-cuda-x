from pathlib import Path
import json,numpy as np
from xml.sax.saxutils import escape
ROOT=Path(__file__).resolve().parent

def render():
 r=json.loads((ROOT/'results/module9_results.json').read_text());s=['''<svg xmlns="http://www.w3.org/2000/svg" width="640" height="1510" viewBox="0 0 640 1510" font-family="'WenQuanYi Micro Hei','Noto Sans CJK SC',sans-serif"><rect width="640" height="1510" fill="#ffffff"/>''']
 def text(x,y,t,size=18,col='#253747',bold=False):s.append(f'<text x="{x}" y="{y}" font-size="{size}" fill="{col}" font-weight="{"bold" if bold else "normal"}">{escape(str(t))}</text>')
 def line(a,b,c,d,col='#b9c6d0'):s.append(f'<line x1="{a}" y1="{b}" x2="{c}" y2="{d}" stroke="{col}"/>')
 text(30,43,'PME + 1,000 步真实柔性分子动力学',25,bold=True);text(30,76,'CPU / CUDA / HIP 原创同源物理模型 · float64',17)
 text(30,117,'① 4阶 B样条：电荷 → 64³网格 → 3D FFT',20,bold=True)
 x=np.fromfile(ROOT/'results/module9_data/atoms.bin').reshape(-1,8)[:,:3];q=np.fromfile(ROOT/'results/module9_data/atoms.bin').reshape(-1,8)[:,6]
 for xx in range(8):line(40+xx*30,150,40+xx*30,350)
 for yy in range(8):line(40,150+yy*28,250,150+yy*28)
 for p,qq in zip(x[:27],q[:27]):
  s.append(f'<circle cx="{40+p[0]/24*210:.2f}" cy="{150+p[1]/24*196:.2f}" r="5" fill="{"#d06060" if qq<0 else "#347ca4"}"/>')
 # True spectral magnitude slice from original charge spreading using the actual initial positions.
 import exp7_pme_and_fused_md_hip as mod
 m=64;u=x*m/24;f=np.floor(u);t=u-f;b=f.astype(int)-1;w=np.stack([(1-t)**3/6,(3*t**3-6*t*t+4)/6,(-3*t**3+3*t*t+3*t+1)/6,t**3/6],axis=-1);rho=np.zeros((m,m,m))
 import itertools
 for a,z,d in itertools.product(range(4),repeat=3):np.add.at(rho,((b[:,0]+a)%m,(b[:,1]+z)%m,(b[:,2]+d)%m),q*w[:,0,a]*w[:,1,z]*w[:,2,d])
 spec=np.log1p(abs(np.fft.fftshift(np.fft.fftn(rho))[:,:,32]));spec=spec[20:44,20:44];spec/=max(spec.max(),1e-20)
 for i in range(24):
  for j in range(24):
   v=spec[i,j];col=f'#{int(240-210*v):02x}{int(245-125*v):02x}{int(250-70*v):02x}';s.append(f'<rect x="{340+j*8}" y="{150+i*8}" width="8" height="8" fill="{col}"/>')
 text(55,383,'初始电荷 xy 投影',16);text(334,383,'真实 |ρ(k)| 对数切片',16)
 text(30,419,'反卷积格林函数 → 逆FFT → B样条导数插值',18)
 text(30,448,'同一离散能量的解析力；实空间 erfc + 倒易 + 自能',16)
 text(30,490,'② NVE：动能/势能交换，总能量保持',21,bold=True)
 h=np.fromfile(ROOT/'results/module9_cuda/nve_history.bin').reshape(-1,7)
 # Display energy changes so different offsets do not hide conservation.
 ytop=540;hei=220;x0=65;wid=510;vals=h[:,1:4]-h[0,1:4];limit=np.max(abs(vals))*1.05
 for v in [-1,0,1]:
  line(x0,ytop+hei/2-v*hei/2,x0+wid,ytop+hei/2-v*hei/2);text(31,ytop+hei/2-v*hei/2+5,f'{v*limit:.0f}',13)
 for j,col in enumerate(['#d88034','#348ab3','#397d55']):
  coords=' '.join(f'{x0+i/(len(h)-1)*wid:.2f},{ytop+hei/2-vals[i,j]/limit*hei/2:.2f}' for i in range(0,len(h),4));s.append(f'<polyline points="{coords}" fill="none" stroke="{col}" stroke-width="{2.8 if j==2 else 1.5}"/>')
 text(30,525,'相对初始能量变化（kJ/mol）',16);text(72,784,'0 ps',15);text(520,784,'0.5 ps',15);text(86,813,'橙：势能  蓝：动能  绿：总能量',17)
 cuda=r['backends']['cuda'];dr=cuda['trajectories']['nve']['energy_relative_max_drift'];ratio=cuda['nve_dt_halving_drift_ratio'];text(30,847,f'最大相对漂移 {dr:.2e}；步长减半改善 {ratio:.2f} 倍',17)
 text(30,885,'③ 1,000 步 NVE 墙钟耗时（含每步 PME）',20,bold=True)
 cpu=r['backends']['cpu']['trajectories']['nve']['seconds']
 for i,(b,label,col) in enumerate([('cpu','CPU NumPy/SciPy','#8795a2'),('cuda','NVIDIA GB10','#3185a4'),('hip','AMD RX 9070 XT','#bc555c')]):
  y=929+i*45;t=r['backends'][b]['trajectories']['nve']['seconds'];text(30,y,label,17);s.append(f'<rect x="220" y="{y-16}" width="{max(3,270*t/cpu):.1f}" height="21" fill="{col}"/>');text(507,y,f'{t:.3f} s',16)
 text(30,1070,'④ BAOAB 热浴：300 K、1,000 步 / 1 ps',20,bold=True)
 for i,b in enumerate(['cpu','cuda','hip']):
  t=r['backends'][b]['trajectories']['nvt'];text(30,1105+i*29,f'{b.upper()}：后半段 {t["temperature_mean_second_half"]:.2f} ± {t["temperature_std_second_half"]:.2f} K',17)
 v=r['validation'];e=v['mesh_sizes']['64'];text(30,1220,'独立直接 Ewald 对照（含实空间/倒易/自能）',18,bold=True)
 text(30,1252,f'64³网格受力误差 L2={e["force_relative_l2"]:.2e}，非机器精度',17)
 text(30,1282,f'128³改善到 {v["mesh_sizes"]["128"]["force_relative_l2"]:.2e}；有限差分 {v["mesh_gradient_max_abs_error"]:.2e}',16)
 text(30,1312,f'CUDA/HIP 初始受力 L2={r["hip_vs_cuda_initial_force_relative_l2"]:.2e}',17)
 text(30,1352,'81 原子 / 27 个合成中性三原子柔性分子',17)
 text(30,1382,'有谐振键角；无 SETTLE、硬约束或生产级优化',16)
 text(30,1412,'GPU循环设备驻留；CPU为Python/FFT基线，计时非等实现',16)
 text(30,1442,'网格导数保能量一致性；净力有有限网格残差，详见报告',16)
 text(30,1474,'原创 MIT prototype，不宣称复刻或兼容 libcupo',15,col='#687b89');s.append('</svg>')
 out=ROOT/'visuals/fig8_pme_kspace_and_1000step_md_trajectory.svg';out.write_text('\n'.join(s));import cairosvg;cairosvg.svg2png(url=str(out),write_to=str(out.with_suffix('.png')))
if __name__=='__main__':render()
