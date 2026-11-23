#!/usr/bin/env python3
from pathlib import Path
import json,math
from xml.sax.saxutils import escape
ROOT=Path(__file__).resolve().parent

def render():
 r=json.loads((ROOT/'results/module7_results.json').read_text());s=['''<svg xmlns="http://www.w3.org/2000/svg" width="640" height="1500" viewBox="0 0 640 1500" font-family="'WenQuanYi Micro Hei','Noto Sans CJK SC',sans-serif"><rect width="640" height="1500" fill="#ffffff"/>''']
 def text(x,y,t,size=18,col='#263546',bold=False):s.append(f'<text x="{x}" y="{y}" font-size="{size}" fill="{col}" font-weight="{"bold" if bold else "normal"}">{escape(str(t))}</text>')
 def line(x1,y1,x2,y2,c='#c9d2dc',w=1.3):s.append(f'<line x1="{x1:.2f}" y1="{y1:.2f}" x2="{x2:.2f}" y2="{y2:.2f}" stroke="{c}" stroke-width="{w}"/>')
 text(30,44,'开源非键力场：CUDA 与 HIP 同算',25,bold=True);text(30,76,'原创 LJ + 反应场库仑 · 无 libcupo 依赖',18)
 text(30,117,'① 周期盒子 + 最小镜像 + 27 邻格',21,bold=True)
 for z in range(3):
  ox=60+z*115;oy=160+z*32
  for yy in range(3):
   for xx in range(3):
    x=ox+xx*60;y=oy+yy*35
    fill='#f5dfbe' if z==1 and xx==1 and yy==1 else ['#eaf0f8','#e1f2ed','#edf3f7'][z]
    s.append(f'<rect x="{x}" y="{y}" width="59" height="34" fill="{fill}" stroke="#99aebb"/>')
    if (xx+yy+z)%2==0:s.append(f'<circle cx="{x+27}" cy="{y+16}" r="4" fill="#3478ad"/>')
 text(30,350,'每格边长 ≥ 10 Å；只查自身与周边 26 格',18);text(30,381,'原子有电荷 q、尺寸 σ、势阱深度 ε；不含键项',17)
 text(30,423,'② 融合势能与解析力：F = −∇E',21,bold=True)
 text(30,455,'E_LJ = 4ε[(σ/r)^12 − (σ/r)^6]',19);text(30,484,'E_RF = k qi qj[1/r + k_rf r² − c_rf]',19)
 # Shared curve axes, r 2.7..10 A, potential -1..1.5 kJ/mol.
 x0=60;y0=540;w=520;h=170
 def px(x):return x0+(x-2.7)/(10-2.7)*w
 def py(v):return y0+(1.5-v)/2.5*h
 line(x0,y0,x0,y0+h);line(x0,py(0),x0+w,py(0));text(65,532,'kJ/mol',14);text(520,736,'r (Å)',15)
 for x in [3,5,7,10]:text(px(x)-7,730,str(x),14)
 for val in [-1,0,1]:text(32,py(val)+5,str(val),14)
 for mode,col in [('lj','#d67b27'),('rf','#2879ae')]:
  xy=[]
  for i in range(240):
   x=2.7+(10-2.7)*i/239;v=4*.2*((3/x)**12-(3/x)**6) if mode=='lj' else 1389.354576*(-.05*.05)*(1/x+(78.3-1)/(2*78.3+1)/1000*x*x-3*78.3/(2*78.3+1)/10)
   xy.append(f'{px(x):.2f},{py(v):.2f}')
  s.append(f'<polyline points="{" ".join(xy)}" fill="none" stroke="{col}" stroke-width="2.4"/>')
 text(330,545,'橙：LJ  蓝：异号电荷 RF',15);text(30,765,'示例 σ=3 Å、ε=0.2 kJ/mol、q=±0.05e',16)
 text(30,805,'③ 真机：索引 + 能量 + 全原子受力',21,bold=True)
 colors={'cpu':'#81919f','cuda':'#2b819f','hip':'#be4b50'}
 for k,sys in enumerate(r['systems']):
  y=845+k*196;text(30,y,f'{sys["n"]:,} 原子 · float64',19,bold=True);maxms=sys['backends']['cpu']['pipeline_median_ms']
  for j,(b,label) in enumerate([('cpu','CPU 单线程C++'),('cuda','NVIDIA GB10'),('hip','AMD RX 9070 XT')]):
   yy=y+32+j*43;ms=sys['backends'][b]['pipeline_median_ms'];text(30,yy,label,16);s.append(f'<rect x="218" y="{yy-14}" width="{max(3,ms/maxms*270):.2f}" height="18" fill="{colors[b]}" rx="3"/>');text(500,yy,f'{ms:.2f} ms',16)
  text(30,y+169,f'CUDA/HIP 全受力相对 L2：{sys["hip_vs_cuda"]["force_relative_l2"]:.2e}',16,col='#226c57')
 v=r['validation'];text(30,1270,'独立校验：全对全暴力计算、有限差分、周期平移',18,bold=True)
 text(30,1301,f'216 原子：暴力对照受力 L2 = {v["force_relative_l2"]:.2e}',16)
 text(30,1331,f'有限差分最大绝对误差 = {v["finite_difference_max_abs_error"]:.2e}',16)
 text(30,1363,'合成中性电荷晶格；非真实溶剂拓扑、非完整分子动力学',16)
 text(30,1393,'计时为设备驻留单次非键评估；传入/传出另列 JSON',16)
 text(30,1423,'RF ε_r=78.3；10 Å 硬截断，未做 LJ 平滑或 PME',16)
 text(30,1454,'CPU/CUDA/HIP 原创同源 MIT；不兼容或替代 libcupo API',15,col='#687b89');s.append('</svg>')
 out=ROOT/'visuals/fig6_open_cupotential_md_forcefield_amd_vs_nvidia.svg';out.write_text('\n'.join(s))
 import cairosvg;cairosvg.svg2png(url=str(out),write_to=str(out.with_suffix('.png')))
if __name__=='__main__':render()
