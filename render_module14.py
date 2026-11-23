from pathlib import Path
import json,numpy as np
from xml.sax.saxutils import escape
ROOT=Path(__file__).resolve().parent

def render():
 r=json.loads((ROOT/'results/module14_results.json').read_text());s=['''<svg xmlns="http://www.w3.org/2000/svg" width="640" height="1450" viewBox="0 0 640 1450" font-family="'WenQuanYi Micro Hei','Noto Sans CJK SC',sans-serif"><rect width="640" height="1450" fill="#ffffff"/>''']
 def text(x,y,t,size=18,col='#243749',bold=False):s.append(f'<text x="{x}" y="{y}" font-size="{size}" fill="{col}" font-weight="{"bold" if bold else "normal"}">{escape(str(t))}</text>')
 def line(a,b,c,d,col='#bdcbd5'):s.append(f'<line x1="{a}" y1="{b}" x2="{c}" y2="{d}" stroke="{col}"/>')
 text(28,42,'10,000 个新构象：CPU / CUDA / HIP',25,bold=True);text(28,77,'真实ESOL拓扑 + 原创简化力场；不是MMFF94平替',17)
 text(28,120,'① 重新ETKDG：1,000分子 × 10构象',21,bold=True)
 p=r['preparation'];text(28,156,f'8进程新起点+参数准备 {p["preparation_wall_seconds"]:.3f} 秒',18);text(28,186,'未做MMFF优化；另加0.06 Å明确位置扰动',17)
 text(28,223,'bond / cos-angle / torsion / LJ+Coulomb',18);text(28,254,'非键从相隔3根键起；MMFF API仅读平衡几何参数',16)
 text(28,297,'② 每构象一个Block，200步回溯下降',21,bold=True)
 for i,(label,fill) in enumerate([('共享坐标 X','#e0eef7'),('解析梯度 g','#e4f0e4'),('回溯候选 X′','#f8ead8')]):
  x=30+i*202;s.append(f'<rect x="{x}" y="320" width="184" height="61" rx="8" fill="{fill}" stroke="#b4c5cf"/>');text(x+10,357,label,17)
 text(28,414,'坐标/候选/梯度常驻Shared；参数仍读Global',17);text(28,445,'无每步CPU干预；能量诊断也有Global输出',16)
 text(28,486,'③ 真实能量轨迹（硝苯地平，第1构象）',20,bold=True)
 h=np.fromfile(ROOT/'results/module14_cuda/history.bin').reshape(10000,201)[480];x0=68;y0=525;w=505;hh=190;lo=h.min();hi=h.max();pad=max((hi-lo)*.08,.1);lo-=pad;hi+=pad
 for i in range(3):
  val=lo+(hi-lo)*i/2;yy=y0+hh-(val-lo)/(hi-lo)*hh;line(x0,yy,x0+w,yy);text(25,yy+4,f'{val:.1f}',13)
 pts=' '.join(f'{x0+i/200*w:.2f},{y0+hh-(v-lo)/(hi-lo)*hh:.2f}' for i,v in enumerate(h));s.append(f'<polyline points="{pts}" fill="none" stroke="#2b849f" stroke-width="2.3"/>');text(67,742,'0步',15);text(535,742,'200步',15);text(28,773,'纵轴：本原型定义的能量（kJ/mol）',16)
 c=r['backends']['cpu'];text(28,813,f'全体平均能量 {c["energy_initial_mean"]:.3f} → {c["energy_final_mean"]:.3f}',19);text(28,845,f'10,000/10,000不增；{c["strict_decrease_count"]:,}严格下降',17)
 text(28,883,f'平均|键长偏差| {c["bond_abs_deviation_initial_mean_A"]:.4f} → {c["bond_abs_deviation_final_mean_A"]:.4f} Å',17)
 text(28,915,f'非键碰撞计数 {c["nonbonded_collision_pairs_initial"]:,} → {c["nonbonded_collision_pairs_final"]:,}',17)
 text(28,955,'④ 同源同任务：10,000 × 200步计时',21,bold=True)
 maxs=c['seconds']
 for j,(b,label,col) in enumerate([('cpu','CPU 单线程C++','#8a99a6'),('cuda','NVIDIA GB10','#3185a6'),('hip','AMD RX9070XT','#bb5860')]):
  yy=998+j*45;sec=r['backends'][b]['seconds'];text(28,yy,label,17);s.append(f'<rect x="218" y="{yy-16}" width="{275*sec/maxs:.2f}" height="21" fill="{col}"/>');text(506,yy,f'{sec:.3f} s',16)
 text(28,1140,f'CUDA/HIP最终坐标L2={r["comparison"]["hip_vs_cuda"]["coordinate_relative_l2"]:.1e}，最终能量L2={r["comparison"]["hip_vs_cuda"]["energy_relative_l2"]:.1e}',16,col='#246d54')
 text(28,1175,f'18方向独立有限差分：max误差 {r["gradient_validation"]["max_abs_error"]:.2e}',16)
 text(28,1218,'旧RDKit 66.608秒：ETKDG + 真正MMFF94，任务不同',16,bold=True)
 text(28,1251,'旧8进程9.987秒也不是本原型CPU基线，不能直接算加速',16)
 text(28,1286,'本原型无氢显式优化、无参数校准；未宣称等价nvMolKit',16)
 text(28,1321,'碰撞：图距≥3且r<0.8σ；力场参数/误差详见JSON',16)
 text(28,1357,'计时为单次完整优化；传入/传出、新ETKDG时间另列',16)
 text(28,1397,'同源FP64、固定原子求和次序、禁fastmath；原创MIT',15,col='#677989');s.append('</svg>')
 out=ROOT/'visuals/fig13_open_nvmolkit_10000_conformers_amd_vs_nvidia.svg';out.write_text('\n'.join(s));import cairosvg;cairosvg.svg2png(url=str(out),write_to=str(out.with_suffix('.png')))
if __name__=='__main__':render()
