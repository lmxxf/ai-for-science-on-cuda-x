#!/usr/bin/env python3
from pathlib import Path
import json, math
import numpy as np
from xml.sax.saxutils import escape
ROOT=Path(__file__).resolve().parent

def render():
 r=json.loads((ROOT/'results/module3_results.json').read_text());amd_path=ROOT/'results/module5_amd9070_results.json';amd=json.loads(amd_path.read_text()) if amd_path.exists() else None;d=np.load(ROOT/'results/module3_preview.npz');p=d['polys'];pts=d['points'];genes=d['genes'];s=[]
 s.append('''<svg xmlns="http://www.w3.org/2000/svg" width="640" height="1390" viewBox="0 0 640 1390" font-family="'WenQuanYi Micro Hei','Noto Sans CJK SC',sans-serif"><rect width="640" height="1390" fill="#ffffff"/>''')
 def txt(x,y,text,size=18,color='#263546',weight='normal'):s.append(f'<text x="{x}" y="{y}" font-size="{size}" fill="{color}" font-weight="{weight}">{escape(str(text))}</text>')
 def line(x1,y1,x2,y2,color='#c9d2dc',dash=''):s.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" stroke-width="1.4" stroke-dasharray="{dash}"/>')
 txt(30,43,'cuSpa：把基因点分配到细胞',27,weight='bold');txt(30,75,'CUDA → HIP：GB10 与 RX 9070 XT 真机对账' if amd else '原版 CUDA 真机编译 · DGX Spark / GB10',18)
 txt(30,113,'① 网格索引：先找少量候选细胞',22,weight='bold');txt(30,143,'合成八边形与转录点：演示计算，不冒充组织样本',16)
 ox,oy,scale=40,170,13
 for x in range(0,41,10):line(ox+x*scale,oy,ox+x*scale,oy+30*scale)
 for y in range(0,31,10):line(ox,oy+y*scale,ox+40*scale,oy+y*scale)
 for poly in p:
  if poly[:,0].mean()>40:continue
  xy=' '.join(f'{ox+x*scale:.2f},{oy+y*scale:.2f}' for x,y in poly)
  s.append(f'<polygon points="{xy}" fill="#e8f5f3" stroke="#217b74" stroke-width="1.8"/>')
 for xy,g in zip(pts[:120],genes[:120]):s.append(f'<circle cx="{ox+xy[0]*scale:.2f}" cy="{oy+xy[1]*scale:.2f}" r="2.8" fill="{["#dd5361","#3478cc"][g%2]}"/>')
 # Highlight one point/ray in the first polygon, through its center.
 x,y=ox+5*scale,oy+5*scale
 line(x,y,ox+9.6*scale,y,'#e18725','5 4');s.append(f'<circle cx="{x}" cy="{y}" r="5" fill="#e18725"/>')
 txt(30,594,'② 射线法：奇数次穿边界＝点在细胞内',22,weight='bold');txt(30,624,'③ (细胞ID, 基因ID) 排序 → 游程计数 → CSR',18)
 txt(30,653,'源码纠偏：atomicAdd 用于网格计数，非基因聚合',16,color='#8b4b19')
 txt(30,696,'50,000 细胞 × 2,000,000 点 × 500 基因',20,weight='bold')
 cpu=r['cpu_pipeline_median_s']*1000;gpu=r['gpu_pipeline_median_s']*1000
 txt(30,735,'索引 + 查询 + 稀疏计数（稳态中位数）',18)
 bars=[(773,'CPU 单线程编译代码',cpu,'#788897'),(838,'NVIDIA GB10 · Python 原入口',gpu,'#227eaf')]
 if amd:bars.append((903,'AMD RX 9070 XT · HIP C++',amd['amd']['pipeline_median_wall_ms'],'#b84040'))
 for y,title,value,col in bars:
  txt(30,y,title,18);s.append(f'<rect x="30" y="{y+9}" width="{max(5,value/cpu*420):.2f}" height="18" rx="3" fill="{col}"/>');txt(465,y+25,f'{value:.2f} ms',18)
 if amd:
  aa=amd['amd'];nv=amd['gb10_same_harness']
  txt(30,970,'同源 C++ 入口：计时口径再对齐',21,weight='bold')
  txt(30,1002,f"GB10 {nv['pipeline_median_wall_ms']:.2f} ms；RX 9070 XT {aa['pipeline_median_wall_ms']:.2f} ms",18)
  txt(30,1035,'AMD 各阶段：索引 / 查询 / CSR 聚合',17)
  txt(30,1066,f"{aa['index_median_wall_ms']:.2f} / {aa['query_median_wall_ms']:.2f} / {aa['aggregate_median_wall_ms']:.2f} ms",20,color='#a42e34',weight='bold')
  txt(30,1100,f"AMD 传入 {aa['h2d_ms']:.2f} ms；回传 {aa['d2h_ms']:.2f} ms（另计）",17)
  txt(30,1137,'200 万点归属 + 全部 CSR：逐字节零差异',20,color='#136d58',weight='bold')
  txt(30,1168,'四个输出数组的 SHA256 与 GB10 全部相同',17)
  txt(30,1207,'float64；输入为同一份二进制；种子 20261123',16)
  txt(30,1237,'驻留稳态墙钟中位数；传输、冷启动分别记录',16)
  txt(30,1267,'Python / C++ 入口有开销差异；同源入口单列',16)
  txt(30,1307,'已真机移植：通用算术 + HIP / hipCUB',19,weight='bold')
  txt(30,1337,'未用 Tensor Core；不将此结果外推为所有科学库',16)
  txt(30,1367,'原版 cuspa commit 2df9fa78 · 详细原始数组见 results/',15,color='#667789')
 else:
  txt(30,1000,f'设备驻留 {r["pipeline_speedup"]:.1f} 倍 · 单查询 {r["query_speedup"]:.1f} 倍',23,color='#136d58',weight='bold')
  txt(30,1040,f'完整校验：{r["points"]:,} 个归属、CSR 计数均零差异',17)
  txt(30,1080,'双方 float64，同类网格 + 射线法；非纯 Python 对比',16)
  txt(30,1120,'传入 14.50 ms；加驻留流程约 24.14 ms（推算）',16)
 s.append('</svg>');out=ROOT/'visuals/fig4_cuspa_cell_polygon_raycasting.svg';out.write_text('\n'.join(s))
 import cairosvg
 cairosvg.svg2png(url=str(out),write_to=str(out.with_suffix('.png')))
if __name__=='__main__':render()
