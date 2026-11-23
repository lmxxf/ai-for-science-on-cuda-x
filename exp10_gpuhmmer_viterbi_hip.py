#!/usr/bin/env python3
"""Real GB1-derived profile and synthetic 100k search database, native CPU/CUDA/HIP."""
import os
os.environ.setdefault('OMP_NUM_THREADS','4')
from pathlib import Path
import argparse,hashlib,json,subprocess,time
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score,roc_curve
ROOT=Path(__file__).resolve().parent;PORT=ROOT/'amd_hip_port/gpuhmmer_viterbi';DAT=ROOT/'results/module12_data';RES=ROOT/'results';BUILD=RES/'module12_build';AA='ACDEFGHIKLMNPQRSTVWY';SEED=20261123

def dump(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
def run(cmd,guard=None):
 if guard:cmd=[str(ROOT/'.venv/bin/python'),str(ROOT/'guard_process.py'),'--audit',str(ROOT/'logs'/guard),'--',*map(str,cmd)]
 subprocess.run(list(map(str,cmd)),check=True,cwd=ROOT)
def reference_score(seq,em,tr):
 # Independent 2D dynamic-programming table, as opposed to in-place GPU recurrence.
 L=len(seq);M=len(em);neg=-1e100
 m=np.full((L+1,M+1),neg);i=m.copy();d=m.copy();m[:,0]=0
 for a in range(1,L+1):
  for b in range(1,M+1):
   m[a,b]=max(m[a-1,b-1]+tr[0],i[a-1,b-1]+tr[3],d[a-1,b-1]+tr[5])+em[b-1,seq[a-1]]
   i[a,b]=max(m[a-1,b]+tr[1],i[a-1,b]+tr[4])
   if b>1:d[a,b]=max(m[a,b-1]+tr[2],d[a,b-1]+tr[6])
 return float(max(m[1:,-1].max(),d[1:,-1].max()))
def prepare(n=100000):
 DAT.mkdir(exist_ok=True);BUILD.mkdir(exist_ok=True)
 csv=ROOT/'data/gb1/splits/sampled.csv'
 if not csv.exists():
  import zipfile
  with zipfile.ZipFile(ROOT/'data/gb1_splits.zip') as z:z.extract('splits/sampled.csv',ROOT/'data/gb1')
 df=pd.read_csv(csv);seqs=df.sequence.str[:56].tolist();assert all(len(s)==56 and set(s)<=set(AA) for s in seqs)
 enc=np.array([[AA.index(c) for c in s] for s in seqs],dtype=np.uint8)
 counts=np.stack([np.bincount(enc[:,j],minlength=20) for j in range(56)]).astype(np.float64)
 emissions=np.log2((counts+.5)/(len(enc)+10)/.05)
 # Alignment has no gaps: only MM is observed. Indel rates are explicit priors, not fabricated gap observations.
 obs=len(enc);mm=(obs+19.2)/(obs+20);mi=.4/(obs+20);md=mi
 trans=np.log2(np.array([mm,mi,md,.7,.3,.7,.3],dtype=np.float64))
 rng=np.random.default_rng(SEED);lengths=rng.integers(128,257,size=n,dtype=np.int32);database=rng.integers(0,20,size=(n,256),dtype=np.uint8);labels=np.zeros(n,dtype=np.int32)
 positive=np.sort(rng.choice(n,500,replace=False));labels[positive]=1;mutated=[]
 for k,idx in enumerate(positive):
  domain=enc[rng.integers(len(enc))].copy();nm=int(k%12);sites=rng.choice(56,nm,replace=False);domain[sites]=rng.integers(20,size=nm,dtype=np.uint8);start=int(rng.integers(0,int(lengths[idx])-56+1));database[idx,start:start+56]=domain;mutated.append(nm)
 arrays={'sequences':database,'lengths':lengths,'emissions':emissions,'transitions':trans,'labels':labels}
 for name,a in arrays.items():a.tofile(DAT/f'{name}.bin')
 (DAT/'params.txt').write_text(f'{n} 256\n')
 meta={'seed':SEED,'sequences':n,'positives':500,'negatives':n-500,'length_range':[int(lengths.min()),int(lengths.max())],'residues':int(lengths.sum()),'real_profile_source':str(csv.relative_to(ROOT)),'real_profile_source_sha256':hashlib.sha256(csv.read_bytes()).hexdigest(),'real_profile_sequences':len(enc),'profile_length':56,'alphabet':AA,'profile':'position-specific emission log2 odds with symmetric 0.5 per-amino-acid pseudocount and uniform0.05 background','transition_probability':{'MM':mm,'MI':mi,'MD':md,'IM':.7,'II':.3,'DM':.7,'DD':.3},'transition_estimation':'gapless experimental GB1 alignments observe MM only; MI/MD and I/D continuation rates supplied by explicit Dirichlet/geometric priors, not inferred wet-experiment indels','database':'synthetic iid amino-acid background, 500 planted experimental-domain-derived sequences with 0..11 random point mutations; not 100000 experimentally annotated proteins','semantics':'glocal full 56-position domain, free restart over sequence; starts at match1, ends at match56 or delete56; insert emissions have background odds1; no HMMER Evalue calibration or MSV filter','hashes':{f'{k}.bin':hashlib.sha256((DAT/f'{k}.bin').read_bytes()).hexdigest() for k in arrays}}
 dump(RES/'module12_input.json',meta)
 run(['g++','-O3','-ffp-contract=off','-std=c++17',PORT/'viterbi.cpp','-o',BUILD/'viterbi_cpu'])
 # Validate both classes on a small independent 2D-table reference before the 100k run.
 ids=np.concatenate([np.arange(3),positive[:3]]);check=DAT/'small';check.mkdir(exist_ok=True)
 database[ids].tofile(check/'sequences.bin');lengths[ids].tofile(check/'lengths.bin');emissions.tofile(check/'emissions.bin');trans.tofile(check/'transitions.bin');(check/'params.txt').write_text(f'{len(ids)} 256\n')
 run([BUILD/'viterbi_cpu',check,RES/'module12_small_cpu','1'],guard='module12_small_memory.json')
 cp=np.fromfile(RES/'module12_small_cpu/scores.bin','<f8');rp=np.array([reference_score(database[ix,:lengths[ix]],emissions,trans) for ix in ids]);err=float(np.max(abs(cp-rp)));assert err==0
 dump(RES/'module12_reference_validation.json',{'checked_sequences':len(ids),'independent_2d_DP_max_abs':err,'sequence_indices':ids.tolist(),'scores':rp.tolist()})
 run([BUILD/'viterbi_cpu',DAT,RES/'module12_cpu','3'],guard='module12_cpu_memory.json')
 return meta

def gpu():
 run(['/usr/local/cuda/bin/nvcc','-x','cu','-O3','--fmad=false','-std=c++17','-arch=sm_121',PORT/'viterbi.cpp','-o',BUILD/'viterbi_cuda'])
 run([BUILD/'viterbi_cuda',DAT,RES/'module12_cuda','3'],guard='module12_cuda_memory.json')
 remote='C:/AI4S-HIP-Lab/gpuhmmer_viterbi'
 run(['ssh','amd9070',r'mkdir C:\AI4S-HIP-Lab\gpuhmmer_viterbi 2>NUL & exit /b 0'])
 run(['scp',PORT/'viterbi.cpp',PORT/'build_windows.cmd',PORT/'run_windows.cmd',f'amd9070:{remote}/'])
 run(['ssh','amd9070',r'mkdir C:\AI4S-HIP-Lab\gpuhmmer_viterbi\input 2>NUL & exit /b 0'])
 run(['scp',*sorted(DAT.glob('*.bin')),DAT/'params.txt',f'amd9070:{remote}/input/'])
 run(['ssh','amd9070',r'C:\AI4S-HIP-Lab\gpuhmmer_viterbi\build_windows.cmd'])
 run(['ssh','amd9070',r'C:\AI4S-HIP-Lab\gpuhmmer_viterbi\run_windows.cmd'])
 (RES/'module12_hip').mkdir(exist_ok=True)
 run(['scp',f'amd9070:{remote}/out_hip/scores.bin',f'amd9070:{remote}/out_hip/timings.json',RES/'module12_hip'])
 run(['scp',f'amd9070:{remote}/build.log',PORT/'build_windows.log'])
 summarize()
def summarize():
 meta=json.loads((RES/'module12_input.json').read_text());labels=np.fromfile(DAT/'labels.bin','<i4');scores={k:np.fromfile(RES/f'module12_{k}/scores.bin','<f8') for k in ['cpu','cuda','hip']};r={'input':meta,'reference_validation':json.loads((RES/'module12_reference_validation.json').read_text()),'backends':{},'comparisons':{}}
 for name,s in scores.items():
  assert len(s)==100000 and np.all(np.isfinite(s));t=json.loads((RES/f'module12_{name}/timings.json').read_text());fpr,tpr,_=roc_curve(labels,s);t.update(ROC_AUC=float(roc_auc_score(labels,s)),positive_score_min=float(s[labels==1].min()),positive_score_max=float(s[labels==1].max()),negative_score_min=float(s[labels==0].min()),negative_score_max=float(s[labels==0].max()),score_sha256=hashlib.sha256(s.astype('<f8').tobytes()).hexdigest(),fpr=fpr.tolist(),tpr=tpr.tolist());r['backends'][name]=t
 for name in ['cuda','hip']:
  difference=scores[name]-scores['cpu'];count=int(np.count_nonzero(difference));r['comparisons'][f'{name}_vs_cpu']={'mismatches':count,'max_abs_diff':float(abs(difference).max()),'byte_identical':scores[name].tobytes()==scores['cpu'].tobytes()};assert count==0
 r['cross_vendor_byte_identical']=scores['cuda'].tobytes()==scores['hip'].tobytes();assert r['cross_vendor_byte_identical']
 np.savez_compressed(RES/'module12_all_scores.npz',labels=labels,cpu=scores['cpu'],cuda=scores['cuda'],hip=scores['hip']);dump(RES/'module12_results.json',r);render(r)
def render(r=None):
 from html import escape
 import cairosvg
 if r is None:r=json.loads((RES/'module12_results.json').read_text())
 z=np.load(RES/'module12_all_scores.npz');score=z['hip'];labels=z['labels'];h=1510;s=[f'<svg xmlns="http://www.w3.org/2000/svg" width="640" height="{h}" font-family="\'WenQuanYi Micro Hei\',\'Noto Sans CJK SC\',sans-serif"><rect width="640" height="{h}" fill="#ffffff"/>']
 def text(x,y,t,size=18,col='#183247'):s.append(f'<text x="{x}" y="{y}" font-size="{size}" fill="{col}">{escape(str(t))}</text>')
 text(25,40,'蛋白序列搜索：10万条 Profile-HMM',24);text(25,76,'真实GB1发射模型 + 合成的阳性/诱饵搜索库',17)
 text(25,120,'① Match / Insert / Delete 动态规划',22)
 for row,letter,color in [(0,'M','#d9eceb'),(1,'I','#e3eaf6'),(2,'D','#f5e5de')]:
  text(30,190+row*70,letter,22)
  for j in range(5):
   x,y=90+j*100,157+row*70;s.append(f'<rect x="{x}" y="{y}" width="60" height="42" rx="8" fill="{color}" stroke="#90a5b1"/>');text(x+15,y+28,str(j+1),18)
   if j<4:s.append(f'<line x1="{x+60}" y1="{y+21}" x2="{x+100}" y2="{y+21}" stroke="#738b98" stroke-width="2"/>')
 text(25,400,'56位点；20AA发射log2几率；GPU按序列并行',17);text(25,432,'无gap实验只识别M→M，插入/删除概率用显式先验',16)
 text(25,480,'② 500个植入同源域 vs 99,500个随机背景',21)
 bins=np.linspace(score.min(),score.max(),65);hn,_=np.histogram(score[labels==0],bins=bins);hp,_=np.histogram(score[labels==1],bins=bins);lo,hi=bins[0],bins[-1]
 s.append('<rect x="60" y="515" width="530" height="230" fill="#fafafa" stroke="#ccd5dd"/>')
 for hist,col in [(hn,'#4787bb'),(hp,'#d96554')]:
  mx=max(hist.max(),1)
  for j,v in enumerate(hist):
   x=60+j*530/64;hh=200*v/mx;s.append(f'<rect x="{x:.3f}" y="{735-hh:.3f}" width="7.8" height="{hh:.3f}" fill="{col}" opacity="0.8"/>')
 text(60,777,f'bit-score {lo:.1f} → {hi:.1f}；两类各归一到自身峰值',16)
 text(25,814,f"AMD 检出ROC-AUC {r['backends']['hip']['ROC_AUC']:.6f}（合成检出任务）",20)
 text(25,862,'③ 真实100,000条完整动态规划计时',22)
 maxms=max(v['median_ms'] for v in r['backends'].values())
 for k,(name,col) in enumerate([('cpu','#8898a6'),('cuda','#2a86b0'),('hip','#c35c61')]):
  t=r['backends'][name];y=907+k*97;label={'cpu':'CPU 单线程C++','cuda':'NVIDIA GB10 CUDA','hip':'AMD RX9070XT HIP'}[name];text(25,y,label,18);w=400*t['median_ms']/maxms;s.append(f'<rect x="25" y="{y+12}" width="{max(w,3):.2f}" height="22" fill="{col}"/>');text(440,y+31,f"{t['median_ms']:.2f} ms",17);text(25,y+65,f"{t['residues_per_s']/1e6:.2f} 百万残基/秒",16)
 text(25,1240,'全部10万条score：CPU/CUDA/HIP逐值与SHA256一致',18,'#137b61');text(25,1277,'独立二维DP参考也通过，未调用HMMER校准/过滤器',16)
 text(25,1316,'驻留缓冲稳态墙钟；输入/输出传输另记JSON',16)
 text(25,1352,'这是glocal三状态原型，非官方GPU-HMMER完整包',17)
 text(25,1390,'阳性为植入域、背景为随机序列；不代表真实注释精度',16)
 text(25,1430,'原profile来自真实GB1实验序列；100k库不是湿实验',16)
 s.append('</svg>');out=ROOT/'visuals/fig11_gpuhmmer_100k_protein_viterbi.svg';out.write_text('\n'.join(s));cairosvg.svg2png(url=str(out),write_to=str(out.with_suffix('.png')))
def main():
 p=argparse.ArgumentParser();p.add_argument('--stage',choices=['prepare','gpu','summarize','render','all'],default='all');a=p.parse_args()
 if a.stage in ['prepare','all']:prepare()
 if a.stage in ['gpu','all']:gpu()
 if a.stage=='summarize':summarize()
 if a.stage=='render':render()
if __name__=='__main__':main()
