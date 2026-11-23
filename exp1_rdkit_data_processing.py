#!/usr/bin/env python3
"""Real MoleculeNet ETKDGv3/MMFF94 measurements; no predicted labels."""
import argparse, concurrent.futures, hashlib, json, pickle, time, urllib.request
from pathlib import Path
import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger, rdBase
from rdkit.Chem import AllChem, Descriptors, Crippen, rdMolDescriptors, SaltRemover, rdDepictor
from rdkit.Chem.Draw import rdMolDraw2D
from html import escape
ROOT=Path(__file__).resolve().parent
RDLogger.DisableLog('rdApp.*')
SOURCES={'delaney-processed.csv':'https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/delaney-processed.csv','BBBP.csv':'https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/BBBP.csv'}
DRUGS={'苯海拉明':'CN(C)CCOC(c1ccccc1)c2ccccc2','盐酸苯海拉明脏记录':'CN(C)CCOC(c1ccccc1)c2ccccc2.Cl','非索非那定':'CC(C)(C1=CC=C(C=C1)C(CCCN2CCC(CC2)C(C3=CC=CC=C3)(C4=CC=CC=C4)O)O)C(=O)O','氯雷他定':'CCOC(=O)N1CCC(=C2c3ccc(Cl)cc3CCc3cccnc32)CC1','西替利嗪':'O=C(O)COCCN1CCN(C(c2ccccc2)c2ccc(Cl)cc2)CC1','阿司匹林':'CC(=O)Oc1ccccc1C(=O)O','咖啡因':'Cn1c(=O)c2c(ncn2C)n(C)c1=O','布洛芬':'CC(C)Cc1ccc(C(C)C(=O)O)cc1'}
def clean(smiles):
 m=Chem.MolFromSmiles(str(smiles))
 if m is None:return None
 m=SaltRemover.SaltRemover().StripMol(m,dontRemoveEverything=True)
 fr=Chem.GetMolFrags(m,asMols=True,sanitizeFrags=True)
 return max(fr,key=lambda q:q.GetNumHeavyAtoms()) if fr else None

def conformer(job):
 idx,smiles,target,nconf=job
 st=time.perf_counter(); m=Chem.AddHs(Chem.MolFromSmiles(smiles))
 p=AllChem.ETKDGv3();p.randomSeed=20261123;p.numThreads=1;p.maxIterations=1000
 ids=list(AllChem.EmbedMultipleConfs(m,numConfs=nconf,params=p))
 states=[]
 if ids and AllChem.MMFFHasAllMoleculeParams(m):
  states=list(AllChem.MMFFOptimizeMoleculeConfs(m,numThreads=1,maxIters=200,mmffVariant='MMFF94'))
 result={'source_index':idx,'smiles':smiles,'target':float(target),'seconds':time.perf_counter()-st,'nconfs':len(ids),'mmff_converged':sum(x[0]==0 for x in states),'mmff_supported':bool(states)}
 if ids:
  # First generated conformer, not label-selected or energy-selected.
  heavy=Chem.RemoveHs(m);result['atoms']=[a.GetSymbol() for a in heavy.GetAtoms()];result['coordinates']=np.asarray(heavy.GetConformer(0).GetPositions(),dtype=np.float32)
 return result

def prepare():
 (ROOT/'data').mkdir(exist_ok=True);(ROOT/'results').mkdir(exist_ok=True);(ROOT/'visuals').mkdir(exist_ok=True)
 stats={}
 for filename,url in SOURCES.items():
  path=ROOT/'data'/filename
  if not path.exists():urllib.request.urlretrieve(url,path)
  df=pd.read_csv(path); rows=[];st=time.perf_counter()
  target='measured log solubility in mols per litre' if filename.startswith('delaney') else 'p_np'
  for i,row in df.iterrows():
   m=clean(row['smiles'])
   if m is not None:rows.append({'source_index':i,'smiles':Chem.MolToSmiles(m),'target':row[target],'MW':Descriptors.MolWt(m),'LogP':Crippen.MolLogP(m),'TPSA':rdMolDescriptors.CalcTPSA(m)})
  name='esol' if filename.startswith('delaney') else 'bbbp'
  pd.DataFrame(rows).to_csv(ROOT/'data'/f'{name}_clean.csv',index=False)
  stats[name]={'raw_rows':len(df),'valid_rows':len(rows),'invalid_rows':len(df)-len(rows),'clean_and_descriptors_seconds':time.perf_counter()-st,'source':url,'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
 drugs={}
 for name,s in DRUGS.items():
  m=clean(s);drugs[name]={'raw_smiles':s,'smiles':Chem.MolToSmiles(m),'MW':Descriptors.MolWt(m),'LogP':Crippen.MolLogP(m),'TPSA':rdMolDescriptors.CalcTPSA(m)}
 stats['drugs']=drugs;stats['invalid_demo']={'smiles':'C(C)(C)(C)(C)C','rejected':Chem.MolFromSmiles('C(C)(C)(C)(C)C') is None};stats['rdkit_version']=rdBase.rdkitVersion
 (ROOT/'results/rdkit_preparation.json').write_text(json.dumps(stats,ensure_ascii=False,indent=2))
 print(json.dumps(stats,ensure_ascii=False,indent=2),flush=True)
 return stats

def benchmark(n=1000,workers=8):
 df=pd.read_csv(ROOT/'data/esol_clean.csv').iloc[:n]
 timings=[]
 for nc,nt in [(1,1),(1,workers),(10,1),(10,workers)]:
  jobs=[(int(r.source_index),r.smiles,r.target,nc) for r in df.itertuples()]
  start=time.perf_counter()
  if nt==1:results=[conformer(j) for j in jobs]
  else:
   with concurrent.futures.ProcessPoolExecutor(max_workers=nt) as pool:results=list(pool.map(conformer,jobs,chunksize=8))
  elapsed=time.perf_counter()-start
  item={'molecules':len(jobs),'requested_conformers_per_molecule':nc,'workers':nt,'wall_seconds':elapsed,'generated_conformers':sum(r['nconfs'] for r in results),'molecule_successes':sum(r['nconfs']>0 for r in results),'mmff_converged_conformers':sum(r['mmff_converged'] for r in results),'sum_worker_seconds':sum(r['seconds'] for r in results)}
  timings.append(item)
  (ROOT/'results/rdkit_timings.json').write_text(json.dumps(timings,indent=2))
  if nc==1 and nt==1:
   extra_df=pd.read_csv(ROOT/'data/esol_clean.csv').iloc[n:]
   extra_start=time.perf_counter()
   extra=[conformer((int(r.source_index),r.smiles,r.target,1)) for r in extra_df.itertuples()]
   with (ROOT/'results/esol_conformers.pkl').open('wb') as f:pickle.dump(results+extra,f)
   (ROOT/'results/rdkit_full_conformer_time.json').write_text(json.dumps({'molecules':len(results)+len(extra),'wall_seconds':elapsed+time.perf_counter()-extra_start,'benchmark_first_n':n,'single_core':True}))
  print(json.dumps(item),flush=True)
 return timings

def svg_start(height):return f'''<svg xmlns="http://www.w3.org/2000/svg" width="640" height="{height}" viewBox="0 0 640 {height}" font-family="'WenQuanYi Micro Hei','Noto Sans CJK SC',sans-serif"><rect width="640" height="{height}" fill="#ffffff"/>'''
def txt(x,y,s,size=18,color='#1e293b'):return f'<text x="{x}" y="{y}" font-size="{size}" fill="{color}">{escape(str(s))}</text>'
def mol_svg(m,x,y,w,h,highlight=None,indices=False):
 m=Chem.Mol(m);rdDepictor.Compute2DCoords(m)
 d=rdMolDraw2D.MolDraw2DSVG(w,h);o=d.drawOptions();o.addAtomIndices=indices;o.clearBackground=False
 d.DrawMolecule(m,highlightAtoms=highlight or []);d.FinishDrawing();s=d.GetDrawingText();s=s[s.index('<svg'):];s=s.replace('<svg ',f'<svg x="{x}" y="{y}" ',1)
 return s

def figures():
 stats=json.loads((ROOT/'results/rdkit_preparation.json').read_text());drugs=stats['drugs']
 s=svg_start(1260)+txt(24,38,'RDKit：从脏记录到可计算的药物',25)+txt(24,72,'真实结构计算；描述符不是临床穿脑结论',17,'#64748b')
 s+=txt(24,114,'① 去盐：分开记录母体与盐型',22)+txt(24,144,DRUGS['盐酸苯海拉明脏记录'],15)
 s+=mol_svg(Chem.MolFromSmiles(DRUGS['盐酸苯海拉明脏记录']),20,155,290,190)+txt(307,252,'→',28)+mol_svg(clean(DRUGS['苯海拉明']),340,155,280,190)
 s+=txt(24,369,'保留有机母体；去掉 .Cl 后不再代表完整盐型质量。',17)+txt(24,410,'非法五价碳：C(C)(C)(C)(C)C → 拒绝解析',17,'#b91c1c')
 s+=txt(24,461,'② 一代与二代抗过敏药：看极性差异',22)
 for y,name,label in [(500,'苯海拉明','第一代：可引起嗜睡'),(805,'非索非那定','第二代：通常较少嗜睡')]:
  m=clean(DRUGS[name]);pol=[a.GetIdx() for a in m.GetAtoms() if a.GetSymbol()=='O']
  s+=txt(24,y,name+'｜'+label,20)+mol_svg(m,30,y+12,570,200,pol)
  d=drugs[name];s+=txt(24,y+240,f"MW {d['MW']:.2f}   LogP {d['LogP']:.2f}   TPSA {d['TPSA']:.2f} Å²",19)
 s+=txt(24,1103,'浅红高亮：含氧极性部位。数值按图中中性结构计算。',16)+txt(24,1141,'血脑屏障还受电离状态、P-gp 外排、剂量等影响。',17)+txt(24,1176,'TPSA 不是“超过 90 就绝不穿脑”的硬开关。',17)+txt(24,1220,'数据：MoleculeNet ESOL / BBBP；结构与数值：RDKit',15,'#64748b')+'</svg>'
 (ROOT/'visuals/fig1_rdkit_cleaning_and_bbb_drugs.svg').write_text(s)
 with (ROOT/'results/esol_conformers.pkl').open('rb') as f:cached=pickle.load(f)
 choices=[]
 for r in cached:
  if 'coordinates' not in r or r['mmff_converged']!=1:continue
  m=Chem.MolFromSmiles(r['smiles']);top=Chem.GetDistanceMatrix(m);xyz=r['coordinates'];dist=np.linalg.norm(xyz[:,None]-xyz[None,:],axis=-1)
  for i,j in zip(*np.where(np.triu((top>=7)&(dist<3.5)&(dist>2.8),1))):
   path=Chem.GetShortestPath(m,int(i),int(j))
   flexible=sum(not m.GetBondBetweenAtoms(a,b).IsInRing() and m.GetBondBetweenAtoms(a,b).GetBondType()==Chem.BondType.SINGLE for a,b in zip(path,path[1:]))
   if flexible>=4:choices.append((float(dist[i,j]),r,i,j,int(top[i,j])))
 choices.sort(key=lambda v:(v[1]['source_index']!=48,v[0]));distance,r,i,j,bonds=choices[0];m=Chem.MolFromSmiles(r['smiles']);xyz=r['coordinates'];dist=np.linalg.norm(xyz[:,None]-xyz[None,:],axis=-1)
 np.savez(ROOT/'results/folded_molecule_distance.npz',coordinates=xyz,distance=dist,atom_symbols=np.array(r['atoms']))
 name=str(pd.read_csv(ROOT/'data/delaney-processed.csv').iloc[r['source_index']]['Compound ID'])
 case={k:v for k,v in r.items() if k!='coordinates'};case['compound_name']=name;case.update({'atom_pair_zero_based':[int(i),int(j)],'topological_bonds':bonds,'distance_angstrom':distance,'coordinates':xyz.tolist()});(ROOT/'results/folded_molecule.json').write_text(json.dumps(case,ensure_ascii=False,indent=2))
 s=svg_start(1580)+txt(24,38,'同一个分子：字符串 → 结构 → 空间',24)+txt(24,74,f'{name}（硝苯地平）｜已收敛 MMFF94' if name.lower()=='nifedipine' else f'{name}｜已收敛 MMFF94',17)
 s+=txt(24,116,'① SMILES 是连接说明，不是空间坐标',21)
 import textwrap
 for k,line in enumerate(textwrap.wrap(r['smiles'],66)):s+=txt(24,150+k*22,line,15)
 s+=txt(24,224,f'② 二维结构：原子 {i} 与 {j} 隔 {bonds} 根键',21)+mol_svg(m,25,240,590,270,[int(i),int(j)],True)
 s+=txt(24,541,f'③ 同一构象的三维距离：{distance:.3f} Å',22)
 # Orthographic projection of actual xyz coordinates, centered and rotated for visibility.
 rot=np.array([[.866,-.25,.433],[0,.866,.5],[-.5,-.433,.75]])
 proj=(xyz-xyz.mean(0))@rot;scale=min(490/(np.ptp(proj[:,0])+1e-6),230/(np.ptp(proj[:,1])+1e-6));px=320+proj[:,0]*scale;py=700-proj[:,1]*scale
 for bond in m.GetBonds():
  a,b=bond.GetBeginAtomIdx(),bond.GetEndAtomIdx();s+=f'<line x1="{px[a]}" y1="{py[a]}" x2="{px[b]}" y2="{py[b]}" stroke="#94a3b8" stroke-width="3"/>'
 s+=f'<line x1="{px[i]}" y1="{py[i]}" x2="{px[j]}" y2="{py[j]}" stroke="#ef4444" stroke-width="3" stroke-dasharray="7 4"/>'
 colors={'C':'#475569','O':'#dc2626','N':'#2563eb','Cl':'#16a34a','S':'#d97706'}
 for a in np.argsort(proj[:,2]):
  color='#f59e0b' if a in [i,j] else colors.get(r['atoms'][a],'#a855f7');s+=f'<circle cx="{px[a]}" cy="{py[a]}" r="8" fill="{color}"/>'
  if a in [i,j]:s+=txt(px[a]+10,py[a]-8,str(a),16)
 s+=txt(24,851,'红虚线：非键合原子接触；球棍来自计算 XYZ 正交投影。',16)
 for k,a in enumerate([i,j]):s+=txt(24,888+32*k,f"原子 {a} ({r['atoms'][a]})  XYZ = ({xyz[a,0]:.3f}, {xyz[a,1]:.3f}, {xyz[a,2]:.3f}) Å",17)
 s+=txt(24,982,'④ 全部重原子的欧氏距离矩阵 D[i,j]',21)
 n=len(xyz);cell=min(11,340/n);ox=145;oy=1005;maxd=dist.max()
 for a in range(n):
  for b in range(n):
   v=float(dist[a,b]/maxd);red=int(245-190*v);green=int(248-120*v);blue=int(255-40*v)
   s+=f'<rect x="{ox+b*cell}" y="{oy+a*cell}" width="{cell+.1}" height="{cell+.1}" fill="rgb({red},{green},{blue})"/>'
 for a,b in [(i,j),(j,i)]:s+=f'<rect x="{ox+b*cell}" y="{oy+a*cell}" width="{cell}" height="{cell}" fill="none" stroke="#ef4444" stroke-width="2"/>'
 s+=txt(24,1415,'D[i,j] = √((xi−xj)² + (yi−yj)² + (zi−zj)²)',18)+txt(24,1452,'颜色越深距离越远；红框定位上面那对原子。',17)+txt(24,1487,'模型生成构象，非实验实测；环境可改变折叠。',17)+txt(24,1525,f"ESOL 原始行 {r['source_index']}；原子索引从 0 开始；单位 Å。",15,'#64748b')+'</svg>'
 (ROOT/'visuals/fig2_1d_smiles_to_3d_conformer_distance.svg').write_text(s)
 import cairosvg
 for name in ['fig1_rdkit_cleaning_and_bbb_drugs','fig2_1d_smiles_to_3d_conformer_distance']:cairosvg.svg2png(url=str(ROOT/'visuals'/f'{name}.svg'),write_to=str(ROOT/'visuals'/f'{name}.png'))
 print('FIGURES COMPLETE',case['atom_pair_zero_based'],distance,flush=True)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--stage',choices=['prepare','benchmark','figures','all'],default='all');p.add_argument('--molecules',type=int,default=1000);p.add_argument('--workers',type=int,default=8);a=p.parse_args()
 if a.stage in ['prepare','all']:prepare()
 if a.stage in ['benchmark','all']:benchmark(a.molecules,a.workers)
 if a.stage in ['figures','all']:figures()
 if (ROOT/'results/rdkit_preparation.json').exists():
  summary=json.loads((ROOT/'results/rdkit_preparation.json').read_text())
  if (ROOT/'results/rdkit_timings.json').exists():summary['benchmarks']=json.loads((ROOT/'results/rdkit_timings.json').read_text())
  if (ROOT/'results/rdkit_full_conformer_time.json').exists():summary['full_single_conformer']=json.loads((ROOT/'results/rdkit_full_conformer_time.json').read_text())
  if (ROOT/'results/drug_structure_audit.json').exists():summary['drug_structure_audit']=json.loads((ROOT/'results/drug_structure_audit.json').read_text())
  if (ROOT/'results/folded_molecule.json').exists():summary['folded_molecule']=json.loads((ROOT/'results/folded_molecule.json').read_text())
  (ROOT/'results/module1_results.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2))
