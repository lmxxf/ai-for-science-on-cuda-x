#!/usr/bin/env python3
"""BBBP full-parameter Uni-Mol fine-tune, held-out evaluation, atom occlusion."""
import os
os.environ.setdefault('OMP_NUM_THREADS','4');os.environ.setdefault('OPENBLAS_NUM_THREADS','4')
from pathlib import Path
ROOT=Path(__file__).resolve().parent;os.environ['UNIMOL_WEIGHT_DIR']=str(ROOT/'weights')
import argparse,concurrent.futures,hashlib,json,pickle,random,time
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from sklearn.model_selection import train_test_split
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score,average_precision_score,accuracy_score,roc_curve
from unimol_tools.models.unimol import UniMolModel
from unimol_tools.data.conformer import coords2unimol
from exp1_rdkit_data_processing import conformer,DRUGS,mol_svg

def metrics(y,p):
 fpr,tpr,_=roc_curve(y,p)
 return {'ROC_AUC':float(roc_auc_score(y,p)),'PR_AUC_AP':float(average_precision_score(y,p)),'Accuracy':float(accuracy_score(y,np.asarray(p)>=.5)),'fpr':fpr.tolist(),'tpr':tpr.tolist()}
def stage_train(epochs=20):
 random.seed(42);np.random.seed(42);torch.manual_seed(42);torch.cuda.manual_seed_all(42);torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=True
 df=pd.read_csv(ROOT/'data/bbbp_clean.csv');groups=df.groupby('smiles',sort=False);conflicts=[s for s,g in groups if g.target.nunique()>1];df=df[~df.smiles.isin(conflicts)].drop_duplicates('smiles',keep='first').reset_index(drop=True)
 cache=ROOT/'results/bbbp_conformers.pkl';t=time.perf_counter()
 if not cache.exists():
  jobs=[(int(r.source_index),r.smiles,float(r.target),1) for r in df.itertuples()]
  with concurrent.futures.ProcessPoolExecutor(max_workers=8) as pool:records=list(pool.map(conformer,jobs,chunksize=8))
  pickle.dump(records,open(cache,'wb'));cpu_seconds=time.perf_counter()-t
  (ROOT/'results/bbbp_conformer_timing.json').write_text(json.dumps({'seconds':cpu_seconds,'workers':8,'attempted':len(records),'successes':sum('coordinates' in r for r in records),'MMFF_converged':sum(r['mmff_converged']>0 for r in records)}))
 else:records=pickle.load(open(cache,'rb'));cpu_seconds=json.loads((ROOT/'results/bbbp_conformer_timing.json').read_text())['seconds']
 records=[r for r in records if 'coordinates' in r];y=np.asarray([r['target'] for r in records],dtype=np.float32);idx=np.arange(len(y));tr,rest=train_test_split(idx,test_size=.2,stratify=y,random_state=42);va,te=train_test_split(rest,test_size=.5,stratify=y[rest],random_state=42)
 (ROOT/'results/module6_split.json').write_text(json.dumps({'train':tr.tolist(),'validation':va.tolist(),'test':te.tolist(),'source_index':[r['source_index'] for r in records],'smiles':[r['smiles'] for r in records],'removed_conflicting_smiles':conflicts},indent=2))
 generator=rdFingerprintGenerator.GetMorganGenerator(radius=2,fpSize=2048);X=np.asarray([generator.GetFingerprintAsNumPy(Chem.MolFromSmiles(r['smiles'])) for r in records]);baseline={}
 for name,m in [('Dummy',DummyClassifier(strategy='prior')),('Morgan_RF',RandomForestClassifier(n_estimators=400,n_jobs=4,random_state=42))]:
  m.fit(X[tr],y[tr]);baseline[name]=metrics(y[te],m.predict_proba(X[te])[:,1])
 model=UniMolModel(output_dim=1,remove_hs=True).cuda();features=[coords2unimol(r['atoms'],r['coordinates'],model.dictionary,remove_hs=True) for r in records]
 def loader(ix,shuffle=False):return DataLoader([(features[i],float(y[i])) for i in ix],batch_size=32,shuffle=shuffle,num_workers=0,collate_fn=model.batch_collate_fn)
 train=loader(tr,True);val=loader(va);test=loader(te);optim=torch.optim.AdamW(model.parameters(),lr=1e-4,weight_decay=.01);sched=torch.optim.lr_scheduler.CosineAnnealingLR(optim,T_max=epochs,eta_min=1e-5)
 def predict(dl):
  model.eval();out=[]
  with torch.no_grad():
   for b,_ in dl:
    with torch.autocast('cuda',dtype=torch.bfloat16):p=model(**{k:v.cuda() for k,v in b.items()}).reshape(-1)
    out.extend(torch.sigmoid(p.float()).cpu().numpy().tolist())
  return np.asarray(out)
 logs=[];best=-1;bestepoch=0;start=time.perf_counter();(ROOT/'results/module6_epochs.jsonl').write_text('')
 for ep in range(1,epochs+1):
  model.train();total=0;count=0;t=time.perf_counter()
  for b,label in train:
   optim.zero_grad(set_to_none=True);label=label.cuda().float()
   with torch.autocast('cuda',dtype=torch.bfloat16):pred=model(**{k:v.cuda() for k,v in b.items()}).reshape(-1);loss=torch.nn.functional.binary_cross_entropy_with_logits(pred.float(),label)
   loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optim.step();total+=float(loss.detach())*len(label);count+=len(label)
  vm=metrics(y[va],predict(val));torch.cuda.synchronize();entry={'epoch':ep,'train_BCE':total/count,'validation':vm,'seconds':time.perf_counter()-t};logs.append(entry);print(json.dumps(entry),flush=True)
  with open(ROOT/'results/module6_epochs.jsonl','a') as f:f.write(json.dumps(entry)+'\n')
  if vm['ROC_AUC']>best:best=vm['ROC_AUC'];bestepoch=ep;torch.save(model.state_dict(),ROOT/'results/unimol_bbbp_best.pt')
  sched.step()
 elapsed=time.perf_counter()-start;model.load_state_dict(torch.load(ROOT/'results/unimol_bbbp_best.pt',map_location='cuda',weights_only=True));tp=predict(test)
 result={'task':'BBBP','raw_clean_valid':2039,'unique_before_conformer':len(df),'n_molecules':len(records),'conflicting_parent_count':len(conflicts),'conflict_policy':'remove every record belonging to canonical-parent groups with conflicting 0/1 labels','duplicates_same_label_removed':2039-sum(len(g) for s,g in groups if s in conflicts)-len(df),'split_sizes':[len(tr),len(va),len(te)],'split':'stratified random80/10/10 seed42 grouped by exact canonical cleaned parent','class_counts':{'negative':int(sum(y==0)),'positive':int(sum(y==1))},'baselines':baseline,'unimol_test':metrics(y[te],tp),'best_epoch':bestepoch,'epochs':logs,'training_seconds':elapsed,'cpu_conformer_seconds':cpu_seconds,'test_source_indices':[records[i]['source_index'] for i in te],'test_labels':y[te].tolist(),'test_probabilities':tp.tolist(),'parameters':sum(p.numel() for p in model.parameters()),'peak_cuda_gib':torch.cuda.max_memory_allocated()/1024**3,'attribution':attribution(model)}
 
 for name,a in result['attribution'].items():
  canonical=Chem.MolToSmiles(Chem.MolFromSmiles(a['smiles']));matches=[i for i,row in enumerate(records) if row['smiles']==canonical];a['dataset_split_membership']=[part for part,ids in [('train',tr),('validation',va),('test',te)] if any(i in ids for i in matches)] or ['not_present']
 result['conformer_failures']=len(df)-len(records);result['MMFF_converged']=sum(row['mmff_converged']>0 for row in records);result['MMFF_unconverged_retained']=sum(row['mmff_converged']==0 for row in records);result['test_evaluation_policy']='Each model test evaluated once, neural checkpoint selected by validation ROC-AUC only.'
 path=ROOT/'results/module6_results.json';path.write_text(json.dumps(result,ensure_ascii=False,indent=2));print('BBBP DONE',result['unimol_test']['ROC_AUC'],flush=True)
 return result

def attribution(model):
 model.eval();out={}
 for name in ['苯海拉明','非索非那定']:
  r=conformer((0,DRUGS[name],0,1));a=r['atoms'];xyz=r['coordinates'];features=[]
  features.append(coords2unimol(a,xyz,model.dictionary,remove_hs=True))
  for i in range(len(a)):
   keep=np.arange(len(a))!=i;features.append(coords2unimol(np.asarray(a)[keep],xyz[keep],model.dictionary,remove_hs=True))
  vals=[]
  with torch.no_grad():
   for offset in range(0,len(features),16):
    b,_=model.batch_collate_fn([(f,0.) for f in features[offset:offset+16]]);vals.extend(model(**{k:v.cuda() for k,v in b.items()}).reshape(-1).float().cpu().tolist())
  out[name]={'smiles':DRUGS[name],'atoms':a,'coordinates':xyz.tolist(),'full_logit':vals[0],'probability':float(torch.sigmoid(torch.tensor(vals[0]))),'occluded_logits':vals[1:],'delta_logit':(vals[0]-np.asarray(vals[1:])).tolist(),'definition':'full_logit minus logit after deleting this heavy-atom token and its distance row/column; no chemical re-sanitization or geometry relaxation','MMFF_converged':r['mmff_converged']}
 return out

def render():
 from html import escape
 from rdkit.Chem.Draw import rdMolDraw2D
 from rdkit.Chem import rdDepictor
 import cairosvg
 r=json.loads((ROOT/'results/module6_results.json').read_text());h=1550;s=[f'<svg xmlns="http://www.w3.org/2000/svg" width="640" height="{h}" font-family="\'WenQuanYi Micro Hei\',\'Noto Sans CJK SC\',sans-serif"><rect width="640" height="{h}" fill="#ffffff"/>']
 def txt(x,y,t,size=17,color='#203449'):s.append(f'<text x="{x}" y="{y}" font-size="{size}" fill="{color}">{escape(str(t))}</text>')
 txt(25,38,'从血脑屏障分类到 AMD 注意力核',24);txt(25,72,f"BBBP：{r['n_molecules']} 个去重分子；训练/验证/测试 "+'/'.join(map(str,r['split_sizes'])),15)
 txt(25,105,f"冲突标签母核 {r['conflicting_parent_count']} 个全部排除；验证集选第 {r['best_epoch']} 轮",16)
 x0,y0,w,hh=65,140,510,250;s.append(f'<rect x="{x0}" y="{y0}" width="{w}" height="{hh}" fill="#fafafa" stroke="#ddd"/>');s.append(f'<line x1="{x0}" y1="{y0+hh}" x2="{x0+w}" y2="{y0}" stroke="#999" stroke-dasharray="4 4"/>')
 for (label,v),col in zip(list(r['baselines'].items())+[('UniMol',r['unimol_test'])],['#999','#4586bd','#129f82']):
  p=' '.join(f'{x0+a*w:.2f},{y0+hh-b*hh:.2f}' for a,b in zip(v['fpr'],v['tpr']));s.append(f'<polyline points="{p}" fill="none" stroke="{col}" stroke-width="2.5"/>')
 for i,(label,v) in enumerate(list(r['baselines'].items())+[('UniMol',r['unimol_test'])]):txt(35,424+27*i,f"{label}  ROC-AUC {v['ROC_AUC']:.3f}  PR-AP {v['PR_AUC_AP']:.3f}",17)
 txt(25,525,'逐重原子遮挡：模型依赖，不是化学因果',21);txt(25,555,'绿：完整 logit 比遮挡后高；红：完整 logit 比遮挡后低',16)
 for k,(name,a) in enumerate(r['attribution'].items()):
  y=594+k*300;txt(25,y,f"{name}：模型穿脑概率 {a['probability']:.3f}",20)
  m=Chem.MolFromSmiles(a['smiles']);rdDepictor.Compute2DCoords(m);d=rdMolDraw2D.MolDraw2DSVG(590,210);d.drawOptions().clearBackground=False;v=np.asarray(a['delta_logit']);scale=max(abs(v).max(),1e-8);colors={i:((1-.65*abs(z)/scale,1.,1-.65*abs(z)/scale) if z>0 else (1.,1-.65*abs(z)/scale,1-.65*abs(z)/scale)) for i,z in enumerate(v)};d.DrawMolecule(m,highlightAtoms=list(colors),highlightAtomColors=colors);d.FinishDrawing();mols=d.GetDrawingText();mols=mols[mols.index('<svg'):].replace('<svg ',f'<svg x="25" y="{y+12}" ',1);s.append(mols);txt(25,y+245,f"Δlogit 范围 [{v.min():.3f}, {v.max():.3f}]；保留原坐标，不优化断裂结构",15)
 txt(25,1200,'CUDA / HIP：同源核心前向算子真机对账',21)
 pb=r.get('pairbias',{})
 if pb:
  txt(25,1235,'真实权重：128高斯 → 128/GELU/64投影 → 首层QKV',16)
  txt(25,1267,f"GB10 {pb['cuda_ms']:.3f} ms｜RX 9070 XT {pb['hip_ms']:.3f} ms",18)
  txt(25,1299,f"跨卡最大绝对误差 {pb['cross_device_max_abs']:.3g}",18)
  txt(25,1331,f"与官方 PyTorch 最大绝对误差 {pb['torch_reference_max_abs']:.3g}",17)
  txt(25,1363,'仅移植 Gaussian/投影/首层注意力核，非全模型 AMD 推理',16)
 txt(25,1410,f"GPU 20轮 {r['training_seconds']:.2f}s；CPU构象 {r['cpu_conformer_seconds']:.2f}s（8进程）",16)
 txt(25,1445,'单随机划分；标签和药物概率不等于临床药效或嗜睡诊断。',16)
 txt(25,1480,'遮挡后的token集合可能不对应有效分子，归因仅作模型探针。',16)
 txt(25,1516,'完整输出、划分和逐轮日志：results/module6_results.json',14)
 s.append('</svg>');p=ROOT/'visuals/fig5_bbbp_unimol_roc_and_atom_attribution.svg';p.write_text('\n'.join(s));cairosvg.svg2png(url=str(p),write_to=str(p.with_suffix('.png')))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--stage',choices=['train','pairbias','render','all'],default='all');p.add_argument('--epochs',type=int,default=20);a=p.parse_args()
 if a.stage in ['train','all']:stage_train(a.epochs)
 if a.stage in ['pairbias','all']:
  import subprocess
  subprocess.run([str(ROOT/'.venv/bin/python'),str(ROOT/'amd_hip_port/unimol_pairbias/run_pairbias.py')],check=True)
 if a.stage in ['render','pairbias','all']:render()
