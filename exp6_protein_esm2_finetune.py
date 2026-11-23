#!/usr/bin/env python3
"""GB1 experimental binding fitness: identical holdout baselines and ESM2 fine-tuning."""
import os
os.environ.setdefault('OMP_NUM_THREADS','4');os.environ.setdefault('OPENBLAS_NUM_THREADS','4');os.environ.setdefault('TOKENIZERS_PARALLELISM','false')
import argparse,hashlib,json,random,time,zipfile,urllib.request
from pathlib import Path
from html import escape
import numpy as np
import pandas as pd
from Bio.SeqUtils.ProtParam import ProteinAnalysis
from scipy.stats import spearmanr
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error,r2_score
ROOT=Path(__file__).resolve().parent;SEED=20261124;AA='ACDEFGHIKLMNPQRSTVWY';SOURCE='https://raw.githubusercontent.com/J-SNACKKB/FLIP/main/splits/gb1/splits.zip';MODEL='facebook/esm2_t6_8M_UR50D';WT='MQYKLILNGKTLKGETTTEAVDAATAEKVFKQYANDNGVDGEWTYDDATKTFTVTE';SITES=[39,40,41,54]
def dump(path,x):
 Path(path).parent.mkdir(parents=True,exist_ok=True);Path(path).write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def feature(s):
 a=ProteinAnalysis(s);L=len(s);count={c:s.count(c)/L for c in AA};dp={a+b:0 for a in AA for b in AA}
 for i in range(L-1):dp[s[i:i+2]]+=1/(L-1)
 bio=[a.molecular_weight(),a.isoelectric_point(),a.aromaticity(),a.instability_index(),a.gravy()]
 return np.array(list(count.values())+list(dp.values())+bio,dtype=np.float64)
def metrics(y,p):
 rho=spearmanr(y,p).statistic if np.std(p)>1e-12 else None
 return {'RMSE':float(np.sqrt(mean_squared_error(y,p))),'Spearman':None if rho is None or not np.isfinite(rho) else float(rho),'R2':float(r2_score(y,p))}
def prepare(n=3000):
 start=time.perf_counter();(ROOT/'data').mkdir(exist_ok=True)
 archive=ROOT/'data/gb1_splits.zip'
 if not archive.exists():urllib.request.urlretrieve(SOURCE,archive)
 path=ROOT/'data/gb1/splits/sampled.csv'
 if not path.exists():
  with zipfile.ZipFile(archive) as z:z.extract('splits/sampled.csv',ROOT/'data/gb1')
 raw=pd.read_csv(path);valid=raw[raw.sequence.map(lambda s:isinstance(s,str) and len(s)>54 and set(s)<=set(AA))].copy();dedup=valid.drop_duplicates('sequence').copy()
 variable=[i+1 for i in range(len(dedup.iloc[0].sequence)) if dedup.sequence.str[i].nunique()>1]
 assert variable==SITES,(variable,SITES)
 # Retain complete 56-residue GB1 domain; discarded fusion tail is identical in every record.
 dedup['full_sequence']=dedup.sequence;dedup['sequence']=dedup.sequence.str[:56]
 assert dedup.sequence.nunique()==len(dedup)
 selected=np.random.default_rng(SEED).choice(len(dedup),size=min(n,len(dedup)),replace=False)
 df=dedup.iloc[selected].reset_index().rename(columns={'index':'source_index'})
 idx=np.arange(len(df));train,rest=train_test_split(idx,test_size=.2,random_state=SEED);val,test=train_test_split(rest,test_size=.5,random_state=SEED)
 df['split']='';df.loc[train,'split']='train';df.loc[val,'split']='val';df.loc[test,'split']='test'
 st=time.perf_counter();X=np.stack([feature(s) for s in df.sequence]);feature_time=time.perf_counter()-st
 df.to_csv(ROOT/'data/gb1_experiment.csv',index=False)
 np.savez(ROOT/'results/module8_prepared.npz',X=X,y=df.target.to_numpy(dtype=np.float32),train=train,val=val,test=test)
 # Domain-level biochemical descriptors, not the invariant expression/fusion tail.
 bio=pd.DataFrame(X[:,-5:],columns=['MW','pI','aromaticity','instability','GRAVY']);bio.insert(0,'source_index',df.source_index);bio.to_csv(ROOT/'results/module8_biopython_descriptors.csv',index=False)
 info={'dataset':'FLIP GB1 sampled; experimental normalized IgG binding fitness (Wu et al. 2016)','source':SOURCE,'paper':'https://elifesciences.org/articles/16965','archive_sha256':sha(archive),'csv_sha256':sha(path),'experiment_csv_sha256':sha(ROOT/'data/gb1_experiment.csv'),'raw_rows':len(raw),'standard_AA_valid_rows':len(valid),'invalid_rows':len(raw)-len(valid),'duplicate_sequences_removed':len(valid)-len(dedup),'unique_rows':len(dedup),'subset_rows':len(df),'subset_method':'seeded uniform sample without replacement before label inspection','original_length':265,'model_length':56,'crop_reason':'all variable positions 39,40,41,54 fall within complete N-terminal GB1 domain; discarded 209AA fusion tail identical across all records','variable_positions_one_based':variable,'feature_dimensions':425,'feature_names':'20 AAC + 400 dipeptide frequencies + MW,pI,aromaticity,instability,GRAVY','feature_seconds':feature_time,'prepare_seconds':time.perf_counter()-start,'seed':SEED,'split_sizes':{k:int((df.split==k).sum()) for k in ['train','val','test']},'split_policy':'new fixed random 80/10/10 within documented 3000 subset, not FLIP official leaderboard split','split_source_indices':{k:df.loc[df.split==k,'source_index'].astype(int).tolist() for k in ['train','val','test']}}
 dump(ROOT/'results/module8_data.json',info);print(json.dumps({k:v for k,v in info.items() if k!='split_source_indices'},ensure_ascii=False),flush=True)
 return info

def load_model():
 import torch
 from transformers import AutoTokenizer,EsmForSequenceClassification
 path=ROOT/'weights/esm2_t6_8M_UR50D'
 if not (path/'config.json').exists():
  from huggingface_hub import snapshot_download
  snapshot_download(MODEL,local_dir=path,allow_patterns=['config.json','vocab.txt','tokenizer_config.json','special_tokens_map.json','model.safetensors'])
 tok=AutoTokenizer.from_pretrained(path,local_files_only=True)
 torch.manual_seed(SEED)
 model=EsmForSequenceClassification.from_pretrained(path,num_labels=1,problem_type='regression',local_files_only=True)
 cfg=model.config
 architecture={'model_id':MODEL,'layers':cfg.num_hidden_layers,'hidden_size':cfg.hidden_size,'attention_heads':cfg.num_attention_heads,'intermediate_size':cfg.intermediate_size,'vocab_size':cfg.vocab_size,'vocabulary':tok.get_vocab(),'position_embedding_type':cfg.position_embedding_type,'rotary_modules':[n for n,m in model.named_modules() if 'rotary' in n],'parameters_total':sum(p.numel() for p in model.parameters()),'parameters_trainable':sum(p.numel() for p in model.parameters() if p.requires_grad),'regression_head':'new randomly initialized classifier; entire pretrained ESM2 encoder unfrozen','standard_AA_tokens':list(AA),'nonstandard_or_ambiguous_AA_tokens':['X','B','U','Z','O'],'other_tokens':['<cls>','<pad>','<eos>','<unk>','.','-','<null_1>','<mask>'],'weight_files_sha256':{p.name:sha(p) for p in path.glob('*') if p.suffix in ['.json','.safetensors','.bin','.txt']},'token_dropout':cfg.token_dropout}
 dump(ROOT/'results/module8_esm2_architecture.json',architecture);print('ARCHITECTURE',json.dumps(architecture,ensure_ascii=False),flush=True)
 return tok,model,architecture

def run(epochs=12):
 import torch
 from torch.utils.data import DataLoader,TensorDataset
 torch.set_num_threads(4);torch.set_num_interop_threads(2);random.seed(SEED);np.random.seed(SEED);torch.manual_seed(SEED);torch.cuda.manual_seed_all(SEED)
 if not (ROOT/'results/module8_prepared.npz').exists():prepare()
 data=np.load(ROOT/'results/module8_prepared.npz');df=pd.read_csv(ROOT/'data/gb1_experiment.csv');X,y=data['X'],data['y'];tr,va,te=data['train'],data['val'],data['test']
 (ROOT/'results/module8_epochs.jsonl').write_text('')
 baseline_st=time.perf_counter();rf=RandomForestRegressor(n_estimators=400,min_samples_leaf=2,max_features=.8,n_jobs=4,random_state=SEED);rf.fit(X[tr],y[tr]);models={'AAC+DP+Bio RF':rf};val_results={'AAC+DP+Bio RF':metrics(y[va],rf.predict(X[va]))};best_ridge=None;best_rmse=float('inf')
 for alpha in [.01,.1,1,10,100]:
  ridge=make_pipeline(StandardScaler(),Ridge(alpha=alpha));ridge.fit(X[tr],y[tr]);m=metrics(y[va],ridge.predict(X[va]))
  if m['RMSE']<best_rmse:best_rmse=m['RMSE'];best_ridge=ridge;best_alpha=alpha
 models['AAC+DP+Bio Ridge']=best_ridge;val_results['AAC+DP+Bio Ridge']=metrics(y[va],best_ridge.predict(X[va]));baseline_seconds=time.perf_counter()-baseline_st
 tok,model,arch=load_model();probe_name='esm.encoder.layer.0.attention.self.query.weight';probe_initial=model.state_dict()[probe_name].detach().cpu().clone();assert torch.cuda.is_available(),'CUDA required for genuine GPU fine-tuning'
 tokens=tok(df.sequence.tolist(),padding=True,return_tensors='pt');ids=tokens['input_ids'];mask=tokens['attention_mask'];mu=float(y[tr].mean());std=float(y[tr].std());z=(y-mu)/std
 loader=DataLoader(TensorDataset(ids[tr],mask[tr],torch.tensor(z[tr])),batch_size=64,shuffle=True,num_workers=0,generator=torch.Generator().manual_seed(SEED))
 device='cuda';model=model.to(device);optim=torch.optim.AdamW(model.parameters(),lr=2e-4,weight_decay=.01);history=[];best_loss=float('inf');best_epoch=0;train_st=time.perf_counter();checkpoint=ROOT/'results/esm2_gb1_best.pt'
 def predict(index):
  model.eval();p=[]
  with torch.no_grad():
   for start in range(0,len(index),128):
    ind=index[start:start+128]
    with torch.autocast('cuda',dtype=torch.bfloat16):out=model(input_ids=ids[ind].to(device),attention_mask=mask[ind].to(device)).logits.squeeze(-1)
    p.extend(out.float().cpu().numpy()*std+mu)
  return np.asarray(p)
 for epoch in range(1,epochs+1):
  torch.cuda.synchronize();t=time.perf_counter();model.train();total=0.;count=0
  for bx,bm,by in loader:
   optim.zero_grad(set_to_none=True)
   with torch.autocast('cuda',dtype=torch.bfloat16):out=model(input_ids=bx.to(device),attention_mask=bm.to(device)).logits.squeeze(-1);loss=torch.nn.functional.mse_loss(out.float(),by.to(device))
   loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optim.step();total+=float(loss.detach())*len(by);count+=len(by)
  vp=predict(va);vm=metrics(y[va],vp);torch.cuda.synchronize();rec={'epoch':epoch,'train_standardized_MSE':total/count,'val':vm,'seconds':time.perf_counter()-t};history.append(rec)
  with (ROOT/'results/module8_epochs.jsonl').open('a') as f:f.write(json.dumps(rec)+'\n')
  print('EPOCH',json.dumps(rec),flush=True)
  if vm['RMSE']<best_loss:
   best_loss=vm['RMSE'];best_epoch=epoch;torch.save({'state_dict':model.state_dict(),'epoch':epoch,'target_mean':mu,'target_std':std},checkpoint)
 train_seconds=time.perf_counter()-train_st;model.load_state_dict(torch.load(checkpoint,map_location=device,weights_only=True)['state_dict'])
 # Final holdout accessed once per final trained model. No test-based model selection.
 preds={'Constant train mean':np.full(len(te),mu)}
 for name,mod in models.items():preds[name]=mod.predict(X[te])
 preds['ESM2 full finetune']=predict(te);scores={k:metrics(y[te],v) for k,v in preds.items()}
 pd.DataFrame({'source_index':df.iloc[te].source_index,'target':y[te],**preds}).to_csv(ROOT/'results/module8_test_predictions.csv',index=False)
 # 80 single mutants on a fixed WT background. These are model predictions, not measured effects.
 scan=[]
 for site in SITES:
  for aa in AA:
   s=WT[:site-1]+aa+WT[site:];scan.append((site,aa,s))
 scanseq=[WT]+[v[2] for v in scan];enc=tok(scanseq,padding=True,return_tensors='pt');model.eval()
 with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):pred=model(input_ids=enc.input_ids.to(device),attention_mask=enc.attention_mask.to(device)).logits.squeeze(-1).float().cpu().numpy()*std+mu
 scanrows=[{'site':site,'WT':WT[site-1],'mutant':aa,'prediction':float(p),'delta_vs_WT':float(p-pred[0]),'sequence':s} for (site,aa,s),p in zip(scan,pred[1:])];pd.DataFrame(scanrows).to_csv(ROOT/'results/module8_mutation_scan.csv',index=False)
 result={'status':'PASS','dataset':json.loads((ROOT/'results/module8_data.json').read_text()),'architecture':arch,'device':torch.cuda.get_device_name(),'epochs':epochs,'batch_size':64,'optimizer_steps':len(loader)*epochs,'encoder_weight_update_audit':{'parameter':probe_name,'L2_change_from_pretrained':float(torch.linalg.norm(model.state_dict()[probe_name].detach().cpu()-probe_initial))},'best_epoch':best_epoch,'validation':val_results,'best_ESM_validation_RMSE':best_loss,'ridge_alpha_chosen_by_val':best_alpha,'test':scores,'test_evaluation_count_per_final_model':1,'baseline_train_seconds':baseline_seconds,'ESM_gpu_train_seconds':train_seconds,'target_standardization':{'train_mean':mu,'train_std':std},'history':history,'scan':{'WT_sequence':WT,'WT_prediction':float(pred[0]),'sites':SITES,'amino_acids':AA,'mutants':80,'meaning':'predicted single-mutant minus predicted WT on fixed WT background; not experimental validation'},'limitations':['single random within-landscape holdout; no unseen-protein generalization claim','3000-row seeded subset of FLIP GB1 sampled, not the official benchmark split','experimental fitness raw scale used; only train-target z-standardization','ESM pretraining homology leakage not audited']}
 dump(ROOT/'results/module8_results.json',result);figures(result);print('FINAL',json.dumps(scores),flush=True)
 del model;torch.cuda.empty_cache()

def txt(x,y,s,size=18,color='#1e293b'):return f'<text x="{x}" y="{y}" font-size="{size}" fill="{color}">{escape(str(s))}</text>'
def figures(r=None):
 if r is None:r=json.loads((ROOT/'results/module8_results.json').read_text())
 s='<svg xmlns="http://www.w3.org/2000/svg" width="640" height="1600" viewBox="0 0 640 1600" font-family="\'WenQuanYi Micro Hei\',\'Noto Sans CJK SC\',sans-serif"><rect width="640" height="1600" fill="#ffffff"/>'
 s+=txt(24,40,'蛋白质专有模型：ESM-2 真机微调',25)+txt(24,76,'GB1 四位点组合突变 → 实验结合适应度',19)+txt(24,117,'① 科学库先处理序列，再交给模型',22)
 s+=txt(24,154,'Biopython：分子量 / 等电点 / 芳香性 / 不稳定指数',17)+txt(24,185,'GRAVY亲疏水指数；仅接受20种标准氨基酸。',17)+txt(24,223,'固定3000条真实实验序列；2400训练 / 300验证 / 300测试',16)
 s+=txt(24,267,'② 不是通用聊天词表，而是氨基酸词表',22)+txt(24,304,'A C D E F G H I K L M N P Q R S T V W Y',19,'#2563eb')
 a=r['architecture'];s+=txt(24,341,f"33词表 / {a['layers']}层 / 隐藏{a['hidden_size']} / {a['attention_heads']}头 / {a['parameters_total']/1e6:.2f}M参数",18)+txt(24,377,'RoPE旋转位置编码；双向注意力 + 新建回归头。',17)+txt(24,413,'56AA完整GB1域；265AA原始序列的恒定融合尾巴裁掉。',16)
 s+=txt(24,457,'③ 同一份测试卷：误差越低越好',22)
 names={'Constant train mean':'常数基线','AAC+DP+Bio RF':'传统特征 + 随机森林','AAC+DP+Bio Ridge':'传统特征 + 岭回归','ESM2 full finetune':'ESM-2 全参数微调'};maximum=max(v['RMSE'] for v in r['test'].values())
 for k,(name,m) in enumerate(r['test'].items()):
  y0=493+k*72;s+=txt(24,y0,names[name],18);w=400*m['RMSE']/maximum;s+=f'<rect x="24" y="{y0+10}" width="{w}" height="22" rx="4" fill="'+('#2563eb' if name.startswith('ESM') else '#94a3b8')+'"/>';s+=txt(445,y0+28,f"RMSE {m['RMSE']:.3f}",17)
 s+=txt(24,805,f"ESM：Spearman {r['test']['ESM2 full finetune']['Spearman']:.3f} / R² {r['test']['ESM2 full finetune']['R2']:.3f}",19)+txt(24,844,f"{r['epochs']}轮训练 {r['ESM_gpu_train_seconds']:.1f}秒；验证集选第{r['best_epoch']}轮。",18)
 s+=txt(24,891,'④ 单点突变扫描：模型认为哪里变好？',22)+txt(24,927,'每格 = 预测突变体适应度 − 预测野生型适应度',17)
 scan=pd.read_csv(ROOT/'results/module8_mutation_scan.csv');mx=max(abs(scan.delta_vs_WT).max(),1e-6);cell=24;ox=90;oy=980
 for j,aa in enumerate(AA):s+=txt(ox+j*cell+5,966,aa,15)
 for i,site in enumerate(SITES):
  s+=txt(24,oy+i*54+26,str(site)+WT[site-1],18)
  for j,aa in enumerate(AA):
   v=float(scan[(scan.site==site)&(scan.mutant==aa)].delta_vs_WT.iloc[0]);q=min(abs(v)/mx,1);rgb=(int(255-180*q),int(255-90*q),255) if v>=0 else (255,int(255-150*q),int(255-150*q));s+=f'<rect x="{ox+j*cell}" y="{oy+i*54}" width="23" height="40" fill="rgb{rgb}"/>'
   if aa==WT[site-1]:s+=f'<rect x="{ox+j*cell}" y="{oy+i*54}" width="23" height="40" fill="none" stroke="#0f172a" stroke-width="2"/>'
 s+=txt(24,1238,f'蓝：预测增加；红：预测降低；色阶范围 ±{mx:.2f}',17)+txt(24,1273,'黑框是原始氨基酸；四个位点共80个单点替换。',17)+txt(24,1320,'这是模型效应图，不是80次新湿实验的验证结果。',18,'#b91c1c')+txt(24,1364,'本次随机留出只检验同一蛋白的突变组合泛化。',17)+txt(24,1401,'不代表预测新蛋白、折叠结构或药物临床效果。',17)+txt(24,1455,'数据：FLIP GB1 / Wu等，eLife 2016；官方Meta ESM-2',15,'#64748b')+txt(24,1491,'测试集不参与早停、超参数选择或标签标准化。',16,'#64748b')+'</svg>'
 out=ROOT/'visuals/fig7_protein_esm2_finetune_and_mutation_scan.svg';out.write_text(s)
 import cairosvg
 cairosvg.svg2png(url=str(out),write_to=str(out.with_suffix('.png')))

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--stage',choices=['prepare','architecture','train','figures','all'],default='all');p.add_argument('--n',type=int,default=3000);p.add_argument('--epochs',type=int,default=12);a=p.parse_args();(ROOT/'results').mkdir(exist_ok=True);(ROOT/'visuals').mkdir(exist_ok=True)
 if a.stage in ['prepare','all']:prepare(a.n)
 if a.stage=='architecture':load_model()
 if a.stage in ['train','all']:run(a.epochs)
 if a.stage=='figures':figures()
