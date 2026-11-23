#!/usr/bin/env python3
"""Real ESOL fine-tuning of official Uni-Mol v1 (15 layers), fixed held-out test."""
import os
os.environ.setdefault('OMP_NUM_THREADS','4')
os.environ.setdefault('OPENBLAS_NUM_THREADS','4')
from pathlib import Path
ROOT=Path(__file__).resolve().parent
os.environ['UNIMOL_WEIGHT_DIR']=str(ROOT/'weights')
import argparse, copy, hashlib, json, pickle, random, subprocess, time
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator, Descriptors
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from unimol_tools.models.unimol import UniMolModel
from unimol_tools.data.conformer import coords2unimol


def metrics(y,p):
    return {'RMSE':float(np.sqrt(mean_squared_error(y,p))),'MAE':float(mean_absolute_error(y,p)),'R2':float(r2_score(y,p))}

def check_memory():
    available=int(next(x.split()[1] for x in Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:')))*1024
    if available<20*1024**3: raise MemoryError('MemAvailable below 20 GiB')

def svg_figure(r):
    from html import escape
    import cairosvg
    width,height=640,1120
    s=[f'<svg xmlns="http://www.w3.org/2000/svg" width="640" height="{height}" viewBox="0 0 640 {height}" font-family="\'WenQuanYi Micro Hei\',\'Noto Sans CJK SC\',sans-serif"><rect width="640" height="{height}" fill="#ffffff"/>']
    def txt(x,y,t,size=17,color='#253449',weight='normal'):
        s.append(f'<text x="{x}" y="{y}" font-size="{size}" fill="{color}" font-weight="{weight}">{escape(str(t))}</text>')
    def rect(x,y,w,h,fill='#edf4fb'):
        s.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="9" fill="{fill}"/>')
    txt(28,40,'3D 分子模型：同一份 ESOL 考卷',25,weight='bold')
    txt(28,70,f"{r['n_molecules']} 个分子｜训练 / 验证 / 测试 = "+' / '.join(map(str,r['split_sizes'])),16)
    txt(28,96,'固定随机种子 42；按母核去重；仅用验证集挑选轮次',15)
    rect(25,115,590,162)
    txt(42,144,'官方 Uni-Mol v1：47,331,139 参数',20,weight='bold')
    txt(42,174,'原子种类 → 512 维向量 → 15 层 Transformer',18)
    txt(42,204,'3D 距离 → 128 个高斯核 → 64 头成对偏置',18)
    txt(42,234,'注意力分数 = Q·K^T / √d + 成对偏置',18)
    txt(42,259,'各层还会更新原子对表示；不是只加一次静态距离',14)
    txt(28,312,'留出测试集：RMSE 越低越好（单位 logS）',20,weight='bold')
    rows=[('常数均值',r['baselines']['Dummy']),('Morgan 2048 + 随机森林',r['baselines']['Morgan_RF']),('Uni-Mol 全参数微调',r['unimol_test'])]
    for k,(label,v) in enumerate(rows):
        y=345+k*66
        txt(30,y,label,17)
        w=v['RMSE']/max(x[1]['RMSE'] for x in rows)*390
        s.append(f'<rect x="30" y="{y+9}" width="{w:.1f}" height="18" fill="{["#9ca3af","#4f8ccb","#27a68a"][k]}"/>')
        txt(440,y+23,f"{v['RMSE']:.3f} / R² {v['R2']:.3f}",15)
    txt(28,566,'Uni-Mol：实测值与预测值',20,weight='bold')
    x0,y0,pw,ph=62,590,515,230
    ys=np.asarray(r['test_targets']); ps=np.asarray(r['test_predictions']); lo=min(ys.min(),ps.min())-.3; hi=max(ys.max(),ps.max())+.3
    def xy(a,b):return x0+(a-lo)/(hi-lo)*pw,y0+ph-(b-lo)/(hi-lo)*ph
    s.append(f'<rect x="{x0}" y="{y0}" width="{pw}" height="{ph}" fill="#fafafa" stroke="#d1d5db"/>')
    s.append(f'<line x1="{x0}" y1="{y0+ph}" x2="{x0+pw}" y2="{y0}" stroke="#999" stroke-dasharray="4 4"/>')
    for a,b in zip(ys,ps):
        x,y=xy(a,b);s.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="2.8" fill="#168f7b" opacity="0.7"/>')
    txt(64,844,f'横轴：实测 logS ({lo:.1f} … {hi:.1f})；纵轴：预测 logS',15)
    txt(28,887,'耗时实账：不预设 CPU 一定是瓶颈',20,weight='bold')
    timings=[('CPU：全量单构象生成',r.get('cpu_conformer_seconds',0)),('GPU：训练 + 每轮验证',r['training_seconds'])]
    for k,(label,v) in enumerate(timings):
        y=919+k*64; txt(30,y,f'{label}：{v:.2f} 秒',16)
        w=max(v,0.01)/max(t[1] for t in timings)*540
        s.append(f'<rect x="30" y="{y+10}" width="{w:.1f}" height="16" fill="{["#d6a746","#27a68a"][k]}"/>')
    txt(28,1064,f"实跑 {len(r['epochs'])} 轮；最佳验证轮次 {r['best_epoch']}；测试只评估一次",16)
    txt(28,1093,'单次随机划分与单构象实验，不能证明 3D 模型普遍优于 2D。',15)
    s.append('</svg>'); path=ROOT/'visuals/fig3_unimol_3d_finetune_vs_baselines.svg';path.write_text('\n'.join(s))
    cairosvg.svg2png(url=str(path),write_to=str(path.with_suffix('.png')))

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--epochs',type=int,default=20);ap.add_argument('--batch-size',type=int,default=32);ap.add_argument('--lr',type=float,default=1e-4);ap.add_argument('--render-only',action='store_true');args=ap.parse_args()
    out=ROOT/'results/module2_results.json'
    if args.render_only: svg_figure(json.loads(out.read_text())); return
    random.seed(42);np.random.seed(42);torch.manual_seed(42);torch.cuda.manual_seed_all(42);torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=True
    check_memory()
    records=pickle.load(open(ROOT/'results/esol_conformers.pkl','rb'))
    if isinstance(records,dict):records=records.get('records',records.get('conformers'))
    # One canonical cleaned parent cannot occur on both sides of a split.
    records=[v for v in records if v.get('coordinates') is not None]
    seen=set(); clean=[]
    for row in records:
        smi=Chem.MolToSmiles(Chem.MolFromSmiles(row['smiles']),isomericSmiles=True)
        if smi not in seen: clean.append(row);seen.add(smi)
    records=clean
    y=np.asarray([v['target'] for v in records],dtype=np.float32)
    idx=np.arange(len(y));tr,rest=train_test_split(idx,test_size=.2,random_state=42);va,te=train_test_split(rest,test_size=.5,random_state=42)
    split={'train':tr.tolist(),'validation':va.tolist(),'test':te.tolist(),'source_indices':[int(v['source_index']) for v in records],'smiles':[v['smiles'] for v in records]}
    (ROOT/'results/module2_split.json').write_text(json.dumps(split,indent=2))
    fp=rdFingerprintGenerator.GetMorganGenerator(radius=2,fpSize=2048)
    X=np.asarray([fp.GetFingerprintAsNumPy(Chem.MolFromSmiles(v['smiles'])) for v in records])
    baselines={}; baseline_seconds={}
    for name,reg in [('Dummy',DummyRegressor()),('Morgan_RF',RandomForestRegressor(n_estimators=400,random_state=42,n_jobs=4,min_samples_leaf=1,max_features=1.0))]:
        t=time.perf_counter();reg.fit(X[tr],y[tr]);baselines[name]=metrics(y[te],reg.predict(X[te]));baseline_seconds[name]=time.perf_counter()-t
    model=UniMolModel(output_dim=1,remove_hs=True)
    architecture={'parameters':sum(p.numel() for p in model.parameters()),'trainable_parameters':sum(p.numel() for p in model.parameters() if p.requires_grad),'layers':model.args.encoder_layers,'hidden_dim':model.args.encoder_embed_dim,'ffn_dim':model.args.encoder_ffn_embed_dim,'attention_heads':model.args.encoder_attention_heads,'gaussian_kernels':128,'args':dict(model.args),'model_repr':str(model)}
    (ROOT/'results/unimol_architecture.json').write_text(json.dumps(architecture,indent=2))
    print(json.dumps({k:v for k,v in architecture.items() if k not in ['model_repr','args']}),flush=True)
    t=time.perf_counter();features=[coords2unimol(v['atoms'],v['coordinates'],model.dictionary,remove_hs=True) for v in records];feature_seconds=time.perf_counter()-t
    mu=float(y[tr].mean());sd=float(y[tr].std());normalized=(y-mu)/sd
    def loader(ix,shuffle=False):
        return DataLoader([(features[i],float(normalized[i])) for i in ix],batch_size=args.batch_size,shuffle=shuffle,num_workers=0,collate_fn=model.batch_collate_fn)
    train_loader=loader(tr,True);val_loader=loader(va);test_loader=loader(te)
    model=model.cuda();optim=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=.01)
    # Same full network receives gradient updates; no frozen backbone / no surrogate model.
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optim,T_max=args.epochs,eta_min=args.lr*.1)
    def predict(dl):
        model.eval();pred=[]
        with torch.no_grad():
            for b,_ in dl:
                b={k:v.cuda() for k,v in b.items()}
                with torch.autocast('cuda',dtype=torch.bfloat16):p=model(**b).reshape(-1)
                pred.extend((p.float().cpu().numpy()*sd+mu).tolist())
        return np.asarray(pred)
    logs=[];best=float('inf');best_epoch=0;t_all=time.perf_counter()
    log_path=ROOT/'results/module2_epochs.jsonl';log_path.write_text('')
    for epoch in range(1,args.epochs+1):
        check_memory();torch.cuda.synchronize();t=time.perf_counter();model.train();total=0.;count=0
        for b,label in train_loader:
            optim.zero_grad(set_to_none=True);b={k:v.cuda() for k,v in b.items()};label=label.cuda().float()
            with torch.autocast('cuda',dtype=torch.bfloat16):pred=model(**b).reshape(-1);loss=torch.nn.functional.mse_loss(pred.float(),label)
            if not torch.isfinite(loss):raise ValueError('Nonfinite training loss')
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optim.step();total+=float(loss.detach())*len(label);count+=len(label)
        vp=predict(val_loader);vm=metrics(y[va],vp);torch.cuda.synchronize();elapsed=time.perf_counter()-t
        entry={'epoch':epoch,'train_normalized_MSE':total/count,'validation':vm,'seconds':elapsed,'lr':optim.param_groups[0]['lr']};logs.append(entry)
        print(json.dumps(entry),flush=True)
        with log_path.open('a') as f:f.write(json.dumps(entry)+'\n')
        if vm['RMSE']<best:
            best=vm['RMSE'];best_epoch=epoch;torch.save(model.state_dict(),ROOT/'results/unimol_esol_best.pt')
        scheduler.step()
    training_seconds=time.perf_counter()-t_all
    model.load_state_dict(torch.load(ROOT/'results/unimol_esol_best.pt',map_location='cuda',weights_only=True));tp=predict(test_loader)
    cpu_time=None
    timing_path=ROOT/'results/rdkit_full_conformer_time.json'
    if timing_path.exists():
        timing=json.loads(timing_path.read_text());cpu_time=timing.get('wall_seconds')
    if not cpu_time:
        timing=json.loads((ROOT/'results/rdkit_timings.json').read_text());cpu_time=timing[0]['wall_seconds']
    source=subprocess.check_output(['git','-C',str(ROOT/'vendor/Uni-Mol'),'rev-parse','HEAD'],text=True).strip()
    weight_path=ROOT/'weights/mol_pre_no_h_220816.pt'
    r={'dataset':'ESOL','n_molecules':len(records),'n_original_cache':len(pickle.load(open(ROOT/'results/esol_conformers.pkl','rb'))),'split_sizes':[len(tr),len(va),len(te)],'split_type':'canonical-parent-deduplicated random 80/10/10 seed42','architecture':{k:v for k,v in architecture.items() if k not in ['model_repr','args']},'official_source_commit':source,'weight_file':str(weight_path.relative_to(ROOT)),'weight_sha256':hashlib.sha256(weight_path.read_bytes()).hexdigest(),'baselines':baselines,'baseline_seconds':baseline_seconds,'unimol_test':metrics(y[te],tp),'epochs':logs,'best_epoch':best_epoch,'best_validation_RMSE':best,'training_seconds':training_seconds,'feature_encoding_seconds':feature_seconds,'cpu_conformer_seconds':cpu_time or 0,'test_targets':y[te].tolist(),'test_predictions':tp.tolist(),'batch_size':args.batch_size,'lr':args.lr,'precision':'BF16 autocast; FP32 weights and optimizer','full_parameter_finetune':True,'max_gpu_allocated_gib':torch.cuda.max_memory_allocated()/1024**3,'notes':['CPU conformer cache supplied by module1','No test-set checkpoint selection; test evaluated only after all epochs','One random split, no scaffold challenge or repeated seeds','UniMol v1 no_h 2022 pretrained checkpoint; training-set overlap with pretraining is not audited']}
    out.write_text(json.dumps(r,indent=2));svg_figure(r);print(json.dumps({'completed':r['unimol_test'],'training_seconds':training_seconds}),flush=True)
if __name__=='__main__':main()
