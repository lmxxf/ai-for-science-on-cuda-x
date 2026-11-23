#!/usr/bin/env python3
"""Full 15-layer official ESOL UniMol checkpoint on native AMD HIP/hipBLAS FP64."""
import os
os.environ.setdefault('OMP_NUM_THREADS','4');os.environ.setdefault('OPENBLAS_NUM_THREADS','4')
from pathlib import Path
ROOT=Path(__file__).resolve().parent;os.environ['UNIMOL_WEIGHT_DIR']=str(ROOT/'weights');PORT=ROOT/'amd_hip_port/unimol_full15_hip'
import argparse,hashlib,json,pickle,subprocess,time,types
import numpy as np
import torch
from rdkit import Chem
from sklearn.metrics import mean_squared_error,r2_score
from unimol_tools.models.unimol import UniMolModel
from unimol_tools.data.conformer import coords2unimol

def run(cmd):print('RUN',cmd,flush=True);subprocess.run(cmd,check=True,cwd=ROOT)
def samples():
 records=pickle.load(open(ROOT/'results/esol_conformers.pkl','rb'));split=json.loads((ROOT/'results/module2_split.json').read_text());lookup={r['source_index']:r for r in records};return [lookup[split['source_indices'][i]] for i in split['test']]
def model():
 m=UniMolModel(output_dim=1,remove_hs=True);m.load_state_dict(torch.load(ROOT/'results/unimol_esol_best.pt',map_location='cpu',weights_only=True),strict=True);m.eval();return m

def prepare():
 torch.set_num_threads(4);inp=PORT/'input';inp.mkdir(parents=True,exist_ok=True);m=model();state=m.state_dict();manifest=[]
 with (inp/'weights_fp64.bin').open('wb') as f:
  for k,v in state.items():a=v.numpy().astype(np.float64);a.tofile(f);manifest.append(f'{k} {a.size}')
 (inp/'weights_manifest.txt').write_text('\n'.join(manifest)+'\n');rs=samples();sizes=[]
 with (inp/'molecules.bin').open('wb') as f:
  for r in rs:
   feature=coords2unimol(r['atoms'],r['coordinates'],m.dictionary,remove_hs=True);sizes.append(len(feature['src_tokens']));feature['src_tokens'].astype(np.int32).tofile(f);feature['src_edge_type'].astype(np.int32).tofile(f);feature['src_distance'].astype(np.float64).tofile(f)
 (inp/'molecules.txt').write_text(str(len(rs))+'\n'+' '.join(map(str,sizes))+'\n')
 (ROOT/'results/module13_manifest.json').write_text(json.dumps({'parameter_count':sum(v.numel() for v in state.values()),'state_tensors':len(state),'test_source_indices':[r['source_index'] for r in rs],'sizes':sizes,'input_sha256':hashlib.sha256((inp/'molecules.bin').read_bytes()).hexdigest(),'checkpoint_sha256':hashlib.sha256((ROOT/'results/unimol_esol_best.pt').read_bytes()).hexdigest(),'head':str(m.classification_head),'encoder_post_ln':m.encoder.layers[0].post_ln,'final_encoder_layernorm':m.encoder.final_layer_norm is not None,'exported_parameters':manifest,'no_predictions_or_labels_sent_to_AMD':True},indent=2))
 run(['ssh','amd9070','mkdir C:\\AI4S-HIP-Lab\\unimol_full15_hip 2>NUL & exit /b 0'])
 run(['scp',str(PORT/'unimol_full.cpp'),str(PORT/'build_windows.cmd'),str(PORT/'run_windows.cmd'),'amd9070:C:/AI4S-HIP-Lab/unimol_full15_hip/'])
 run(['scp','-r',str(inp),'amd9070:C:/AI4S-HIP-Lab/unimol_full15_hip/'])
 run(['ssh','amd9070','C:\\AI4S-HIP-Lab\\unimol_full15_hip\\build_windows.cmd'])
 run(['scp','amd9070:C:/AI4S-HIP-Lab/unimol_full15_hip/build.log',str(PORT/'build_windows.log')])
 print('PREPARE complete; no GPU inference executed',flush=True)

def fp64_gaussian(self,x,edge_type):
 # Same official formulas, all in FP64 rather than the upstream helper's forced float().
 x=self.mul(edge_type)*x.unsqueeze(-1)+self.bias(edge_type);mu=self.means.weight.reshape(-1);std=self.stds.weight.reshape(-1).abs()+1e-5
 return torch.exp(-.5*((x-mu)/std)**2)/((2*3.14159)**.5*std)

def reference():
 torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;m=model().cuda();rs=samples();pred32=[]
 with torch.no_grad():
  for r in rs:
   ft=coords2unimol(r['atoms'],r['coordinates'],m.dictionary,remove_hs=True);b,_=m.batch_collate_fn([(ft,0.)]);b={k:v.cuda() for k,v in b.items()};pred32.append(float(m(**b).reshape(-1)[0]))
 m=m.double();m.gbf.forward=types.MethodType(fp64_gaussian,m.gbf);pred64=[];compute_ms=0;start=time.perf_counter();trace=PORT/'reference_trace_fp64.bin';a=torch.cuda.Event(enable_timing=True);z=torch.cuda.Event(enable_timing=True)
 with trace.open('wb') as f,torch.no_grad():
  for idx,r in enumerate(rs):
   ft=coords2unimol(r['atoms'],r['coordinates'],m.dictionary,remove_hs=True);b,_=m.batch_collate_fn([(ft,0.)]);b={k:(v.cuda().double() if v.dtype.is_floating_point else v.cuda()) for k,v in b.items()};N=b['src_tokens'].shape[1]
   a.record();x=m.encoder.emb_layer_norm(m.embed_tokens(b['src_tokens']));g=m.gbf(b['src_distance'],b['src_edge_type']);pair=m.gbf_proj(g).permute(0,3,1,2).contiguous().reshape(64,N,N);z.record();z.synchronize();compute_ms+=a.elapsed_time(z);x.cpu().numpy().tofile(f);pair.cpu().numpy().tofile(f)
   for layer in m.encoder.layers:
    a.record();x,pair,_=layer(x,attn_bias=pair,padding_mask=None,return_attn=True);z.record();z.synchronize();compute_ms+=a.elapsed_time(z);x.cpu().numpy().tofile(f);pair.cpu().numpy().tofile(f)
   a.record();x=m.encoder.final_layer_norm(x);p=m.classification_head(x[:,0,:]).reshape(-1);z.record();z.synchronize();compute_ms+=a.elapsed_time(z);pred64.append(float(p[0]));print('FP64 reference',idx,pred64[-1],flush=True)
 np.savez(PORT/'reference_predictions.npz',fp32_standardized=np.array(pred32),fp64_standardized=np.array(pred64))
 (PORT/'reference_runtime.json').write_text(json.dumps({'device':torch.cuda.get_device_name(),'dtype':'FP64, including Gaussian formulas (upstream forced .float() removed only for precision)','gpu_compute_ms':compute_ms,'wall_with_trace_s':time.perf_counter()-start,'peak_cuda_allocated_bytes':torch.cuda.max_memory_allocated(),'scope':'per-molecule forward15 layers including Gaussian, linear actual head; traces excluded from device-event sums','TF32':False},indent=2))

def execute():
 reference();run(['ssh','amd9070','C:\\AI4S-HIP-Lab\\unimol_full15_hip\\run_windows.cmd']);run(['scp','-r','amd9070:C:/AI4S-HIP-Lab/unimol_full15_hip/out_hip',str(PORT)]);run(['scp','amd9070:C:/AI4S-HIP-Lab/unimol_full15_hip/run.log',str(PORT/'run_windows.log')]);compare();run([str(ROOT/'.venv/bin/python'),str(PORT/'run_mixed_gaussian.py')]);render()

def compare():
 man=json.loads((ROOT/'results/module13_manifest.json').read_text());ref=np.load(PORT/'reference_predictions.npz');hip=np.fromfile(PORT/'out_hip/predictions_fp64.bin',dtype=np.float64);rs=samples();split=json.loads((ROOT/'results/module2_split.json').read_text());cache=pickle.load(open(ROOT/'results/esol_conformers.pkl','rb'));lookup={r['source_index']:r for r in cache};ytrain=np.array([lookup[split['source_indices'][i]]['target'] for i in split['train']],dtype=np.float32);mu=float(ytrain.mean());sd=float(ytrain.std());y=np.array([r['target'] for r in rs]);old=json.loads((ROOT/'results/module2_results.json').read_text());oldpred=np.array(old['test_predictions']);p64=ref['fp64_standardized']*sd+mu;p32=ref['fp32_standardized']*sd+mu;phip=hip*sd+mu
 layer=[{'stage':'embedding_and_gaussian','x_max_abs':0.,'pair_max_abs':0.}]+[{'stage':f'encoder_layer_{i+1}','x_max_abs':0.,'pair_max_abs':0.} for i in range(15)]
 with (PORT/'reference_trace_fp64.bin').open('rb') as a,(PORT/'out_hip/layer_trace_fp64.bin').open('rb') as b:
  for N in man['sizes']:
   for item in layer:
    for name,n in [('x_max_abs',N*512),('pair_max_abs',64*N*N)]:
     va=np.fromfile(a,np.float64,n);vb=np.fromfile(b,np.float64,n)
     if len(va)!=n or len(vb)!=n:raise ValueError('trace size mismatch')
     item[name]=max(item[name],float(abs(va-vb).max()))
 def met(p):return {'RMSE':float(np.sqrt(mean_squared_error(y,p))),'R2':float(r2_score(y,p))}
 result={'module':13,'parameter_count':man['parameter_count'],'layers':15,'test_samples':len(rs),'checkpoint_sha256':man['checkpoint_sha256'],'test_source_indices':man['test_source_indices'],'AMD_HIP_FP64':met(phip),'GB10_reference_FP64':met(p64),'GB10_official_FP32_no_TF32':met(p32),'original_GB10_BF16_autocast':met(oldpred),'AMD_vs_GB10_FP64_max_abs_logS':float(abs(phip-p64).max()),'AMD_vs_GB10_official_FP32_max_abs_logS':float(abs(phip-p32).max()),'AMD_vs_original_BF16_max_abs_logS':float(abs(phip-oldpred).max()),'layerwise_all112_max_abs':layer,'AMD_runtime':json.loads((PORT/'out_hip/runtime.json').read_text()),'GB10_runtime':json.loads((PORT/'reference_runtime.json').read_text()),'targets':y.tolist(),'predictions_AMD_FP64':phip.tolist(),'predictions_GB10_FP64':p64.tolist(),'predictions_GB10_FP32':p32.tolist(),'preserved_original_BF16_predictions':oldpred.tolist(),'actual_head':man['head'],'post_ln':man['encoder_post_ln'],'final_encoder_layernorm':man['final_encoder_layernorm'],'scope':'full original checkpoint15 layers all47,331,139 parameters nativeHIP/hipBLAS inference; actual LinearHead (no tanh); dropout disabled eval; no padding (single molecule per forward); all pair updates preserved','precision_note':'Original training evaluation used BF16 autocast. This independent exact-weight FP64 inference retains original predictions and reports differences, not force agreement with rounded BF16. FP64 reference changes only Gaussian helper forced FP32 casts to double for mathematical precision.','pass':len(hip)==112 and float(abs(phip-p64).max())<1e-5}
 result['original_reported_metrics_unchanged']=old['unimol_test'];result['FP32_vs_HIP_below_1e5']=result['AMD_vs_GB10_official_FP32_max_abs_logS']<1e-5;result['AMD_runtime']['exact_weight_tensor_bytes']=man['parameter_count']*8;result['AMD_runtime']['memory_note']='weight_device_bytes and peak_estimated_device_bytes are observed free-memory deltas, not exact allocation accounting; exact parameter storage listed separately';result['trace_checked_entries']=sum(16*(n*512+64*n*n) for n in man['sizes']);result['max_layerwise_node_abs']=max(v['x_max_abs'] for v in layer);result['max_layerwise_pair_abs']=max(v['pair_max_abs'] for v in layer);result['pass']=bool(result['pass'] and result['max_layerwise_node_abs']<1e-9 and result['max_layerwise_pair_abs']<1e-9)

 previous_path=ROOT/'results/module13_results.json'
 if previous_path.exists():
  previous=json.loads(previous_path.read_text())
  for key in ['optional_FP32_Gaussian_full15layer','official_fp32_compatibility']:
   if key in previous:result[key]=previous[key]
 (ROOT/'results/module13_results.json').write_text(json.dumps(result,indent=2));print(json.dumps({k:v for k,v in result.items() if k in ['AMD_HIP_FP64','GB10_reference_FP64','GB10_official_FP32_no_TF32','original_GB10_BF16_autocast','AMD_vs_GB10_FP64_max_abs_logS','pass']},indent=2))
 if not result['pass']:raise RuntimeError('full inference agreement failed')

def render():
 from html import escape
 import cairosvg
 r=json.loads((ROOT/'results/module13_results.json').read_text());s=['<svg xmlns="http://www.w3.org/2000/svg" width="640" height="1380" font-family="\'WenQuanYi Micro Hei\',\'Noto Sans CJK SC\',sans-serif"><rect width="640" height="1380" fill="#ffffff"/>']
 def text(x,y,t,size=17,col='#203449'):s.append(f'<text x="{x}" y="{y}" font-size="{size}" fill="{col}">{escape(str(t))}</text>')
 text(25,40,'Uni-Mol 全 15 层：原生 HIP 完整推理',24);text(25,76,'真实 ESOL 检查点 47,331,139 参数；112 个原留出测试分子',16)
 text(25,112,'原子嵌入 → 高斯距离 → 15层成对更新 → 真实线性回归头',16)
 text(25,150,'AMD HIP FP64 与 GB10 FP64：逐点预测',21)
 xs=np.array(r['predictions_GB10_FP64']);ys=np.array(r['predictions_AMD_FP64']);lo=min(xs.min(),ys.min())-.2;hi=max(xs.max(),ys.max())+.2;x0,y0,w,h=60,180,520,300;s.append(f'<rect x="{x0}" y="{y0}" width="{w}" height="{h}" fill="#fafafa" stroke="#ddd"/>');s.append(f'<line x1="{x0}" y1="{y0+h}" x2="{x0+w}" y2="{y0}" stroke="#999" stroke-dasharray="4 4"/>')
 for a,b in zip(xs,ys):s.append(f'<circle cx="{x0+(a-lo)/(hi-lo)*w}" cy="{y0+h-(b-lo)/(hi-lo)*h}" r="3" fill="#159e84" opacity=".7"/>')
 text(60,502,'横轴 GB10 logS；纵轴 AMD HIP logS（同一尺度）',14)
 text(25,535,f"112 个预测最大差 {r['AMD_vs_GB10_FP64_max_abs_logS']:.3g} logS",19)
 text(25,570,'逐层核对：全部112分子的最大绝对误差',20)
 vals=[max(v['x_max_abs'],v['pair_max_abs']) for v in r['layerwise_all112_max_abs']];maximum=max(vals);points=' '.join(f'{55+i*35},{760-(np.log10(max(v,1e-15))+15)/max(np.log10(maximum)+15,1)*150}' for i,v in enumerate(vals));s.append('<rect x="50" y="595" width="540" height="175" fill="#fafafa" stroke="#ddd"/>');s.append(f'<polyline points="{points}" fill="none" stroke="#2482bb" stroke-width="2"/>');text(25,800,f'逐层最大误差 {maximum:.3g}；纵轴 log10(|误差|)，约1e-15至1e-12',14)
 text(25,844,'精度分账：保留 BF16 旧结果，不强行贴数字',20)
 for i,(label,k) in enumerate([('原始 BF16','original_GB10_BF16_autocast'),('GB10 FP64','GB10_reference_FP64'),('AMD HIP FP64','AMD_HIP_FP64')]):text(25,878+i*31,f"{label}：RMSE {r[k]['RMSE']:.6f}，R² {r[k]['R2']:.6f}",17)
 text(25,997,f"与原始 BF16 最大预测差 {r['AMD_vs_original_BF16_max_abs_logS']:.4g} logS",17)
 text(25,1044,'推理耗时：设备事件累计，包含15层，不含权重加载/回传',16)
 text(25,1078,f"GB10 FP64 {r['GB10_runtime']['gpu_compute_ms']/1000:.3f}s｜AMD FP64 {r['AMD_runtime']['gpu_compute_ms']/1000:.3f}s",19)
 text(25,1117,'这是精度优先的单分子参考实现，未经性能优化。',17)
 text(25,1152,'真实检查点回归头是512→1线性层，没有假设中的tanh两层头。',15)
 text(25,1190,'FP64同公式参考；未在AMD训练，也未重新选择测试分子。',16)
 text(25,1230,'证据：results/module13_results.json；逐层原始张量可核查',15)
 c=r.get('official_fp32_compatibility')
 if c:
  text(25,1274,'补齐官方 FP32 目标：仅高斯核沿用原FP32精度',19)
  text(25,1310,f"AMD vs 官方全FP32 最大差 {c['AMD_vs_official_FP32_max_abs_logS']:.3g} &lt; 1e-5".replace('&lt;','<'),17)
  text(25,1346,f"AMD vs 精度匹配参考 最大差 {c['AMD_vs_matched_reference_max_abs_logS']:.3g}",17)
 s.append('</svg>');p=ROOT/'visuals/fig12_unimol_full15layer_end2end_amd9070.svg';p.write_text('\n'.join(s));cairosvg.svg2png(url=str(p),write_to=str(p.with_suffix('.png')))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--stage',choices=['prepare','execute','mixed','compare','render','all'],default='all');a=p.parse_args()
 if a.stage in ['prepare','all']:prepare()
 if a.stage in ['execute','all']:execute()
 if a.stage=='compare':compare();render()
 if a.stage=='mixed':run([str(ROOT/'.venv/bin/python'),str(PORT/'run_mixed_gaussian.py')]);render()
 if a.stage=='render':render()
