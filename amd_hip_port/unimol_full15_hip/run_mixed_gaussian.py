"""Additive mixed-precision evidence; does not overwrite full-FP64 outputs."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from exp11_unimol_full15layer_amd9070 import *

def main():
 torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;m=model().cuda().double();m.gbf.float();official=m.gbf.forward;m.gbf.forward=lambda x,e:official(x.float(),e).double();rs=samples();pred=[]
 with (PORT/'reference_trace_mixed.bin').open('wb') as f,torch.no_grad():
  for r in rs:
   ft=coords2unimol(r['atoms'],r['coordinates'],m.dictionary,remove_hs=True);b,_=m.batch_collate_fn([(ft,0.)]);b={k:(v.cuda().double() if v.dtype.is_floating_point else v.cuda()) for k,v in b.items()};N=b['src_tokens'].shape[1];x=m.encoder.emb_layer_norm(m.embed_tokens(b['src_tokens']));pair=m.gbf_proj(m.gbf(b['src_distance'],b['src_edge_type'])).permute(0,3,1,2).contiguous().reshape(64,N,N);x.cpu().numpy().tofile(f);pair.cpu().numpy().tofile(f)
   for layer in m.encoder.layers:
    x,pair,_=layer(x,attn_bias=pair,padding_mask=None,return_attn=True);x.cpu().numpy().tofile(f);pair.cpu().numpy().tofile(f)
   p=m.classification_head(m.encoder.final_layer_norm(x)[:,0,:]).reshape(-1);pred.append(float(p[0]))
 np.array(pred).tofile(PORT/'reference_predictions_mixed.bin')
 run(['ssh','amd9070','cmd /c "set PATH=C:\\Strata-Lab\\Strata\\.rocm-win\\Lib\\site-packages\\_rocm_sdk_devel\\bin;%PATH%&&cd /d C:\\AI4S-HIP-Lab\\unimol_full15_hip&&unimol_full_hip.exe input out_hip_mixed gaussian_fp32 > run_mixed.log 2>&1"'])
 run(['scp','-r','amd9070:C:/AI4S-HIP-Lab/unimol_full15_hip/out_hip_mixed',str(PORT)]);run(['scp','amd9070:C:/AI4S-HIP-Lab/unimol_full15_hip/run_mixed.log',str(PORT/'run_mixed_windows.log')]);compare_mixed()

def compare_mixed():
 man=json.loads((ROOT/'results/module13_manifest.json').read_text());ref=np.fromfile(PORT/'reference_predictions_mixed.bin',np.float64);hip=np.fromfile(PORT/'out_hip_mixed/predictions_fp64.bin',np.float64);original=np.load(PORT/'reference_predictions.npz')['fp32_standardized'];split=json.loads((ROOT/'results/module2_split.json').read_text());lookup={r['source_index']:r for r in pickle.load(open(ROOT/'results/esol_conformers.pkl','rb'))};yt=np.array([lookup[split['source_indices'][i]]['target'] for i in split['train']],dtype=np.float32);mu=float(yt.mean());sd=float(yt.std());target=np.array([r['target'] for r in samples()]);layer=[{'stage':'embedding_and_pair','node_max_abs':0.,'pair_max_abs':0.}]+[{'stage':f'layer{i+1}','node_max_abs':0.,'pair_max_abs':0.} for i in range(15)]
 with (PORT/'reference_trace_mixed.bin').open('rb') as a,(PORT/'out_hip_mixed/layer_trace_fp64.bin').open('rb') as b:
  for N in man['sizes']:
   for item in layer:
    for field,n in [('node_max_abs',N*512),('pair_max_abs',64*N*N)]:
     x=np.fromfile(a,np.float64,n);y=np.fromfile(b,np.float64,n)
     if len(x)!=n or len(y)!=n:raise ValueError('trace length')
     item[field]=max(item[field],float(abs(x-y).max()))
 diff32=float(abs(hip-original).max()*sd);diffmatch=float(abs(hip-ref).max()*sd)
 r={'scope':'same full15layer HIP/hipBLAS FP64 weights and network, optional only Gaussian arithmetic FP32 matching upstream official; no prediction calibration','samples':len(hip),'AMD_vs_official_GB10_FP32_max_abs_logS':diff32,'AMD_vs_precision_matched_GB10_mixed_max_abs_logS':diffmatch,'GB10_mixed_vs_official_FP32_max_abs_logS':float(abs(ref-original).max()*sd),'AMD_metrics':{'RMSE':float(np.sqrt(mean_squared_error(target,hip*sd+mu))),'R2':float(r2_score(target,hip*sd+mu))},'precision_matched_reference_metrics':{'RMSE':float(np.sqrt(mean_squared_error(target,ref*sd+mu))),'R2':float(r2_score(target,ref*sd+mu))},'layerwise_errors':layer,'AMD_runtime':json.loads((PORT/'out_hip_mixed/runtime.json').read_text()),'AMD_predictions_logS':(hip*sd+mu).tolist(),'GB10_precision_matched_predictions_logS':(ref*sd+mu).tolist(),'pass_official_FP32':len(hip)==112 and diff32<1e-5,'pass_matched_reference':len(hip)==112 and diffmatch<1e-5}
 (ROOT/'results/module13_mixed_precision_results.json').write_text(json.dumps(r,indent=2));full=json.loads((ROOT/'results/module13_results.json').read_text());full['optional_FP32_Gaussian_full15layer']=r;full['official_fp32_compatibility']={'pass':bool(r['pass_official_FP32'] and r['pass_matched_reference']),'AMD_vs_official_FP32_max_abs_logS':diff32,'AMD_vs_matched_reference_max_abs_logS':diffmatch,'AMD_pred_logS':(hip*sd+mu).tolist(),'reference_pred_logS':(original*sd+mu).tolist(),'matched_reference_pred_logS':(ref*sd+mu).tolist(),'reference_definition':'official UniMol PyTorch full FP32, TF32 disabled, unchanged original checkpoint','precision_definition':'only complete Gaussian arithmetic FP32 as upstream; all other layers, weights storage and hipBLAS GEMMs FP64; all15 original layers','metrics':r['AMD_metrics'],'runtime':r['AMD_runtime'],'layerwise_vs_matched_reference':layer};(ROOT/'results/module13_results.json').write_text(json.dumps(full,indent=2));print(json.dumps({k:v for k,v in r.items() if not isinstance(v,list)},indent=2),flush=True)
 if not r['pass_official_FP32'] or not r['pass_matched_reference']:raise RuntimeError('mixed precision did not meet1e-5')
if __name__=='__main__':main()
