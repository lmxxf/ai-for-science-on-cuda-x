#!/usr/bin/env python3
"""Export official UniMol weights/normalized real-drug inputs, compile same CUDA/HIP kernel."""
import os,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT));os.environ['UNIMOL_WEIGHT_DIR']=str(ROOT/'weights')
import json,subprocess,time
import numpy as np
import torch
from exp1_rdkit_data_processing import conformer,DRUGS
from unimol_tools.models.unimol import UniMolModel
from unimol_tools.data.conformer import coords2unimol
PORT=Path(__file__).resolve().parent

def main():
 torch.set_num_threads(4);m=UniMolModel(output_dim=1,remove_hs=True);m.load_state_dict(torch.load(ROOT/'results/unimol_bbbp_best.pt',map_location='cpu',weights_only=True));m.eval()
 records=[conformer((i,s,0,1)) for i,s in enumerate(DRUGS.values())];features=[coords2unimol(r['atoms'],r['coordinates'],m.dictionary,remove_hs=True) for r in records];batch,_=m.batch_collate_fn([(f,0.) for f in features]);layer=m.encoder.layers[0]
 with torch.no_grad():
  emb=m.encoder.emb_layer_norm(m.embed_tokens(batch['src_tokens']));mask=batch['src_tokens'].eq(m.padding_idx);emb=emb*(1-mask.unsqueeze(-1).type_as(emb));x=layer.self_attn_layer_norm(emb)
  gbf=m.gbf(batch['src_distance'],batch['src_edge_type']);bias=m.gbf_proj(gbf);qkv=layer.self_attn.in_proj(x);B,N,_=x.shape;q,k,v=qkv.chunk(3,dim=-1);q=q.reshape(B,N,64,8).transpose(1,2)*8**-.5;k=k.reshape(B,N,64,8).transpose(1,2);v=v.reshape(B,N,64,8).transpose(1,2);scores=q@k.transpose(-1,-2)+bias.permute(0,3,1,2);scores=scores.masked_fill(mask[:,None,None,:],-torch.inf);attention=torch.softmax(scores,dim=-1);context=(attention@v).transpose(1,2).reshape(B,N,512)
 out=PORT/'input';out.mkdir(exist_ok=True)
 arrays={'distance':batch['src_distance'],'edge':batch['src_edge_type'].int(),'mask':mask.int(),'mu':m.gbf.means.weight,'std':m.gbf.stds.weight,'mul':m.gbf.mul.weight,'edge_bias':m.gbf.bias.weight,'w1':m.gbf_proj.linear1.weight,'b1':m.gbf_proj.linear1.bias,'w2':m.gbf_proj.linear2.weight,'b2':m.gbf_proj.linear2.bias,'normalized_input':x,'qkv_weight':layer.self_attn.in_proj.weight,'qkv_bias':layer.self_attn.in_proj.bias}
 for n,a in arrays.items():a.detach().contiguous().numpy().tofile(out/f'{n}.bin')
 (out/'shape.txt').write_text(f'{B} {N} {m.gbf.mul.weight.numel()}\n')
 refs={'gaussian':gbf.numpy(),'bias':bias.numpy(),'qkv':qkv.numpy(),'attention':attention.reshape(B*64,N,N).numpy(),'context':context.numpy()}
 for name,a in refs.items():a.astype('float32').tofile(out/f'reference_{name}.bin')
 shapes={n:list(a.shape) for n,a in refs.items()};(out/'reference_shapes.json').write_text(json.dumps(shapes,indent=2));(out/'molecule_sources.json').write_text(json.dumps({'smiles':list(DRUGS.values()),'note':'8 real-drug molecules including salt-string example; RDKit heavy-atom candidate conformers','layer_norm':'official embedding LayerNorm -> zero pad -> official layer0 pre-attention LayerNorm; computed in PyTorch and exported'},ensure_ascii=False,indent=2))
 def run(cmd):print('RUN',cmd,flush=True);subprocess.run(cmd,check=True,cwd=ROOT)
 run(['/usr/local/cuda/bin/nvcc','-O3','--fmad=false','-std=c++17','-arch=sm_121','-x','cu',str(PORT/'pairbias.cpp'),'-o',str(PORT/'pairbias_cuda')])
 run([str(PORT/'pairbias_cuda'),str(out),str(PORT/'out_cuda'),'7'])
 run(['ssh','amd9070','mkdir C:\\AI4S-HIP-Lab\\unimol_pairbias 2>NUL & exit /b 0'])
 run(['scp',str(PORT/'pairbias.cpp'),str(PORT/'build_windows.cmd'),str(PORT/'run_windows.cmd'),'amd9070:C:/AI4S-HIP-Lab/unimol_pairbias/'])
 run(['scp','-r',str(out),'amd9070:C:/AI4S-HIP-Lab/unimol_pairbias/'])
 run(['ssh','amd9070','C:\\AI4S-HIP-Lab\\unimol_pairbias\\build_windows.cmd'])
 run(['ssh','amd9070','C:\\AI4S-HIP-Lab\\unimol_pairbias\\run_windows.cmd'])
 run(['scp','-r','amd9070:C:/AI4S-HIP-Lab/unimol_pairbias/out_hip',str(PORT)])
 run(['scp','amd9070:C:/AI4S-HIP-Lab/unimol_pairbias/build.log',str(PORT/'build_windows.log')])
 comparisons={};maxcross=0;maxref=0
 for name,ref in refs.items():
  a=np.fromfile(PORT/'out_cuda'/f'{name}.bin',dtype=np.float32).reshape(ref.shape);b=np.fromfile(PORT/'out_hip'/f'{name}.bin',dtype=np.float32).reshape(ref.shape)
  c=float(np.max(abs(a-b)));ra=float(np.max(abs(a-ref)));rb=float(np.max(abs(b-ref)));comparisons[name]={'CUDA_vs_HIP_max_abs':c,'CUDA_vs_official_PyTorch_max_abs':ra,'HIP_vs_official_PyTorch_max_abs':rb,'CUDA_vs_HIP_relative_L2':float(np.linalg.norm(a-b)/(np.linalg.norm(a)+1e-30))};maxcross=max(maxcross,c);maxref=max(maxref,ra,rb)
 result={'cuda':json.loads((PORT/'out_cuda/timing.json').read_text()),'hip':json.loads((PORT/'out_hip/timing.json').read_text()),'comparisons':comparisons,'cuda_ms':json.loads((PORT/'out_cuda/timing.json').read_text())['median_ms'],'hip_ms':json.loads((PORT/'out_hip/timing.json').read_text())['median_ms'],'cross_device_max_abs':maxcross,'torch_reference_max_abs':maxref,'input_shape':{'batch':B,'padded_atoms':N,'hidden':512,'head_dim':8,'heads':64},'scope':'naive educational Gaussian128->Linear128/GELU/Linear64 + first-layer QKV with biases + masked softmax + AV. LayerNorm output is exported from official PyTorch; excludes all subsequent Transformer layers, output projection, FFN, and AMD full-model training.','precision':'float32; FMA disabled; CUDA and HIP same C++ source','timing_scope':'resident device tensors; five forward kernels after one warmup, median7; excludes H2D,D2H,load,normalization and compile','pass':maxcross<1e-3 and maxref<1e-3}
 (ROOT/'results/module6_pairbias_results.json').write_text(json.dumps(result,indent=2));r=json.loads((ROOT/'results/module6_results.json').read_text());r['pairbias']=result;(ROOT/'results/module6_results.json').write_text(json.dumps(r,ensure_ascii=False,indent=2));print(json.dumps(result,indent=2),flush=True)
 if not result['pass']:raise RuntimeError('numerical check failed')
if __name__=='__main__':main()
