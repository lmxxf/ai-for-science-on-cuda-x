from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from exp11_unimol_full15layer_amd9070 import *
import types
m=model().cuda().double();m.gbf.float();official_gbf=m.gbf.forward
m.gbf.forward=lambda x,edge_type:official_gbf(x.float(),edge_type).double()
rs=samples();p=[];torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
with torch.no_grad():
 for r in rs:
  ft=coords2unimol(r['atoms'],r['coordinates'],m.dictionary,remove_hs=True);b,_=m.batch_collate_fn([(ft,0.)]);b={k:(v.cuda().double() if v.dtype.is_floating_point else v.cuda()) for k,v in b.items()};p.append(float(m(**b).reshape(-1)[0]))
ref=np.load(PORT/'reference_predictions.npz');split=json.loads((ROOT/'results/module2_split.json').read_text());lookup={r['source_index']:r for r in pickle.load(open(ROOT/'results/esol_conformers.pkl','rb'))};yt=np.array([lookup[split['source_indices'][i]]['target'] for i in split['train']],dtype=np.float32);mu=float(yt.mean());sd=float(yt.std());p=np.array(p);diff=abs(p-ref['fp32_standardized'])*sd
r={'max_abs_logS_vs_official_FP32':float(diff.max()),'mean_abs_logS':float(diff.mean()),'worst_index':int(diff.argmax()),'mixed_standardized_predictions':p.tolist(),'mixed_logS_predictions':(p*sd+mu).tolist(),'scope':'Gaussian all FP32 exactly official, rest official double; GB10 CUDA'}
(PORT/'mixed_precision_probe.json').write_text(json.dumps(r,indent=2));print(r['max_abs_logS_vs_official_FP32'])
