#!/usr/bin/env python3
"""Independent saved-artifact checks for Phase 3, including raw numerical outputs."""
from pathlib import Path
import json,hashlib,statistics,xml.etree.ElementTree as ET
import numpy as np
from sklearn.metrics import roc_auc_score,mean_squared_error,r2_score
R=Path(__file__).resolve().parent
load=lambda p:json.loads((R/p).read_text())
def pme():
 r=load('results/module9_results.json');v=r['validation'];assert v['mesh_sizes']['128']['force_relative_l2']<v['mesh_sizes']['64']['force_relative_l2']<v['mesh_sizes']['32']['force_relative_l2']
 assert v['mesh_gradient_max_abs_error']<1e-6
 for bk,b in r['backends'].items():
  assert b['initial_force_relative_l2_vs_cpu']<1e-10
  assert b['nve_dt_halving_drift_ratio']>2
  for name,t in b['trajectories'].items():
   h=np.fromfile(R/f'results/module9_{bk}/{name}_history.bin',dtype='<f8').reshape(-1,7)
   assert h.shape[0]==t['steps']+1 and np.all(np.isfinite(h))
   assert np.allclose(h[:,1]+h[:,2],h[:,3],rtol=1e-12,atol=1e-12)
   drift=np.max(abs(h[:,3]-h[0,3]))/max(abs(h[0,3]),1.)
   assert np.isclose(drift,t['energy_relative_max_drift'])
  assert b['trajectories']['nve']['steps']==1000 and b['trajectories']['nvt']['steps']==1000
  assert 200<b['trajectories']['nvt']['temperature_mean_second_half']<400
 return r

def hmm():
 r=load('results/module12_results.json');z=np.load(R/'results/module12_all_scores.npz');y=z['labels']
 assert len(y)==100000 and sum(y)==500
 for bk in ['cpu','cuda','hip']:
  a=z[bk];assert a.size==100000 and np.all(np.isfinite(a))
  assert np.array_equal(a,z['cpu'])
  assert hashlib.sha256(a.astype('<f8').tobytes()).hexdigest()==r['backends'][bk]['score_sha256']
  assert np.isclose(roc_auc_score(y,a),r['backends'][bk]['ROC_AUC'])
  t=r['backends'][bk];assert np.isclose(statistics.median(t['score_ms']),t['median_ms'])
 assert r['reference_validation']['independent_2d_DP_max_abs']==0
 return r

def davidson():
 r=load('results/module10_results.json');ref=np.load(R/'results/module10_data/reference_eigenvalues.npy');H=np.fromfile(R/'results/module10_data/H.bin',dtype='<f8').reshape(64,384,384)
 assert hashlib.sha256((R/'results/module10_data/H.bin').read_bytes()).hexdigest()==r['metadata']['H_sha256']
 for bk,b in r['backends'].items():
  ev=np.fromfile(R/f'results/module10_{bk}_eigenvalues.bin',dtype='<f8').reshape(64,16)
  x=np.fromfile(R/f'results/module10_{bk}_eigenvectors.bin',dtype='<f8').reshape(64,384,16)
  assert np.max(abs(ev-ref))<1e-10
  residue=np.linalg.norm(H@x-x*ev[:,None,:],axis=1).max();assert residue<1e-7
  assert np.isclose(residue,b['independent_full_H_max_residual'],rtol=1e-6,atol=1e-12)
  gram=np.transpose(x,(0,2,1))@x;assert np.max(abs(gram-np.eye(16)))<1e-10
 return r

def fx():
 r=load('results/module11_results.json');shape=(64,120,1024)
 cpu=np.memmap(R/'results/module11_cpu_vis.bin',mode='r',dtype='<c16',shape=shape)
 allvis={'cpu':cpu}
 for bk in ['cuda','hip']:
  a=np.memmap(R/f'results/module11_{bk}_vis.bin',mode='r',dtype='<c16',shape=shape);allvis[bk]=a
  numerator=denominator=0.
  for i in range(64):
   assert np.all(np.isfinite(a[i]));numerator+=float(np.sum(abs(a[i]-cpu[i])**2));denominator+=float(np.sum(abs(cpu[i])**2))
  rel=np.sqrt(numerator/denominator);assert rel<1e-12
  assert np.isclose(rel,r['backends'][bk]['visibility_relative_l2_vs_cpu'],atol=1e-25)
 images={bk:np.load(R/f'results/module11_{bk}_images.npz')['clean'] for bk in ['cpu','cuda','hip']}
 rel=np.linalg.norm(images['hip']-images['cuda'])/np.linalg.norm(images['cuda']);assert rel<1e-12
 assert r['metadata']['voltage_samples_complex']==16*4096*1024
 assert max(v['abs_diff'] for v in r['independent_direct_fourier_pixel_sum_checks'])<1e-12
 return r

def full_model():
 r=load('results/module13_results.json');m=load('results/module13_manifest.json');old=load('results/module2_results.json');split=load('results/module2_split.json')
 assert r['parameter_count']==m['parameter_count']==47331139 and r['layers']==15 and r['test_samples']==112
 assert r['test_source_indices']==[split['source_indices'][i] for i in split['test']]
 assert np.array_equal(r['preserved_original_BF16_predictions'],old['test_predictions'])
 for k,metric in [('predictions_AMD_FP64','AMD_HIP_FP64'),('predictions_GB10_FP64','GB10_reference_FP64'),('predictions_GB10_FP32','GB10_official_FP32_no_TF32')]:
  pred=np.array(r[k]);target=np.array(r['targets']);assert len(pred)==112 and np.all(np.isfinite(pred))
  assert np.isclose(np.sqrt(mean_squared_error(target,pred)),r[metric]['RMSE'])
  assert np.isclose(r2_score(target,pred),r[metric]['R2'])
 error=np.max(abs(np.asarray(r['predictions_AMD_FP64'])-np.asarray(r['predictions_GB10_FP64'])))
 assert error<1e-5 and np.isclose(error,r['AMD_vs_GB10_FP64_max_abs_logS'],atol=1e-20)
 port=R/'amd_hip_port/unimol_full15_hip'
 assert sum(int(v.split()[1]) for v in m['exported_parameters'])==47331139
 assert (port/'input/weights_fp64.bin').stat().st_size==47331139*8
 assert hashlib.sha256((port/'input/molecules.bin').read_bytes()).hexdigest()==m['input_sha256']
 assert m['no_predictions_or_labels_sent_to_AMD']
 checks=[{'x_max_abs':0.,'pair_max_abs':0.} for _ in range(16)];count=0
 with (port/'reference_trace_fp64.bin').open('rb') as a,(port/'out_hip/layer_trace_fp64.bin').open('rb') as b:
  for n in m['sizes']:
   for stage in checks:
    for key,num in [('x_max_abs',n*512),('pair_max_abs',64*n*n)]:
     aa=np.fromfile(a,dtype='<f8',count=num);bb=np.fromfile(b,dtype='<f8',count=num)
     assert aa.size==bb.size==num
     stage[key]=max(stage[key],float(np.max(abs(aa-bb))));count+=num
  assert a.read(1)==b.read(1)==b''
 assert count==r['trace_checked_entries']==50501632
 for i,t in enumerate(checks):
  for key,value in t.items():
   assert value<1e-9 and np.isclose(value,r['layerwise_all112_max_abs'][i][key],atol=1e-20)
 c=r['official_fp32_compatibility'];assert c['pass']
 a=np.asarray(c['AMD_pred_logS']);b=np.asarray(c['reference_pred_logS']);matched=np.asarray(c['matched_reference_pred_logS'])
 assert a.size==b.size==matched.size==112
 assert np.array_equal(b,r['predictions_GB10_FP32'])
 assert np.max(abs(a-b))<1e-5 and np.max(abs(a-matched))<1e-5
 assert np.isclose(np.max(abs(a-b)),c['AMD_vs_official_FP32_max_abs_logS'],atol=1e-20)
 assert np.isclose(np.sqrt(mean_squared_error(r['targets'],a)),c['metrics']['RMSE'])
 return r

def figures():
 expected=['fig8_pme_kspace_and_1000step_md_trajectory','fig9_daviator_batched_davidson_vasp_band','fig10_cuphoton_eht_blackhole_correlator','fig11_gpuhmmer_100k_protein_viterbi','fig12_unimol_full15layer_end2end_amd9070']
 for name in expected:
  p=R/'visuals'/f'{name}.svg';text=p.read_text();root=ET.fromstring(text)
  assert 560<=float(root.attrib['width'].replace('px',''))<=680
  assert 'WenQuanYi Micro Hei' in root.attrib.get('font-family','')
  assert '\u1d40' not in text
  white=any(el.attrib.get('fill','').lower()=='#ffffff' or '#ffffff' in el.attrib.get('style','').lower() for el in root.iter())
  assert white and p.with_suffix('.png').exists()

def main():
 pme();hmm()
 davidson();fx();full_model()
 figures();print('PASS Phase3: PME mesh/gradient/trajectory/dt/thermostat, complete 100k HMM scores/ROC/hash/timing, all eigenpairs/full residuals, complete complex FX visibility and CLEAN images, all112 full15-layer traces/metrics, 5 white SVG/PNG')
if __name__=='__main__':main()
