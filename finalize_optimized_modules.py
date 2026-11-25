#!/usr/bin/env python3
"""Version current steady-state measurements without overwriting original experiment snapshots."""
from pathlib import Path
import json,datetime,hashlib,copy
R=Path(__file__).resolve().parent;REC='same_source_optimization_20261125_warm5_rep7'
load=lambda p:json.loads((R/p).read_text())
def main():
 fx=load('results/module11_optimized_results.json');cf=load('results/module14_optimized_results.json');variant=fx['selected_common_variant'];assert variant=='v2' and fx['status']==cf['status']=='PASS'
 ledger={'record_id':REC,'updated':datetime.datetime.now().astimezone().isoformat(),'timing_scope':'resident computation, synchronized wall medians; transfers/allocation/I/O separate; F/X and full F+X independently timed','FP64_noFMA':True,'FX':{'variant':variant,'source_sha256':fx['new_source_sha256'],'protocol':fx['protocol'],'backends':{},'validation':fx['comparisons']},'conformer':{'source_sha256':cf['source_sha256'],'protocol':cf['thermal_retest'],'backends':copy.deepcopy(cf['backends']),'validation':cf['comparisons'],'gradient_validation':cf['gradient_validation']}}
 cpu=fx['CPU'];ledger['FX']['backends']['cpu']={'FFT_seconds':cpu['FFT_wall_median_seconds'],'X_seconds':cpu['X_wall_median_seconds'],'F_plus_X_seconds':cpu['pipeline_wall_median_seconds'],'raw_FFT_seconds':cpu['FFT_wall_seconds'],'raw_X_seconds':cpu['X_wall_seconds'],'raw_F_plus_X_seconds':cpu['pipeline_wall_seconds'],'warmup_passes':cpu['warmup_passes'],'repeats':cpu['repeats']}
 for bk,t in fx['backends'].items():
  v=t['variants'][variant];ledger['FX']['backends'][bk]={'FFT_seconds':v['FFT_wall_median_seconds'],'X_seconds':v['X_wall_median_seconds'],'F_plus_X_seconds':v['pipeline_wall_median_seconds'],'raw_FFT_seconds':v['FFT_wall_seconds'],'raw_X_seconds':v['X_wall_seconds'],'raw_F_plus_X_seconds':v['pipeline_wall_seconds'],'warmup_passes':t['warmup_passes'],'repeats':t['repeats'],'h2d_seconds':t['h2d_seconds'],'d2h_seconds':v['d2h_seconds'],'transfer_plus_pipeline_estimate_seconds':v['host_transfer_plus_pipeline_estimate_seconds'],'same_harness_v0_F_plus_X_seconds':t['variants']['v0']['pipeline_wall_median_seconds']}
 out=R/'results/optimization_20261125';out.mkdir(exist_ok=True);(out/'latest_ledger.json').write_text(json.dumps(ledger,indent=2)+'\n')
 for n,record,path in [(11,ledger['FX'],'results/module11_optimized_results.json'),(14,ledger['conformer'],'results/module14_optimized_results.json')]:
  p=R/f'results/module{n}_results.json';old=json.loads(p.read_text());baseline=out/f'baseline_module{n}_results.json'
  if not baseline.exists():baseline.write_bytes(p.read_bytes())
  old['active_performance_record']=REC
  old['current_steady_state']=record
  old['original_backends_scope']='Original experiment snapshot retained in backends; current steady-state timing is authoritative in current_steady_state, not a mutation of historical end-to-end measurements.'
  history=old.setdefault('optimization_history',[])
  entry={'record_id':REC,'record_path':path,'baseline_record_path':str(baseline.relative_to(R)),'source_sha256':record['source_sha256']}
  history[:]=[e for e in history if e.get('record_id')!=REC]+[entry]
  p.write_text(json.dumps(old,indent=2,ensure_ascii=False)+'\n')
 print(json.dumps({k:{bk:{n:v[n] for n in (['FFT_seconds','X_seconds','F_plus_X_seconds'] if k=='FX' else ['seconds'])} for bk,v in ledger[k]['backends'].items()} for k in ['FX','conformer']},indent=2))
if __name__=='__main__':main()
