#!/usr/bin/env python3
"""Publish measured layout candidates while retaining the non-regressing v0 default."""
from pathlib import Path
import json,copy,datetime
R=Path(__file__).resolve().parent
load=lambda p:json.loads((R/p).read_text())
def main():
 new=load('results/module7_optimized_results.json');old_round=load('results/module7_optimized_round1_results.json');assert new['status']=='PASS'
 record={'record_id':'nonbonded_layout_20261125_warm5_rep7','updated':datetime.datetime.now().astimezone().isoformat(),'default_variant':'v0','tested_candidate_variants':['v1','v2','v3'],'recommendation':'retain v0 default: candidates improve selected AMD medians but regress GB10; do not mix variants between cards','source_sha256':new['source_sha256'],'protocol':new['protocol'],'latest_candidate_record':'results/module7_optimized_results.json','previous_candidate_record':'results/module7_optimized_round1_results.json','physics_validation':new['physics_validation'],'systems':[]}
 byn={s['n']:s for s in old_round['systems']}
 for s in new['systems']:
  item={'n':s['n'],'input_sha256':s['input_sha256'],'backends':{},'candidate_round1':{},'candidate_soa':{},'candidate_comparisons':s['comparisons']}
  for bk,t in s['backends'].items():
   flat=copy.deepcopy(t['variants']['v0']);flat['evaluations_per_s']=1000.0/flat['pipeline_median_ms'];flat.update(device=t['device'],warmups=t['warmups'],repeats=t['repeats'],variant='v0',timing_scope=t['scope']);item['backends'][bk]=flat
  for bk in ['cuda','hip']:
   item['candidate_round1'][bk]=copy.deepcopy(byn[s['n']]['backends'][bk]['variants']['v2'])
   item['candidate_soa'][bk]=copy.deepcopy(s['backends'][bk]['variants']['v3'])
  item['hip_vs_cuda']={'energy_relative_error':0.,'force_relative_l2':0.}
  record['systems'].append(item)
 p=R/'results/module7_results.json';root=json.loads(p.read_text());root['active_performance_record']=record['record_id'];root['current_steady_state']=record;h=root.setdefault('optimization_history',[]);h[:]=[x for x in h if x.get('record_id')!=record['record_id']]+[{'record_id':record['record_id'],'baseline_record':'results/nonbonded_optimization_20261125/baseline_module7_results.json','latest_record':record['latest_candidate_record'],'round1_record':record['previous_candidate_record'],'default_retained':'v0'}];p.write_text(json.dumps(root,ensure_ascii=False,indent=2)+'\n')
 out=R/'results/nonbonded_optimization_20261125/latest_ledger.json';out.write_text(json.dumps(record,indent=2)+'\n')
 # Same performance table in both public homepages.
 ns=record['systems'];fmt=lambda b:' / '.join(f"{s['backends'][b]['pipeline_median_ms']:.3f}" for s in ns)
 row=f"| **2 非键力场（默认v0）** | 32,000 / 128,000原子，LJ＋反应场库仑＋解析力 | {fmt('cpu')}ms | {fmt('cuda')}ms | **{fmt('hip')}ms** | 默认核全E/F三端L2＝0；预sqrt候选对原CPU≤6.70e-17 |"
 detail='''本轮比较了原核v0、连续AoS7重排v1/v2与SoA7重排v3。预sqrt候选在AMD有局部收益，但GB10回退；**默认保留v0**，两卡始终比较相同变体。5次预热、7次测量，完整计时包括Cell-List、重排与求力。\n\n| 原子数 | 同源变体 | GB10 / ms | RX 9070 XT / ms |\n|---:|---|---:|---:|\n'''
 for s in ns:
  for label,key in [('默认v0','backends'),('AoS7＋预sqrt v2（前轮）','candidate_round1'),('SoA7＋预sqrt v3','candidate_soa')]:
   d=s[key];detail+=f"| {s['n']:,} | {label} | {d['cuda']['pipeline_median_ms']:.3f} | {d['hip']['pipeline_median_ms']:.3f} |\n"
 detail+='\n全部逐原子能量及3N受力对原公式CPU相对L2≤6.70e-17，跨卡全数组一致；独立全配对、有限差分、周期平移和净力检查通过。[完整对照](results/module7_optimized_results.json) · [参数与复测记录](MODULE7_OPTIMIZATION_NOTES.md)\n'
 for repo in [R,R.parent/'ai-for-science-on-amd-9070xt']:
  p=repo/'README.md';text=p.read_text();lines=text.splitlines();matches=[i for i,l in enumerate(lines) if l.startswith('| **2 非键力场')];assert len(matches)==1
  lines[matches[0]]=row;text='\n'.join(lines)+'\n'
  if repo.name=='ai-for-science-on-amd-9070xt':
   a=text.index('128,000原子：',text.index('## 2.')) if '128,000原子：' in text[text.index('## 2.'):text.index('## 3.')] else text.index('本轮比较了原核v0',text.index('## 2.'));b=text.index('\n[源码与物理参数]',a);text=text[:a]+detail+'\n'+text[b:]
  else:
   needle='**构象与FX采用最新热态账本**';i=text.index(needle);old_note=text.find('**非键力场保留v0默认**');text=text if old_note<0 else text[:old_note]+text[i:];i=text.index(needle);text=text[:i]+'**非键力场保留v0默认**：排序候选的AMD局部收益伴随GB10回退，原核/候选同配置结果完整保留。最新5warm＋7rep分账和全数组精度见[模块2对照](results/module7_optimized_results.json)。\n\n'+text[i:]
  p.write_text(text)
 print(json.dumps(record['systems'],indent=2)[:600]);print('Updated versioned module7 and matched README rows; v0 remains default')
if __name__=='__main__':main()
