#!/usr/bin/env python3
"""Append Phase 2 measured outcomes to the base report, without changing metrics."""
from pathlib import Path
import json
ROOT=Path(__file__).resolve().parent
load=lambda p:json.loads((ROOT/p).read_text())
def report_text():
 if not all((ROOT/f'results/module{n}_results.json').exists() for n in [6,7,8]):return ''
 b,f,p=[load(f'results/module{n}_results.json') for n in [6,7,8]]
 pb=b['pairbias'];v=f['validation'];bio=p['dataset'];arch=p['architecture']
 text=f'''

# 第二阶段：血脑屏障、开源非键力场与蛋白质模型

## 模块六：BBBP 分类、真实原子遮挡归因、AMD 注意力核

清洗表2039行，canonical母核有{b['conflicting_parent_count']}组标签冲突，整组剔除而不任意挑标签；同标签母核去重后构象尝试{b['unique_before_conformer']}个，成功{b['n_molecules']}个。具体构象成功/收敛统计在`results/bbbp_conformer_timing.json`；未收敛候选不能称为实验测量几何。固定seed42分层80/10/10为{b['split_sizes']}，canonical母核不跨集。

| 模型 | 留出ROC-AUC | PR-AUC（average precision） | Accuracy |
|---|---:|---:|---:|
'''
 for name,m in list(b['baselines'].items())+[('Uni-Mol全参数微调',b['unimol_test'])]:text+=f"| {name} | {m['ROC_AUC']:.6f} | {m['PR_AUC_AP']:.6f} | {m['Accuracy']:.6f} |\n"
 text+=f'''
官方Uni-Mol v1全{b['parameters']:,}参数训练20轮，BCEWithLogitsLoss、batch32、BF16 autocast/FP32权重、AdamW与cosine调度。仅validation ROC-AUC选第{b['best_epoch']}轮，最终test一次。训练{b['training_seconds']:.3f}s，CPU8进程构象{b['cpu_conformer_seconds']:.3f}s。**本次Uni-Mol与RF基本打平，RF在PR-AP与准确率上略高，不能延续ESOL的“3D明显胜出”叙事。**随机划分不是scaffold外推。

逐重原子遮挡：删除对应原子token与距离行/列，保留其余原坐标、不重优化几何，贡献定义`delta_logit = full_logit - occluded_logit`；正负按真实模型结果染色，未强制亲脂为正或含氧为负。

'''
 for name,a in b['attribution'].items():text+=f"- {name}：原模型概率{a['probability']:.6f}；逐原子贡献范围[{min(a['delta_logit']):.4f}, {max(a['delta_logit']):.4f}]。\n"
 text+='''
两种展示药物均出现在本次训练集合（membership记录于模块六JSON/NOTES），因此它们是模型敏感性解释示例，不是两个独立外推测试。遮挡后的token集合可能不是有效化学分子，概率也未作临床校准，不能当嗜睡诊断或化学因果。

### 官方权重核心前向算子：CUDA / HIP / PyTorch 三方验证

从本次训练checkpoint导出真实高斯参数、原子对仿射、128→128/GELU→64投影和首层QKV参数。官方embedding LayerNorm、首层pre-attention LayerNorm及padding零化在PyTorch执行后导出输入，C++运行Gaussian、投影、QKV（含bias）、缩放点积、padding key mask、softmax、attention×V。8个真实药物输入，包含BOS/EOS及padding；不是假设全原子都是普通词元。

'''
 text+=f"输入batch{pb['input_shape']['batch']}、pad长度{pb['input_shape']['padded_atoms']}、hidden512、64头、每头8维。FP32，关闭FMA contraction，无fastmath。驻留核中位数GB10 **{pb['cuda_ms']:.4f}ms**，RX9070XT **{pb['hip_ms']:.4f}ms**。这是朴素教育实现，不调用BLAS/TensorCore，不用于评价厂商成熟库或整模型性能。\n\n"
 text+='| 张量 | CUDA/HIP max_abs | 对官方PyTorch最坏max_abs |\n|---|---:|---:|\n'
 for name,c in pb['comparisons'].items():text+=f"| {name} | {c['CUDA_vs_HIP_max_abs']:.6g} | {max(c['CUDA_vs_official_PyTorch_max_abs'],c['HIP_vs_official_PyTorch_max_abs']):.6g} |\n"
 text+='''
没有移植后续14层、FFN、attention输出投影或整模型训练；只证明这个真实权重核心前向路径可在AMD原生执行。源码/可复现驱动在`amd_hip_port/unimol_pairbias/`，结果`results/module6_pairbias_results.json`。

![BBBP与原子归因](visuals/fig5_bbbp_unimol_roc_and_atom_attribution.svg)

## 模块七：原创 MIT 开源非键力场（无需 libcupo）

代码`amd_hip_port/cupotential_open/`，驱动`exp5_open_cupotential_hip.py`。支持CPU C++、CUDA sm_121、HIP gfx1201，**是独立实现的非键算子，不宣称二进制/API兼容NVIDIA cuPotential，也不是完整分子动力学引擎**。

输入为固定种子合成中性电荷扰动晶格，不冒称真实溶剂化分子拓扑。坐标Å、能量kJ/mol、受力kJ/(mol·Å)、电荷e；每原子参数q/sigma/epsilon，Lorentz-Berthelot混合。立方周期盒子/最小镜像，cutoff10Å，每格边长>=cutoff且每轴>=3格，GPU分桶、基数排序、扫描，再查27邻格。

`E_LJ=4epsilon[(sigma/r)^12-(sigma/r)^6]`

`E_RF=ke*q_i*q_j[1/r+k_rf*r^2-c_rf]`

`k_rf=(epsilon_RF-1)/(2epsilon_RF+1)/rc^3`，`c_rf=3epsilon_RF/(2epsilon_RF+1)/rc`，epsilon_RF=78.3，ke=1389.354576 kJ·Å/(mol·e²)。

对d=r_i-r_j，力系数为`24epsilon[2(sigma/r)^12-(sigma/r)^6]/r^2 + ke*q_i*q_j[1/r^3-2k_rf]`，乘以d。每原子独立遍历所有邻居，能量除2消除双计数，受力不除2。不含键、角、排除规则或1-4缩放；LJ未shift硬截断，RF能量在cutoff处为零，非PME/Ewald。单次求力没有验证长期能量守恒，不输出伪造ns/day。

| 原子数 | CPU C++ / ms | GB10 CUDA / ms | RX9070XT HIP / ms | HIP/CUDA全受力相对L2 |
|---:|---:|---:|---:|---:|
'''
 active_nonbonded=f.get('current_steady_state');display_systems=active_nonbonded['systems'] if active_nonbonded else f['systems']
 for a in display_systems:
  t=a['backends'];text+=f"| {a['n']:,} | {t['cpu']['pipeline_median_ms']:.4f} | {t['cuda']['pipeline_median_ms']:.4f} | {t['hip']['pipeline_median_ms']:.4f} | {a['hip_vs_cuda']['force_relative_l2']:.3g} |\n"
 text+='\n阶段拆账（独立计时，不与完整流程简单相加）：\n\n| 原子数 | 设备 | Cell-List / ms | 能量+全部受力 / ms | 非键求值次数/s |\n|---:|---|---:|---:|---:|\n'
 for a in display_systems:
  for name,t in a['backends'].items():text+=f"| {a['n']:,} | {name} | {t['index_median_ms']:.4f} | {t['force_median_ms']:.4f} | {t['evaluations_per_s']:.2f} |\n"
 if active_nonbonded:
  text+='\n2026-11-25同源布局对照：5次预热＋7次计时，表中为保留默认的v0全流程（index＋reorder接口＋force）。CPU是原公式单线程参考。原始单次成绩仍保存在JSON systems；当前记录在current_steady_state。\n\n| 原子数 | 候选 | GB10 / ms | AMD / ms |\n|---:|---|---:|---:|\n'
  for a in display_systems:
   for label,key in [('AoS7预sqrt v2（前轮）','candidate_round1'),('SoA7预sqrt v3','candidate_soa')]:
    t=a[key];text+=f"| {a['n']:,} | {label} | {t['cuda']['pipeline_median_ms']:.6f} | {t['hip']['pipeline_median_ms']:.6f} |\n"
  text+='\n两候选对原CPU的全部逐原子E/3N受力相对L2≤6.70e-17，跨GPU全数组一致；两卡使用相同候选，不混基线与优化。候选在AMD有局部收益、GB10回退，128K AMD时序也有波动，因此保留v0默认。原始逐项/完整重复、阴性结果见module7_optimized_results.json与Round1快照。\n'
 text+=f'''
所有N个能量分量与3N个受力分量，在CPU/CUDA/HIP之间逐值完全一致（L2=0），不是空数组：32k体系力范围约[-8.17,8.06]、全力范数458.39；128k全力范数912.24。相同求和顺序、FP64与禁止FMA使本负载达成逐值相同，不扩展成所有体系必然零误差。单独由C++ long double累加的总E日志在Linux/Windows末位可有约1.1e-14相对差（long double宽度不同）；用同一NumPy归约全部能量分量后再对账，总E一致，原始与canonical总E均保留。

独立216原子NumPy暴力全配对参考：能量相对差{v['energy_relative_error']:.3g}、受力相对L2 {v['force_relative_l2']:.3g}；六个方向有限差分（h={v['finite_difference_step']}Å，避开cutoff）最大绝对差{v['finite_difference_max_abs_error']:.3g}；总力L2 {v['net_force_l2']:.3g}；周期平移受力相对差{v['periodic_translation_force_relative_l2']:.3g}。验证代码与全部原始数组留本机，汇总`results/module7_cpu_validation.json`与`results/module7_results.json`。

GPU先预热再重复，驻留同步墙钟；分桶与力核均真实在对应GPU执行。H2D/D2H和冷启动另存JSON；CPU为Spark单线程编译基线，非CPU并行上限，结果不是通用GPU排名。

![非键力场机制](visuals/fig6a_nonbonded_potential.svg)\n\n![非键力场最新布局对照](visuals/fig6b_nonbonded_benchmark.svg)

## 模块八：真实 GB1 实验与官方 ESM-2 全参数微调

原始来源：[FLIP GB1 split archive]({bio['source']})，实验论文[Wu等，eLife 2016]({bio['paper']})。archive SHA256 `{bio['archive_sha256']}`，原CSV `{bio['csv_sha256']}`。真实实验结合适应度，未生成伪标签。

原采样表{bio['raw_rows']}条，全部通过20标准AA验证，去重{bio['duplicate_sequences_removed']}条。原序列265AA，只有39/40/41/54四位点变异；保留完整N端56AA GB1域，裁掉209AA在全部记录恒定的融合尾段，不是按标签挑选片段。固定seed{bio['seed']}，在查看标签前均匀无放回抽{bio['subset_rows']}条，随机2400/300/300。同序列不跨集，但不是FLIP官方leaderboard split，也不是未见蛋白泛化。

Biopython `ProteinAnalysis`计算MW、pI、芳香性、不稳定指数、GRAVY；加20维AAC、400维二肽频次形成425维传统特征，CPU特征时间{bio['feature_seconds']:.3f}s。原序列、数据划分、source_index和描述符均保留。

官方`{arch['model_id']}`实际{arch['parameters_total']:,}可训练参数、{arch['layers']}层、hidden{arch['hidden_size']}、{arch['attention_heads']}头、FFN{arch['intermediate_size']}、词表33、RoPE。33词表不全是“20AA+特殊符号”：还包括X/B/U/Z/O等非标准/模糊残基；token全表与权重SHA在`results/module8_esm2_architecture.json`。编码器全部解冻，新建回归头。

| 同一测试集模型 | RMSE | Spearman | R² |
|---|---:|---:|---:|
'''
 for name,m in p['test'].items():
  rho='未定义（常数）' if m['Spearman'] is None else f"{m['Spearman']:.5f}"
  text+=f"| {name} | {m['RMSE']:.5f} | {rho} | {m['R2']:.5f} |\n"
 text+=f'''
ESM-2全参数训练{p['epochs']}轮、batch64，GPU训练与逐轮验证{p['ESM_gpu_train_seconds']:.3f}s；validation RMSE选第{p['best_epoch']}轮。目标标准化只拟合train，Ridge的alpha只用val挑选，各最终模型test各评一次；逐轮日志`results/module8_epochs.jsonl`，独立test预测`results/module8_test_predictions.csv`。

固定WT背景、39/40/41/54四位点×20AA共80个单点替换，报告`pred(mutant)-pred(WT)`。原AA自替换的变化为零；模型热图不是80次新湿实验验证，不能宣称已证明某突变提高真实蛋白功能。

![ESM-2与单点突变扫描](visuals/fig7_protein_esm2_finetune_and_mutation_scan.svg)

## 第二阶段一键入口、验证与 AMD 同步

```bash
.venv/bin/python run_pipeline.py --phase 2
.venv/bin/python run_pipeline.py --phase 2 --reuse-results
.venv/bin/python verify_phase2.py
```

第一条真实顺序重跑BBBP（含远端注意力核）→开源力场→ESM-2；第二条只核对已保存产物并汇编报告。原权重、张量缓存和编译产物留本机，不随Git发布；真实小型CSV、日志、指标、split、哈希和源码随仓保存。AMD源码与轻量结果同步至`/home/lmxxf/work/ai-for-science-on-amd-9070xt`，远端仍只使用独立实验目录。

新增图5/6/7均640px白底中文SVG，禁Unicode上标ᵀ，CairoSVG转PNG逐图目检。`verify_phase2.py`从test预测重算指标，检查split无交叉、最佳轮次、真实归因差值、原始能量/受力数组、序列/突变、数据哈希与图表格式；不能把“JSON存在”当作实验验证。
'''
 return text
if __name__=='__main__':print(report_text())
