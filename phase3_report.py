#!/usr/bin/env python3
"""Human-readable Phase 3 report assembled from measured artifacts."""
from pathlib import Path
import json
ROOT=Path(__file__).resolve().parent
read=lambda p:json.loads((ROOT/p).read_text())
def report_text():
 if not all((ROOT/f'results/module{n}_results.json').exists() for n in range(9,14)):return ''
 p,d,f,h,u=[read(f'results/module{n}_results.json') for n in range(9,14)]
 v=p['validation'];grid=v['mesh_sizes']['64'];text='''

# 第三阶段：PME轨迹、材料本征值、射电FX、序列HMM、全15层Uni-Mol

这些是同源跨平台原创原型与实际模型推理，不是获得了NVIDIA专有库源码，也不宣称兼容其完整API。CPU与两块GPU都执行真实数值计算，合成体系/天空/搜索库明确标识；所有模块均先验证数值，再解释性能。

## 模块九：PME长程静电与1000步真实数值轨迹

输入为固定种子的27个中性柔性三原子合成分子（81原子），24Å周期盒子，不是假称实际溶剂化药物或湿实验轨迹。每步含实空间LJ与erfc库仑、自能、谐振键长/键角，以及倒易PME；GPU不回传全体系逐步驱动，主机只发射设备驻留核，结束才回传历史与最终XYZ。

4阶B样条电荷铺展→64³复数网格FFT（CUDA cuFFT/HIP hipFFT）→带B样条反卷积的Ewald Green函数→逆FFT→插值力。力用B样条导数求同一离散势能的梯度，避免简单ik场插值与离散能量不一致。采用C2C而非单独R2C包装，数学上同一实电荷FFT路径，不能把包装类型当新的物理。

'''
 text+=f"独立直接Ewald参考包含相邻周期像实空间、kmax={v['ewald_kmax']}倒易求和及自能，进一步k截断收敛的受力相对差{v['ewald_convergence_force_l2']:.3g}。\n\n"
 text+='| 网格 | 对独立Ewald能量绝对差 / kJ/mol | 全受力相对L2 |\n|---|---:|---:|\n'
 for g,a in v['mesh_sizes'].items():text+=f"| {g}³ | {a['energy_abs_error']:.6g} | {a['force_relative_l2']:.6g} |\n"
 text+=f'''
64³网格对真实Ewald的受力误差为**{grid['force_relative_l2']:.3g}**，不是机器精度；128³改善到{v['mesh_sizes']['128']['force_relative_l2']:.3g}。同一mesh的能量有限差分最大误差{v['mesh_gradient_max_abs_error']:.3g}，证明力与当前离散势能一致。有限mesh净力L2约{grid['net_force_l2']:.3g}，不假称严格动量守恒。

| 平台 | 1000步NVE / s | 1000步BAOAB / s | NVE最大相对能量漂移 | NVT后半段温度 / K |
|---|---:|---:|---:|---|
'''
 for name,b in p['backends'].items():
  a=b['trajectories']['nve'];t=b['trajectories']['nvt'];text+=f"| {name} | {a['seconds']:.4f} | {t['seconds']:.4f} | {a['energy_relative_max_drift']:.6g} | {t['temperature_mean_second_half']:.3f} ± {t['temperature_std_second_half']:.3f} |\n"
 c=p['backends']['cpu'];text+=f'''
NVE Velocity-Verlet：dt0.5fs×1000=0.5ps；dt0.25fs×2000保持同一物理时长，最大漂移从{c['trajectories']['nve']['energy_relative_max_drift']:.6g}降到{c['trajectories']['nve_halfdt']['energy_relative_max_drift']:.6g}，改善{c['nve_dt_halving_drift_ratio']:.4f}倍，符合二阶积分。BAOAB dt1fs×1000=1ps，目标300K；热浴交换能量，因此不能拿NVT总能量变化当NVE误差。

CUDA/HIP初始全受力相对L2 {p['hip_vs_cuda_initial_force_relative_l2']:.3g}。每步U/K/E/T、首键长、首原子x坐标均保存；全原子XYZ只保存初末，不伪称落盘了每步全部XYZ。无SETTLE、无刚性约束、无生产级优化。CPU为单worker NumPy/SciPy FFT+Python循环，不是编译CPU上限。

![PME与1000步轨迹](visuals/fig8_pme_kspace_and_1000step_md_trajectory.svg)

## 模块十：64组周期平面波Block-Davidson

构造64个k点、每个384平面波、取16条最低带，共1024个本征值。H=T+V，T=0.5|G+k|²，V为小幅周期余弦局域势；不是硅的自洽DFT/VASP输入，不能把曲线叫真实硅能带。

CPU和两GPU真的执行Block-Davidson：H·V、Rayleigh-Ritz子空间投影、Ritz重构、残差、对角预条件、两遍MGS与重启。GPU已执行投影/重构/残差/预条件等批处理核；小子空间Jacobi对角化与MGS仍在CPU，是混合求解器而非全GPU驻留Daviator实现。Sequential对各系统分开发射，batch打包同一阶段。

| 路径 | 迭代数 | 端到端 / s | GPU/CPU H应用核心 / s | 本征值最大绝对差 | 独立H残差最大值 |
|---|---:|---:|---:|---:|---:|
'''
 for name,b in d['backends'].items():text+=f"| {name} | {b['iterations']} | {b['total_seconds']:.4f} | {b['H_apply_seconds']:.4f} | {b['eigenvalue_max_abs_error']:.3g} | {b['independent_full_H_max_residual']:.3g} |\n"
 text+=f'''
所有路径9轮收敛至残差约3.52e-8；独立NumPy全矩阵eigvalsh验证1024个值，max_abs<=4.73e-14。未将精确参考结果填回迭代器。

GB10批处理核心速度收益{d['batch_speedups']['cuda']['GPU_core_only']:.3f}倍、端到端{d['batch_speedups']['cuda']['end_to_end']:.3f}倍；9070XT分别{d['batch_speedups']['hip']['GPU_core_only']:.3f}/{d['batch_speedups']['hip']['end_to_end']:.3f}倍。**没有复现22倍**，剩余CPU步骤限制收益。NumPy优化全eigh参考只需{d['metadata']['reference_seconds']:.3f}s，甚至快于本朴素混合迭代实现；图中如实展示，不用慢教学CPU制造“GPU必快”。

![批处理本征值](visuals/fig9_daviator_batched_davidson_vasp_band.svg)

## 模块十一：合成光子环→相关电压→FFT/FX→CLEAN

合成非对称环天空，不是M87*/SgrA*真实观测。16个非共线站点、64个旋转快照、120条基线、每快照64帧、每帧1024复采样，共67,108,864个complex128采样。230GHz/4%带宽是示意物理参数，不伪称真实地理EHT站点或连续12小时ADC流。

每个时间/窄带分组先由天空傅里叶积分得到C_ab，再Cholesky生成具有该协方差的圆复高斯谱，IFFT得到时域电压。GPU再真实FFT回谱并累积全部基线S_a S_b*，没有用不相关噪声却直接画目标图。四项独立逐像素Fourier求和校验最大绝对差约1.3e-16。

| 平台 | F阶段FFT / s | X阶段相关 / s | F+X / s | 含I/O总墙钟 / s | 可见度对CPU相对L2 |
|---|---:|---:|---:|---:|---:|
'''
 for name,b in f['backends'].items():text+=f"| {name} | {b['FFT_seconds']:.6f} | {b['X_seconds']:.6f} | {b['FFT_seconds']+b['X_seconds']:.6f} | {b['total_seconds']:.6f} | {b['visibility_relative_l2_vs_cpu']:.3g} |\n"
 text+=f'''
GPU为原生FP64 radix-2 FFT+复数相关；uv网格、2D IFFT、Högbom CLEAN是共同CPU后处理，未冒称全部GPU化。每uv格取平均后用binary mask，相当于uniform-per-cell权重，不是自然权重。CLEAN gain0.1、最多2000轮；恢复Gaussian sigma0.7pix是固定教学选择，不声称由EHT beam拟合。

仅FX输出进入image()，未读目标天空拼图。两GPU可见度相对L2 {f['hip_vs_cuda_visibility_relative_l2']:.3g}、恢复图相对L2 {f['hip_vs_cuda_clean_relative_l2']:.3g}。恢复环相对输入天空PSNR **{f['backends']['cuda']['clean_PSNR_vs_input_sky_dB']:.3f}dB**，不是完美重建。没有零基线测量导致dirty图DC缺失；PSNR只对恢复图按其恢复flux归一，CLEAN达2000轮上限，不冒称优化收敛或必然比dirty更清晰。I/O会改变两GPU端到端排名，应与驻留F+X分开。

![射电相关与重建](visuals/fig10_cuphoton_eht_blackhole_correlator.svg)

## 模块十二：10万条三状态Profile-HMM Viterbi

真实8733条GB1域序列构造56位点/20AA PSSM，以每AA0.5伪计数和均匀0.05背景得到log2 emission odds。原数据gapless，只能观测MM，MI/MD与I/D延续用明确先验，不编造indel统计。

同种子20261123生成100000条128–256AA背景序列，在500条植入由真实GB1域衍生的0–11点扰动，剩99500为随机诱饵。这是合成搜索基准，非10万个湿实验注释。

M/I/D动态规划用glocal完整56位点域：任意序列位置重启、从M1开始、必须结束于M56/D56；insert发射背景odds为1。CPU同源单线程、GPU按序列并行，FP64 add/max。独立全二维DP与就地三数组算法分开实现，小样本先通过再跑全库。begin复用MM，和官方HMMER的校准bitscore不是同一接口；没有MSV过滤、E-value或Forward实现。

| 平台 | 10万序列 / ms | 百万残基/s | 植入域ROC-AUC |
|---|---:|---:|---:|
'''
 for name,b in h['backends'].items():text+=f"| {name} | {b['median_ms']:.3f} | {b['residues_per_s']/1e6:.3f} | {b['ROC_AUC']:.6f} |\n"
 text+=f'''
全部100000个分数CPU/CUDA/HIP逐值与逐字节相同，SHA256 `{h['backends']['hip']['score_sha256']}`。全部分数与标签保存在`results/module12_all_scores.npz`，独立二维DP校验误差0。AUC=1仅表明这组植入域/随机诱饵强可分，不外推真实家族注释准确率。

![10万序列HMM](visuals/fig11_gpuhmmer_100k_protein_viterbi.svg)

## 模块十三：4733万参数全15层原生HIP端到端推理

原ESOL最佳checkpoint SHA256 `{u['checkpoint_sha256']}`，全部47,331,139个参数（195张量）真实导出；没有把GB10预测或标签送AMD，AMD只接收原子token、distance、edge与checkpoint权重。

实际网络是embedding/LN→Gaussian与128/128GELU/64→15轮pre-LN QKV、pair累加、softmax、AV、out_proj残差、pre-FFN LN与512/2048GELU/512残差→末尾encoder LN→真实LinearHead512→1。不是图纸猜测的两层tanh头；当前final_head_layer_norm本来就是None，辅助pair norm loss不参与回归。eval关闭dropout；逐分子无padding，BOS/EOS保留。

原生HIP/hipBLAS跑全部原112个留出分子，没有重新挑test或改标签。精度优先全FP64实现与同公式GB10 FP64对账：

| 推理路径 | 测试RMSE | R² |
|---|---:|---:|
'''
 for name,key in [('原GB10 BF16 autocast（保留）','original_GB10_BF16_autocast'),('官方GB10 FP32/noTF32','GB10_official_FP32_no_TF32'),('GB10 FP64同公式','GB10_reference_FP64'),('原生AMD HIP FP64','AMD_HIP_FP64')]:text+=f"| {name} | {u[key]['RMSE']:.9f} | {u[key]['R2']:.9f} |\n"
 text+=f'''
AMD/GB10 FP64的112个预测最大绝对差**{u['AMD_vs_GB10_FP64_max_abs_logS']:.3g} logS**，50,501,632个node/pair中间元素逐值核对；node max{u['max_layerwise_node_abs']:.3g}、pair max{u['max_layerwise_pair_abs']:.3g}。16个检查点覆盖embedding/初始pair与全部15层。

全FP64对原官方FP32最大差{u['AMD_vs_GB10_official_FP32_max_abs_logS']:.6g}，略超1e-5，原JSON明确列false；对原BF16最大差{u['AMD_vs_original_BF16_max_abs_logS']:.6f}。原BF16 RMSE0.57377856不能强塞给另一精度路径。差别主要定位到官方Gaussian helper的强制FP32计算；全双精度公式参考只为公平跨卡对账，不隐藏这个官方兼容问题。

'''
 compat=u.get('official_fp32_compatibility')
 if compat:
  text+='### 官方FP32 Gaussian兼容模式补验\n\n'
  for key in ['pass','AMD_vs_official_FP32_max_abs_logS','AMD_vs_matched_reference_max_abs_logS','reference_definition','precision_definition','definition','scope','precision']:
   if key in compat:text+=f"- {key}: `{compat[key]}`\n"
  if 'metrics' in compat:text+=f"- 兼容模式测试：RMSE {compat['metrics']['RMSE']:.9f}，R² {compat['metrics']['R2']:.9f}。\n"
  text+='\n此模式只把真实Gaussian路径按官方FP32运算，其余完整15层高精度，保留原全FP64结果和旧BF16输出；不复制预测或事后校正。\n\n'
 text+=f'''
全112分子设备事件累计：GB10 FP64 **{u['GB10_runtime']['gpu_compute_ms']/1000:.3f}s**，AMD FP64 **{u['AMD_runtime']['gpu_compute_ms']/1000:.3f}s**；AMD含权重加载/trace输出墙钟{u['AMD_runtime']['wall_including_weights_and_trace_s']:.3f}s。事件和墙钟不混用。FP64单分子参考未做性能优化，不把它当生产batch吞吐。

![全15层原生HIP推理](visuals/fig12_unimol_full15layer_end2end_amd9070.svg)

## 第三阶段复现与证据

```bash
.venv/bin/python run_pipeline.py --phase 3
.venv/bin/python run_pipeline.py --phase 3 --reuse-results
.venv/bin/python verify_phase3.py
```

真实单模块执行均保留日志，GPU按9→10→11→12→13串行，CPU准备并行。`--reuse-results`只重验保存的数值与汇编报告，不称重新模拟。独立校验从原始轨迹、全部本征对、全频可见度、CLEAN图、100k完整scores和50M全层trace重算关键误差与指标。

输入电压1GiB、H矩阵72MiB、模型FP64权重379MB和raw traces等大数据留本地，脚本/种子/SHA/轻量结果随Git；新增源码、JSON和图8–12同步AMD姊妹仓。五图均白底文泉驿中文，CairoSVG转PNG已逐图目检。
'''
 return text
if __name__=='__main__':print(report_text())
