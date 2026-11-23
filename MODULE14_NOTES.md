# 模块十四：真实ESOL × fresh ETKDG × 原创批量简化力场

交付：`exp12_open_nvmolkit_batched_conformer_hip.py`、`amd_hip_port/nvmolkit_batched_conformer/`原创MIT源码/README/许可/Windows编译日志、`results/module14_results.json`，图13`visuals/fig13_open_nvmolkit_10000_conformers_amd_vs_nvidia.svg`。图已转PNG目检：640宽、白底、文泉驿，无缺字/重叠。

## 新起点，真实拓扑，明确参数

ESOL清洗表前1,000个分子，每个重新ETKDGv3生成10个构象，全部成功、无需retry，最大55重原子。**没有读取模块一优化后的坐标缓存，没有调用MMFFOptimize/UFFOptimize。** 每分子seed20261124+row，另加明确的sigma0.06Å确定性随机位置扰动作为受控松弛测试。总133,450个重原子（所有构象），400,350个有效XYZ标量。

8进程ETKDG+jitter+每分子拓扑/参数提取worker阶段真实wall3.533790s；嵌入+jitter工人累计26.480680s。这不含后续扁平化打包/磁盘序列化，不是“纯ETKDG端到端”或单核时间，scope已写元数据。

真实图拓扑项：13,754条bond、18,808条angle、23,549条torsion、73,697条nonbonded（每分子共享拓扑，10构象复用）。非键图距离>=3根键，包含1–4。显式H用于ETKDG与Gasteiger分配，H电荷折到对应重原子；无NaN电荷回退。优化只动重原子，原型不是完整氢坐标力场。

平衡键长和角仅从MMFF参数API读取（支持时），不代表使用了MMFF94能量模型。原创模型具体为：

- bond `.5*400*(r-r0)^2`；
- angle `.5*40*(cos(theta)-cos(theta0))^2`；
- torsion显式的二/三周期余弦多项式，k0.2，normal convention和phase见README，不称真实MMFF扭转剖面；
- nonbonded full LJ12–6，最终epsilon0.5kJ/mol、sigma=.85*(元素Rvdw之和)，加库仑1389.354576qiqj/r；使用r=sqrt(distance²+.01)平滑极近接，不屏蔽、不截断、不对1–4特殊缩放。单位Å、kJ/mol。

这是一套明确且可微的简化原型，**没有参数化学标定或独立构象真实性验证**。不得称真实nvMolKit或MMFF94等价替代。

## 真正踩坑：能量下降≠构象质量通过

第一轮epsilon0.05过弱，虽然三平台逐位一致、平均能量下降、键长更接近目标，但声明的近碰撞条件（图距>=3，r<.8sigma）从47增至1138。这个失败没有被藏掉或通过改阈值“修好”。

保留失败完整三后端坐标、能量历史、时间：`results/module14_failed_weak_lj_{cpu,cuda,hip}/`，失败汇总`results/module14_failed_weak_lj_results.json`，旧参数`results/module14_failed_weak_lj_data/terms.bin`。

**仅把LJ epsilon改为0.5，原始fresh ETKDG坐标一字节未改，碰撞阈值未变。** 修正后CPU先验通过，等待模块十五完全跑完GPU才按顺序复测两卡，最终碰撞47→0。此参数调整是对这个诊断失败的原型修正，不是假装经过独立化学数据集校准。

输入坐标before/after SHA256均为：

`3716b9a08c14a52e55a6633f1c62533d67950af5c963be29c16ba22cf5dd09ee`

旧terms hash `910a74cab95fbc3ccb9125c8bd07a60e4d0a3571ee0f8fb2084d50efbb481990`；新terms `0ed00b2a3f1e795eb3d5a89641ffab7501054c49a452809030f6fab86075a22e`。最终JSON嵌`failure_case`，包括输入哈希、原失败47→1138和修正说明。默认prepare已同步epsilon0.5。

## 同源、shared驻留和梯度校验

64threads一Block负责一构象，最多64重原子（实测55）。坐标、候选坐标、梯度、逐原子能量驻留shared，200步回溯下降，无每步CPU干预。**参数/邻接表仍从global读取，能量历史还会写global**，不能说全程零global访问。固定incident-term顺序逐原子gather、thread0固定原子顺序相加，不用浮点atomic。CPU执行相同次序、同源C++数学；GPU仅做并行分工，不换能量模型。

初始step.003，全局最大原子梯度超过10时缩放；最多16次回溯，接受不增能量后step×1.15（上限.02），失败step×.5。200次尝试不等于求得全局极小或完全驻点，最终平均最大原子梯度仍1.1348，已如实报告。

对独立NumPy能量表达式，以h2e-4Å五点差分验证三个真实分子18方向；C++解析梯度最大绝对误差 **1.95901e-10 <1e-8**。epsilon改后已重做同18方向；未拿旧梯度结果充新模型。

## 最终真实结果

| 后端 | 10,000构象×200步秒数 | H2D(s) | D2H(s) |
|---|---:|---:|---:|
| CPU单线程同源C++ | 10.522151 | 0 | 0 |
| NVIDIA GB10 CUDA | 3.229019 | 0.000535565 | 0.000531837 |
| AMD RX9070XT HIP | 3.853299 | 0.0120884 | 0.0028555 |

单次完整优化wall，不是稳态重复中位数；初始化/分配/输入和结果传输在优化计时外，另列JSON，历史诊断写入计入。相对同源CPU优化速度约GB103.26倍、AMD2.73倍；不是等价MMFF94加速倍数。

三平台的**全部400,350个有效坐标、10,000个最终能量、完整10,000×201能量历史逐位完全一致**。CUDA/HIP最终坐标相对L2=0、maxabs=0，能量L2=0。无降精度、无fastmath、禁FMA contraction，输出padding亦零初始化。

- 平均定义势能：17.329141 → -27.427460kJ/mol；
- 10,000/10,000不增，9,990严格下降；剩10个是ESOL row934单重原子`C`（隐含H的甲烷重原子表达），没有这些重原子力场项，因此零能量保持不变；
- 平均|键长偏差|：0.072064 → 0.011286Å；最终95分位0.025527Å，最大0.100807Å（初始最大0.469016Å）；
- 近碰撞：47 → 0，完全相同阈值；
- 全部坐标/能量有限，所有2,000,000次迭代能量增量最大0；
- 完整逐构象/逐步原始数据存binary，所有统计可重算。

## 旧RDKit66.608秒的正确口径

模块一`conformer()`实际做ETKDG+真正RDKit MMFF94 200步，单核66.607885s、8进程9.986772s；它不是“200步弹簧松弛”。本原型只有简化重原子力场，不能拿66.608除以3.229去声称等价MMFF/nvMolKit加速20倍。图13把旧数作为**任务不同的文字参考**，公平柱图只画本原型同源CPU/CUDA/HIP。

## 复现与纪律

`exp12... --stage prepare/run/gpu/summarize`：新数据/全优化/复用CPU基线只跑两GPU/只读汇总出图。重进程均guard，远端仅独立`C:\AI4S-HIP-Lab\nvmolkit_batched_conformer`，现有Strata/ROCm只读。源码附MIT，库依赖许可另计。修正版结果生成后同步AMD独立仓，未修改主报告/流水线。
