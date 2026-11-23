# 模块六：BBBP 真机微调、逐原子归因、同源 CUDA/HIP 注意力核

## 数据与固定考卷

MoleculeNet BBBP原始2050行，模块一RDKit合法清洗2039行。按清洗后canonical SMILES母体分组：11个母体有互相冲突的0/1标签，**整组22行全部排除**；同标签重复再去62行，得到1955个唯一母体。ETKDGv3成功1952个，失败3个排除。MMFF94在200迭代上限内收敛1058个；其余894个保留最后坐标，不能写成“1952个全部力场收敛”。

按标签分层固定seed42，train1561 / validation195 / test196，同母体不会跨集。`results/module6_split.json`保留所有source_index与SMILES及冲突母体列表。去重按准确canonical结构，并非Bemis-Murcko scaffold划分，近似骨架可能跨集。

8进程全体构象CPU12.447064秒；真实缓存 `results/bbbp_conformers.pkl`，计时与MMFF状态 `results/bbbp_conformer_timing.json`。

## 同考卷真实结果

| 选手 | 测试ROC-AUC | 测试PR-AUC（Average Precision） | Accuracy |
|---|---:|---:|---:|
| 训练集先验常数 | 0.500000 | 0.765306 | 0.765306 |
| Morgan2048半径2 + RF400树 | 0.932609 | **0.975935** | **0.903061** |
| 官方UniMol v1全参数微调 | **0.933333** | 0.975010 | 0.892857 |

结果基本打平，不能写成UniMol全面胜出。PR-AUC使用sklearn `average_precision_score`，不是梯形积分面积。

官方47,331,139参数，15层/512宽/64头/head_dim8/FFN2048，128Gaussian。BCEWithLogitsLoss，20轮，AdamW lr1e-4 cosine至1e-5，batch32，BF16 autocast+FP32权重，num_workers0，全参数更新。仅validation ROC-AUC选第4轮，最终test对每个模型各评估一次；未用test改参重跑。训练+每轮验证+最佳模型保存112.259057秒，峰值CUDA分配等信息在JSON；逐轮 `results/module6_epochs.jsonl`。

训练和后续核实验分开串行执行，均由guard_process包裹。训练全程最低MemAvailable99.672GiB，核阶段109.251GiB，未触发20GiB阈值。

## 逐重原子遮挡：真实正负贡献

保留RDKit原构象，逐次删除一个重原子token和对应距离矩阵行列，不重新生成构象；计算 `Δlogit_i = 完整logit - 删除该原子后的logit`。颜色来自实际正负，**没有先验规定“碳一定绿/氧一定红”**。这是模型依赖探针，不是化学可行反应或因果药理：删token后的结构可能根本不是合法分子。

- 苯海拉明：预测穿脑概率0.957338；重原子Δlogit范围[-1.406480,+1.778018]。
- 非索非那定：预测穿脑概率0.038494；Δlogit范围[-2.151298,+0.254791]。
- 命名药物是否出现在训练/验证/测试中，在 `attribution.*.dataset_split_membership`明确记录，不能当独立外推验证。
- 染色以各分子自己的最大绝对Δlogit归一化，两图颜色深浅不直接可比。完整每个原子结果在JSON。

## 官方真实权重：同源 CUDA/HIP 核

源码 `amd_hip_port/unimol_pairbias/pairbias.cpp` 同一份文件分别由nvcc sm_121与Windows HIP clang gfx1201编译。不是第三方替代模型；导出的是本次BBBP最佳官方UniMol模型真实参数：

1. 原子对 `mul/bias` 仿射距离；可训练128个mu/std（`abs(std)+1e-5`）；按官方常数3.14159计算Gaussian。
2. **真实非线性投影为128→128→GELU→64**，含两层bias，不能简写成直接128→64。
3. 第一层真实1536×512 QKV权重及偏置，head_dim8，scale1/sqrt8。
4. padding key mask、softmax、attention×V，输出512维context。BOS/EOS是保留的正常特殊token，不应误mask。
5. 输入先由官方embedding LayerNorm、pad置零、第一层pre-attention LayerNorm计算，**归一化后的输入从PyTorch导出**。手写GPU核没有重复实现LayerNorm。

输入8个真实药物记录对应的RDKit候选构象，padding到39原子token，含特殊token。每个阶段输出都与**官方PyTorch CPU FP32**参考及跨GPU逐元素对账，非只核对最终标量。

| 核对张量 | CUDA↔HIP最大绝对差 |
|---|---:|
| Gaussian128 | 1.525879e-5 |
| 成对偏置64头 | 7.629395e-6 |
| QKV | 0 |
| masked softmax attention | 1.519918e-6 |
| 最终context | 2.503395e-6 |

与官方PyTorch CPU参考所有阶段最大绝对差1.525879e-5；context差CUDA2.801418e-6、HIP3.933907e-6。全部PASS（全张量maxabs<1e-3，具体真实误差见JSON）。

- GB10 CUDA：**5.238368 ms**。
- AMD RX9070XT HIP：**12.101480 ms**。
- 均为输入和权重常驻显存、先warmup一次、重复7次GPU event中位数，五个自写forward kernels总耗时，排除H2D/D2H、编译、加载、LayerNorm。

**这是未经优化的教育参考核，无法代表厂商库/整卡性能。** 这里串行loop GEMV、没有共享内存分块或BLAS，且不同硬件对其映射不同。没有移植15层全模型、output projection、FFN、训练反向；绝不能写“UniMol整模型已在AMD跑通”。

## 路径与复现

```bash
.venv/bin/python guard_process.py --audit logs/module6_memory.json -- .venv/bin/python exp4_bbbp_unimol_and_amd.py
```

默认all依次构象、训练、归因、编译/执行CUDA与远端HIP、对账、画图。`--stage pairbias`只重做核与图，不重新训练；`--stage render`只重画；`--stage train`只训练归因。主入口需要可连接 `ssh amd9070` 与当前Windows HIP SDK路径。

远端独立工作区 `C:/AI4S-HIP-Lab/unimol_pairbias`。输出：`results/module6_results.json`、`results/module6_pairbias_results.json`、`results/module6_split.json`、`results/module6_epochs.jsonl`、`visuals/fig5_bbbp_unimol_roc_and_atom_attribution.svg`。PNG已渲染目检：纯白底、640宽、WenQuanYi中文清晰、无Unicode上标ᵀ。
