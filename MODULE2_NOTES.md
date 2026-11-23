# 模块二：官方 Uni-Mol 真机全参数微调

执行脚本：`exp2_unimol_finetune.py`；环境 `.venv`；输入 `results/esol_conformers.pkl` 来自模块一的真实 ESOL 1128 行 ETKDGv3/MMFF94 重原子构象。

## 安装、来源与兼容

- 官方源码 `https://github.com/deepmodeling/Uni-Mol`，本地 `vendor/Uni-Mol`，commit `90f52c41299a1a582da0f9765e9f87aa21faa16a`。
- 使用 `pip install --no-deps -e vendor/Uni-Mol/unimol_tools`，避免旧版包对 numpy/pandas 的版本范围迫使 Python 3.12/aarch64 构建过时库。额外安装 addict，复用现有 torch、sklearn 等；numpy 在任务独立 venv 中解析为2.3.5，基础venv没有修改。
- 权重官方 Hugging Face `https://huggingface.co/dptech/Uni-Mol-Models/resolve/main/mol_pre_no_h_220816.pt`，词表同目录 `mol.dict.txt`。成功下载并真实加载181 MiB预训练权重；sha256写入结果JSON。
- 使用 Uni-Mol **v1 no_h**，不是 Uni-Mol2。真实打印参数 **47,331,139**，15层、512隐藏维、2048 FFN、64注意力头、128 Gaussian 核。图纸的84M不是本次所加载模型。

## 真实结构与实验协议

- 原子种类是token；按原子对类型对距离做可学习仿射，再用128个可学习高斯核展开，经非线性投影到64个注意力头。
- 距离成对表示初始化注意力偏置；官方 `TransformerEncoderWithPair` 会在各层传递/更新成对表示。因此“唯一差别只是每层静态加一个距离bias”过于简化。
- 采用整个ESOL，按清洗后的canonical母核去重再作固定seed42的80/10/10随机train/val/test划分。去重后数量及具体source_index存于 `results/module2_split.json`，避免相同分子跨集。
- 比较训练均值Dummy、2048位半径2 Morgan指纹+400树RandomForest、官方UniMol全参数微调。三者测试集完全相同。
- 20轮、batch32、AdamW lr1e-4、cosine退火、BF16 autocast，FP32权重与优化器；num_workers=0；梯度裁剪1.0。目标标准化仅取训练集均值/标准差。
- 验证集RMSE选最佳轮次；测试集在全部训练完成后只评估一次。不以测试集分数改超参或重跑。逐轮训练归一化MSE和验证RMSE/MAE/R²在 `results/module2_epochs.jsonl`，最终结果 `results/module2_results.json`。
- 包裹 `guard_process.py` 做0.25秒统一内存监控，低于20GiB可用内存终止。

## 复现命令

```bash
.venv/bin/python guard_process.py --audit logs/module2_memory.json -- .venv/bin/python exp2_unimol_finetune.py
```

可视化图 `visuals/fig3_unimol_3d_finetune_vs_baselines.svg` 与PNG自检预览。重绘不重训：`.venv/bin/python exp2_unimol_finetune.py --render-only`。

## 解释边界

- 本次是单随机划分、单构象、单次seed的完整真机演示，不是系统基准，不外推成“3D一定吊打2D”或“临床药效预测”。
- ESOL预训练分子重叠情况未审计。随机划分一般比scaffold划分容易；论文横评应增加scaffold与多seed。
- 不预设CPU构象生成比GPU训练更慢：真实全1128单构象CPU为8.870秒，20轮GPU训练时间按JSON实录。
- 没有把通用LLM当化学预测对手；Dummy只是零知识常数基线。分子距离输入也不是实验测得的真实构象，而是RDKit构象算法给出的候选几何。

## 本次最终实测（2026-11-23 22:39）

1128条原始构象按canonical母核去重后1117分子；train893 / validation112 / test112。

| 选手 | 测试RMSE | 测试MAE | 测试R² |
|---|---:|---:|---:|
| 训练集均值 | 2.175708 | 1.664649 | -0.000286 |
| Morgan2048 + RF400树 | 1.144189 | 0.820279 | 0.723358 |
| 官方Uni-Mol全参数微调 | **0.573779** | **0.411979** | **0.930432** |

验证集最优第13轮；20轮训练+验证+最佳模型保存合计37.183264秒，单轮1.57–3.29秒；全1128单核构象生成8.870285秒。本机GB10上此规模GPU训练反而更耗时，CPU不是主要耗时。峰值torch分配2.234GiB；内存看护全程最小可用107.197GiB、未触发阈值。

本次并不能隔离“3D信息”本身带来的收益：Uni-Mol还拥有预训练与不同模型结构/规模；要证明3D贡献，需要固定架构做距离打乱/2D距离等消融。单纯比较三种预测器展示工作流，而非机制归因。
