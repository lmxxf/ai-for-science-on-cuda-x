# 模块八：Biopython + ESM-2 蛋白质实验记录

## 数据来源与任务

使用FLIP官方仓库的GB1 `sampled.csv`，来源归档为：https://raw.githubusercontent.com/J-SNACKKB/FLIP/main/splits/gb1/splits.zip 。实验原始研究是 Wu等，eLife 2016，https://elifesciences.org/articles/16965 ，测量的是GB1四位点组合突变的标准化IgG结合适应度，不是荧光、热稳定性或绝对结合自由能。

原ZIP位于`data/gb1_splits.zip`，解压原CSV位于`data/gb1/splits/sampled.csv`。CSV SHA256 `19bcb946fb9f8fe5ac1d5ed0aa85d4a85848b44744560d9684cc278935f3bd16`；ZIP SHA256 `00a3c3d5be56ce35adfbec838db8ff549be369525e151a6978133da34b9fd86f`。FLIP说明原始数据CC-BY4.0，仓库修改版AFL-3.0，说明文件保存在`data/gb1/README.md`。

所有8733行均为265个标准氨基酸，无非法AA、无重复序列。逐列扫描全8733条序列得到唯一变异位置为39/40/41/54。前56AA是完整GB1域，其后209AA表达融合尾巴对全库恒定，因此对全部选手统一裁剪到56AA。裁剪前后仍为8733唯一序列。并非按目标值挑选可预测样本。

使用随机种子20261124，不放回均匀抽取3000条，去重与AA校验在划分之前；固定新随机2400训练/300验证/300测试。该子集和划分用于一晚跑通的教学实战，**不是FLIP官方排行榜划分**，不是跨蛋白泛化。完整样本索引、split证据与CSV哈希在`results/module8_data.json`；实际同考卷为`data/gb1_experiment.csv`。

## 生化特征和基线

Biopython1.88 `ProteinAnalysis`真实计算每条56AA域的MW、pI、芳香性、不稳定指数与GRAVY。原始特征表保存`results/module8_biopython_descriptors.csv`。特征总425维：20种AA组成频率，400种相邻二肽频率，5个生化指标。

- 常数基线：训练集标签均值。
- 随机森林：400棵树，`min_samples_leaf=2,max_features=.8,n_jobs=4`，固定种子。
- 岭回归：StandardScaler只拟合训练特征，alpha从0.01/0.1/1/10/100按验证RMSE选。
- 所有选手共享同一训练/验证/测试索引。测试集不用于标签缩放、训练轮次或超参选择。常数预测的Spearman无定义，保存null，不捏造为0。

## ESM-2官方权重与架构

模型`facebook/esm2_t6_8M_UR50D`，已下载官方safetensors与配置到`weights/esm2_t6_8M_UR50D`。本地实际打印：6层、隐藏320、20个注意力头、FFN1280、`position_embedding_type=rotary`，运行对象存在`esm.rotary_embeddings`。新增回归头后总参数7,512,122，全部可训练；不是冻结编码器后只训练线性探针。预训练语言模型头未使用、新建回归`classifier`缺失权重是正常任务适配，明确记录，不假装该回归头预训练过。

词表33个token完整列表：`<cls>, <pad>, <eos>, <unk>, L,A,G,V,S,E,R,T,I,D,P,K,Q,N,F,Y,M,H,W,C, X,B,U,Z,O, ., -, <null_1>, <mask>`。其中20种是标准AA，X/B/U/Z/O是非标准/模糊AA符号，剩余才是标点/任务特殊token，不能说“20AA之外全是特殊符号”。真实架构及权重SHA256在`results/module8_esm2_architecture.json`。

训练默认12轮，AdamW学习率0.0002、weight_decay0.01、batch64、BF16自动混合精度、梯度裁剪1、workers0、CPU线程4。训练目标仅使用训练标签均值/标准差做z标准化，预测再还原原始实验fitness量纲。每轮仅验证RMSE选checkpoint，最终选定后对测试集预测一次，保存RMSE/Spearman/R²。逐轮记录在`results/module8_epochs.jsonl`，最终JSON `results/module8_results.json`，测试逐样本预测`results/module8_test_predictions.csv`。

## 突变扫描的边界

固定野生型56AA背景，对39/40/41/54四个位点各换20种AA，得到80条（包括4个原型对角项）模型预测。热图是`预测突变fitness−预测WT fitness`，不是该80条的新湿实验验证，不是残基注意力，也不是因果归因。行位点/列AA、完整序列/预测/差值保存在`results/module8_mutation_scan.csv`。外推新蛋白、蛋白折叠、临床效果均不在本实验验证范围内。预训练是否接触GB1同源序列未审计。

## 执行与安全

按用户顺序模块六→模块七→模块八；本模块数据/权重/生化CPU准备先做，GPU训练等模块七完成后启动。运行命令：

```bash
.venv/bin/python guard_process.py --audit logs/module8_train_guard.json -- .venv/bin/python exp6_protein_esm2_finetune.py
```

250ms统一内存看护低于20GiB熔断。脚本支持`--stage prepare|architecture|train|figures|all`，默认all会真实准备与训练，不是占位模板。SVG目标`visuals/fig7_protein_esm2_finetune_and_mutation_scan.svg`，640宽白底中文文泉驿，无Unicode上标ᵀ，并转PNG检查。

## 本次真机结果（2026-11-23 23:48 +08:00）

默认`all`完整执行成功，guard进程退出0，总墙钟22.055秒，最低可用统一内存108.486GiB。12轮GPU微调（含逐轮验证与最佳权重写盘）15.016秒；456个优化步骤，每步batch64，末批32。验证集RMSE最低为第11轮0.456265，按其选checkpoint后测试300条：

| 同考卷模型 | RMSE ↓ | Spearman ↑ | R² ↑ |
|---|---:|---:|---:|
| 训练均值常数 | 1.243687 | 无定义 | -0.000351 |
| AAC20+二肽400+生化5 随机森林 | 0.870472 | 0.798806 | 0.509950 |
| 相同425维特征 岭回归 | 0.931356 | 0.742779 | 0.439001 |
| 官方ESM2全参数微调 | 0.538136 | 0.898290 | 0.812710 |

这些指标仅支持本次随机同蛋白变异组合留出结果，不能延伸为ESM2必胜任何蛋白数据、胜更强位点感知基线，或与GB1官方排行榜直接比。

真实token化另核对：56AA加CLS/EOS后张量长度58，未误把整条蛋白吞成一个词元。微调后第一层`esm.encoder.layer.0.attention.self.query.weight`与官方预训练权重L2差1.657597，证明编码器确实更新而非只训回归头。3000条序列唯一且split互斥；测试预测全为有限值；扫描四个原型AA格Δfitness均为0。图7已用PNG目检：中文正常、无覆盖越界、所有单元格和色阶说明清晰。
