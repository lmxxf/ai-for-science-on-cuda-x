# 模块十三：全15层Uni-Mol原生HIP/hipBLAS端到端推理

## 实现与原模型对账

源模型是模块二真实ESOL微调最佳检查点 `results/unimol_esol_best.pt`，全部 **47,331,139** 参数导出为FP64二进制（原参数仍是原来的FP32值，只有运算精度提升），无冻结、蒸馏、裁层。原112个测试分子的source_index与顺序来自 `results/module2_split.json`，没有重新挑测试集。

图纸的“tanh两层回归头”与当前检查点不符，实物是 `LinearHead`，只有`classification_head.out_proj`的512→1线性层；没有tanh。按实际代码执行，不按假设造一层头。

实际前向链：

1. atom token查表512维，embedding LayerNorm（epsilon1e-5）。
2. 真实原子对mul/bias距离仿射；128个可训练Gaussian均值/标准差，`abs(std)+1e-5`，官方常数3.14159。
3. 官方gbf_proj：128→128→精确GELU→64，投影为64头初始pair bias。
4. 全15层，每层先pre-attention LayerNorm；512→1536真实QKV权重含bias，64头、head_dim8、scale1/sqrt8。将QK分数**累加进上一层pair表示**，softmax后乘V，真实out_proj再加残差。
5. pre-FFN LayerNorm；512→2048→精确GELU→512，再加残差。
6. encoder末尾final LayerNorm，取CLS向量，经实际512→1线性回归头输出标准化logS，再用原训练集均值/标准差还原。

推理eval状态禁用dropout；逐个分子运行所以没有padding，BOS/EOS正常保留。所有原子全量输入，不裁剪token。模型用于辅助训练的norm_loss诊断不影响回归前向；final_head_layer_norm在本检查点配置中为None。

原生GPU实现 `amd_hip_port/unimol_full15_hip/unimol_full.cpp`：每个线性层用 `hipblasDgemm`，LayerNorm/Gaussian/GELU/softmax/pair更新等用原生HIP C++自写核。AMD端不调用Python/PyTorch、不读取任何参考预测或标签。远端只接收真实模型权重和atom/edge/distance输入。

## 精度协议

模块二旧结果使用BF16 autocast，旧预测原样保留。不能要求同一个权重的FP32/FP64推理强行重现BF16截断结果；本次独立记录：

- 原始GB10 BF16预测与原始RMSE/R²。
- GB10官方模型FP32、禁用TF32、无autocast的独立预测。
- GB10 FP64同公式参考，及原生HIP FP64逐点一致性。

官方Gaussian helper会显式 `.float()` 中间均值/标准差，本次FP64参考将这两处强制FP32转换移除，保持公式、参数和架构相同，仅使计算精度一致。所有encoder层仍调用官方PyTorch实现，不另造模型。

每个测试分子在初始embedding/Gaussian以及15层后，分别导出完整残差流x和完整64头pair表示。对全部112分子所有层张量比较最大绝对差，不仅比较RMSE或单一示例。最终预测目标max_abs_diff<1e-5；与旧BF16差异另外报告。

## 执行与调试

- 2026-11-24 CPU导出362MiB全参数FP64权重，全部112测试输入。manifest和checkpoint SHA256在 `results/module13_manifest.json`。
- Windows HIP SDK：`C:/Strata-Lab/Strata/.rocm-win/Lib/site-packages/_rocm_sdk_devel`，gfx1201，链接amdhip64+hipblas。
- 初次编译CK宏的临时变量`e`与edge数组同名，clang报“auto variable cannot appear in own initializer”；改为独立`hip_status_`/`blas_status_`后编译成功。没有删算子绕过错误。
- 先完成源码/CPU导出/编译，按模块9→10→11→12→13顺序等待放行，不与其他GPU基准混跑。

## 复现命令

```bash
.venv/bin/python guard_process.py --audit logs/module13_memory.json -- .venv/bin/python exp11_unimol_full15layer_amd9070.py
```

默认all：prepare（导出与编译）→execute（GB10参考、远端HIP全网、对账、画图）。`--stage prepare`不进行GPU推理；`--stage execute`使用已编译产物执行；`--stage compare`复算结果；`--stage render`只重画。

GPUevent累计时间排除权重加载、H2D/D2H和逐层trace拷贝，另记录包含权重加载与trace的端到端wall time，不混用。单分子/FP64是精度优先参考实现，未做批量化/融合优化；不能据此声称整卡性能排名。

## 最终真机实测（2026-11-24 00:17）

全部112个原留出药物，原生HIP完整15层预测通过：

| 数值路径 | 测试RMSE | 测试R² |
|---|---:|---:|
| 原GB10 BF16 autocast（原预测保留） | 0.573778549480 | 0.930431657285 |
| GB10官方模型FP32，无TF32 | 0.573633378384 | 0.930466855655 |
| GB10 FP64同公式参考 | 0.573633555896 | 0.930466812621 |
| AMD9070XT原生HIP/hipBLAS FP64 | **0.573633555896** | **0.930466812621** |

- **AMD FP64与GB10 FP64全112个logS预测最大差7.105427e-15**，远小于1e-5目标。
- 与未经改公式的官方GB10 FP32最大差1.918932e-5；这一条**略超过1e-5**，没有隐藏或强行舍入成PASS；1e-5通过项是精度对齐的FP64↔FP64。
- 与原始BF16最大预测差0.03684296 logS，RMSE只相差0.000145。这正是混合精度路径不同导致的数值变化，不是更换标签/测试集/权重。原模块二报告保存的float32指标还会与这里统一float64重算末尾约1e-8不同，`original_reported_metrics_unchanged`保留原字段。
- 全112个分子的embedding/初始pair及15个encoder层后node/pair张量全部逐值对账。所有层node最大绝对差9.086065e-13；pair最大差7.704948e-13。

设备计算event累计：GB10 FP64 **1.756469秒**，AMD9070XT FP64 **2.291475秒**。GB10参考wall含trace2.088727秒；AMD完整进程wall含权重加载/逐层trace **8.169713秒**。前者是PyTorch官方各层、后者是原生HIP/hipBLAS；该单分子FP64精度实现不作公平优化性能排名。

全权重原始显存数据378,649,112字节（361.108MiB）。AMD运行时free-memory差估约434,831,360字节，属于可用显存查询的估计差值，含driver/hipBLAS影响，不等同精确分配统计；GB10 torch peak allocated439,619,072字节。统一内存看护全程最低可用108.622GiB，未触发熔断。

图12已渲染PNG并目检：白底640宽、中文清晰、无文字重叠。112点预测重合图、逐层误差曲线与旧BF16/新FP64分账均来自真实JSON。

## 官方FP32兼容项：窄范围修复也已通过（2026-11-24 00:21）

上面的全FP64对账是第一份证据，保持全部原输出不动。继续定位官方FP32的1.9189e-5差异：在GB10只把Gaussian完整算术（原子对仿射、std绝对值与1e-5、标准化、平方、指数、归一化）保留官方FP32，其余15层仍FP64，112个预测对官方全FP32最大差降至1.358384e-6。主要差异因此来自原库在Gaussian路径上的强制FP32，而不是网络结构遗漏。

原生HIP增加可选参数`gaussian_fp32`，只改变这一小段计算精度，保留完全相同权重与其余15层双精度实现。重新真实跑全部112分子，独立输出 `out_hip_mixed/`，不覆盖原 `out_hip/`：

- **AMD mixed vs 官方GB10全FP32最大logS差2.268455e-6 < 1e-5，原图纸这一项现已满足。**
- AMD mixed vs 精度匹配GB10 mixed最大logS差1.571899e-6 < 1e-5。
- AMD mixed RMSE0.573633402294、R²0.930466849858；GB10匹配参考RMSE0.573633385521、R²0.930466853925。
- AMD mixed累计设备计算5.405542秒，进程含加载/trace10.816014秒。此轮的实际耗时单列，不替换前一轮全FP64时账。
- 两种模式没有预测校准、label输入或输出修饰。模型权重/测试分子不变，差异来自官方Gaussian浮点路径与跨GPU float math实现。

新结果在`results/module13_mixed_precision_results.json`，主JSON增加 `official_fp32_compatibility`，包含pass、官方/匹配参考误差、112预测数组、精度定义和运行时。`optional_FP32_Gaussian_full15layer`保留完整逐层mixed对账。原FP64及旧BF16仍保留。

`--stage mixed`只运行这一额外路径与重绘；默认all/execute在原FP64之后也串行执行mixed，完整复现两个成功条件。图12末尾补充官方FP32目标的真实PASS结果，PNG再次目检。
