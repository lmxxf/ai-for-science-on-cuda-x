# 模块十二：真实 GB1 Profile + 十万条序列 Viterbi

2026-11-24，源 `exp10_gpuhmmer_viterbi_hip.py`、`amd_hip_port/gpuhmmer_viterbi/viterbi.cpp`。原创MIT教育原型，不是官方HMMER/GPU-HMMER源码或完整校准流水线。

## 真实模型与明确的合成数据库

- 发射模型来自模块八的真实FLIP GB1 sampled原表8733条实验序列的前56AA域；PSSM按20AA逐位置频次、每AA 0.5伪计数和0.05背景转成log2 odds。原CSV SHA保存在 `results/module12_input.json`。
- 数据无gap，只能从数据观测M→M。MM由观测计数和20强度伪计数得到，MI/MD与I/D延续显式使用先验；没有编造插入/删除实验频数。M行、I行、D行概率分别归一。
- seed20261123生成100000条128–256AA背景（共19179665个残基），均匀随机AA；无放回选500条植入随机真实GB1突变域，再施加0–11处随机点扰动，剩99500为诱饵。这不是100000个真实已注释蛋白，也不是未知蛋白家族泛化基准。
- 采用完整56位点glocal域搜索：可在序列任意位置重启，只从M1开始，必须到M56或D56结束。Insert发射与背景相同，log-odds为0；无HMMER MSV/SSV过滤、背景校正、E-value或Forward算法。begin→M1复用MM这个全局常数，分数有极小共同偏置，所有后端和独立参考一致；不能将原型分数冒充官方HMMER bit-score。

## 独立正确性

- GPU按序列并行，一线程完整扫描一个序列；M/I/D三组就地滚动数组，FP64只用add/max，禁FMA contraction，无fastmath。
- 独立Python实现使用完整二维DP表，而非抄就地滚动更新，随机阴性/植入阳性共6条逐分数完全相同。完整独立表证据 `results/module12_reference_validation.json`。
- 所有100000条CPU/CUDA/HIP输出逐值、逐字节一致，SHA256均为 `aaf96d109052bc6d5e83077f67d5b3289269b8086a2198dc86e290bc57ccb429`，完整分数与标签压缩保存在 `results/module12_all_scores.npz`。
- 植入检出ROC-AUC三方均1.0。该简单阳性/背景构造本就强可分，不把它当真实注释效果。

## 真机计时

同源C++、同一份little-endian二进制输入，单次冷启动后3次稳态，取同步墙钟中位数，传入/传出另列JSON。

| 平台 | 100k序列 / ms | 百万残基/s |
|---|---:|---:|
| DGX Spark单线程C++ CPU | 1136.210 | 16.880 |
| NVIDIA GB10 CUDA | 234.400 | 81.825 |
| AMD RX9070XT HIP | 60.701 | 315.969 |

具体长度总数以JSON为准。不是CPU并行上限、不是GPU-HMMER官方内核性能，硬件/OS/编译器不同。

## 编译与执行

`--stage prepare`下载/复用真实GB1表、生成固定输入、CPU独立参考与100k基准；`--stage gpu`编译CUDA并用SSH在 `C:\AI4S-HIP-Lab\gpuhmmer_viterbi` 编译HIP、逐值对账和出图；默认all完整顺序。

```bash
.venv/bin/python guard_process.py --audit logs/module12_prepare_memory.json -- .venv/bin/python exp10_gpuhmmer_viterbi_hip.py --stage prepare
.venv/bin/python guard_process.py --audit logs/module12_gpu_memory.json -- .venv/bin/python exp10_gpuhmmer_viterbi_hip.py --stage gpu
```

本次GPU等模块十一完全释放后执行，模块十二结束才放行模块十三。HIP工具链借用现成Strata SDK与vcvars64，不写Strata源树。

首次NVCC编译发现三处host驱动命名问题：CUDA属性类型应为cudaDeviceProp、CHECK宏局部e与设备emission缓冲变量冲突、ofstream s与设备sequence缓冲同名。已修正命名后真机通过，早期失败记录在工程上下文；最终编译日志与全部原始运行证据落盘。

`visuals/fig11_gpuhmmer_100k_protein_viterbi.svg`是640px文泉驿中文白底，PNG已逐图目检。两类直方图各归一到自己峰值，不能从图高误读类比例。
