# 模块十：原始双后端 Block-Davidson 教学原型

## 真实问题与原型范围

这是从零编写的MIT教学实现，不是NVIDIA Daviator源码、不调用其闭源二进制，也不是VASP自洽密度泛函计算或真实硅晶体。64个k点，每个384个截断平面波，求最低16个本征对。哈密顿量由动能`0.5|G+k|²`和实空间周期势`V(r)=-0.07[cos(x)+cos(y)+cos(z)]`构成，倒空间近邻耦合-0.035，无量纲单位。它是周期势中的单粒子薛定谔本征问题，不能等同材料性质预测。

`results/module10_data/H.bin`保存完整64×384×384 FP64矩阵，SHA256在`metadata.json`。G向量和k点单独保存。独立NumPy `eigvalsh`对全部64个完整矩阵求解，不把参考答案传入迭代器。初始子空间由动能最低的24个坐标基向量构成。

## 真正执行的算法

Rayleigh-Ritz投影 → 小子空间Jacobi对角化 → Ritz向量重构 → 残差 → 对角能量差预条件（分母幅度下限0.08）→ 两遍修正Gram-Schmidt → 扩张子空间；超过96维则从16个Ritz向量重启。每轮检查全部64×16个本征对最大残差，小于1e-7才停止，本次实际9轮，不预设15轮或22倍。

CUDA和HIP同源C++中，**GPU真实计算H·V、V^T HV投影、V*C/HV*C重构、残差与对角预条件**。顺序路径每阶段发射64个系统的独立kernel；批处理路径每阶段一个覆盖全部系统的kernel。没有把CPU预解答案复制到GPU冒充计算。

小子空间Jacobi eigensolve与两遍MGS仍在CPU，波函数/投影矩阵/预条件残差每轮传输，已计入端到端和独立transfer时间。不是全设备驻留、不是cuBLAS/hipBLAS调优库版本；此处自写FP64求和核用于解释批处理调度机制。结果JSON历史字段`H_apply_seconds`现在包含所有GPU核心阶段，不仅H·V，速比字段命名`GPU_core_only`澄清这一点。

## 真机结果

全部5条路径（CPU、GB10 sequential/batch、RX9070XT sequential/batch）都真实求解完整64系统，9轮收敛。独立NumPy参考本征值最大误差<=4.73e-14，直接用完整H重算的最大残差约3.524e-8，正交性最大误差4.35e-14；满足1e-10本征值与1e-7残差验收。

| 路径 | 端到端秒 | GPU/CPU H核心秒 | CPU子空间/正交秒 | 传输秒 |
|---|---:|---:|---:|---:|
| 教学C++ CPU | 3.231638 | 2.082833 | 1.090179 | 0 |
| GB10顺序 | 0.856452 | 0.103269 | 0.690524 | 0.004649 |
| GB10批处理 | 0.811427 | 0.058208 | 0.690507 | 0.004600 |
| RX9070XT顺序 | 1.195314 | 0.235031 | 0.866810 | 0.035863 |
| RX9070XT批处理 | 1.040538 | 0.083876 | 0.867562 | 0.032025 |

GB10批处理核心1.774倍、端到端1.055倍；RX9070XT批处理核心2.802倍、端到端1.149倍。CPU剩余工作压住了端到端收益，不能写复现22倍。

**独立NumPy全矩阵eigvalsh参考只用0.231秒，比本原型所有GPU混合求解更快。** 教学C++ CPU三重循环+自写Jacobi不是优化过的CPU线性代数库天花板，不能用其3.23秒去宣称GPU普遍胜CPU。GB10与Windows AMD主机CPU不同，绝对端到端时间还混有主机性能差异，跨GPU对比应优先同机seq/batch而不是把两机比值全归于GPU。

## 文件与复现

入口`exp8_daviator_batched_davidson_hip.py`，同源C++ `amd_hip_port/daviator_batched_dft/davidson.cpp`。Windows构建最初错误使用用户文本中的假设ROCm路径，修正为真实Strata `_rocm_sdk_devel`和vcvars64；成功build/seq/batch日志保留在源码目录。

- `results/module10_results.json`：全部指标与明确范围。
- `results/module10_*_history.csv`：逐轮全系统最大残差。
- `results/module10_*_eigenvalues.bin`、`eigenvectors.bin`：完整1024本征值及波函数。
- `visuals/fig9_daviator_batched_davidson_vasp_band.svg`及PNG：白底640px文泉驿，已目检。

一键默认真实执行CPU准备/独立参考/五路径求解/图表；支持`--prepare-only --gpu-only --summarize-only`分阶段。重运行进程由主入口和内部guard包裹，20GiB阈值、250ms看护。
