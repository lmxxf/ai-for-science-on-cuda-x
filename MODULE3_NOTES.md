# 模块三：cuSpa 真机编译 / cuPotential 源码审计

2026-11-23 22:35–22:41 (+08:00)。执行目录为本目录；结果入口 `results/module3_results.json`，源码审计明细 `results/module3_source_audit.json`。图为 `visuals/fig4_cuspa_cell_polygon_raycasting.svg`，PNG 已渲染并肉眼自检：白底、中文无缺字、无重叠。

## 真实负载与校验

固定种子 20261123，**程序合成** 50,000 个互不重叠八边形、2,000,000 个均匀坐标点、500 种基因编号。它具有空间转录组量级，但**不是实验采集的肿瘤组织**。其中 793,204 个点落在细胞内，得到 50,000×500 CSR 矩阵，780,563 个非零项。双方 float64，完整比对所有两百万归属、整个 CSR 矩阵，均零差异。

CPU 是单线程 Numba 编译后的**网格 AABB 索引 + 射线法**，并非逐点遍历全部细胞，更非 Python 解释器循环；聚合为 NumPy sort/unique 计数再建 scipy CSR。GPU 为原版 cuspa 实际编译出的索引、查询、聚合三个 CUDA 文件，未重写替代 GPU 核。

| 阶段 | CPU 秒（中位数） | GPU 秒（中位数） |
|---|---:|---:|
| 建索引 | 0.002168 | 0.003277 |
| 射线查询 | 0.236064 | 0.001512 |
| CSR聚合 | 0.034043 | 0.005965 |
| 完整设备驻留流程 | 0.272274（各阶段中位数求和） | 0.009636（流程整体中位数） |

**设备驻留流程 28.26×，仅查询 156.10×。** 输入 CPU→GPU 另计 0.014502 秒；与驻留流程相加约 0.024138 秒、推算约 11.28×，不是另做的端到端计时，且不含将结果拷回 CPU。GPU 第一轮建索引含冷启动 34.82ms，后两轮约 3.2ms，原始数组保留。CPU JIT 1.46s、CUDA 编译、导入和数据生成均不计入稳态；这里没有证明在更复杂多边形/重叠组织/多线程 CPU 上同样加速。

统一内存看护 `logs/module3_memory.json`：250ms 间隔，20GiB 阈值；成功运行最低可用内存 >110GiB，未触发杀进程。Torch 记录的峰值分配116MB**不涵盖原生CUDA直接分配的所有CUB scratch**，总统一内存情况以看护为准。

## 实际编译与接口踩坑

1. `cmake -S cuspa -B results/cuspa_build -DPython_EXECUTABLE=/home/lmxxf/work/interp-curvature-exp/venv/bin/python -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=121 -DCMAKE_LIBRARY_OUTPUT_DIRECTORY="$PWD/cuspa/python/cuspa"`。配置需 `PYTHONPATH="$PWD/results/module3_builddeps"`。`cmake --build results/cuspa_build -j 4`，CUDA13.0.88 / GB10 sm_121 成功；编译日志在 `results/cuspa_build_*.log`。
2. 系统已有 Numba 与基础 NumPy2.5 不兼容。为避免污染主环境，在 `results/module3_builddeps` 隔离安装 nanobind3.1.0、NumPy2.3.x。脚本主动将该目录加到模块搜索路径。
3. README 提到可使用Torch，但高层 `aggregate_to_cells` 实际仍强制 CuPy。我们使用官方 `cuspa.tl.assign_points` 配官方 `cuspa._core.aggregate_to_cells`，Torch 提供 CUDA buffer，仍是相同原生 GPU 算法。
4. 初次 float32 全量校验发现 **16/2,000,000** 归属不同。原核使用四ULP共线判定、严格边界处理，CPU简化射线公式在近边界 float32 上并不等价；这说明边界精度必须验证。未删除/筛掉这16个点，而是统一双方为 float64 后重新生成同种子全部负载，完整零差异。**float32差异的精确逐点根因尚未定位，不能宣称“已证明全部由ULP造成”**，本次展示结论限定为通过严核的 float64。
5. 一键执行：`python guard_process.py --audit logs/module3_memory.json -- python exp3_cudax_cuspa_benchmark.py`。默认写 JSON、preview NPZ，再调用 `render_module3.py` 输出SVG和PNG。

## cuSpa 源码事实：对原图纸的两处修正

commit `2df9fa78a7ac23993695807a27d97b025a2d359e`，70 tracked 文件，11个 C++/CUDA 源头文件共 **2,378行**。这不是“单个2,378行CUDA文件”：3个 `.cu` 合计871行，`.cuh` 1459行，其余为绑定/接口。

- `cpp/src/common/geometry.cuh`：`is_point_in_polygon` 水平射线法，多环与边界语义；
- `cpp/src/index/kernels_index.cuh`：AABB映射网格、`atomicAdd` 网格候选数量、CUB扫描/排序；
- `cpp/src/aggregate/kernels_aggregate.cuh`：**64-bit(细胞ID,基因ID)编码 → CUB radix sort → run-length encode → CSR**。不是每个基因点 `atomicAdd` 到稠密细胞×基因矩阵；
- 源码未检出 `wmma`、`mma`、`tcgen05`、内联 `asm`、`__shfl`。本次主要用通用算术、分支、整数排序和扫描，不依赖 Tensor Core。

**不依赖Tensor Core ≠ 已能直接跑AMD**：CUDA Runtime/streams/alloc、CUDA launches、CUB、nanobind CUDA/DLPack设备检查都仍需 HIP/hipCUB 和绑定适配；RX9070XT 尚未实测。只证明此次源码无上述专用指令路径，不对编译器/闭源CUDA运行时的内部实现做零私有指令承诺。

## cuPotential：不只是“空壳”，但核心运行时确为专有

commit `fbfc4cb6282cd52d33f7adbbcbaeaa5c19b47d74`，233 tracked文件，62个C++/CUDA头/源文件、16,650行。仓库根 `LICENSE` 明确区分：OpenMM插件为Apache2.0；`include/cupo/`、`libcupo.so`、LTO-IR math对象为NVIDIA专有许可。当前clone没有 `libcupo.so`，`fetch_cupo` 在构建时从release下载；wheel会打包运行时，不能说“所有安装都要额外单独下载”。

开源部分确有实质优化：`integrator/CudaLangevinMiddleKernel.cu`、`FusedChains.cuh` 把 BAOAB Langevin步、约束处理（SETTLE/SHAKE）、动量去除等合并/组织到融合执行路径（具体选择受精度与拓扑影响）；`bonded/GroupedBonded.cu` 批处理键相互作用；barostat与非键模块有OpenMM桥接。

`nonbonded/CudaNonbondedKernels.cu` 调用 `cupo_cell_list_create`、`cupo_dspace_create`、`cupo_kspace_create`、`cupo_kspace_configure_convolution` 等，把直接/倒空间非键作用与邻居表工作交给 **libcupo专有运行时**。因此可以审计集成、调度与部分可见融合核，但不能仅靠这个clone完整审计/重建cuPotential核心力计算。本模块只作源码解剖，未声称已安装OpenMM运行cuPotential分子动力学。
