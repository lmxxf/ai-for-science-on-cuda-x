# cuSpa 原核移植：CUDA/CUB → HIP/hipCUB

2026-11-23，NVIDIA 原仓 commit `2df9fa78a7ac23993695807a27d97b025a2d359e`。内核继承 NVIDIA Apache-2.0，已保留各文件版权头，并附原 `LICENSE`。

## 已验证结果

Windows 11 / AMD RX 9070 XT gfx1201，现有 TheRock ROCm 工具链，原版几何/网格内核机械转换并通过 hipCUB 执行排序、扫描、归约和游程计数。不是CPU fallback、不是自造简化核。

固定相同二进制输入：50,000细胞八边形、2,000,000转录点、500基因，所有坐标float64，无降精度近似。**这里的细胞与点位置是固定种子合成负载，不是实验生物数据。** 命中793,204点，CSR非零780,563。全部2百万assignment与完整CSR的indptr/indices/data，均与GB10上官方cuspa Python入口参考**逐字节全等、SHA256全等**。见 `output_comparison.json`。

AMD同步host wall稳态中位数（3次完整预热、7次采样）：

| 阶段 | 毫秒 |
|---|---:|
| 原核网格索引 | 2.3869 |
| 原核射线查询 | 0.9207 |
| 原核hipCUB CSR聚合 | 3.7683 |
| 整体设备驻留流程 | 6.0502 |
| H2D（另测） | 11.6143 |
| D2H（另测） | 1.5167 |
| 首次冷启动流程 | 34.1987 |

原始wall和GPU event数组：`../results/module5_amd9070/timings.json`。事件耗时比host wall低，二者均保留；不混用口径。整体中位数不能用独立阶段中位数简单相加替代。H2D/D2H是单次传输测量，不是传输稳态中位数。

**同源C++ harness** 本机 GB10设备驻留5.759897ms，对AMD6.0502ms；这是接近的结果。旧官方Python入口GB109.636ms有不同接口/分配/校验开销，不能据此宣称AMD硬件比GB10快1.59倍。CPU272.274ms为先前单线程Numba网格射线+NumPy计数基线，不是CPU多线程上限。

## 文件清单与改动边界

机械转换的五个上游头文件：

- `common/geometry.cuh`：四ULP共线/边界语义、逐环射线判断；
- `common/cuda_utils.cuh`：设备暂存内存、异步分配释放、检查宏；
- `index/kernels_index.cuh`：AABB归约、候选网格计数、发射、排序、直方图扫描；
- `query/kernels_query.cuh`：assignment/overlap射线核；
- `aggregate/kernels_aggregate.cuh`：64位pair键、排序、游程计数、CSR与batch merge辅助。

`hipify_headers.py` 在本目录生成副本，**不改上游cuspa**。只替换 runtime头、`cudaXxx→hipXxx`、`cudaSuccess→hipSuccess`、CUB device includes→hipCUB统一头、`cub::→hipcub::`，及错误信息CUDA→HIP。没有更改几何公式、线程块大小、精度、网格分桶或排序计数算法。逐行变更在 `mechanical_translation.diff`，每份上游与移植文件SHA256在 `translation_manifest.json`。

`cuspa_harness.cpp` 是新写的独立C++输入/输出/计时驱动，不依赖nanobind/CuPy/PyTorch。HIP编译时使用此处转换头；NVCC编译时直接包含原仓CUDA头。保留原内核中的设备分配和标量回传同步。为固定本次负载，输出缓冲预分配N，网格pair容量P×16，并有容量检查；不是对所有任意复杂多边形的完整通用库封装。输入grid参数由官方Python库 `ensure_index` 导出，避免两端浮点参数生成差异；AABB、网格、射线、聚合均在GPU运行。

## 输入与输出接口

INPUT目录：

- `params.txt`：一行 `P N G nx ny origin_x origin_y cell_size_x cell_size_y`，浮点17有效位。
- `polys.bin`：little-endian float64，C contiguous `(P,9,2)`，8边且首尾闭环。
- `points.bin`：little-endian float64 `(N,2)`。
- `genes.bin`：int32 `(N,)`。
- `part.bin`、`ring.bin`：int32 `(P+1,)`，GeoArrow风格ring分段，part每polygon一个ring，ring步长9。
- `header.json`：供外部导出/校验的shape/dtype/hash规范，harness不自行解释JSON。

OUTPUT目录：`assignment.bin` int32(N)、`indptr.bin` int32(P+1)、`indices.bin` int32(nnz)、`data.bin` int32(nnz)，全部canonical CSR；`timings.json`同时记录host wall与GPU events，3warmups+7repeats。

## 复现

Linux（生成转换头）：

```bash
python3 amd_hip_port/hipify_headers.py
```

远端路径独立为 `C:\AI4S-HIP-Lab`。原有 `C:\Strata-Lab` 只读使用编译器/SDK/VS文件，无安装、改配置或写回原仓。`build_windows.cmd` 指向已安装 SDK 与现有MSVC；换机器需改这两个路径。

```bat
C:\AI4S-HIP-Lab\amd_hip_port\build_windows.cmd
```

运行前应先查用户进程及游戏本体，**不杀任何进程**。本次主线程全面检查进程后确认无游戏本体，子代理启动前再次列出可见窗口；Steam与FLiNGTrainer后台不作为游戏本体证据。短跑完成即退出。

```bat
C:\AI4S-HIP-Lab\amd_hip_port\run_windows.cmd
```

脚本等价于设置现有ROCm的DLL搜索路径后执行：

```bat
cuspa_hip.exe ..\data ..\out_amd 7
```

Linux GB10同源harness：

```bash
nvcc -x cu -std=c++17 -O3 --fmad=false -arch=sm_121 amd_hip_port/cuspa_harness.cpp -o results/module5_build/cuspa_gb10
results/module5_build/cuspa_gb10 results/module5_data results/module5_gb10_cpp 7
```

WindowsClang选项 `-x hip --offload-arch=gfx1201 -std=c++17 -O3 -ffp-contract=off`。无fast-math，无float32降精度。现有编译器携hipCUB/rocPRIM，**首次编译直接通过**，仅有忽略析构/事件释放返回值的nodiscard警告（和原实现的释放风格一致），见 `build_windows.log`。harness输入输出日志 `run_windows.log`。

此结果只验证这组非重叠八边形与随机点、默认strict contains语义，不替代对孔洞/重叠/边界/极端多边形的跨厂商完整回归测试；源码保留相关逻辑，本次并未覆盖全部路径。
