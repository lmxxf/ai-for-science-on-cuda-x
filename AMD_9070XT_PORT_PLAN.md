# 模块五：将 NVIDIA SC26 `CUDA-X` 科学库 `cuSpa` (2,378 行 CUDA) 移植到 `AMD RX 9070 XT` (`gfx1201` HIP) 真机实测图纸

**目标机器**：`ssh amd9070`（`192.168.31.44`，Windows 11，AMD Radeon RX 9070 XT `gfx1201` RDNA4 16GB）  
**现成 HIP / ROCm 工具链路径（已验证可用）**：
- `ROCM_VENV = C:\Strata-Lab\Strata\.rocm-win`
- `ROCM_ROOT = C:\Strata-Lab\Strata\.rocm-win\Lib\site-packages\_rocm_sdk_devel`（可通过 `"C:\Strata-Lab\Strata\.rocm-win\Scripts\rocm-sdk.exe" path --root` 获取）
- 参考编译脚本：`C:\Strata-Lab\Strata-speed\tools\hip\build_windows.bat` 与 `C:\Strata-Lab\build-speed.cmd`（使用 Visual Studio `vcvars64.bat` + ROCm 的 `clang++.exe -x hip --offload-arch=gfx1201` 及 `hipcub / rocprim`）
- **安全规矩**：在 `amd9070` 上跑 GPU 前先查有无游戏进程；在 `amd9070` 上建立工作目录 `C:\AI4S-HIP-Lab\`，不要污染 `C:\Strata-Lab` 原有文件。

---

## 执行步骤（闇执行清单）

1. **将 `NVIDIA/cuspa` 的核心 CUDA 算子（`geometry.cuh`、`kernels_index.cuh / index.cu`、`kernels_query.cuh / query.cu`、`kernels_aggregate.cuh / aggregate.cu`）移植为 AMD HIP（`amd_hip_port/`）**：
   - 替换 `<cuda_runtime.h>` $\rightarrow$ `<hip/hip_runtime.h>`，`<cub/cub.cuh>` $\rightarrow$ `<hipcub/hipcub.hpp>`（或 `rocprim`，若 Windows `_rocm_sdk_devel` 下有 `hipcub` 直接用 `hipcub::DeviceRadixSort` / `DeviceRunLengthEncode` / `DeviceScan`；若无 `hipcub` 头文件则直接用 `thrust` 或几行纯 HIP parallel radix/bitonic sort + scan kernel 实现等价的 `(cell_id, gene_id)` 64-bit 排序与游程编码 CSR 聚合，确保 100% 在 GPU `gfx1201` 上原生执行）；
   - 将 `cudaMalloc / cudaMemcpy / cudaEventElapsedTime / cudaStream_t` 等映射为 `hipMalloc / hipMemcpy / hipEventElapsedTime / hipStream_t`（可用一个干净的 `cuda_to_hip.h` 宏头文件，保持原 `cuspa` 核函数几何与索引算法逐行原汁原味！）。
2. **在 `amd9070`（`AMD RX 9070 XT, gfx1201`）上编译并运行完全同考卷的 `50,000 细胞 × 2,000,000 基因点 × 500 基因`（固定种子 `20261123`，`float64`）基准测试**：
   - 测量 `AMD RX 9070 XT` 上三个阶段（① 网格包围盒索引 `Grid Index`、② 水平射线法 `Ray-Crossing Query`、③ 稀疏矩阵聚合 `CSR Aggregate` 及设备驻留全流程）的稳态中位数毫秒耗时（以及含 H2D 传输耗时）；
   - **严格逐位对账**：把 `AMD RX 9070 XT` 算出的 2,000,000 个点的 `cell_id` 归属与 `50,000 × 500` CSR 稀疏计数矩阵哈希值，跟 `DGX Spark (NVIDIA GB10)` 的输出做比对，确认 **100% 零差异（0 mismatch）**！
3. **（加分项）顺手把第二步分子 3D 距离矩阵 + Lennard-Jones / 库仑非键对势能（`cuPotential` 核心计算的分子非键相互作用 `E_LJ + E_Coulomb`）也写一个 CUDA vs AMD HIP (`gfx1201`) 同源算子基准**：
   - 对 100,000 对/百万原子对的 3D 非键势能与受力计算，实测 CPU vs `DGX Spark (GB10)` vs `AMD RX 9070 XT (gfx1201)`，直观证明“第二步古典 CUDA Core 物理几何算子在 AMD RDNA4 上同样快如闪电”！
4. **落盘产物与更新图表**：
   - 将 AMD 移植源码放入 `/home/lmxxf/work/ai-for-science-on-cuda-x/amd_hip_port/`；
   - 将实测 JSON 存入 `results/module5_amd9070_results.json`；
   - 更新 `visuals/fig4_cuspa_cell_polygon_raycasting.svg`（及 `.png`）：在柱状图里直接并排展示 **`CPU (272.27 ms)` vs `NVIDIA DGX Spark GB10 (9.64 ms)` vs `AMD RX 9070 XT HIP (实测 ms)`** 的三方真机对比！
   - 更新 `REPORT.md` 与 `DevHistory.md`。
