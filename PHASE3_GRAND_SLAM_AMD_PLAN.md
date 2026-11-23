# Phase 3 终极满贯实验图纸：把 NVIDIA SC26 `CUDA-X` 推文四大科学领域（黑洞 `cuPhoton`、材料量子力学 `Daviator`、分子动力学 `cuPotential PME + 融合积分器`、生物序列 `GPU-HMMER`）及深势 15 层 `Uni-Mol` 全网推理 100% 搬上 `AMD RX 9070 XT (gfx1201 HIP)`！

**主工作目录**：`/home/lmxxf/work/ai-for-science-on-cuda-x`  
**AMD 独立同步仓**：`/home/lmxxf/work/ai-for-science-on-amd-9070xt`  
**远端 AMD 靶机**：`ssh amd9070`（`C:\AI4S-HIP-Lab\`，Windows 11 + ROCm 10.2 `gfx1201` `clang++.exe` HIP 工具链，内含 `hipblas`、`hipfft / rocfft`、`hipcub / rocprim`）  
**总目标**：Zero 要去洗澡和打游戏了，点名给你多派重活！我们要把英伟达 SC26 推文《NVIDIA CUDA-X 库为智能体提供加速科学发现的工具》里提到的**其余所有模块**，全部在 `NVIDIA DGX Spark (GB10 CUDA)` 和 `AMD Radeon RX 9070 XT (gfx1201 HIP)` 上写出同源双后端真机实现、跑出 CPU vs GB10 vs RX 9070 XT 三方基准与逐位/高精度物理守恒对账，并画出 `fig8`～`fig12` 五张纯白底中文可视化大图！

---

## 模块九：`cuPotential` 完整闭环 —— 倒易空间 `PME (3D-FFT)` 长程静电 + 融合朗之万热浴积分器（`Fused Langevin Integrator + SETTLE/Velocity-Verlet`）跑出 1,000 步真实分子动力学轨迹（`exp7_pme_and_fused_md_hip.py` & `amd_hip_port/cupotential_pme_md/`）

在模块七中我们实现了实空间 27 邻格 Cell-List + LJ 12-6 + 反应场库仑。现在补齐 `cuPotential` 的另外两大核心组件：
1. **倒易空间 PME（Smooth Particle Mesh Ewald）长程静电算子（`CUDA cuFFT` vs `AMD hipFFT / 自写 3D-FFT`）**：
   - **Step 1（B-Spline 电荷网格铺展 `Charge Spreading`）**：将三维周期盒子中 $N$ 个原子的点电荷 $q_i$ 用 4 阶 B-Spline 插值铺展到 $64 \times 64 \times 64$ 均匀三维网格 $\rho(g_x, g_y, g_z)$ 上；
   - **Step 2（三维正反傅里叶变换 + 倒易空间高斯核卷积 `K-Space Green's Function`）**：对电荷网格做 3D R2C FFT $\hat{\rho}(\vec{k})$，在频域乘以 Ewald 倒易格林函数 $\frac{4\pi}{k^2}\exp(-k^2 / 4\alpha^2)$ 算出倒易静电势能 $E_{\text{recip}}$ 与频域电场，再做 3D C2R IFFT；
   - **Step 3（电场插值求原子长程静电力 `Force Interpolation`）**：将网格电场插值回每个原子位置得到长程库仑力 $\vec{F}_{i,\text{recip}}$，并与 Ewald 直接求和高精度参考对账！
2. **融合积分器与 1,000 步真分子动力学轨迹（参考 `cuPotential/plugins/openmm/` 的 Fused Integrator）**：
   - 将实空间非键力 + 谐振键长/键角弹簧力 + **Velocity-Verlet（NVE 微正则系综，严格检验总能量 $E_{\text{kin}} + E_{\text{pot}}$ 守恒漂移！）** 与 **BAOAB 朗之万热浴积分器（NVT 恒温系综，300K 温度涨落统计）** 融合在 GPU 设备驻留循环中，不回传 CPU 连续跑 **1,000 步 MD 轨迹**！
   - 对比 `CPU` vs `NVIDIA GB10` vs `AMD RX 9070 XT` 跑 1,000 步 MD 的墙钟耗时、能量守恒漂移率 $\Delta E / E$，并生成 **`visuals/fig8_pme_kspace_and_1000step_md_trajectory.svg`**（展示 PME 电荷铺展频域图 + NVE 1000 步动能/势能反相振荡与总能量守恒平线 + 三平台耗时对比）。

---

## 模块十：材料科学 `VASP Daviator` 原型复现 —— 批处理平面波量子力学 `Batched Davidson Eigensolver`（`CUDA` vs `AMD RX 9070 XT HIP`，`exp8_daviator_batched_davidson_hip.py` & `amd_hip_port/daviator_batched_dft/`）

英伟达推文第三节吹嘘：*“Daviator 是一组专为 VASP 打造的批处理 GPU 内核……将量子力学计算速度最高提升 22 倍，包括此前在 CPU 上运行速度更快的小型系统。”*
为什么小型原子系统（单 k 点平面波基组维数 $N_{\text{pw}} \approx 256 \sim 512$，求前 $N_b = 16$ 条能带本征值 $H\psi_n = \varepsilon_n\psi_n$）以前在 GPU 上比 CPU 还慢？因为原版按单个 k 点串行调几十次小矩阵 GEMM 和小 3D-FFT，全卡在 Kernel Launch 开销上！
1. **我们直接用 `CUDA (GB10)` 和 `AMD HIP (RX 9070 XT)` 实现 `Daviator` 的核心——`Batched Block-Davidson 平面波哈密顿量本征值求解器`**：
   - 构造 $B = 64$ 个独立小型晶胞/k 点（每个系统在三维平面波网格 $16 \times 16 \times 16$ 或紧束缚/平面波截断维数 $N_{\text{pw}} = 384$ 上具有动能对角项 $\frac{1}{2}|\vec{G}+\vec{k}|^2$ + 实空间局域周期晶格势 $V(\vec{r})$）；
   - 实现 **两条执行路径正面 PK**：
     - **旧路径（`Sequential Unbatched`，模拟传统 VASP 单 k 点串行）**：对 64 个 k 点逐个循环发射 Rayleigh-Ritz 子空间投影、残差 $R = H\Psi - \Psi E$、对角预条件子 $P = (T - \varepsilon_n I)^{-1} R$ 与正交化；
     - **新路径（`Daviator-Style Batched GPU Kernels`）**：将 64 个 k 点的波函数打包为三维张量 $[B, N_{\text{pw}}, N_b]$，使用 **Batched 算子 / 融合 Block-Davidson 核函数** 一次性并发迭代 15 步收敛到机器精度（残差 $\|H\psi - \varepsilon\psi\|_2 < 10^{-7}$）！
2. **三方真机对决（CPU vs `GB10 Sequential/Batched` vs `AMD RX 9070 XT Sequential/Batched`）**：
   - 实测为什么 `Unbatched` 在 GPU 上极慢、而改成 `Batched` 后在 `GB10` 和 `AMD RX 9070 XT` 上直接暴涨十几到二十几倍（复现老黄“提速 22 倍”的物理机制！），并验证 `GB10` 与 `RX 9070 XT` 解出的 $64 \times 16 = 1,024$ 个量子能带本征值 $\varepsilon_{k,n}$ 与 `scipy/numpy.linalg.eigh` 精确解的最大绝对误差 `< 10^{-10}`！
   - 生成 **`visuals/fig9_daviator_batched_davidson_vasp_band.svg`**（展示硅/周期晶胞能带本征值收敛曲线 + `Unbatched vs Batched` 在 `GB10` 与 `RX 9070 XT` 上的二十倍加速对比）。

---

## 模块十一：天体物理 `cuPhoton.xBeam` 原型复现 —— 事件视界望远镜（EHT）黑洞干涉阵列 FX 交叉关联与洁化成像（`CUDA` vs `AMD RX 9070 XT HIP`，`exp9_cuphoton_blackhole_fx_hip.py` & `amd_hip_port/cuphoton_xbeam_fx/`）

英伟达推文第一节吹嘘：*“借助 cuPhoton.xBeam 模块，事件视界望远镜（EHT）的黑洞观测数据关联处理速度实现了数量级的飞跃。”*
射电天文学里拍黑洞照片的 **FX 相关器（FX Correlator）+ 傅里叶逆变换孔径综合成像（Aperture Synthesis & Högbom CLEAN）** 在算什么？
1. **我们用 `CUDA (GB10)` 和 `AMD HIP (RX 9070 XT)` 真机实现完整的 `cuPhoton.xBeam` 射电干涉流水线**：
   - 构造一个真实的 **M87* / Sgr A* 非对称光子环（Black-Hole Asymmetric Photon Ring）** 天空亮度分布，模拟全球 $N_{\text{ant}} = 16$ 座射电望远镜（共 $N_{\text{ant}}(N_{\text{ant}}-1)/2 = 120$ 条基线 Baselines）在地球自转孔径综合下的高频电压采样信号 $s_a(t)$（例如 $N_{\text{frames}} = 4,096$ 时间帧 × 每帧 $N_{\text{chan}} = 1,024$ 采样点 × 双极化/多子带）；
   - **F 阶段（Channelizer，批处理 1D FFT）**：在 GPU 上把 16 座天线的高频时域电压流并行变换为频域复数谱 $S_a(f, t)$；
   - **X 阶段（Cross-Correlator，`xBeam` 核心复数外积矩阵乘累加）**：对全部基线对 $(a, b)$ 在时间窗口上并行累加复数可见度函数（Visibilities）$V_{ab}(f) = \sum_t S_a(f, t) S_b^*(f, t)$，再经 $(u, v)$ 平面网格化（Gridding）与 2D IFFT + Högbom CLEAN 去卷积，**直接从合成射电电压信号里重建出清晰标的黑洞光子环图像**！
2. **三方真机对决（CPU vs `NVIDIA GB10` vs `AMD RX 9070 XT`）**：
   - 对比三平台处理数千万采样点 FX 交叉关联的毫秒耗时，校验 `GB10` 与 `RX 9070 XT` 重建出的复数可见度矩阵与黑洞光子环图像 PSNR / 相对 L2 误差（`< 10^{-12}` in `float64`）；
   - 生成 **`visuals/fig10_cuphoton_eht_blackhole_correlator.svg`**（展示 16 台望远镜 $(u,v)$ 覆盖轨迹 $\rightarrow$ 脏图 Dirty Image $\rightarrow$ `CUDA/HIP` 重建出的清晰黑洞光子环对比 + 三方耗时条形图）。

---

## 模块十二：生物信息学 `GPU-HMMER` 原型复现 —— 100,000 条蛋白质序列的 Profile-HMM 维特比（Viterbi / Forward）GPU 并行搜索（`CUDA` vs `AMD RX 9070 XT HIP`，`exp10_gpuhmmer_viterbi_hip.py` & `amd_hip_port/gpuhmmer_viterbi/`）

英伟达推文第四节末尾提到：*“开源的 NVIDIA GPU 加速 HMMER 库可加速预测蛋白质序列的注释工作，为生物学基础模型的数据准备提供支持。”*
1. **我们用 `CUDA (GB10)` 和 `AMD HIP (RX 9070 XT)` 实现 Profile-HMM（隐马尔可夫轮廓模型，含 `Match M / Insert I / Delete D` 三状态转移与 20 种氨基酸发射对数几率得分）的 GPU 大规模并行 Viterbi 对数几率打分算子**：
   - 以模块八的 **56 氨基酸 `GB1` 免疫球蛋白结合域（Immunoglobulin-binding B1 domain）** 构建真实的 `L = 56` 位点 Profile-HMM 模型（从真实多序列/突变集统计 20 种氨基酸位点特异打分矩阵 PSSM 与 `M/I/D` 状态转移对数概率）；
   - 生成包含 **`100,000` 条蛋白质序列（每条长 `128 ~ 256` 氨基酸，其中混入 500 条含不同突变度 GB1 同源结构域的阳性序列与 99,500 条随机/诱饵背景蛋白质序列）** 的大规模待注释数据库；
   - 在 `CPU`、`NVIDIA GB10 (CUDA)` 与 `AMD RX 9070 XT (HIP)` 上并行跑完 100,000 条蛋白质序列的 Profile-HMM Viterbi 动态规划扫描（每秒扫描百万残基级别），验证 **100,000 条序列的 Viterbi bit-score 在 `GB10` 与 `RX 9070 XT` 上 100% 零误差（并达到接近 1.000 的同源域检出 ROC-AUC）**！
   - 生成 **`visuals/fig11_gpuhmmer_100k_protein_viterbi.svg`**（展示 Profile-HMM 的 `M/I/D` 状态格网 + 10 万条序列阳性同源域与背景得分分布双峰图 + `CPU vs GB10 vs RX 9070 XT` 扫描吞吐对比）。

---

## 模块十三：把微调好的深势 **全 15 层 `47.3M` 参数 `Uni-Mol`** 完整搬上 `AMD RX 9070 XT (gfx1201 HIP)` 跑通端到端测试集推理！（`exp11_unimol_full15layer_amd9070.py` & `amd_hip_port/unimol_full15_hip/`）

在模块六中我们只在 `AMD RX 9070 XT` 上跑通了 `Uni-Mol` 的首层高斯距离核。现在干票大的：
1. **把我们在模块二微调好的 `Uni-Mol` `ESOL` 最佳模型（全部 15 层 Transformer Encoder + 128 高斯核 + 逐层 `pair_repr` 传播更新 + Pre/Post LayerNorm + FFN `512→2048→512` GELU + 2 层 `tanh` 回归预测头，共 `47,331,139` 个真实权重参数）全部导出并在 `AMD RX 9070 XT (gfx1201)` 上用原生 HIP C++ / `hipBLAS` 完整跑通全部 112 个 `ESOL` 留出测试药物分子的端到端 15 层前向推理！**
2. **终极闭环对账**：
   - 对比同一份 112 个测试药物分子在 **`NVIDIA DGX Spark (GB10 PyTorch / CUDA)`** 与 **`AMD RX 9070 XT (原生 HIP C++ 全 15 层)`** 上预测出的 112 个水溶性 `logS` 数值：
     - 验证 `AMD RX 9070 XT` 跑出的测试集 `RMSE` 与 `R²` 跟 `DGX Spark` 完全一致（`RMSE = 0.5738, R² = 0.9304`，112 个分子预测值的最大绝对差 `< 10^{-5}`）！
   - 生成 **`visuals/fig12_unimol_full15layer_end2end_amd9070.svg`**（展示全 15 层 `47.3M` 参数在 `AMD RX 9070 XT` vs `NVIDIA GB10` 上对 112 个测试药物分子的端到端预测逐点重合图、层间残差误差曲线与推理耗时）。

---

## 交付与同步纪律
1. 每完成一个模块，立即将数据写入 `results/`、画好白底 SVG 并转 PNG 目检；
2. 全部 5 个模块跑通后，更新 `REPORT.md`、`DevHistory.md`、`run_pipeline.py`（支持 `--phase 3`）与 `verify_phase3.py`；
3. 将全部新增的 AMD HIP 算子源码、脚本、JSON 证据与 `fig8..fig12` 同步至 `/home/lmxxf/work/ai-for-science-on-amd-9070xt`，本地 commit 两个仓库并汇报！
