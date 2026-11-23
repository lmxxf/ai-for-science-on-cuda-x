# Phase 4 攻坚图纸：手写开源版 `open-nvMolKit`（10,000 个真实药物 3D 构象 GPU 批量力场松弛 `CUDA vs AMD RX 9070 XT HIP`）、`Daviator` 100% 全 GPU 驻留化升级、以及 `wechat/381.md` 终审

**主工作目录**：`/home/lmxxf/work/ai-for-science-on-cuda-x`  
**AMD 独立同步仓**：`/home/lmxxf/work/ai-for-science-on-amd-9070xt`  
**远端 AMD 靶机**：`ssh amd9070`（`C:\AI4S-HIP-Lab\`，Windows 11 + ROCm 10.2 `gfx1201` `clang++.exe` HIP 工具链）

---

## 模块十四：手写开源版 `open-nvMolKit` —— 攻克第一章 `66.61 秒` 的 CPU `RDKit` 10,000 个 3D 分子构象力场松弛瓶颈（`exp12_open_nvmolkit_batched_conformer_hip.py` & `amd_hip_port/nvmolkit_batched_conformer/`）

在模块一中，我们实测出 AI for Science 分子微调管线里最卡脖子的环节：给 `1,000` 个真实 `ESOL` 药物分子每个生成 `10` 个 3D 构象（共 **`10,000` 个 3D 分子构象**）并做 200 步弹簧力场能量最小化，单核 CPU `RDKit` 耗时高达 **`66.608 秒`**（8 核也需 **`9.987 秒`**），直接反超了 GPU 微调 20 轮 `Uni-Mol` 的 `37.18 秒`！
现在我们直接用 **`CUDA (GB10 sm_121)` 与 `AMD HIP (RX 9070 XT gfx1201)`** 编写一个 **100% 开源、Block-per-Conformer 的批量 3D 分子构象力场优化器 `open-nvMolKit`**：

1. **真实药物分子拓扑与力场参数提取（来自 `ESOL` 前 `1,000` 个真实药物分子，`10,000` 个初始 3D 构象）**：
   - 从 `RDKit` 为这 `1,000` 个真实药物分子提取真实拓扑项（或者直接从 `RDKit MMFFGetMoleculeProperties` / 标准谐振键长项 $E_{\text{bond}} = \sum_{(i,j)\in\text{bonds}} \frac{1}{2}k_{b,ij}(r_{ij} - r_{0,ij})^2$、谐振键角弯曲项 $E_{\text{angle}} = \sum_{(i,j,k)\in\text{angles}} \frac{1}{2}k_{\theta,ijk}(\cos\theta_{ijk} - \cos\theta_{0,ijk})^2$、四原子二面角扭转项 $E_{\text{torsion}}$ 以及非键 `1-4 / >=1-4` 原子对的范德华排斥与库仑项 $E_{\text{nb}}$ 导出每个分子的真实力场参数表與 `10,000` 组带扰动/ETKDG 初始 3D 坐标 `X_init`）；
2. **GPU 批量并行构象能量最小化核函数（`One Thread-Block per Conformer`，`10,000` 个 Block 并发执行 200 步梯度下降 / 带回溯线搜索或动量松弛）**：
   - 每个 Thread-Block 负责 1 个 3D 分子构象，将该构象的重原子坐标 `(x_i, y_i, z_i)` 加载进 **GPU Shared Memory（共享内存）**；
   - Block 内的线程并行计算所有成键项（Bond / Angle / Torsion）与非键对（Nonbonded Pairs）的势能及每个原子的解析三维受力梯度 $\vec{g}_i = \nabla_{\vec{r}_i} E$（带梯度裁剪/线搜索步长控制），**在 Shared Memory 内部直接连续迭代 200 步**，全程 0 次全局显存往返、0 次 CPU 干预！
3. **三方真机对决与物理校验（`1,000` 真实分子 × `10` 构象 = `10,000` 构象 × `200` 步松弛，`float64`）**：
   - **物理与梯度校验**：在小样本分子上对比解析梯度 $\vec{g}_i$ 与 6 方向有限差分（最大绝对误差 `< 1e-8`），确认 10,000 个构象优化后的平均势能大幅下降、键长/空间碰撞显著消除；
   - **跨架构精度与速度对决**：
     - 对比 ① `RDKit` CPU（单核 `66.608 s` / 8 核 `9.987 s`）、② 同源力场 `CPU C++` 编译基线、③ **`NVIDIA DGX Spark (GB10 CUDA)`**、④ **`AMD Radeon RX 9070 XT (gfx1201 HIP)`** 跑完 10,000 个构象×200 步的毫秒级耗时；
     - 校验 `GB10 (CUDA)` 与 `RX 9070 XT (HIP)` 优化完的 10,000 个构象最终能量与全部原子三维坐标 `(x, y, z)` 的一致性（`L2 = 0`）！
   - 生成 **`visuals/fig13_open_nvmolkit_10000_conformers_amd_vs_nvidia.svg`**（展示单个分子构象优化前后能量下降与键长恢复曲线 + 10,000 个构象在 `RDKit CPU vs CPU C++ vs NVIDIA GB10 vs AMD RX 9070 XT` 上的耗时对比）。

---

## 模块十五：将模块十 `VASP Daviator` 批处理量子 `Block-Davidson` 升级为 **100% 全 GPU 驻留版**（消除 CPU 子空间对角化与 MGS 往返！）

在模块十中，虽然我们把 `H·V`、Rayleigh-Ritz 投影、残差和预条件写成了 GPU 批处理核，但小子空间（`32×32`）的 Jacobi 特征值分解与 Modified Gram-Schmidt (MGS) 正交化还留在 CPU 上，导致每轮迭代都要把波函数在 CPU 与 GPU 之间来回拷贝，端到端批处理提速只有 `1.15×`（核内 `2.80×`）。
1. **现在把剩下的两步也 100% 搬进 `CUDA (GB10)` 和 `AMD HIP (RX 9070 XT)` 的 GPU 核函数里**：
   - 在 GPU 上实现 **Batched Modified Gram-Schmidt（两遍 MGS 正交归一化核函数，每个 Block 负责 1 个 k 点系统的 `384 × 32` 波函数列正交化）**；
   - 在 GPU 上实现 **Batched Shared-Memory Jacobi 实对称子空间特征值求解器（每个 Block 在共享内存里对自己的 `32 × 32` 投影哈密顿量跑循环 Jacobi 旋转，直接算出前 16 个最小特征值与 Ritz 特征向量！）**；
   - 这样整整 9 轮 Block-Davidson 迭代 **100% 在 GPU 显存内闭环完成（0 次 PCIe/主机内存往返）**！
2. **重新实测 `64` 个 k 点（以及扩展到 `256` 个 k 点 = `4,096` 个量子本征值）下**：
   - 对比 `CPU` vs `GPU Sequential（逐 k 点分开发射）` vs **`GPU 100% Resident Batched（全驻留批处理）`** 在 `NVIDIA GB10` 和 `AMD RX 9070 XT` 上的真实端到端加速比（看全 GPU 驻留批处理能把加速比从原来的 `1.15×` 拉升到多少倍！），验证全部本征值与 `eigvalsh` 误差 `< 1e-13`；
   - 更新 `visuals/fig9_daviator_batched_davidson_vasp_band.svg`（及 `.png`）与 `results/module10_results.json`。

---

## 模块十六：对上篇公众号 `/home/lmxxf/work/ai-theorys-study/wechat/381.md` 进行终审（只写 `wechat/assets/381/审稿意见.md`，不改 `381.md` 正文）

按照我们的公众号终审规矩：
1. 读取 `/home/lmxxf/work/ai-theorys-study/wechat/381.md` 及 `/home/lmxxf/work/ai-theorys-study/wechat/assets/381/` 下的 5 张 SVG、`/home/lmxxf/work/ai-theorys-study/wechat/cover381.svg`；
2. 逐项核对 `381.md` 中的每一个数字、药物 CID/分子量/LogP/TPSA、硝苯地平原子编号与 `3.453 Å` 距离、`Uni-Mol` 架构参数与 `ESOL / BBBP` 测试集指标、`ESM-2` 架构参数与 `GB1` 测试集指标、所有外部链接与往期微信链接；
3. **不要直接修改 `wechat/381.md`**，把完整核查结论与任何需要微调的建议写入 `/home/lmxxf/work/ai-theorys-study/wechat/assets/381/审稿意见.md`！

---

## 交付与同步
完成后更新 `REPORT.md`、`DevHistory.md`、`run_pipeline.py`、`verify_phase3.py`，将新增的 `amd_hip_port/nvmolkit_batched_conformer/`、升级后的 `daviator_batched_dft/`、`fig9`、`fig13` 同步到 `/home/lmxxf/work/ai-for-science-on-amd-9070xt`，本地 commit 两个实验仓并告诉我！
