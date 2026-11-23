# Phase 2 满汉全席实验图纸：补齐 `BBBP` 血脑屏障 `Uni-Mol` 分类微调與原子归因、开源版 `cuPotential` 分子力场 (`CUDA vs AMD RX 9070 XT HIP`)、深势 `Uni-Mol` 算子 AMD 移植与蛋白质大模型 `ESM-2` 真机微调

**主工作目录**：`/home/lmxxf/work/ai-for-science-on-cuda-x`  
**AMD 独立同步仓**：`/home/lmxxf/work/ai-for-science-on-amd-9070xt`  
**远端 AMD 靶机**：`ssh amd9070`（`C:\AI4S-HIP-Lab\`，Windows 11 + ROCm 10.2 `gfx1201` `clang++.exe` HIP 工具链；Zero 已去客厅 `5090` 玩游戏，书房这台 `AMD RX 9070 XT` 今晚全权归我们跑实验！）  
**安全红线**：`DGX Spark` 上所有重进程继续包裹 `guard_process.py`（250ms 监控 `MemAvailable < 20GiB` 自动熔断）；所有产出图表必垫 `#ffffff` 纯白底、用 `WenQuanYi Micro Hei` 字体、严禁用 Unicode 上标 `ᵀ`（用 `^T`），并用 `cairosvg` 转 PNG 目检。

---

## 模块六：`BBBP`（血脑屏障穿透）深势 `Uni-Mol` 3D 分类微调 + 原子级归因染色图 + `AMD RX 9070 XT` 跑通 `Uni-Mol` 3D 高斯距离注意力核（`exp4_bbbp_unimol_and_amd.py`）

1. **`BBBP`（2,039 个清洗后合法药物分子，0/1 二分类）真机微调**：
   - 按去重 canonical 母核做 `80 / 10 / 10` 固定种子划分（确保同一母核不跨 train/val/test）；
   - 为全部 `BBBP` 有效分子生成或复用 `RDKit ETKDGv3 + MMFF94` 3D 构象缓存；
   - 同考卷对比三名选手：
     - ① 多数类常数/逻辑回归基线；
     - ② `RDKit` 2048 维 Morgan 2D 指纹 + `RandomForestClassifier (400 trees)`；
     - ③ **深势官方 `Uni-Mol v1`（`47.3M` 参数）全参数二分类微调**（`BCEWithLogitsLoss`，20 epochs，仅用 validation `ROC-AUC` 挑选最佳 checkpoint，在 test 上只评一次 `ROC-AUC`、`PR-AUC`、`Accuracy`）。
2. **原子级归因高亮染色（Atomic Attribution on 1st-Gen vs 2nd-Gen Antihistamines）**：
   - 用微调好的 `Uni-Mol` `BBBP` 模型，对 **苯海拉明（一代嗜睡，易穿脑）** 和 **非索非那定（二代少嗜睡，难穿脑）** 做推理，并通过 **逐个重原子遮挡/扰动（Leave-One-Atom-Out / Distance Masking）或注意力权重归因**，算出每个重原子对“穿脑概率 logit”的贡献 $\Delta \text{logit}_i$；
   - 用 `RDKit MolDraw2DSVG` 画出真实的模型原子归因染色图（推高穿脑的亲脂原子染绿/蓝，拉低穿脑的极性 `-OH / -COOH` 原子染红）！
3. **在 `AMD RX 9070 XT (gfx1201 HIP)` 上编译并跑通 `Uni-Mol` 核心的“3D 距离矩阵 $\rightarrow$ 128 高斯核展开 $\rightarrow$ 64 头 Pair-Bias 注意力”前向算子**：
   - 提取 `Uni-Mol` 的真实权重张量（高斯核参数 $\mu, \sigma$、原子对仿射参数 $\gamma, \beta$、非线性投影层 $128 \to 64$ 及第 1 层 QKV 注意力权重），编写同源 CUDA (`GB10`) 与 HIP (`AMD RX 9070 XT`) C++ 核函数，在真实药物分子 Batch 上实测 `GB10` vs `RX 9070 XT` 的执行耗时与 `max_abs_diff` 数值一致性！
4. **生成可视化大图 5**：
   - `visuals/fig5_bbbp_unimol_roc_and_atom_attribution.svg`（含 `BBBP` 测试集 ROC 曲线/AUC 对比 + 苯海拉明与非索非那定的 `Uni-Mol` 真实原子归因高亮分子图 + `Uni-Mol` 3D 高斯距离核在 `GB10 vs RX 9070 XT` 上的验证结果）。

---

## 模块七：手写开源版 `open-cupotential` —— 打破老黄 `libcupo.so` 闭源锁，在 `NVIDIA GB10` 与 `AMD RX 9070 XT (HIP)` 上同台实测 3D 分子非键力场算子（`exp5_open_cupotential_hip.py` & `amd_hip_port/cupotential_open/`）

既然我们在模块三查出老黄的 `NVIDIA/cuPotential` 把最核心的非键相互作用运行时 `libcupo.so`（Cell-List + Direct-Space Lennard-Jones & Electrostatic 势能与受力）闭源了，我们直接亲手写一个 **100% 开源、同时支持 `CUDA (sm_121)` 与 `AMD HIP (gfx1201)` 的 3D 分子非键力场算子包 `open-cupotential`**！

1. **物理与数学实现（真实分子动力学非键相互作用 Nonbonded Forcefield）**：
   - 在周期性边界条件（Periodic Boundary Conditions, PBC，三维盒子 $L_x \times L_y \times L_z$，最小镜像约定 Minimum Image Convention）下，给定 $N$ 个三维原子坐标 $\vec{r}_i = (x_i, y_i, z_i)$、部分电荷 $q_i$、Lennard-Jones 参数 $(\sigma_i, \epsilon_i)$ 及截断半径 $r_{\text{cut}} = 10.0\text{ \AA}$：
   - **Kernel 1（3D 空间网格邻居列表构建 `3D Cell-List Construction`）**：将三维模拟盒子划分为边长 $\ge r_{\text{cut}}$ 的 3D 网格单元，把每个原子分桶并排序，使每个原子只需遍历自身及相邻 $3 \times 3 \times 3 = 27$ 个网格单元内的候选邻居原子（复杂度从 $O(N^2)$ 降至 $O(N)$，正是 `cuPotential` 的 `cupo_cell_list_create` 做的事！）；
   - **Kernel 2（融合非键势能与解析受力梯度核 `Fused Lennard-Jones 12-6 + Reaction-Field Coulomb Energy & Force Kernel`）**：
     对距离 $r_{ij} < r_{\text{cut}}$ 的原子对，采用 Lorentz-Berthelot 混合规则 $\sigma_{ij} = (\sigma_i+\sigma_j)/2, \epsilon_{ij} = \sqrt{\epsilon_i\epsilon_j}$，计算：
     - 范德华势（Lennard-Jones 12-6）：$E_{\text{LJ}}(r_{ij}) = 4\epsilon_{ij}\left[\left(\frac{\sigma_{ij}}{r_{ij}}\right)^{12} - \left(\frac{\sigma_{ij}}{r_{ij}}\right)^6\right]$
     - 反应场库仑静电势（Reaction-Field Electrostatics）：$E_{\text{elec}}(r_{ij}) = k_e q_i q_j \left[\frac{1}{r_{ij}} + k_{\text{rf}} r_{ij}^2 - c_{\text{rf}}\right]$
     - **同时解析求导算出每个原子受到的三维合力向量** $\vec{F}_i = -\sum_j \nabla_{\vec{r}_i} (E_{\text{LJ}} + E_{\text{elec}}) \in \mathbb{R}^3$！
2. **三方真机跑分与精度对账（`N = 32,000` 及 `N = 128,000` 溶剂化原子体系，对应 `cuPotential` README 的 `24k~409k` 真实规模！）**：
   - 对比 **① CPU 编译基线（Numba / C++）** vs **② `NVIDIA DGX Spark (GB10 CUDA)`** vs **③ `AMD Radeon RX 9070 XT (gfx1201 HIP)`**；
   - 记录 Cell-List 构建耗时、Force+Energy 核函数耗时、每秒可模拟步数（Steps/s 或 ns/day），并校验 `GB10` 与 `RX 9070 XT` 算出的总势能 $E_{\text{total}}$ 与全部 $N$ 个原子的三维受力向量 $\vec{F}_i$ 的相对 L2 误差（`< 1e-12` in `float64` / `< 1e-6` in `float32`）！
3. **生成可视化大图 6**：
   - `visuals/fig6_open_cupotential_md_forcefield_amd_vs_nvidia.svg`（展示 3D 周期盒子 `27` 邻格 Cell-List 原理 + `LJ 12-6 & 库仑势` 曲线 + `32K / 128K` 原子体系在 `CPU vs NVIDIA GB10 vs AMD RX 9070 XT` 上的真实耗时与受力精度对账）。

---

## 模块八：补齐“蛋白质专有模型”拼图——科学库 `Biopython` 序列处理 + Meta 蛋白质语言模型 `ESM-2` 真机微调（`exp6_protein_esm2_finetune.py`）

陆昀原话专门提了**“分子，蛋白质等专有模型”**。我们已经跑透了 3D 分子模型 `Uni-Mol`，现在把**蛋白质专有模型 `ESM-2`** 也真机跑通！

1. **真实蛋白质数据集与科学库 `Biopython` 处理**：
   - 安装 `biopython`（以及加载 HuggingFace 官方蛋白质语言模型 `facebook/esm2_t6_8M_UR50D`［7.5M 参数，6 层］或 `facebook/esm2_t12_35M_UR50D`［35M 参数，12 层］）；
   - 使用经典的真实蛋白质实验数据集（如 **GB1 蛋白质结合亲和力/适应度突变数据集**、**GFP 绿色荧光蛋白荧光强度突变数据集** 或 **蛋白质热稳定性 $\Delta T_m$ / 溶解度真实实验数据集**，若网络拉取慢可从 HuggingFace Datasets / ProteinGym / FLIP 标准子集拉取真实实验 CSV 并记录 SHA256）；
   - 用 `Biopython`（`Bio.SeqUtils.ProtParam.ProteinAnalysis`）校验 20 种标准氨基酸字母表、过滤非法非标准残基字符，计算蛋白质分子量、等电点（`pI`）、芳香性、不稳定指数（Instability Index）和 GRAVY 亲疏水指数。
2. **同考卷三方真机 PK**：
   - ① 常数基线；
   - ② 传统生物信息学特征：氨基酸组成（AAC / Dipeptide 400 维频次）+ `Biopython` 理化性质 + `RandomForest / Ridge`；
   - ③ **真机微调蛋白质专有大模型 `ESM-2`**：拆解打印 `ESM-2` 的词表（33 个 Token = 20 种标准氨基酸 `L, A, G, V, S, E, R, T, I, D, P, K, Q, N, F, Y, M, H, W, C` + 特殊符号）与 RoPE 架构，跑 10~15 轮微调（仅用 validation 挑选最佳轮次），记录测试集 `RMSE / Spearman ρ / R²` 或 `ROC-AUC`，并画出 `ESM-2` 对蛋白质关键突变位点（Wild-Type vs Mutant）的注意力或突变效应热力图！
3. **生成可视化大图 7**：
   - `visuals/fig7_protein_esm2_finetune_and_mutation_scan.svg`（展示 `ESM-2` 的 20 氨基酸词表架构 + 同考卷微调对比 + 蛋白质序列突变位点扫描热力图）。

---

## 收尾交付要求
1. 全部完成后更新 `REPORT.md`、`DevHistory.md`、`run_pipeline.py` 及 `skill/ai4s-mol-skill/SKILL.md`；
2. 将新增的 AMD HIP 源码（`amd_hip_port/cupotential_open/` 与 `amd_hip_port/unimol_pairbias/`）同步拷入 `/home/lmxxf/work/ai-for-science-on-amd-9070xt/`，确保两个仓库工作区干净、全部测试 PASS！
